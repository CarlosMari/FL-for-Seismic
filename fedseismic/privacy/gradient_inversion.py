"""One-step gradient inversion on a single labeled image.

This is the attack privacy papers run when the server receives a gradient:
match a dummy input's gradient to the true one (Geiping et al., NeurIPS 2020).
The label is given to the attacker, so this is an upper bound. Our server
actually receives a five-epoch weight delta, which this does not invert.
"""

import torch
import torch.nn.functional as F

from fedseismic.eval.metrics import _logits
from fedseismic.federated.fedkper import _distillation_kl


def _parameter_grad(model, images, targets, teacher=None, lambda_cap=10.0, skip_prefixes=(),
                    log_prior=None):
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name == prefix or name.startswith(prefix + ".") for prefix in skip_prefixes):
            continue
        params.append(param)
    logits = _logits(model(images))
    scored = logits if log_prior is None else logits + log_prior.view(1, -1)
    loss = F.cross_entropy(scored, targets)
    if teacher is not None:
        with torch.no_grad():
            teacher_logits = _logits(teacher(images))
            teacher_ce = F.cross_entropy(teacher_logits, targets)
            lam = min(float(lambda_cap), 1.0 / float(teacher_ce.clamp(min=1e-8)))
        loss = loss + lam * _distillation_kl(logits, teacher_logits)
    grads = torch.autograd.grad(loss, params, create_graph=images.requires_grad)
    return torch.cat([grad.reshape(-1) for grad in grads])


def _total_variation(images):
    dy = (images[:, :, 1:, :] - images[:, :, :-1, :]).abs().mean()
    dx = (images[:, :, :, 1:] - images[:, :, :, :-1]).abs().mean()
    return dx + dy


def to_unit_interval(image, mean, std):
    """Map a normalized ``(1, C, H, W)`` tensor back to ``[0, 1]`` pixels."""
    center = image.new_tensor(mean).view(1, -1, 1, 1)
    scale = image.new_tensor(std).view(1, -1, 1, 1)
    return (image * scale + center).clamp(0, 1)


def psnr(pred, target):
    """Peak signal-to-noise ratio for tensors already scaled to ``[0, 1]``."""
    mse = (pred - target).pow(2).mean().clamp(min=1e-12)
    return float((10.0 * torch.log10(1.0 / mse)).item())


def ssim(pred, target, window=7):
    """Mean structural similarity over channels. Inputs are ``[0, 1]``."""
    pad = window // 2
    pred = pred.detach()
    target = target.detach()
    channels = []
    for channel in range(pred.size(1)):
        left = pred[:, channel:channel + 1]
        right = target[:, channel:channel + 1]
        mu_x = F.avg_pool2d(left, window, stride=1, padding=pad)
        mu_y = F.avg_pool2d(right, window, stride=1, padding=pad)
        sigma_x = F.avg_pool2d(left * left, window, stride=1, padding=pad) - mu_x * mu_x
        sigma_y = F.avg_pool2d(right * right, window, stride=1, padding=pad) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(left * right, window, stride=1, padding=pad) - mu_x * mu_y
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        channels.append(float((numerator / denominator).mean().item()))
    return float(sum(channels) / max(len(channels), 1))


def reconstruct(model, image, target, steps=400, lr=0.1, tv_weight=1e-3,
                teacher=None, lambda_cap=10.0, skip_prefixes=(), attacker_teacher=None,
                log_prior=None):
    """Return ``(dummy, mse)``. ``image`` is one example, shape ``(1, C, H, W)``.

    ``teacher`` is used for the true gradient. ``attacker_teacher`` is what the
    reconstruction is allowed to use. FedPer drops ``skip_prefixes`` from both.
    ``log_prior`` is added to the logits before cross-entropy, matching FedLC.
    """
    model.eval()
    true = _parameter_grad(
        model, image.detach(), target.detach(), teacher, lambda_cap, skip_prefixes,
        log_prior,
    ).detach()
    attack_teacher = teacher if attacker_teacher is None else attacker_teacher
    dummy = torch.randn_like(image, device=image.device, requires_grad=True)
    optimizer = torch.optim.Adam([dummy], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        guess = _parameter_grad(
            model, dummy, target.detach(), attack_teacher, lambda_cap, skip_prefixes,
            log_prior,
        )
        match = 1.0 - F.cosine_similarity(guess, true, dim=0)
        objective = match + tv_weight * _total_variation(dummy)
        objective.backward()
        optimizer.step()
    with torch.no_grad():
        mse = float((dummy - image).pow(2).mean().item())
    return dummy.detach(), mse
