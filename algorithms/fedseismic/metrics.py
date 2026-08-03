import copy
import numpy as np
import torch
from sklearn.metrics import jaccard_score

def eval_model(model, data_loader, previous_acc, round_idx, save_file=None, device="cuda:0"):
    model.eval()
    model.to(device)
    amount = 0
    total_forgets = 0

    if save_file is not None:
        prediction = np.zeros(save_file.shape)
        test_labels = copy.deepcopy(save_file)
    else:
        prediction = np.array([])

    with torch.no_grad():
        for i, (image, target, idx) in enumerate(data_loader):
            image = image.to(device).type(torch.float)
            target = target.to(device).type(torch.long)
            output, recon = model(image)

            batch_size = image.size(0)
            for b in range(batch_size):
                s = int(idx[b])
                prediction[:, s, :] = output[b].argmax(0).detach().cpu().numpy().T
                pred_labels = output[b].argmax(0).detach().cpu().numpy().T
                prev_state = previous_acc[:, s, :]
                label = target[b].detach().cpu().numpy().T
                if round_idx != 0:
                    mask = (pred_labels != label) & (prev_state == label)
                    num_forgets = np.count_nonzero(mask)
                    total_forgets += num_forgets
                else:
                    total_forgets += 0
                amount += (image.shape[-1] * image.shape[-2])

    mean_iou = jaccard_score(test_labels.flatten(), prediction.flatten(), labels=list(range(6)), average='weighted')
    mean_iou_class = jaccard_score(test_labels.flatten(), prediction.flatten(), labels=list(range(6)), average=None)
    nfr = total_forgets / amount

    return prediction, mean_iou, mean_iou_class, nfr