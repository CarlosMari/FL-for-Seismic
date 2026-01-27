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
        for i, (image, target) in enumerate(data_loader):
            image = image.to(device).type(torch.float)
            target = target.to(device).type(torch.long)
            output, recon = model(image)
            # Save predicted output for current image
            prediction[:, i, :] = output.argmax(1).detach().cpu().numpy().T.squeeze()
            # figure out how many pixels switched from current class to incorrect class after update
            pred_labels = output.argmax(1).detach().cpu().numpy().T.squeeze()
            prev_state = previous_acc[:, i, :]
            label = target.detach().cpu().numpy().squeeze().T
            # no forgets in round 0
            if round_idx != 0:
                mask = (pred_labels != label) & (prev_state == label)
                num_forgets = np.count_nonzero(mask)
                total_forgets += num_forgets
            else:
                total_forgets += 0
            # count total # of pixels in image to get NFR
            amount += (image.shape[-1] * image.shape[-2])

    mean_iou = jaccard_score(test_labels.flatten(), prediction.flatten(), labels=list(range(6)), average='weighted')
    mean_iou_class = jaccard_score(test_labels.flatten(), prediction.flatten(), labels=list(range(6)), average=None)
    nfr = total_forgets / amount

    return prediction, mean_iou, mean_iou_class, nfr