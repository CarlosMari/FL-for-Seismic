"""FedPer+BN: head and BN stay local, only the body is uploaded."""

import unittest
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fedseismic.federated.fedperbn import FedPerBNClientTrainer
from fedseismic.models import UNet


class _Seg(Dataset):
    def __init__(self, n, lo, hi):
        rng = np.random.RandomState(0)
        self.images = torch.from_numpy(rng.randn(n, 1, 32, 32).astype(np.float32))
        lab = rng.randint(lo, hi, (1, n, 32)).astype(np.int64)
        self.label = lab
        self.indices = list(range(n))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        lab = torch.from_numpy(self.label[0, i, :].repeat(32).reshape(32, 32))
        return self.images[i], lab, i


def _trainer(lo=6, hi=12):
    torch.manual_seed(0)
    model = UNet(1, 12, bilinear=False, norm="batch", groups=8)
    loader = DataLoader(_Seg(4, lo, hi), batch_size=2, shuffle=False)
    return FedPerBNClientTrainer(model, loader, nn.CrossEntropyLoss(), device="cpu",
                                 local_epochs=1, lr=1e-3, weight_decay=0.0,
                                 optimizer="sgd", momentum=0.0, num_classes=12)


class FedPerBNTests(unittest.TestCase):
    def test_local_keys_are_head_and_bn_only(self):
        t = _trainer()
        keys = set(t._local_keys(t.model.state_dict()))
        # Head and BN (including up4's BN) are local.
        self.assertTrue(any(k.startswith("outc.") for k in keys))
        self.assertIn("up4.conv.double_conv.1.weight", keys)  # BN -> local
        # Non-BN body (conv weights) is aggregated.
        self.assertNotIn("up4.conv.double_conv.0.weight", keys)
        self.assertNotIn("inc.double_conv.0.weight", keys)

    def test_upload_body_differs_head_bn_reset_to_global(self):
        t = _trainer()
        global_state = deepcopy(t.model.state_dict())
        t.download(global_state)
        t.train(global_state=global_state)
        uploaded = t._upload_state
        personal = t.personal_state
        # Upload's head + BN equal the global (so aggregation leaves them alone).
        for key in t._local_keys(global_state):
            self.assertTrue(torch.allclose(uploaded[key], global_state[key]))
        # Deployed keeps the client's own head + BN.
        changed = [k for k in t._local_keys(global_state)
                   if not torch.allclose(personal[k], global_state[k])]
        self.assertTrue(len(changed) > 0)

    def test_own_classes_mapping(self):
        t = _trainer(lo=6, hi=12)
        own = t._own_classes()
        self.assertTrue(torch.equal(own, torch.arange(6, 12)))


if __name__ == "__main__":
    unittest.main()
