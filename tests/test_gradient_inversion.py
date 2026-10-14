"""Gradient inversion returns a finite reconstruction of one image."""

import unittest

import torch
import torch.nn as nn

from fedseismic.privacy.gradient_inversion import psnr, reconstruct, ssim, to_unit_interval


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.classifier = nn.Linear(2 * 8 * 8, 3)

    def forward(self, images):
        features = torch.relu(self.conv(images))
        return self.classifier(features.reshape(features.size(0), -1))


class GradientInversionTests(unittest.TestCase):
    def test_reconstruction_is_finite(self):
        torch.manual_seed(0)
        image = torch.randn(1, 1, 8, 8)
        target = torch.tensor([1])
        dummy, mse = reconstruct(_Tiny(), image, target, steps=3, lr=0.05, tv_weight=0.0)
        self.assertEqual(tuple(dummy.shape), tuple(image.shape))
        self.assertTrue(torch.isfinite(dummy).all())
        self.assertGreaterEqual(mse, 0.0)

    def test_identical_pixels_score_as_a_match(self):
        image = torch.rand(1, 3, 8, 8)
        unit = to_unit_interval(image, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        self.assertGreater(psnr(unit, unit), 40.0)
        self.assertGreater(ssim(unit, unit), 0.99)


if __name__ == "__main__":
    unittest.main()
