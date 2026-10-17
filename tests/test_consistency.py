"""The linearised consistency loss."""

from __future__ import annotations

import torch

from fusioneq.consistency import consistency_loss


class TestConsistency:
    def test_consistency_loss_is_the_linearised_target_error_over_its_variance(self):
        g = torch.tensor([[[1.0, 0.5], [2.0, 0.0]]])  # one frame, coefficients (0, 2), targets (volume, li)
        index, step, var = torch.tensor([0, 2]), torch.tensor([0.5, 1.0]), torch.tensor([4.0, 1.0])
        d = torch.tensor([[0.25, 9.0, 1.0]])  # coefficient 1 has no gradient and must not count
        # delta = g . (d[index] / step) = (0.5 * 1 + 1 * 2, 0.5 * 0.5 + 1 * 0) = (2.5, 0.25)
        assert torch.isclose(consistency_loss(d, g, index, step, var), torch.tensor(2.5**2 / 4 + 0.25**2))
