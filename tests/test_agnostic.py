"""The machine-agnostic network: the normalised frame, the current-to-flux model and its losses."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fusioneq.agnostic import MachineFrame, ellipse_current, machine_current, vacuum_fields
from fusioneq.agnostic_model import (
    FULL_WEIGHT,
    N_OFFSET_IN,
    N_SCALARS,
    OFFSET_WEIGHT,
    AgnosticConfig,
    AgnosticPredictor,
    MachineData,
    correction_penalty,
    current_loss,
    gradient_loss,
    psi_loss,
)
from fusioneq.physics import Calibration
from fusioneq.plasma import PlasmaShape


class TestMachineFrame:
    @pytest.mark.parametrize("machine", ["DIII-D", "MAST"])
    def test_round_trip_keeps_a_smooth_field_and_the_frame_covers_the_envelope(self, machine):
        f = MachineFrame.for_machine(machine)
        R, Z = np.meshgrid(f.grid_R, f.grid_Z)
        psi = torch.as_tensor(
            (np.exp(-(((R - f.r_c) / f.h) ** 2) - (Z / (1.6 * f.h)) ** 2) + 0.3 * R)[None], dtype=torch.float64
        )
        back = f.from_frame(f.to_frame(psi))[0].numpy()
        env = f.envelope
        assert np.abs(back - psi[0].numpy())[env].max() < 1e-3 * np.ptp(psi[0].numpy()[env])
        covered = f.from_frame(torch.ones(1, f.size, f.size, dtype=torch.float64))[0].numpy()
        assert np.allclose(covered[env], 1.0)

    def test_machine_current_is_nonnegative_inside_the_envelope_and_carries_ip(self):
        f = MachineFrame.for_machine("DIII-D")
        density = ellipse_current(f, np.array([[1.7, 0.6, 0.0], [1.6, 0.5, 0.2]]), PlasmaShape(kappa=1.8))
        ip = torch.tensor([1e6, 8e5])
        J = machine_current(f, density, ip)
        assert (J >= 0).all() and torch.allclose(J.sum(1), ip)
        assert J.view(2, *f.envelope.shape)[:, ~torch.as_tensor(f.envelope)].abs().max() == 0

    def test_vacuum_moments_recover_the_coefficients_of_a_vacuum_field(self):
        f = MachineFrame.for_machine("MAST")
        R, Z = f.frame_RZ
        fields = vacuum_fields(R / f.r_c, Z / f.r_c)
        coef = np.array([0.3, -1.2, 0.5, 0.1, 0.02, -0.4])
        field = torch.as_tensor(np.tensordot(coef, fields, 1)[None], dtype=torch.float64)
        assert np.allclose(f.vacuum_moments(field)[0].numpy(), coef, atol=1e-6)

    @pytest.mark.parametrize("sign", [1, -1])
    def test_label_current_inverts_grad_shafranov_and_ignores_vacuum_fields(self, sign):
        f = MachineFrame.for_machine("DIII-D")
        R, Z = np.meshgrid(f.grid_R, f.grid_Z)
        vacuum = np.tensordot([0.3, -1.2, 0.5, 0.1, 0.02, -0.4], vacuum_fields(R / f.r_c, Z / f.r_c), 1)
        psi = torch.as_tensor(sign * (-(R**4) + vacuum).reshape(1, -1), dtype=torch.float64)
        expected = R * f.envelope  # Delta* R^4 = 8 R^2, so J ~ -Delta* psi / R ~ R
        assert np.allclose(f.label_current(psi)[0].numpy(), (expected / expected.sum()).ravel(), rtol=1e-2, atol=1e-7)

    def test_channels_are_dimensionless_vacuum_toroidicity_envelope_and_height(self):
        f = MachineFrame.for_machine("MAST")
        vac = torch.ones(2, len(f.grid_Z), len(f.grid_R), dtype=torch.float64)
        ch = f.channels(vac, torch.tensor([600.0, -600.0], dtype=torch.float64))
        assert ch.shape == (2, 4, f.size, f.size)
        assert torch.allclose(ch[0, 0], ch[1, 0])  # the Ip sign does not flip the normalised vacuum
        assert 0 < ch[0, 1].min() < 0.2  # R / R_c reaches toward MAST's centre column
        assert ch[0, 2].max() == 1  # envelope
        assert abs(float(ch[0, 3].mean())) < 1e-9  # height, symmetric about the midplane


class TestAgnosticModel:
    @pytest.mark.parametrize(("offset", "correction"), [("none", False), ("bottleneck", False), ("vacuum", True)])
    def test_psi_loss_trains_the_net_through_the_current_and_green_step(self, offset, correction):
        frame = MachineFrame.for_machine("DIII-D")
        n = frame.envelope.size
        zeros = torch.zeros(2)
        data = MachineData(
            "DIII-D", frame, torch.eye(n) * 1e-7, torch.zeros(frame.envelope.shape), zeros, zeros, torch.zeros(2, 3)
        )
        calib = Calibration(scale=-0.9, plasma_ratio=0.9, offset=0.01, ecoil_turns=0.0)
        cfg = AgnosticConfig(base=8, levels=3, offset=offset, correction=correction)
        model = AgnosticPredictor(cfg, calib, "cpu")
        x = torch.randn(2, 4 + N_SCALARS, frame.size, frame.size)
        x[:, 2] = 1.0  # envelope channel
        offset_in = torch.randn(2, N_OFFSET_IN) if offset == "vacuum" else None
        psi, _, field = model.psi(data, x, torch.zeros(2, *frame.envelope.shape), torch.tensor([1e6, 5e5]), offset_in)
        psi.square().mean().backward()
        assert psi.shape == (2, n) and model.net.down[0][0].weight.grad.abs().sum() > 0
        if offset != "none":
            assert any(p.grad.abs().sum() > 0 for p in model.net.offset.parameters())
        if correction:  # the output conv's second channel is the flux correction
            assert model.net.out.weight.grad[1].abs().sum() > 0
        else:
            assert not field.any()

    @pytest.mark.parametrize(("saved", "offset"), [({}, "none"), ({"offset_head": True}, "bottleneck")])
    def test_configs_saved_before_the_offset_mode_still_load(self, saved, offset):
        assert AgnosticConfig.from_saved({"epochs": 8, **saved}).offset == offset

    def test_shape_loss_down_weights_a_pure_offset_but_not_a_shape_error(self):
        target = torch.randn(3, 50)
        env = torch.zeros(50)
        env[10:40] = 1.0
        offset = torch.tensor([[0.5], [-1.0], [2.0]])
        var = target.var(1, keepdim=True)
        pure_offset = (offset**2 / var).mean()
        assert torch.isclose(psi_loss("psi", target + offset, target, env), 2 * pure_offset)
        assert torch.isclose(
            psi_loss("shape", target + offset, target, env), (OFFSET_WEIGHT + FULL_WEIGHT) * pure_offset
        )
        bump = torch.zeros(50)
        bump[20] = 1.0  # a shape error inside the envelope keeps full weight in the shape term
        assert psi_loss("shape", target + bump, target, env) > psi_loss("shape", target + bump.mean(), target, env) * 5
        with pytest.raises(ValueError, match="unknown loss mode"):
            psi_loss("shap", target, target, env)

    def test_correction_penalty_prices_the_correction_inside_the_envelope_like_a_psi_error(self):
        target = torch.randn(3, 50)
        env = torch.zeros(50)
        env[10:40] = 1.0
        inside, outside = torch.zeros(3, 50), torch.zeros(3, 50)
        inside[:, 10:40], outside[:, :10] = 0.5, 7.0
        assert torch.isclose(correction_penalty(inside, target, env), (0.25 / target.var(1)).mean())
        assert correction_penalty(outside, target, env) == 0

    def test_gradient_loss_is_offset_free_and_current_loss_is_zero_at_the_label_current(self):
        f = MachineFrame.for_machine("DIII-D")
        R, Z = np.meshgrid(f.grid_R, f.grid_Z)
        target = torch.as_tensor(-(R**4) - 2 * Z**2 * R**2, dtype=torch.float64).reshape(1, -1)
        assert gradient_loss(f, target + 3.0, target) < 1e-20
        assert gradient_loss(f, target * 1.1, target) > 1e-3
        shares = f.label_current(target)
        assert current_loss(f, shares, target) < 1e-20
        assert current_loss(f, shares.roll(5, 1), target) > 1e-2
