"""Nonclinical public-interface examples for the frozen execution snapshot.

Run inside the standalone review ZIP with PYTHONPATH=execution/src.
No patient records, trained weights, downloads or fitting are required.
"""
import torch
from cpet_foundation.data.wearable import WearableSimulationConfig, simulate_wearable_sample
from cpet_foundation.model.wearable_student import wearable_student_loss


def test_valid_zero_missing_hr_and_padding_are_distinct():
    x = torch.tensor([[0., 0., 0., 0.], [10., 1., 999., 25.],
                      [20., 2., 90., 0.], [0., 0., 999., 999.]])
    mask = torch.ones_like(x)
    mask[1, 2] = 0
    mask[3] = 0
    q = simulate_wearable_sample(
        {'x': x, 'mask': mask, 'time_mask': torch.tensor([1., 1., 1., 0.])},
        ('Time', 'Load_Phase', 'HR', 'Power_Load'),
        WearableSimulationConfig(selected_channels=('HR', 'signal_quality'),
                                 protocol_channels=('Power_Load',)),
    )
    assert q.x[0, 0].item() == 0. and q.mask[0, 0].item() == 1.
    assert q.mask[1].tolist() == [0., 1.]
    assert q.x[1, 1].item() == 0.
    assert q.protocol_x[2, 0].item() == 0. and q.protocol_mask[2, 0].item() == 1.
    assert not q.mask[3].any() and not q.protocol_mask[3].any()


def test_missing_neighbour_and_padding_placeholder_cannot_change_observations():
    x = torch.tensor([[float(i), 1., float(i), float(i)] for i in range(9)])
    mask = torch.ones_like(x)
    mask[3, 2:] = 0
    cfg = WearableSimulationConfig(
        selected_channels=('HR', 'HR_smooth', 'HR_slope', 'HR_acceleration', 'HR_rolling_std'),
        protocol_channels=('ramp_slope',))

    def simulate(placeholder, padding):
        values = x.clone()
        values[3, 2:] = placeholder
        return simulate_wearable_sample(
            {'x': torch.cat([values, torch.full((padding, 4), placeholder)]),
             'mask': torch.cat([mask, torch.zeros(padding, 4)]),
             'time_mask': torch.cat([torch.ones(9), torch.zeros(padding)])},
            ('Time', 'Load_Phase', 'HR', 'Power_Load'), cfg)

    a, b = simulate(0., 0), simulate(10000., 3)
    assert torch.equal(a.x, b.x[:9]) and torch.equal(a.mask, b.mask[:9])
    assert a.mask[:, 1].tolist() == [1., 0., 0., 0., 0., 0., 1., 1., 1.]
    assert a.mask[:, 2].tolist() == [1., 1., 0., 0., 0., 1., 1., 1., 1.]
    assert torch.equal(a.protocol_x, b.protocol_x[:9])
    assert not b.mask[9:].any() and not b.protocol_mask[9:].any()


def test_missing_at_does_not_localize_curve_loss():
    pred = torch.tensor([[[1.], [0.], [0.], [0.], [2.]]], requires_grad=True)
    outputs = {'trajectory_mu': pred, 'trajectory_logvar': torch.zeros_like(pred),
               'at_norm': torch.tensor([.5], requires_grad=True)}
    batch = {'mask': torch.ones(1, 5), 'shape_mask': torch.ones(1, 5),
             'target_trajectory': torch.zeros_like(pred), 'target_channel_mask': torch.ones_like(pred),
             'core_derived_feature_indices': torch.tensor([0])}
    weights = dict(latent_alignment=0., trajectory_distillation=0., endpoint=0.,
                   uncertainty_distillation=0., physiology_consistency=0., core_derived_trajectory=1.)
    reference, _ = wearable_student_loss(outputs, batch, weights)
    for at in [0., .8, float('nan')]:
        actual, _ = wearable_student_loss(outputs, {**batch, 'at_norm': torch.tensor([at]),
                                                   'at_supervision_mask': torch.zeros(1)}, weights)
        assert torch.allclose(actual, reference)
    reference.backward()
    assert torch.isfinite(pred.grad).all()


def test_masked_peak_and_slope_cells_cannot_supply_supervision():
    pred = torch.tensor([[[-1.], [-1.], [10000.]]], requires_grad=True)
    outputs = {'trajectory_mu': pred, 'trajectory_logvar': torch.zeros_like(pred)}
    batch = {'mask': torch.ones(1, 3), 'shape_mask': torch.ones(1, 3),
             'target_trajectory': torch.tensor([[[-1.], [-1.], [9999.]]]),
             'target_channel_mask': torch.tensor([[[1.], [1.], [0.]]]),
             'peak_loss_feature_indices': torch.tensor([0]), 'slope_loss_feature_indices': torch.tensor([0])}
    weights = dict(latent_alignment=0., trajectory_distillation=0., endpoint=0.,
                   uncertainty_distillation=0., physiology_consistency=0., peak=1., slope=1.)
    loss, parts = wearable_student_loss(outputs, batch, weights)
    assert parts['peak'] == 0. and parts['slope'] == 0.
    loss.backward()
    assert torch.equal(pred.grad, torch.zeros_like(pred))
