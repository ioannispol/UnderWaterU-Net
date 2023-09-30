import pytest

import torch

from utils.dice_score import dice_coefficient, multiclass_dice_coefficient, dice_loss


def test_dice_coefficient():
    # 1. Test perfect overlap
    pred = torch.Tensor([[1, 1, 0, 0]])
    target = torch.Tensor([[1, 1, 0, 0]])
    assert dice_coefficient(pred, target) == 1.0, "Expected Dice score of 1.0 for perfect overlap"

    # 2. Test no overlap
    pred = torch.Tensor([[1, 1, 1, 1]])
    target = torch.Tensor([[0, 0, 0, 0]])
    assert dice_coefficient(pred, target) == pytest.approx(0.0, abs=1e-6), \
           "Expected Dice score close to 0.0 for no overlap"

    # 3. Test partial overlap
    pred = torch.Tensor([[1, 1, 0, 0]])
    target = torch.Tensor([[1, 0, 0, 0]])
    dice_score = dice_coefficient(pred, target)
    assert 0 < dice_score < 1, f"Expected Dice score between 0 and 1 for partial overlap, but got {dice_score}"

    # 4. Test with real numbers (not just 0 and 1), which should be thresholded
    pred = torch.Tensor([[0.8, 0.2, 0.6, 0.1]])
    target = torch.Tensor([[1, 0, 1, 0]])
    dice_score_real = dice_coefficient((pred > 0.5).float(), target)
    assert 0 <= dice_score_real <= 1, f"Expected Dice score between 0 and 1 for real values, but got {dice_score_real}"


def test_multiclass_dice_coefficient():
    # 1. Test perfect overlap for multiclass
    pred = torch.Tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
    target = torch.Tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
    assert multiclass_dice_coefficient(pred, target) == 1.0, \
           "Expected Dice score of 1.0 for perfect overlap in multiclass"

    # 2. Test no overlap for multiclass
    pred = torch.Tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
    target = torch.Tensor([[0, 0, 0, 0], [0, 0, 0, 0]])
    assert multiclass_dice_coefficient(pred, target) == pytest.approx(0.0, abs=1e-6), \
           "Expected Dice score close to 0.0 for no overlap in multiclass"

    # 3. Test partial overlap for multiclass
    pred = torch.Tensor([[1, 1, 0, 0], [1, 0, 1, 1]])
    target = torch.Tensor([[1, 0, 0, 0], [0, 0, 1, 1]])
    dice_score = multiclass_dice_coefficient(pred, target)
    assert 0 < dice_score < 1, f"Expected Dice score between 0 and 1 for partial overlap in multiclass, \
                                but got {dice_score}"


def test_dice_loss():
    # 1. Perfect overlap
    pred = torch.Tensor([1, 1, 0, 0])
    target = torch.Tensor([1, 1, 0, 0])
    assert dice_loss(pred, target) == 0.0, "Expected Dice loss of 0.0 for perfect overlap"

    # 2. No overlap
    pred = torch.Tensor([1, 1, 1, 1])
    target = torch.Tensor([0, 0, 0, 0])
    assert dice_loss(pred, target) == pytest.approx(1.0, abs=1e-6), "Expected Dice loss close to 1.0 for no overlap"

    # 3. Partial overlap
    pred = torch.Tensor([1, 1, 0, 0])
    target = torch.Tensor([1, 0, 0, 0])
    loss = dice_loss(pred, target)
    assert 0 < loss < 1, f"Expected Dice loss between 0 and 1 for partial overlap, but got {loss}"

    # 4. Multiclass dice loss
    pred = torch.Tensor([[1, 1, 0, 0], [1, 0, 1, 1]])
    target = torch.Tensor([[1, 0, 0, 0], [0, 0, 1, 1]])
    loss_multiclass = dice_loss(pred, target, multiclass=True)
    assert 0 < loss_multiclass < 1, f"Expected Dice loss between 0 and 1 for multiclass, but got {loss_multiclass}"
