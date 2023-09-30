import torch


def dice_coefficient(input: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    Compute the Dice coefficient between two tensors. Returns a float.
    """
    # Ensure the tensors are of the same size
    assert input.size() == target.size(), "Input and target tensors must have the same shape"

    # If the tensor has only one dimension, add a batch dimension
    if len(input.shape) == 1:
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)

    # Flatten the tensors. If they are batched, we compute the Dice coefficient for each item in the batch,
    # then average them.
    input = input.contiguous().view(input.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    # Compute the intersection (AND) and sum it over all the batches
    intersection = (input * target).sum(dim=1)

    # For the denominator, we take the sum of the input and target tensors
    # and subtract the intersection (this is equivalent to OR).
    sets_sum = input.sum(dim=1) + target.sum(dim=1)

    # Handle the case where both input and target are zero
    sets_sum = torch.where(sets_sum == 0, 2 * intersection + epsilon, sets_sum)

    dice_scores = (2 * intersection + epsilon) / (sets_sum + epsilon)
    return dice_scores.mean().item()


def multiclass_dice_coefficient(input: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    Compute the Dice coefficient for multiclass tensors.
    """
    # Ensure the tensors are of the same size
    assert input.size() == target.size(), "Input and target tensors must have the same shape"

    # Number of items
    num_items = input.size(0) // 2

    dice_scores = []

    for i in range(num_items):
        dice_scores.append(dice_coefficient(input[i::num_items], target[i::num_items], epsilon))

    return sum(dice_scores) / num_items


def dice_loss(input: torch.Tensor, target: torch.Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coefficient if multiclass else dice_coefficient
    return 1 - fn(input, target)
