import torch
import torch.nn.functional as F
from torch import Tensor


def generate_smooth_mask(
    shape: tuple[int, ...],
    blur_iterations: int = 4,
    threshold: float = 0.5,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Generates a smooth spatial binary mask M in {0, 1}.

    Procedure:
    1. Start with a random binary Bernoulli(0.5) image of shape (B, 1, H, W).
    2. Repeatedly convolve with separable filter [1/4, 1/2, 1/4] horizontally & vertically.
    3. Threshold at 0.5 to form contiguous, smoothly bounded spatial regions.
    """
    if len(shape) == 2:
        shape = (1, 1, shape[0], shape[1])
    elif len(shape) == 3:
        shape = (shape[0], 1, shape[1], shape[2])
    elif len(shape) == 4 and shape[1] != 1:
        shape = (shape[0], 1, shape[2], shape[3])

    b, _, h, w = shape
    # Random bit image
    mask = (torch.rand((b, 1, h, w), device=device) > 0.5).float()

    # Separable blur filter [1/4, 1/2, 1/4]
    kernel_1d = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32, device=device)
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, 3, 3)

    for _ in range(blur_iterations):
        mask = F.conv2d(mask, kernel_2d, padding=1)

    # Threshold at 0.5
    binary_mask = (mask >= threshold).float()
    return binary_mask


def create_hybrid_negative_batch(
    images: Tensor,
    blur_iterations: int = 4,
    threshold: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Creates hybrid negative images x_neg = M * x_1 + (1 - M) * x_2.

    Preserves short-range correlations (strokes, edges, textures) while destroying
    long-range shape correlations (semantic digit identity).

    Returns:
        hybrid_negatives: Tensor of same shape as images
        smooth_masks: The binary masks used for hybridization
        permuted_images: The secondary images used as x_2
    """
    device = images.device
    b = images.size(0)

    # Permute batch to get a different digit image for each sample
    perm_indices = (
        torch.arange(b, device=device)
        + torch.randint(1, max(b, 2), (1,), device=device)
    ) % b
    permuted = images[perm_indices]

    # Ensure shape is (B, 1, 28, 28)
    if images.dim() == 2:
        img_4d = images.view(b, 1, 28, 28)
        perm_4d = permuted.view(b, 1, 28, 28)
    elif images.dim() == 3:
        img_4d = images.unsqueeze(1)
        perm_4d = permuted.unsqueeze(1)
    else:
        img_4d = images
        perm_4d = permuted

    masks = generate_smooth_mask(
        img_4d.shape,
        blur_iterations=blur_iterations,
        threshold=threshold,
        device=device,
    )

    hybrid = masks * img_4d + (1.0 - masks) * perm_4d

    if images.dim() == 2:
        return hybrid.view(b, -1), masks.view(b, -1), permuted
    return hybrid, masks, permuted


def inject_label_border(
    images: Tensor,
    labels: Tensor,
    wrong_labels: Tensor | None = None,
    num_classes: int = 10,
    label_scale: float | None = None,
) -> Tensor:
    """Injects class label into input representation.

    - For images (dim >= num_classes, e.g. MNIST 784):
      Follows Hinton (2022) Section 3.3 by writing 1-of-N into the top-border pixels.
    - For low-dimensional toy data (dim < num_classes, e.g. 2D points):
      Concatenates the one-hot class vector directly.
    """
    device = images.device
    b = images.size(0)
    flat_images = images.view(b, -1).clone().to(dtype=torch.float32)

    active_labels = labels if wrong_labels is None else wrong_labels
    active_labels = active_labels.to(device=device, dtype=torch.long)

    # Handle 2D or low-dimensional toy datasets by concatenation
    if flat_images.size(1) < num_classes:
        one_hot = F.one_hot(active_labels, num_classes).to(
            device=device, dtype=torch.float32
        )
        return torch.cat([flat_images, one_hot], dim=1)

    # Image inputs (MNIST): replace first num_classes pixels
    if label_scale is None:
        max_val = flat_images.max().item()
        min_val = flat_images.min().item()
        scale = max_val if max_val > 0.5 else 1.0
        base_val = min_val if min_val < 0 else 0.0
    else:
        scale = label_scale
        base_val = 0.0

    flat_images[:, :num_classes] = base_val
    row_indices = torch.arange(b, device=device)
    flat_images[row_indices, active_labels] = scale
    return flat_images
