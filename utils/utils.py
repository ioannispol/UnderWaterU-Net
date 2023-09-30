import numpy as np
import matplotlib.pyplot as plt


def plot_image_and_mask(img: np.ndarray, mask: np.ndarray) -> None:
    """
    Plots an input image and its corresponding segmentation masks.

    Args:
    - img (np.ndarray): The input image.
    - mask (np.ndarray): The segmentation mask. Each unique value represents a different class.
    """

    # Ensure mask is an integer array, and img is in the range [0, 1]
    mask = mask.astype(int)
    img = img.astype(float) / 255

    # Number of unique classes in the mask
    classes = mask.max() + 1

    # Create subplots
    fig, ax = plt.subplots(1, classes + 1, figsize=(15, 5))

    # Plot input image
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[0].axis('off')

    # Plot masks for each class
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i})')
        ax[i + 1].imshow(mask == i, cmap=plt.cm.gray)
        ax[i + 1].axis('off')

    plt.tight_layout()
    plt.show()
