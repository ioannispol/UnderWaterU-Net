import numpy as np
import matplotlib.pyplot as plt

from utils.utils import plot_image_and_mask


def test_plot_image_and_mask():
    # Create a dummy image and mask
    img = np.random.rand(100, 100, 3)
    mask = np.random.randint(0, 2, (100, 100))  # Two classes: 0 and 1

    # Call the plotting function
    plot_image_and_mask(img, mask)

    # Retrieve the current figure and its axes
    fig = plt.gcf()
    axes = fig.axes

    # Assert that the number of subplots is correct
    assert len(axes) == 3, "Expected 3 subplots: 1 for image and 2 for masks"

    # Check the title of the first subplot (the image)
    assert axes[0].get_title() == "Input image", "First subplot should be the input image"

    # Check the titles of the mask subplots
    for i, ax in enumerate(axes[1:]):
        assert ax.get_title() == f"Mask (class {i})", f"Unexpected title for mask {i}"

    # Close the plot to free up resources
    plt.close(fig)
