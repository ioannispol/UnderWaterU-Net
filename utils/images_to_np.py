import cv2
import numpy as np
import os
import argparse


def images_to_npy(image_directory, output_filename):
    """
    Convert images from a specified directory into a .npy file.

    Parameters:
    - image_directory (str): Path to the directory containing the images.
    - output_filename (str): Name of the .npy file to save the images to.

    Returns:
    - int: Number of images saved to the .npy file.
    """

    # Get a list of all image paths from the specified directory
    image_paths = [
        os.path.join(image_directory, f)
        for f in os.listdir(image_directory)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    # Load all images into memory
    images = [cv2.imread(path) for path in image_paths]

    # Ensure all images are loaded successfully
    images = [img for img in images if img is not None]

    # Stack the images into a single array
    dataset = np.stack(images)

    # Save the dataset to a .npy file
    np.save(output_filename, dataset)

    return len(images)


def display_image_from_npy(npy_path, image_index):
    """
    Load and display an image from a .npy file.

    Parameters:
    - npy_path (str): Path to the .npy file containing the images.
    - image_index (int): 0-based index of the image to display from the .npy file.
    """

    # Load the dataset from the .npy file
    dataset = np.load(npy_path)

    # Check if the image_index is valid
    if image_index < 0 or image_index >= len(dataset):
        print(
            f"Invalid image index. Please provide an index between 0 and {len(dataset) - 1}."
        )
        return

    # Get the desired image
    image = dataset[image_index]

    # Display the image using OpenCV
    cv2.imshow(f"Image {image_index}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert images from a directory into a .npy file."
    )
    parser.add_argument(
        "image_directory", type=str, help="Path to the directory containing the images."
    )
    parser.add_argument(
        "output_filename", type=str, help="Name of the .npy file to save the images to."
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display an image from the .npy file after saving it.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="0-based index of the image to display from the .npy file.",
    )

    args = parser.parse_args()

    num_images_saved = images_to_npy(args.image_directory, args.output_filename)
    print(f"Saved {num_images_saved} images to {args.output_filename}")
    if args.display:
        display_image_from_npy(args.output_filename, args.index)
