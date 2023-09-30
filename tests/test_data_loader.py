from pathlib import Path
import pytest

import numpy as np
import torch
import cv2
from utils.data_load import load_image, mask_values_unique, BasicDataset, UnderwaterDataset


def test_load_npy_file():
    test_array = np.array([[1, 2], [3, 4]])
    np.save("test.npy", test_array)
    loaded_image = load_image("test.npy")
    assert np.array_equal(loaded_image, test_array)


def test_load_pt_file():
    test_tensor = torch.tensor([[5, 6], [7, 8]])
    torch.save(test_tensor, "test.pt")
    loaded_image = load_image("test.pt")
    assert np.array_equal(loaded_image, test_tensor.numpy())


def test_load_image_file():
    # Here, we'll create a dummy image using OpenCV and save it.
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.imwrite("test.jpg", test_image)
    loaded_image = load_image("test.jpg")
    assert np.array_equal(loaded_image, test_image)


# Create a temporary directory for masks
mask_dir = Path('./temp_masks')
mask_dir.mkdir(exist_ok=True)


def test_2d_mask():
    idx = "test_2d"
    mask_suffix = "_mask"
    mask_array = np.array([[1, 2], [3, 1]])
    mask_file = mask_dir / (idx + mask_suffix + '.npy')
    np.save(mask_file, mask_array)
    unique_values = mask_values_unique(idx, mask_dir, mask_suffix)
    assert np.array_equal(unique_values, np.array([1, 2, 3]))


def test_3d_mask():
    idx = "test_3d"
    mask_suffix = "_mask"
    mask_array = np.array([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]]])
    mask_file = mask_dir / (idx + mask_suffix + '.npy')
    np.save(mask_file, mask_array)
    unique_values = mask_values_unique(idx, mask_dir, mask_suffix)
    expected_values = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    assert np.array_equal(unique_values, expected_values)


def test_invalid_mask_dimensions():
    idx = "test_invalid"
    mask_suffix = "_mask"
    mask_array = np.array([[[[1, 0], [0, 1]], [[0, 0], [1, 0]]]])
    mask_file = mask_dir / (idx + mask_suffix + '.npy')
    np.save(mask_file, mask_array)
    try:
        mask_values_unique(idx, mask_dir, mask_suffix)
        assert False, "Expected a ValueError due to invalid mask dimensions"
    except ValueError as e:
        assert str(e) == "Loaded masks should have 2 or 3 dimensions, found 4"


# Create temporary directories for mock data
IMG_DIR = Path('./temp_images')
MASK_DIR = Path('./temp_masks')
IMG_DIR.mkdir(exist_ok=True)
MASK_DIR.mkdir(exist_ok=True)


# Create mock data
def create_mock_data():
    # Mock image data
    img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(IMG_DIR / "image1.png"), img_array)

    # Mock mask data
    mask_array = np.random.randint(0, 3, (100, 100), dtype=np.uint8)  # Assuming 3 classes
    cv2.imwrite(str(MASK_DIR / "image1_mask.png"), mask_array)


create_mock_data()


def test_dataset_length():
    dataset = BasicDataset(str(IMG_DIR), str(MASK_DIR), mask_suffix='_mask')
    assert len(dataset) == 1


def test_dataset_item():
    dataset = BasicDataset(str(IMG_DIR), str(MASK_DIR), mask_suffix='_mask')
    item = dataset[0]
    assert "image" in item
    assert "mask" in item

    # Check shapes and data types
    assert item["image"].shape == (3, 100, 100)  # Channels, Height, Width
    assert item["mask"].shape == (100, 100)

    assert item["image"].dtype == torch.float32
    assert item["mask"].dtype == torch.int64


def test_resize():
    dataset = BasicDataset(str(IMG_DIR), str(MASK_DIR), resize_to=(50, 50), mask_suffix='_mask')
    item = dataset[0]
    assert item["image"].shape == (3, 50, 50)
    assert item["mask"].shape == (50, 50)


def test_invalid_directory():
    with pytest.raises(RuntimeError):
        _ = BasicDataset('./invalid_dir', str(MASK_DIR))


# Tests fot the UnderwaterDataset
def test_underwaterdataset_length():
    dataset = UnderwaterDataset(str(IMG_DIR), str(MASK_DIR), mask_suffix='_mask')
    assert len(dataset) == 1


def test_underwaterdataset_item():
    dataset = UnderwaterDataset(str(IMG_DIR), str(MASK_DIR), mask_suffix='_mask')
    item = dataset[0]
    assert "image" in item
    assert "mask" in item

    # Check shapes and data types
    assert item["image"].shape == (3, 100, 100)  # Channels, Height, Width
    assert item["mask"].shape == (100, 100)


def test_underwaterdataset_resize():
    dataset = UnderwaterDataset(str(IMG_DIR), str(MASK_DIR), resize_to=(50, 50), mask_suffix='_mask')
    item = dataset[0]
    assert item["image"].shape == (3, 50, 50)
    assert item["mask"].shape == (50, 50)


def test_underwaterdataset_invalid_directory():
    with pytest.raises(RuntimeError):
        _ = UnderwaterDataset('./invalid_dir', str(MASK_DIR))

# More tests can be added, like testing augmentations, unique mask values, etc.


# Cleanup after tests
def teardown_module():
    import shutil
    shutil.rmtree(IMG_DIR)
    shutil.rmtree(MASK_DIR)
