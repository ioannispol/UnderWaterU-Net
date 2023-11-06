from pathlib import Path
from os import listdir
from os.path import splitext, isfile, join
from typing import Optional, Tuple, Union
import logging


from torch.utils.data import Dataset
import torch
import numpy as np
import cv2


def load_image(filename: Union[str, Path]) -> Union[np.ndarray, None]:
    filename = str(filename)
    extension = splitext(filename)[1]
    if extension == ".npy":
        return np.load(filename)
    elif extension in ['.pt', '.pth']:
        return torch.load(filename).numpy()
    else:
        return cv2.imread(filename, cv2.IMREAD_COLOR)  # Reading the image in color mode.


def mask_values_unique(idx: int, mask_dir: Path, mask_suffix: str) -> np.ndarray:
    mask_file = list(mask_dir.glob('mask-' + idx + mask_suffix + '.*'))
    if not mask_file:
        raise FileNotFoundError(f"No mask file found for index {idx} with suffix '{mask_suffix}' in {mask_dir}")
    mask = np.asarray(load_image(mask_file[0]))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: Path, mask_dir: Path,
                 resize_to: Optional[Tuple[int, int]] = None,
                 mask_suffix: str = '', augmentations=None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        if not self.images_dir.exists():
            raise RuntimeError(f"No input directory found at {images_dir}. Ensure your images are located here.")
        self.resize_to = resize_to
        self.mask_suffix = mask_suffix
        self.augmentations = augmentations

        self.ids = [splitext(file)[0] for file in listdir(images_dir)
                    if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}. Ensure your images are located here.')

        logging.info(f'Creating dataset with {len(self.ids)} examples')

        # Lazy load unique mask values
        self._mask_values = None

    @property
    def mask_values(self):
        if self._mask_values is None:
            logging.info('Scanning mask files to determine unique values')
            # For simplicity, I'm removing the multiprocessing part here. It can be added back if needed.
            unique = [mask_values_unique(id_, self.mask_dir, self.mask_suffix) for id_ in self.ids]
            self._mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
            logging.info(f'Unique mask values: {self._mask_values}')
        return self._mask_values

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, img: np.ndarray, resize_to: Tuple[int, int], is_mask: bool) -> np.ndarray:
        # Resize image or mask using OpenCV
        if resize_to:
            interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
            img = cv2.resize(img, resize_to, interpolation=interpolation)

        # Process mask
        if is_mask:
            mask = np.zeros(img.shape[:2], dtype=np.int64)  # Assuming the image is HxW or HxWxC
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask

        # Process image
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))
            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx: int) -> dict:
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob('mask-' + name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.resize_to, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.resize_to, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class UnderwaterDataset(BasicDataset):
    def __init__(self,
                 images_dir: Path,
                 mask_dir: Path,
                 resize_to:  Union[Tuple[int, int], None] = None,
                 mask_suffix: str = '',
                 augmentations=None
                 ):
        super().__init__(images_dir, mask_dir, resize_to, mask_suffix, augmentations)
