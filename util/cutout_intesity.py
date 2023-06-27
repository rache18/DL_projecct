import torch
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The maximum length (in pixels) of each patch.
        max_intensity (float): The maximum intensity of the cutout patches (between 0 and 1).
    """
    def __init__(self, n_holes, length, max_intensity=1.0):
        self.n_holes = n_holes
        self.length = length
        self.max_intensity = max_intensity

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of varying shapes and intensities cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            # Randomly generate the length and width of the cutout patch
            length = np.random.randint(1, self.length + 1)
            width = np.random.randint(1, self.length + 1)

            # Randomly select the center coordinates of the cutout patch
            y = np.random.randint(h)
            x = np.random.randint(w)

            # Calculate the corner coordinates of the cutout patch
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - width // 2, 0, w)
            x2 = np.clip(x + width // 2, 0, w)

            # Randomly generate the intensity level
            intensity = np.random.uniform(0, self.max_intensity)
            if intensity > 0.5:
                intensity = 1.0

            # Apply the cutout patch with varying intensity
            mask[y1:y2, x1:x2] = intensity

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
