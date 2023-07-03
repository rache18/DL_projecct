import torch
import numpy as np


class Cutout_Circle(object):
    """Randomly mask out one or more circular patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The radius (in pixels) of each circular patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.radius = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension 2 * radius x 2 * radius cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            # Generate circular mask
            y_grid, x_grid = np.ogrid[:h, :w]
            mask_circle = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= self.radius ** 2
            mask[y_grid[mask_circle], x_grid[mask_circle]] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
