import torch
import numpy as np


class Cutout_Shape(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        shape (str): Shape of the mask. Can be 'square', 'circle', or 'triangle'.
    """
    def __init__(self, n_holes, length, shape='circle'):
        self.n_holes = n_holes
        self.length = length
        self.shape = shape

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            if self.shape == 'square':
                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                mask[y1: y2, x1: x2] = 0.

            elif self.shape == 'circle':
                size = np.random.randint(8,16)
                radius = min(h, w) // size  # Set radius to 1/size:2 of the image size
                y_grid, x_grid = np.ogrid[:h, :w]
                mask_circle = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius ** 2
                mask[mask_circle] = 0.

            elif self.shape == 'triangle':
                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                triangle = np.tri(y2 - y1, x2 - x1, self.length // 2, dtype=bool)
                mask[y1: y2, x1: x2] = triangle

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img