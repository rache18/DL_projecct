import torch
import numpy as np


class Cutout_intensity_shapes(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The maximum length (in pixels) of each patch.
        max_intensity (float): The maximum intensity of the cutout patches (between 0 and 1).
    """
    def __init__(self, n_holes, length, max_intensity=1.0 , shape = 'square'):
        self.n_holes = n_holes
        self.length = length
        self.max_intensity = max_intensity
        self.shape = shape

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
            y = np.random.randint(h)
            x = np.random.randint(w)

            # Randomly generate the intensity level
            intensity = np.random.uniform(0, self.max_intensity)
            if intensity > 0.85:
              intensity = 0.5

            if self.shape == 'square':
                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                mask[y1:y2, x1:x2] = intensity

            elif self.shape == 'circle':
                size = np.random.randint(8,16)
                radius = min(h, w) // size  # Set radius to 1/size:2 of the image size
                y_grid, x_grid = np.ogrid[:h, :w]
                mask_circle = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius ** 2
                mask[mask_circle] = intensity

            elif self.shape == 'triangle':
                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                triangle = np.tri(y2 - y1, x2 - x1, self.length // 2)
                # Replace zeros with the intensity value
                mask[y1: y2, x1: x2] = np.where(triangle == 0, intensity, triangle)


        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
