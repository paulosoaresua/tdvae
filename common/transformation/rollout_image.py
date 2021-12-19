import torch
import numpy as np


class RolloutImage(object):
    """
    Rollout an image by shifting it to a random direction by the number of pixels defined by the offset. This will
    result in a final dataset of size: num_samples, time_step, height, width
    """

    def __init__(self, time_steps: int, offset: int = 1):
        self._time_steps = time_steps
        self._offset = offset

    def __call__(self, image):
        # Choose a random direction. The model should learn it.
        direction = -1 # np.random.choice([-1, 1])  # left and right
        images = []
        for t in range(self._time_steps):
            images.append(torch.roll(image, shifts=direction * self._offset * t, dims=2))

        images = torch.cat(images, dim=0)
        return images

    def __repr__(self):
        return self.__class__.__name__ + '()'
