# Author: Yinghua Zhou
# Creation Date: 2023/09/16

import numpy as np


def add_gaussian_noise(img, mean=0, var=10):
    """
     In a sense, the standard deviation in gaussian noise can be thought of
     as the degree of noise, and mean as the standard noise.

    :param img: Original image.
    :param mean: Mean of the gaussian noise.
    :param var: Standard deviation of the gaussian noise.
    :return: Noisy image.
    """
    noise = np.random.normal(mean, var, img.shape)

    return np.clip(img + noise, 0, 255)


def add_salt_pepper_noise(img, noise_prob=0.3, salt_prob=0.5):
    """
    Add salt and pepper noise to the image.
    :param img: Original image.
    :param noise_prob: Probability of the noise.
    :param salt_prob: Probability of the salt noise. Pepper noise is 1 - salt_prob.
    :return: Noisy image.
    """
    # Create a mask for the pixels that will have noise
    noise_mask = np.random.rand(*img.shape) < noise_prob

    # Create a mask for salt (within the previously chosen noise pixels)
    salt_mask = np.random.rand(*img.shape) < salt_prob

    # Copy the original image
    noisy_img = img.copy()

    # Apply salt and pepper noise using the masks
    noisy_img[noise_mask & salt_mask] = 255  # apply salt noise
    noisy_img[noise_mask & ~salt_mask] = 0  # apply pepper noise

    return noisy_img
