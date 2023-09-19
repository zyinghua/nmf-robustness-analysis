# Author: Yinghua Zhou
# Creation Date: 2023/09/16

import numpy as np


def add_gaussian_noise(img, mean=0, sd=0.05):
    """
     In a sense, the standard deviation in gaussian noise can be thought of
     as the degree of noise, and mean as the standard noise.

    :param img: Original images.
    :param mean: Mean of the gaussian noise.
    :param sd: Standard deviation of the gaussian noise.
    :return: Noisy images.
    """
    noise = np.random.normal(mean, sd, img.shape)

    return np.clip(img + noise, 0, 1)


def add_salt_pepper_noise(img, noise_prob=0.2, salt_prob=0.5):
    """
    Add salt and pepper noise to the image.
    :param img: Original images.
    :param noise_prob: Probability of the noise.
    :param salt_prob: Probability of the salt noise. Pepper noise is 1 - salt_prob.
    :return: Noisy images.
    """
    # Create a mask for the pixels that will have noise
    noise_mask = np.random.rand(*img.shape) < noise_prob

    # Create a mask for salt (within the previously chosen noise pixels)
    salt_mask = np.random.rand(*img.shape) < salt_prob

    # Copy the original image
    noisy_img = img.copy()

    # Apply salt and pepper noise using the masks
    noisy_img[noise_mask & salt_mask] = 1  # apply salt noise
    noisy_img[noise_mask & ~salt_mask] = 0  # apply pepper noise

    return noisy_img


def add_block_occlusion_noise(img, img_size, width=0.3, height=0.3):
    """
    Add block occlusion noise to the image.
    :param img: Original images.
    :param img_size: Size of the image.
    :param width: Width of the block.
    :param height: Height of the block.
    :return: Noisy images.
    """
    assert width < 1, "width needs to be less than 1"
    assert height < 1, "height needs to be less than 1"

    img_copy = img.copy()

    num_samples = img_copy.shape[1]

    img_copy = img_copy.T.reshape(num_samples, img_size[1], img_size[0])

    block_height = int(img_size[1] * height)
    block_width = int(img_size[0] * width)

    for i in range(num_samples):
        start_x = np.random.randint(0, img_size[0] - block_width)
        start_y = np.random.randint(0, img_size[1] - block_height)

        img_copy[i, start_y:start_y + block_height, start_x:start_x + block_width] = 1  # Set to white

    return img_copy.reshape(num_samples, img_size[0] * img_size[1]).T