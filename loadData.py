import os
import numpy as np
from PIL import Image


def load_data(root, reduce=4):
    """
    Load ORL (or Extended YaleB) dataset to numpy array.

    Args:
        root: path to dataset.
        reduce: scale factor for zooming out images.

    Acknowledgement: This function is completely from the assignment 1 instruction ipynb file of COMP4328/5328
    Advanced Machine Learning course at University of Sydney. For the purpose of loading data only.
    """
    images, labels = [], []

    for i, person in enumerate(sorted(os.listdir(root))):

        if not os.path.isdir(os.path.join(root, person)):
            continue

        for fname in os.listdir(os.path.join(root, person)):

            # Remove background images in Extended YaleB dataset.
            if fname.endswith('Ambient.pgm'):
                continue

            if not fname.endswith('.pgm'):
                continue

            # load image.
            img = Image.open(os.path.join(root, person, fname))
            img = img.convert('L')  # grey image.

            # reduce computation complexity.
            img = img.resize([s // reduce for s in img.size])

            # TODO: preprocessing.

            # convert image to numpy array.
            img = np.asarray(img).reshape((-1, 1))

            # collect data and label.
            images.append(img / 255.0)  # normalize to [0, 1], easier to work with for many algorithms.
            labels.append(i)

    # concat all images and labels.
    images = np.concatenate(images, axis=1)
    labels = np.array(labels)

    return images, labels


if __name__ == '__main__':
    X_hat, Y_hat = load_data('../data/ORL', reduce=3)
    print('X_hat.shape={}, Y_hat.shape={}'.format(X_hat.shape, Y_hat.shape))

    # Add Noise.
    X_noise = np.random.rand(*X_hat.shape) * 40
    X = X_hat + X_noise

    import matplotlib.pyplot as plt

    img_size = [i // 3 for i in (92, 112)]  # ORL  30 x 37
    ind = 2  # index of demo image.
    plt.figure(figsize=(10, 3))
    plt.subplot(131)
    plt.imshow(X_hat[:, ind].reshape(img_size[1], img_size[0]),
               cmap=plt.cm.gray)  # reshape converts a 1D vector to an image matrix shape
    plt.title('Image(Original)')
    plt.subplot(132)
    plt.imshow(X_noise[:, ind].reshape(img_size[1], img_size[0]), cmap=plt.cm.gray)
    plt.title('Noise')
    plt.subplot(133)
    plt.imshow(X[:, ind].reshape(img_size[1], img_size[0]), cmap=plt.cm.gray)
    plt.title('Image(Noise)')
    plt.show()
