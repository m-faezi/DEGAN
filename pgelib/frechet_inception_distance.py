'''
From https://github.com/tsc2017/Frechet-Inception-Distance
Code derived from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
Usage:
    Call get_fid(images1, images2)
Args:
    images1, images2: Numpy arrays with values ranging from 0 to 255 and shape in the form [N, 3, HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be arbitrary.
    dtype of the images is recommended to be np.uint8 to save CPU memory.
Returns:
    Frechet Inception Distance between the two image distributions.
'''

import tensorflow as tf
import numpy as np
import time
from scipy.linalg import sqrtm


BATCH_SIZE = 64


def load_inception_model():

    """Load InceptionV3 model without the top classification layer"""

    inception_model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(299, 299, 3),
        pooling='avg'
    )

    return inception_model


inception_model = load_inception_model()


def preprocess_images(images):

    """Preprocess images for Inception model"""

    if images.dtype != np.uint8:

        images = np.clip(images, 0, 255).astype(np.uint8)

    if images.shape[1] == 3:

        images = np.transpose(images, (0, 2, 3, 1))

    resized_images = []

    for img in images:

        img_resized = tf.image.resize(img, [299, 299]).numpy()
        resized_images.append(img_resized)

    resized_images = np.array(resized_images)


    processed_images = tf.keras.applications.inception_v3.preprocess_input(
        resized_images.astype(np.float32)
    )

    return processed_images

def get_inception_activations(images):

    """Get Inception activations (features) for images"""

    n_batches = int(np.ceil(float(images.shape[0]) / BATCH_SIZE))
    activations = np.zeros([images.shape[0], 2048], dtype=np.float32)

    processed_images = preprocess_images(images)

    for i in range(n_batches):

        batch = processed_images[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batch_activations = inception_model.predict(batch, verbose=0)
        activations[i * BATCH_SIZE : i * BATCH_SIZE + batch.shape[0]] = batch_activations

    return activations

def calculate_fid(act1, act2):

    """Calculate Frechet Inception Distance between two sets of activations"""

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):

        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid

def get_fid(images1, images2):

    """
    Calculate Frechet Inception Distance between two sets of images

    Args:
        images1: numpy array with shape [N, 3, H, W] with values in [0, 255]
        images2: numpy array with shape [N, 3, H, W] with values in [0, 255]

    Returns:
        fid: Frechet Inception Distance
    """

    print('Calculating FID with %i images from each distribution' % (images1.shape[0]))
    start_time = time.time()

    if len(images1.shape) != 4 or len(images2.shape) != 4:

        raise ValueError('Images must be 4D arrays')

    if images1.shape[1] != 3 and images1.shape[3] != 3:

        raise ValueError('Images must have 3 channels')

    if images1.shape[0] != images2.shape[0]:

        print('Warning: Different number of images in distributions')

    act1 = get_inception_activations(images1)
    act2 = get_inception_activations(images2)

    fid = calculate_fid(act1, act2)

    print('FID calculation time: %f s' % (time.time() - start_time))

    return fid


def get_fid_tfhub(images1, images2):

    """
    Alternative implementation using TensorFlow Hub
    Requires: pip install tensorflow_hub
    """

    try:

        import tensorflow_hub as hub

        module = hub.load('https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4')

        def get_activations_tfhub(images):

            processed_images = preprocess_images(images)
            n_batches = int(np.ceil(float(images.shape[0]) / BATCH_SIZE))
            activations = np.zeros([images.shape[0], 2048], dtype=np.float32)

            for i in range(n_batches):

                batch = processed_images[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                batch_activations = module(batch).numpy()
                activations[i * BATCH_SIZE : i * BATCH_SIZE + batch.shape[0]] = batch_activations

            return activations

        act1 = get_activations_tfhub(images1)
        act2 = get_activations_tfhub(images2)
        fid = calculate_fid(act1, act2)

        return fid

    except ImportError:

        print("TensorFlow Hub not installed. Using Keras implementation.")

        return get_fid(images1, images2)

