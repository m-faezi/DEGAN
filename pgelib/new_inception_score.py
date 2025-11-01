'''
From https://github.com/tsc2017/Inception-Score
Code derived from https://github.com/openai/improved-gan/blob/master/inception_score/model.py and https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
Usage:
    Call get_inception_score(images, splits=10)
Args:
    images: A numpy array with values ranging from 0 to 255 and shape in the form [N, 3, HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be arbitrary. A dtype of np.uint8 is recommended to save CPU memory.
    splits: The number of splits of the images, default is 10.
Returns:
    Mean and standard deviation of the Inception Score across the splits.
'''

import tensorflow as tf
import os
import numpy as np
import time

BATCH_SIZE = 64


def load_inception_model():

    """Load the pre-trained Inception model"""

    inception_model = tf.keras.applications.InceptionV3(
        include_top=True,
        weights='imagenet',
        input_shape=(299, 299, 3)
    )

    logits_model = tf.keras.Model(
        inputs=inception_model.input,
        outputs=inception_model.layers[-1].output  # Logits layer
    )

    return logits_model


inception_model = load_inception_model()


def preprocess_images(images):

    """Preprocess images for Inception model"""

    if images.dtype != np.uint8:

        images = np.clip(images, 0, 255).astype(np.uint8)

    resized_images = []

    for img in images:

        if img.shape[0] == 3:

            img = np.transpose(img, (1, 2, 0))

        img_resized = tf.image.resize(img, [299, 299]).numpy()
        resized_images.append(img_resized)

    resized_images = np.array(resized_images)

    processed_images = tf.keras.applications.inception_v3.preprocess_input(
        resized_images.astype(np.float32)
    )

    return processed_images


def get_inception_probs(images):

    """Get Inception model probabilities for images"""

    n_batches = int(np.ceil(float(images.shape[0]) / BATCH_SIZE))
    preds = np.zeros([images.shape[0], 1000], dtype=np.float32)

    processed_images = preprocess_images(images)

    for i in range(n_batches):

        batch = processed_images[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batch_preds = inception_model.predict(batch, verbose=0)
        preds[i * BATCH_SIZE : i * BATCH_SIZE + batch.shape[0]] = batch_preds

    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)

    return preds


def preds2score(preds, splits=10):

    """Calculate Inception Score from probabilities"""

    scores = []

    for i in range(splits):

        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)


def get_inception_score(images, splits=10):

    """
    Calculate Inception Score for generated images

    Args:
        images: numpy array with shape [N, 3, H, W] or [N, H, W, 3] with values in [0, 255]
        splits: number of splits to calculate score over

    Returns:
        mean: mean Inception Score
        std: standard deviation of Inception Score
    """

    print('Calculating Inception Score with %i images in %i splits' % (images.shape[0], splits))

    start_time = time.time()

    if len(images.shape) != 4:

        raise ValueError('Images must be 4D array')

    if images.shape[1] == 3:

        pass

    elif images.shape[3] == 3:

        pass

    else:

        raise ValueError('Images must have 3 channels in position 1 or 3')

    preds = get_inception_probs(images)
    mean, std = preds2score(preds, splits)

    print('Inception Score calculation time: %f s' % (time.time() - start_time))

    return mean, std


def get_inception_score_tfhub(images, splits=10):

    """
    Alternative implementation using TensorFlow Hub
    Requires: pip install tensorflow_hub
    """

    try:

        import tensorflow_hub as hub

        module = hub.load('https://tfhub.dev/google/tf2-preview/inception_v3/classification/4')

        n_batches = int(np.ceil(float(images.shape[0]) / BATCH_SIZE))
        preds = np.zeros([images.shape[0], 1000], dtype=np.float32)

        processed_images = preprocess_images(images)

        for i in range(n_batches):

            batch = processed_images[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            batch_preds = module(batch).numpy()
            preds[i * BATCH_SIZE : i * BATCH_SIZE + batch.shape[0]] = batch_preds

        preds = tf.nn.softmax(preds).numpy()
        mean, std = preds2score(preds, splits)

        return mean, std

    except ImportError:

        print("TensorFlow Hub not installed. Using Keras implementation.")

        return get_inception_score(images, splits)

