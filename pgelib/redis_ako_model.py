import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from ops import *
from discriminator import Discriminator
from generator import Generator

params = dict()


def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)


def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)


def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def change_gradent_format(new_values, previous_grads):

    modified_grads = list()

    for v, ref in zip(new_values, previous_grads):

        modified_grads.append((v, ref[1]))

    return modified_grads


def init_parameters():

    params["weights"] = dict()
    params["data"] = dict()
    params["gradient"] = list()
    params["new_g"] = dict()
    params["loss"] = list()
    params["optimizer"] = list()

    params["weights_d"] = dict()
    params["data_d"] = dict()
    params["gradient_d"] = list()
    params["new_g_d"] = dict()
    params["loss_d"] = list()
    params["optimizer_d"] = list()


def build_model(cfg):

    init_parameters()

    with tf.device("/job:%s/task:%d" % (cfg.job_name, cfg.nID)):

        with tf.variable_scope("%s%d" % (cfg.job_name, cfg.nID)):

            generator = Generator(cfg.img_shape, cfg.batch_size)
            discriminator = Discriminator(cfg.img_shape)

            W_conv1 = generator.W1
            W_conv2 = generator.W2
            W_conv3 = generator.W3
            W_conv4 = generator.W4

            W_conv1_d = discriminator.W1
            W_conv2_d = discriminator.W2
            W_conv3_d = discriminator.W3
            W_conv4_d = discriminator.W4
            W_conv5_d = discriminator.W5

            phX = tf.placeholder(tf.float32, [None, cfg.rows, cfg.cols])
            phZ = tf.placeholder(tf.float32, [None, cfg.z_shape])
            new_g_W_conv1 = tf.placeholder(tf.float32, shape=[100, 7*7*512])
            new_g_W_conv2 = tf.placeholder(tf.float32, shape=[3, 3, 512, 256])
            new_g_W_conv3 = tf.placeholder(tf.float32, shape=[3, 3, 256, 128])
            new_g_W_conv4 = tf.placeholder(tf.float32, shape=[3, 3, 128, 1])

            new_g_W_conv1_d = tf.placeholder(tf.float32, shape=[5, 5, 1, 64])
            new_g_W_conv2_d = tf.placeholder(tf.float32, shape=[3, 3, 64, 64])
            new_g_W_conv3_d = tf.placeholder(tf.float32, shape=[3, 3, 64, 128])
            new_g_W_conv4_d = tf.placeholder(tf.float32, shape=[2, 2, 128, 256])
            new_g_W_conv5_d = tf.placeholder(tf.float32, shape=[7*7*256, 1])

            gen_out = generator.forward(phZ)

            disc_logits_fake = discriminator.forward(gen_out)
            disc_logits_real = discriminator.forward(phX)

            disc_fake_loss = cost(tf.zeros_like(disc_logits_fake), disc_logits_fake)
            disc_real_loss = cost(tf.ones_like(disc_logits_real), disc_logits_real)

            disc_loss = tf.add(disc_fake_loss, disc_real_loss)
            gen_loss = cost(tf.ones_like(disc_logits_fake), disc_logits_fake)

            train_vars = tf.trainable_variables()

            disc_vars = [var for var in train_vars if 'd' in var.name]

            trainable_vars_d = [W_conv1_d, W_conv2_d, W_conv3_d, W_conv4_d, W_conv5_d]

            trainable_vars = [W_conv1, W_conv2, W_conv3, W_conv4]

            disc_train = tf.train.AdamOptimizer(cfg.lr_disc, beta1=cfg.beta1).minimize(disc_loss, var_list=disc_vars)
            # Todo: Fix commented optimizer
            optimizer = tf.train.AdamOptimizer(cfg.lr_gen, beta1=cfg.beta1)

            grads_d = optimizer.compute_gradients(disc_loss, trainable_vars_d)

            loss = cost(tf.ones_like(disc_logits_fake), disc_logits_fake)
            grads = optimizer.compute_gradients(loss, trainable_vars)
            new_grads = [new_g_W_conv1, new_g_W_conv2, new_g_W_conv3, new_g_W_conv4]
            new_grads_d = [new_g_W_conv1_d, new_g_W_conv2_d, new_g_W_conv3_d, new_g_W_conv4_d, new_g_W_conv5_d]

            modified_grads = change_gradent_format(new_grads + new_grads_d, grads + grads_d)
            train_op = optimizer.apply_gradients(modified_grads)

            params["weights"]["W_conv1"] = W_conv1
            params["weights"]["W_conv2"] = W_conv2
            params["weights"]["W_conv3"] = W_conv3
            params["weights"]["W_conv4"] = W_conv4
            params["new_g"]["W_conv1"] = new_g_W_conv1
            params["new_g"]["W_conv2"] = new_g_W_conv2
            params["new_g"]["W_conv3"] = new_g_W_conv3
            params["new_g"]["W_conv4"] = new_g_W_conv4
            params["data"]["z"] = phZ

            params["weights_d"]["W_conv1_d"] = W_conv1_d
            params["weights_d"]["W_conv2_d"] = W_conv2_d
            params["weights_d"]["W_conv3_d"] = W_conv3_d
            params["weights_d"]["W_conv4_d"] = W_conv4_d
            params["weights_d"]["W_conv5_d"] = W_conv5_d
            params["new_g_d"]["W_conv1_d"] = new_g_W_conv1_d
            params["new_g_d"]["W_conv2_d"] = new_g_W_conv2_d
            params["new_g_d"]["W_conv3_d"] = new_g_W_conv3_d
            params["new_g_d"]["W_conv4_d"] = new_g_W_conv4_d
            params["new_g_d"]["W_conv5_d"] = new_g_W_conv5_d
            params["data"]["z"] = phZ
            params["gradient_d"] = grads_d
            params["gradient"] = grads
            params["loss_d"] = disc_loss
            params["loss"] = loss
            params["optimizer"] = train_op

    return gen_out, params, phZ, phX

