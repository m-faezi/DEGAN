import tensorflow.compat.v1 as tf
import os
import sys
import time
import json
import redis_ako_config
from redis_ako_cluster import build_cluster
from redis_ako_model import build_model
from redis_ako_queue import GradientExchange
from frechet_inception_distance import get_fid

tf.disable_v2_behavior()

job_name = sys.argv[1]
nID = int(sys.argv[2])

json_directory = 'results/worker_%d_loss' % nID

if not os.path.exists(json_directory):

    os.makedirs(json_directory)

directory = 'results/worker_%d_samples/' % nID

if not os.path.exists(directory):

    os.makedirs(directory)

directory = 'results/worker_%d_results/' % nID

if not os.path.exists(directory):

    os.makedirs(directory)

json_inception_score_directory = 'results/worker_%d_inception_score' % nID

if not os.path.exists(json_inception_score_directory):

    os.makedirs(json_inception_score_directory)

json_frechet_inception_distance_directory = 'results/worker_%d_fid_score' % nID

if not os.path.exists(json_frechet_inception_distance_directory):

    os.makedirs(json_frechet_inception_distance_directory)


cfg = redis_ako_config.Config(job_name=job_name, nID=nID)

cluster, server, workers, term_cmd = build_cluster(cfg)
model = build_model(cfg)
params = model[1]
generator = model[0]
phZ = model[2]
phX = model[3]


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


train_mask_even = np.isin(y_train, [0, 2, 4, 6, 8])
test_mask_even = np.isin(y_test, [0, 2, 4, 6, 8])

x_train_even, y_train_even = x_train[train_mask_even], y_train[train_mask_even]
x_test_even, Y_test_even = x_test[test_mask_even], y_test[test_mask_even]

X_even = np.concatenate([x_train_even, x_test_even])

train_mask_odd = np.isin(y_train, [1, 3, 5, 7, 9])
test_mask_odd = np.isin(y_test, [1, 3, 5, 7, 9])

x_train_odd, y_train_odd = x_train[train_mask_odd], y_train[train_mask_odd]
x_test_odd, Y_test_odd = x_test[test_mask_odd], y_test[test_mask_odd]

X_odd = np.concatenate([x_train_odd, x_test_odd])

X_total = np.concatenate([x_train, x_test])

if nID == 0 or nID == 2:

    X = X_even

else:

    X = X_odd

X = X / 127.5 - 1

print("mnist data loaded")

loss_data_list = list()
inception_score_list = list()
frechet_inception_distance_list = list()


with tf.compat.v1.Session("grpc://" + workers[nID]) as mySess:

    mySess.run(tf.compat.v1.global_variables_initializer())
    myQueue = GradientExchange(mySess, cfg)

    myQueue.send_ready()
    myQueue.check_all_ready()
    myQueue.receive_go_sign()

    if cfg.synchronous_training:

        if nID == 0:

            myQueue.set_pongs()

    accuracies = list()
    elapsed_time = 0.0
    iteration = -1
    flag_stop_training = False

    for i in range(cfg.epochs):

        print("*** epoch %d ***" % (i + 1))

        for j in range(cfg.num_batches):

            if (j % cfg.num_workers) == nID:

                start_time = time.time()
                iteration += 1
                idx = np.random.randint(0, len(X), cfg.batch_size)
                idx_fid_500 = np.random.randint(0, len(X), 500)
                batch_X_fid_500 = X[idx_fid_500]
                batch_X = X[idx]
                batch_Z = np.random.uniform(-1, 1, (cfg.batch_size, cfg.z_shape))

                _grads_d, d_loss = mySess.run(
                    [params["gradient_d"], params["loss_d"]],
                    feed_dict={phX: batch_X, phZ: batch_Z}
                )

                _grads = mySess.run(params["gradient"], feed_dict={params["data"]["z"]: batch_Z})
                myQueue.enqueue(_grads + _grads_d, iteration)

                if myQueue.get_stop() == "True":

                    flag_stop_training = True

                    break

                if cfg.synchronous_training:

                    myQueue.receive_pong()

                total_grads = myQueue.get_others_grads()

                for w in range(len(cfg.weights)):

                    total_grads[w] = np.add(total_grads[w], (_grads + _grads_d)[w][0])

                _ = mySess.run(
                    params["optimizer"],
                    feed_dict={
                         params["new_g"]["W_conv1"]: total_grads[cfg.weights["W_conv1"]["wid"]],
                         params["new_g"]["W_conv2"]: total_grads[cfg.weights["W_conv2"]["wid"]],
                         params["new_g"]["W_conv3"]: total_grads[cfg.weights["W_conv3"]["wid"]],
                         params["new_g"]["W_conv4"]: total_grads[cfg.weights["W_conv4"]["wid"]],
                         params["new_g_d"]["W_conv1_d"]: total_grads[cfg.weights["W_conv1_d"]["wid"]],
                         params["new_g_d"]["W_conv2_d"]: total_grads[cfg.weights["W_conv2_d"]["wid"]],
                         params["new_g_d"]["W_conv3_d"]: total_grads[cfg.weights["W_conv3_d"]["wid"]],
                         params["new_g_d"]["W_conv4_d"]: total_grads[cfg.weights["W_conv4_d"]["wid"]],
                         params["new_g_d"]["W_conv5_d"]: total_grads[cfg.weights["W_conv5_d"]["wid"]]
                    }
                )

                _loss = mySess.run(params["loss"], feed_dict={params["data"]["z"]: batch_Z})

                print("[Node ID: %d] iter: %d, generator_loss: %f, discriminator_loss: %f" % \
                      (nID, iteration, _loss, d_loss))

                if cfg.testing:

                    if iteration == cfg.testing_iteration:

                        break

                elapsed_time += (time.time() - start_time)

        loss_data = {
            "epoch": i,
            "generator_loss": float(_loss),
            "discriminator_loss": float(d_loss)
        }

        loss_data_list.append(loss_data)

        with open('results/worker_%d_loss/worker_%d_loss.json' % (nID, nID), 'w') as f:

            json.dump(loss_data_list, f, indent=2, ensure_ascii=False)

        if (i % 50 == 0 and i < 500) or i % 500 == 0:

            z_500 = np.random.uniform(-1, 1, (500, cfg.z_shape))
            is_sample = mySess.run(generator, feed_dict={phZ: z_500})
            is_sample = np.stack((is_sample,) * 3, axis=3)
            is_sample = is_sample.reshape((500, 28, 28, 3))
            is_sample = np.moveaxis(is_sample, 3, 1)

            is_sample = (is_sample + 1) * 127.5

            batch_X_fid_500 = np.stack((batch_X_fid_500,) * 3, axis=3)
            batch_X_fid_500 = batch_X_fid_500.reshape((500, 28, 28, 3))
            batch_X_fid_500 = np.moveaxis(batch_X_fid_500, 3, 1)
            batch_X_fid_500 = (batch_X_fid_500 + 1) * 127.5
            FID = get_fid(is_sample, batch_X_fid_500)

            frechet_inception_score_data = {"epoch": i, "FID": float(FID)}
            frechet_inception_distance_list.append(frechet_inception_score_data)

            with open('results/worker_%d_fid_score/worker_%d_fid_score.json' % (nID, nID), 'w') as f:

                json.dump(frechet_inception_distance_list, f, indent=2, ensure_ascii=False)

        if flag_stop_training:

            break

    myQueue.terminate_threads()

    myQueue.send_ready()
    myQueue.check_all_ready()
    myQueue.receive_go_sign()

    os.system(term_cmd)

    print("Terminating server" + str(nID))

