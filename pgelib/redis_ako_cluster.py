import tensorflow.compat.v1 as tf
import subprocess


tf.disable_v2_behavior()


def build_cluster(cfg):

    workers = list()

    if cfg.remote is None:

        for i in range(cfg.num_workers):

            ipport = cfg.local_ip + ":" + str(cfg.worker_port + i)
            workers.append(ipport)

    else:

        for i in range(cfg.num_workers):

            ipport = cfg.remote_ip[cfg.remote[i]] + ":" + str(cfg.worker_port)
            workers.append(ipport)

    cluster = tf.train.ClusterSpec({"wk": workers})
    server = tf.train.Server(cluster, job_name=cfg.job_name, task_index=cfg.nID)
    print("Starting server /job:{}/task:{}".format(cfg.job_name, cfg.nID))

    redis_start_cmd = "redis-server --port %s &" % str(cfg.redis_port + cfg.nID)
    redis_process = subprocess.Popen(redis_start_cmd, shell=True)
    term_cmd = "kill -9 %s" % str(redis_process.pid + 1)

    return cluster, server, workers, term_cmd

