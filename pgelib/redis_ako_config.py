class Config():
    def __init__(self, **kwargs):

        self.job_name = kwargs.get("job_name", "wk")
        self.nID = kwargs.get("nID", 0)
        self.local_ip = "localhost"
        self.remote_ip = ["111.222.333.444", "111.222.333.555", "111.222.333.666", "111.222.333.777", "111.222.333.888"]
        self.remote = None
        self.num_workers = 4
        self.worker_port = 2523
        self.redis_port = 6380
        self.training_epochs = 10
        self.batch_size = 100
        self.num_batches = 10
        self.img_shape = (28, 28, 1)
        self.rows, self.cols, self.channels = (28, 28, 1)
        self.epochs = 50100
        self.z_shape = 100
        self.epochs_for_sample = 500
        self.beta1 = 0.5
        self.lr_gen = 0.0001
        self.lr_disc = 0.0001
        self.synchronous_training = False
        self.train_until_fixed_accuracy = False
        self.testing = False

        if self.train_until_fixed_accuracy:

            self.target_accuracy = 0.4
            self.iteration_to_check_accuracy = 5
            self.stop_time = 3000

        else:

            self.stop_time = 300

        if self.testing:

            self.training_epochs = 1
            self.testing_iteration = 10

        self.p = kwargs.get("p", [4, 4, 4, 4])
        self.synch_max_diff = 0
        self.fine_grained_partition = True
        self.num_dqthreads = 2
        self.weights = dict()

        self.weights["W_conv1"] = {"wid": 0, "num_parts": 4, "shape": (100, 7*7*512), "range": [0, 25, 50, 75, 100]}
        self.weights["W_conv2"] = {"wid": 1, "num_parts": 1, "shape": (3, 3, 512, 256), "range": [0, 256]}
        self.weights["W_conv3"] = {"wid": 2, "num_parts": 1, "shape": (3, 3, 256, 128), "range": [0, 128]}
        self.weights["W_conv4"] = {"wid": 3, "num_parts": 1, "shape": (3, 3, 128, 1), "range": [0, 1]}

        self.weights["W_conv1_d"] = {"wid": 4, "num_parts": 1, "shape": (5, 5, 1, 64), "range": [0, 64]}
        self.weights["W_conv2_d"] = {"wid": 5, "num_parts": 1, "shape": (3, 3, 64, 64), "range": [0, 64]}
        self.weights["W_conv3_d"] = {"wid": 6, "num_parts": 1, "shape": (3, 3, 64, 128), "range": [0, 128]}
        self.weights["W_conv4_d"] = {"wid": 7, "num_parts": 1, "shape": (2, 2, 128, 256), "range": [0, 256]}
        self.weights["W_conv5_d"] = {"wid": 8, "num_parts": 1, "shape": (7 * 7 * 256, 1), "range": [0, 1]}

        self.subweights = dict()
        self.subweights["1@W_conv1"] = {"wid": 9, "part": 1, "shape": (25, 7*7*512)}
        self.subweights["2@W_conv1"] = {"wid": 10, "part": 2, "shape": (25, 7*7*512)}
        self.subweights["3@W_conv1"] = {"wid": 11, "part": 3, "shape": (25, 7*7*512)}
        self.subweights["4@W_conv1"] = {"wid": 12, "part": 4, "shape": (25, 7*7*512)}

        if self.fine_grained_partition:

            # TODO: check these dimensions

            self.partitions = dict()
            self.partitions[1] = [
                ["W_conv1",
                "W_conv2",
                "W_conv3",
                "W_conv4",
                "W_conv1_d",
                "W_conv2_d",
                "W_conv3_d",
                "W_conv4_d",
                "W_conv5_d"]
            ]
            self.partitions[2] = [
                ["W_conv1", "W_conv2", "1@W_conv1", "2@W_conv1", "W_conv1_d", "W_conv2_d", "W_conv3_d"],
                ["W_conv3", "W_conv4", "3@W_conv1", "4@W_conv1", "W_conv4_d", "W_conv5_d"]
            ]
            self.partitions[4] = [
                ["W_conv1", "1@W_conv1", "W_conv1_d", "W_conv2_d"],
                ["W_conv2", "2@W_conv1", "W_conv3_d"],
                ["W_conv3", "3@W_conv1", "W_conv4_d"],
                ["W_conv4", "4@W_conv1", "W_conv5_d"]
            ]

        else:

            # TODO: check these dimensions
            self.partitions = dict()
            self.partitions[1] = [["W_conv1", "W_conv2", "W_conv3", "W_conv4"]]
            self.partitions[2] = [["W_conv1", "W_conv2", "W_conv3"], ["W_conv4"]]
            self.partitions[4] = [["W_conv1"], ["W_conv4"], ["W_conv2"], ["W_conv3"]]

