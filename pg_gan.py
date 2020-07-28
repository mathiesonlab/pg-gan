"""
Application entry point for PG-GAN. Options include:
-grid search
-simulated annealing

Author: Sara Mathieson, Zhanpeng Wang, Jiaping Wang
Date 7/25/20
"""

# python imports
import datetime
import math
import numpy as np
import random
import sys
import tensorflow as tf

# our imports
import models
import real_data_random
import simulation
import util

from real_data_random import Region

# globals for simulated annealing
NUM_ITER = 200
BATCH_SIZE = 50
NUM_TEST = 500
NUM_BATCH = 200
print("NUM_ITER", NUM_ITER)
print("BATCH_SIZE", BATCH_SIZE)
print("NUM_TEST", NUM_TEST)
print("NUM_BATCH", NUM_BATCH)

# globals for data
NUM_SNPS = 36       # number of seg sites, should be divisible by 4
L = 50000           # length of each region
NUM_CLASSES = 2     # "real" vs "simulated"
NUM_CHANNELS = 2    # SNPs and distances
print("NUM_SNPS", NUM_SNPS)
print("L", L)
print("NUM_CLASSES", NUM_CLASSES)
print("NUM_CHANNELS", NUM_CHANNELS)

def main():
    """Parse args and run simulated annealing"""

    opts = util.parse_args()
    print(opts)
    model_type, samples, demo_file, simulator, iterator, parameters = \
        process_opts(opts)

    is_range = (opts.is_range == "range")

    # grid search
    if opts.grid == 'grid':
        posterior, test_acc_lst = grid_search(model_type, samples, \
            demo_file, simulator, iterator, parameters, is_range)
    # simulated annealing
    else:
        posterior, test_acc_lst = simulated_annealing(model_type, samples, \
            demo_file, simulator, iterator, parameters, is_range, toy=opts.toy)

    print(posterior)
    print(test_acc_lst)

def process_opts(opts):
    # TODO make file names command-line parameters

    # parse model and simulator
    model_type = models.SinglePopModel()
    simulator = simulation.simulate_exp # TODO make option for simulate_const
    demo_file = None # not used right now
    region_file = None

    # out-of-Africa model (3 populations)
    if opts.model == 'ooa':
        samples = [66,66,66]
        model_type = models.OOAmodel(samples[0], samples[1], samples[2])
        simulator = simulation.simulate_ooa
        region_file = real_data_random.BIG_DATA + "yri_ceu_chb_s36.npy"

    # isolation-with-migration model (2 populations)
    elif opts.model == 'im':
        samples = [98,98]
        model_type = models.IMmodel(samples[0], samples[1])
        simulator = simulation.simulate_im
        region_file = real_data_random.BIG_DATA + "yri_chb_s36.npy"

    # out-of-Africa model (2 populations)
    elif opts.model == 'ooa2':
        samples = [98,98]
        model_type = models.IMmodel(samples[0], samples[1])
        simulator = simulation.simulate_ooa2
        region_file = real_data_random.BIG_DATA + "yri_chb_s36.npy"

    # single-pop model (YRI)
    elif opts.model == 'yri':
        samples = [198,0,0]
        demo_file = 'hist/yri_4N0.psmc.h'
        region_file = real_data_random.BIG_DATA + "yri_s36.npy"

    # single-pop model (CEU)
    elif opts.model == 'ceu':
        samples = [0,198,0]
        demo_file = 'hist/fre_4N0.psmc.h'
        region_file = real_data_random.BIG_DATA + "ceu_s36.npy"

    # single-pop model (CHB)
    elif opts.model == 'chb':
        samples = [0,0,198]
        demo_file = 'hist/han_4N0.psmc.h'
        region_file = real_data_random.BIG_DATA + "chb_s36.npy"

    # simulations in the style of real data
    elif opts.model == 'exp':
        samples = [0,198,0]
        region_file = simulate_real.BIG_DATA + "sim_exp_s36.npy"

    # no other options
    else:
        sys.exit(opts.model + " is not recognized")

    # real vs. simulated
    iterator = None
    if opts.sim_real == 'real':
        # we don't really use this option anymore
        if opts.model == 'exp':
            print('region file', region_file)
            iterator = simulate_real.SimRealIterator(samples, NUM_SNPS, \
                L, region_file, num_test=NUM_TEST//2) # only need half from real

        # most typical case for real data
        else:
            iterator = real_data_random.RealDataRandomIterator(samples, \
                NUM_SNPS, L, region_file, num_test=NUM_TEST//2) # half from real

    # not using range right now
    if opts.is_range == 'range':
        sys.exit("Unsupported option range")

    # parameter defaults
    all_params = util.ParamSet()
    parameters = util.parse_params(opts.params, all_params) # desired params

    return model_type, samples, demo_file, simulator, iterator, parameters


################################################################################
# SIMULATED ANNEALING
################################################################################


def simulated_annealing(model_type, samples, demo_file, simulator, iterator, \
        parameters, is_range, toy=False):
    """Main function that drives GAN updates"""

    # initialize params
    if is_range:
        s_current = [param.start_range() for param in parameters]
    else:
        s_current = [param.start() for param in parameters]
    print("init", s_current)

    # compute "likelihood"
    model = TrainingModel(model_type, samples, demo_file, simulator, iterator, \
            parameters, is_range)
    model.train(s_current, NUM_BATCH, BATCH_SIZE)
    test_acc, conf_mat = model.test(s_current, NUM_TEST)
    likelihood_prev = likelihood(test_acc)
    print("params, test_acc", s_current, test_acc)

    posterior = [s_current]
    test_acc_lst = [test_acc]

    # simulated-annealing iterations
    num_iter = NUM_ITER + len(parameters)*150 # more iterations for more params
    # for toy example
    if toy:
        num_iter = 2
    for i in range(num_iter):
        print("\nITER", i)
        print("time", datetime.datetime.now().time())
        T = temperature(i, num_iter) # reduce width of proposal over time

        # propose 5 updates and pick the best one based on test accuracy
        s_best = None
        likelihood_best = -float('inf')
        k = random.choice(range(len(parameters))) # random param for this iter
        for j in range(5):
            if is_range:
                s_proposal = [parameters[k].proposal_range(s_current[k], T) for\
                    k in range(len(parameters))]
            else:
                # can update all the parameters at once, or choose one at a time
                #s_proposal = [parameters[k].proposal(s_current[k], T) for k in\
                #    range(len(parameters))]
                s_proposal = [v for v in s_current] # copy
                s_proposal[k] = parameters[k].proposal(s_current[k], T)

            #model.train(s_proposal, NUM_BATCH, BATCH_SIZE) # don't train first
            test_acc_proposal, conf_mat_proposal = model.test(s_proposal, \
                NUM_TEST)
            likelihood_proposal = likelihood(test_acc_proposal)

            print(j, "proposal", s_proposal, test_acc_proposal, \
                likelihood_proposal)
            if likelihood_proposal > likelihood_best:
                likelihood_best = likelihood_proposal
                s_best = s_proposal

        # decide whether to accept or not (reduce accepting bad state later on)
        if likelihood_best >= likelihood_prev: # unsure about this equal here
            p_accept = 1
        else:
            p_accept = (likelihood_best / likelihood_prev) * T
        rand = np.random.rand()
        accept = rand < p_accept

        # if accept, retrain
        if accept:
            print("ACCEPTED")
            s_current = s_best
            model.train(s_current, NUM_BATCH, BATCH_SIZE) # train only if accept
            # redo testing to get an accurate value, but save old for comparison
            likelihood_prev = likelihood_best
            test_acc, conf_mat = model.test(s_current, NUM_TEST)

        print("T, p_accept, rand, s_current, test_acc, conf_mat", end=" ")
        print(T, p_accept, rand, s_current, test_acc, conf_mat)
        posterior.append(s_current)
        test_acc_lst.append(test_acc)

    return posterior, test_acc_lst

def likelihood(test_acc):
    """
    Compute pseudo-likelihood based on test accuracy (triangle shaped):
    0 -> 0
    0.5 - 1
    1.0 -> 0
    """
    return 1 - abs(2*test_acc - 1)

def temperature(i, num_iter):
    """Temperature controls the width of the proposal and acceptance prob."""
    return 1 - i/num_iter # start at 1, end at 0

def grid_search(model_type, samples, demo_file, simulator, iterator, \
        parameters, is_range):
    """Simple grid search"""

    # can only do one param right now
    assert len(parameters) == 1
    param = parameters[0]

    all_values = []
    all_likelihood = []
    for fake_value in np.linspace(param.min, param.max, num=30):
        fake_params = [fake_value]
        model = TrainingModel(model_type, samples, demo_file, simulator, \
            iterator, parameters, is_range)

        # train more for grid search
        model.train(fake_params, NUM_BATCH*10, BATCH_SIZE)
        test_acc, conf_mat = model.test(fake_params, NUM_TEST)
        like_curr = likelihood(test_acc)
        print("params, test_acc, likelihood", fake_value, test_acc, like_curr)

        all_values.append(fake_value)
        all_likelihood.append(like_curr)

    return all_values, all_likelihood


################################################################################
# TRAINING
################################################################################


class TrainingModel:

    def __init__(self, model, sample_sizes, demo_file, simulator, iterator, \
            parameters, is_range):
        """Setup the model and training framework"""

        # set up simulator
        sim_real = "real"
        if iterator == None:
            sim_real = "sim"
        param_names = [p.name for p in parameters]
        self.simulator = simulation.Simulator(sim_real, simulator, param_names,\
            sample_sizes, NUM_SNPS, L)

        self.model = model
        self.sample_sizes = sample_sizes
        self.num_samples = sum(sample_sizes)
        self.demo_file = demo_file
        self.iterator = iterator
        self.parameters = parameters
        self.is_range = is_range

        # this checks and prints the model (1 is for the batch size)
        self.model.build_graph((1,self.num_samples,NUM_SNPS,NUM_CHANNELS))
        self.model.summary()

        self.loss_object = tf.keras.losses.CategoricalCrossentropy()

        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name = \
            'train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name = \
            'test_accuracy')

    def train(self, fake_values, num_batches, batch_size):
        """Train using fake_values for the simulated data"""

        # need to reinitialize the data since real/fake are different
        train_ds = tf.data.Dataset.from_generator(
            self.simulation_iterator,
            args = [batch_size, fake_values, True],
            output_types=(tf.float32, tf.float32),
            output_shapes=([batch_size,self.num_samples,NUM_SNPS,NUM_CHANNELS],\
                [batch_size,NUM_CLASSES]))

        for epoch in range(num_batches):
            # reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            include_conf = False
            if (epoch+1) % 100 == 0:
                include_conf = True

            regions, labels = next(iter(train_ds))
            conf_mat = train_step(regions, labels, self.model, \
                self.loss_object, self.optimizer, self.train_loss, \
                self.train_accuracy, confusion=include_conf)

            if (epoch+1) % 100 == 0:
                template = 'Epoch {}, Loss: {}, Accuracy: {}, Conf Matrix: {}'
                print(template.format(epoch + 1,
                                self.train_loss.result(),
                                self.train_accuracy.result() * 100,
                                np.array(conf_mat).tolist()))

    def test(self, fake_values, num_test):
        """No training, just test"""

        test_ds = tf.data.Dataset.from_generator(
            self.simulation_iterator,
            args = [num_test, fake_values, False],
            output_types=(tf.float32, tf.float32),
            output_shapes=([num_test,self.num_samples,NUM_SNPS,\
                NUM_CHANNELS], [num_test,NUM_CLASSES]))

        # finally: test and return
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        test_regions, test_labels = next(iter(test_ds))
        conf_mat = test_step(test_regions, test_labels, self.model, \
            self.loss_object, self.test_loss, self.test_accuracy)

        # this is the "value" of the function (can be transformed)
        return self.test_accuracy.result().numpy(), np.array(conf_mat).tolist()

    def simulation_iterator(self, batch_size, fake_values, is_train):
        """Simulate a batch that contains both "real" and "fake" values"""
        while True:
            # real data
            if self.iterator != None:
                x_batch,y_batch = self.simulator.simulate_batch_real( \
                    batch_size, fake_values, self.iterator, is_train)
            # simulated data
            else:
                x_batch,y_batch = self.simulator.simulate_batch(batch_size, \
                    fake_values)
            yield x_batch,y_batch

def train_step(regions, labels, model, loss_object, optimizer, train_loss, \
        train_accuracy, confusion=False):
    """From TF2 tutorial"""
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(regions, training=True)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

    # periodically print confusion matrix
    if confusion:
        labels_arr = tf.argmax(labels, axis = 1)
        predictions_arr = tf.argmax(predictions, axis = 1)
        return tf.math.confusion_matrix(labels_arr, predictions_arr, \
            num_classes=2)

def test_step(regions, labels, model, loss_object, test_loss, test_accuracy):
    """From TF2 tutorial"""
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(regions, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

    labels_arr = tf.argmax(labels, axis = 1)
    predictions_arr = tf.argmax(predictions, axis = 1)
    return tf.math.confusion_matrix(labels_arr, predictions_arr, num_classes=2)

if __name__ == "__main__":
    main()
