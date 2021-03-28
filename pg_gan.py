"""
Application entry point for PG-GAN.
Author: Sara Mathieson, Zhanpeng Wang, Jiaping Wang
Date 2/4/21
"""

# python imports
import datetime
import math
import numpy as np
import random
import sys
import tensorflow as tf
from scipy.optimize import basinhopping

# our imports
import discriminators
import real_data_random
import simulation
import util

from real_data_random import Region

# globals for simulated annealing
NUM_ITER = 300
BATCH_SIZE = 50
NUM_BATCH = 100
print("NUM_ITER", NUM_ITER)
print("BATCH_SIZE", BATCH_SIZE)
print("NUM_BATCH", NUM_BATCH)

# globals for data
NUM_SNPS = 36       # number of seg sites, should be divisible by 4
L = 50000           # heuristic to get enough SNPs for simulations
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

    # set up seeds
    if opts.seed != None:
        np.random.seed(opts.seed)
        random.seed(opts.seed)
        tf.random.set_seed(opts.seed)

    generator, discriminator, iterator, parameters = process_opts(opts)

    # grid search
    if opts.grid:
        print("Grid search not supported right now")
        sys.exit()
        #posterior, loss_lst = grid_search(discriminator, samples, simulator, \
        #    iterator, parameters, opts.seed)
    # simulated annealing
    else:
        posterior, loss_lst = simulated_annealing(generator, discriminator,\
            iterator, parameters, opts.seed, toy=opts.toy)

    print(posterior)
    print(loss_lst)

def process_opts(opts):

    # parameter defaults
    all_params = util.ParamSet()
    parameters = util.parse_params(opts.params, all_params) # desired params
    param_names = [p.name for p in parameters]

    # if real data provided
    real = False
    if opts.data_h5 != None:
        # most typical case for real data
        iterator = real_data_random.RealDataRandomIterator(NUM_SNPS, \
            opts.data_h5, opts.bed)
        num_samples = iterator.num_samples # TODO use num_samples below
        real = True

    filter = False # for filtering singletons

    # parse model and simulator
    model_sample_sizes = {
        'const': [198],
        'exp': [198],
        'im': [98,98],
        'ooa2': [98,98],
        'post_ooa': [98,98],
        'ooa3': [66,66,66],
    }
    if opts.model not in model_sample_sizes:
        sys.exit(opts.model + " is not recognized")
    sample_sizes = model_sample_sizes[opts.model]
    discriminator = discriminators.PopModel(*sample_sizes)
    simulator = getattr(simulation, "simulate_" + opts.model, None)
    if simulator == None:
        sys.exit("simulate_" + opts.model + " is not recognized")

    # generator
    generator = simulation.Generator(simulator, param_names, sample_sizes,\
        NUM_SNPS, L, opts.seed, mirror_real=real, reco_folder=opts.reco_folder,\
        filter=filter)

    # "real data" is simulated with fixed params
    if opts.data_h5 == None:
        iterator = simulation.Generator(simulator, param_names, sample_sizes, \
            NUM_SNPS, L, opts.seed, filter=filter) # don't need reco_folder

    return generator, discriminator, iterator, parameters

################################################################################
# SIMULATED ANNEALING
################################################################################

def simulated_annealing(generator, discriminator, iterator, parameters, seed, \
    toy=False):
    """Main function that drives GAN updates"""

    # main object for pg-gan
    pg_gan = PG_GAN(generator, discriminator, iterator, parameters, seed)

    # find starting point through pre-training (update generator in method)
    if not toy:
        s_current = pg_gan.disc_pretraining(800, BATCH_SIZE)
    else:
        s_current = [param.start() for param in pg_gan.parameters]
        pg_gan.generator.update_params(s_current)

    loss_curr = pg_gan.generator_loss(s_current)
    print("params, loss", s_current, loss_curr)

    posterior = [s_current]
    loss_lst = [loss_curr]
    real_acc_lst = []
    fake_acc_lst = []

    # simulated-annealing iterations
    num_iter = NUM_ITER
    # for toy example
    if toy:
        num_iter = 2

    # main pg-gan loop
    for i in range(num_iter):
        print("\nITER", i)
        print("time", datetime.datetime.now().time())
        T = temperature(i, num_iter) # reduce width of proposal over time

        # propose 10 updates per param and pick the best one
        s_best = None
        loss_best = float('inf')
        for k in range(len(parameters)): # trying all params!
            #k = random.choice(range(len(parameters))) # random param
            for j in range(10): # trying 10

                # can update all the parameters at once, or choose one at a time
                #s_proposal = [parameters[k].proposal(s_current[k], T) for k in\
                #    range(len(parameters))]
                s_proposal = [v for v in s_current] # copy
                s_proposal[k] = parameters[k].proposal(s_current[k], T)
                loss_proposal = pg_gan.generator_loss(s_proposal)

                print(j, "proposal", s_proposal, loss_proposal)
                if loss_proposal < loss_best: # minimizing loss
                    loss_best = loss_proposal
                    s_best = s_proposal

        # decide whether to accept or not (reduce accepting bad state later on)
        if loss_best <= loss_curr: # unsure about this equal here
            p_accept = 1
        else:
            p_accept = (loss_curr / loss_best) * T
        rand = np.random.rand()
        accept = rand < p_accept

        # if accept, retrain
        if accept:
            print("ACCEPTED")
            s_current = s_best
            generator.update_params(s_current)
            # train only if accept
            real_acc, fake_acc = pg_gan.train_sa(NUM_BATCH, BATCH_SIZE)
            loss_curr = loss_best

        # don't retrain
        else:
            print("NOT ACCEPTED")

        print("T, p_accept, rand, s_current, loss_curr", end=" ")
        print(T, p_accept, rand, s_current, loss_curr)
        posterior.append(s_current)
        loss_lst.append(loss_curr)

    return posterior, loss_lst

def temperature(i, num_iter):
    """Temperature controls the width of the proposal and acceptance prob."""
    return 1 - i/num_iter # start at 1, end at 0

# not used right now
"""
def grid_search(model_type, samples, demo_file, simulator, iterator, \
        parameters, is_range, seed):

    # can only do one param right now
    assert len(parameters) == 1
    param = parameters[0]

    all_values = []
    all_likelihood = []
    for fake_value in np.linspace(param.min, param.max, num=30):
        fake_params = [fake_value]
        model = TrainingModel(model_type, samples, demo_file, simulator, \
            iterator, parameters, is_range, seed)

        # train more for grid search
        model.train(fake_params, NUM_BATCH*10, BATCH_SIZE)
        test_acc, conf_mat = model.test(fake_params, NUM_TEST)
        like_curr = likelihood(test_acc)
        print("params, test_acc, likelihood", fake_value, test_acc, like_curr)

        all_values.append(fake_value)
        all_likelihood.append(like_curr)

    return all_values, all_likelihood
"""

################################################################################
# TRAINING
################################################################################

class PG_GAN:

    def __init__(self, generator, discriminator, iterator, parameters, seed):
        """Setup the model and training framework"""
        print("parameters", type(parameters), parameters)


        # set up generator and discriminator
        self.generator = generator
        self.discriminator = discriminator
        self.iterator = iterator # for training data (real or simulated)
        self.parameters = parameters

        # this checks and prints the model (1 is for the batch size)
        self.discriminator.build_graph((1, iterator.num_samples, NUM_SNPS, \
            NUM_CHANNELS))
        self.discriminator.summary()

        self.cross_entropy =tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.disc_optimizer = tf.keras.optimizers.Adam()

    def disc_pretraining(self, num_batches, batch_size):
        """Pre-train so discriminator has a chance to learn before generator"""
        s_best = []
        max_acc = 0
        k = 0

        # try with several random sets at first
        while max_acc < 0.9 and k < 10:
            s_trial = [param.start() for param in self.parameters]
            print("trial", k+1, s_trial)
            self.generator.update_params(s_trial)
            real_acc, fake_acc = self.train_sa(num_batches, batch_size)
            avg_acc = (real_acc + fake_acc)/2
            if avg_acc > max_acc:
                max_acc = avg_acc
                s_best = s_trial
            k += 1

        # now start!
        self.generator.update_params(s_best)
        return s_best

    def train_sa(self, num_batches, batch_size):
        """Train using fake_values for the simulated data"""

        for epoch in range(num_batches):

            real_regions = self.iterator.real_batch(batch_size, True)
            real_acc, fake_acc, disc_loss = self.train_step(real_regions)

            if (epoch+1) % 100 == 0:
                template = 'Epoch {}, Loss: {}, Real Acc: {}, Fake Acc: {}'
                print(template.format(epoch + 1,
                                disc_loss,
                                real_acc/BATCH_SIZE * 100,
                                fake_acc/BATCH_SIZE * 100))

        return real_acc/BATCH_SIZE, fake_acc/BATCH_SIZE

    def generator_loss(self, proposed_params):
        """ Generator loss """
        generated_regions = self.generator.simulate_batch(BATCH_SIZE, \
            params=proposed_params)
        # not training when we use the discriminator here
        fake_output = self.discriminator(generated_regions, training=False)
        loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        return loss.numpy()

    def discriminator_loss(self, real_output, fake_output):
        """ Discriminator loss """
        # accuracy
        real_acc = np.sum(real_output >= 0) # positive logit => pred 1
        fake_acc = np.sum(fake_output <  0) # negative logit => pred 0

        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        # add on entropy regularization (small penalty)
        real_entropy = self.cross_entropy(real_output, real_output)
        fake_entropy = self.cross_entropy(fake_output, fake_output)
        entropy = tf.math.scalar_mul(0.001/2, tf.math.add(real_entropy, \
            fake_entropy)) # can I just use +,*? TODO
        total_loss -= entropy # maximize entropy

        return total_loss, real_acc, fake_acc

    def train_step(self, real_regions):
        """One mini-batch for the discriminator"""

        with tf.GradientTape() as disc_tape:
            # use current params
            generated_regions = self.generator.simulate_batch(BATCH_SIZE)

            real_output = self.discriminator(real_regions, training=True)
            fake_output = self.discriminator(generated_regions, training=True)

            disc_loss, real_acc, fake_acc = self.discriminator_loss( \
                real_output, fake_output)

        # gradient descent
        gradients_of_discriminator = disc_tape.gradient(disc_loss, \
            self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, \
            self.discriminator.trainable_variables))

        return real_acc, fake_acc, disc_loss

if __name__ == "__main__":
    main()
