"""
Simulate data for training or testing using msprime.
Author: Sara Matheison, Zhanpeng Wang, Jiaping Wang
Date: 7/16/20
"""

# python imports
import math
import msprime
import numpy as np
import random
import sys
import time

# our imports
import real_data_random
import util

################################################################################
# SIMULATION
################################################################################

class Simulator:

    def __init__(self, sim_real, simulator, param_names, sample_sizes, \
            num_snps, L):
        self.sim_real = sim_real
        self.simulator = simulator
        self.param_names = param_names
        self.sample_sizes = sample_sizes
        self.num_snps = num_snps
        self.L = L

        # for real data, use HapMap
        self.prior = []
        self.weights = []
        if sim_real == "real":
            files = [real_data_random.BIG_DATA+"genetic_map_GRCh37_chr" + \
                str(i) + ".txt" for i in range(1,23)]
            self.prior, self.weights = util.parse_hapmap_empirical_prior(files)


    def simulate_batch(self, num_data, fake_values):

        # initialize 4D matrix (two channels for distances)
        all_regions = np.zeros((num_data, sum(self.sample_sizes), \
            self.num_snps, 2), dtype=np.float32) # two channels
        # initialize labels (one-hot, real/fake)
        all_labels = np.zeros((num_data, 2), dtype=np.float32) # two classes

        # set up parameters for real and fake
        real_params = util.ParamSet()
        sim_params = util.ParamSet()
        sim_params.update(self.param_names, fake_values)

        # simulate each region
        for i in range(num_data):

            # decide real or fake
            fake = np.random.randint(0,2)

            if fake >= 1: # label 1 (sim)
                all_regions[i] = self.simulator(sim_params, self.sample_sizes, \
                    self.num_snps, self.L)
            else: # label 0 (real)
                all_regions[i] = self.simulator(real_params, self.sample_sizes,\
                    self.num_snps, self.L)

            # compute label
            label = np.zeros(2)
            label[fake] = 1
            all_labels[i] = label

        return all_regions, all_labels

    def simulate_batch_real(self, num_data, fake_values, real_iterator, \
        is_train):

        # initialize 4D matrix (two channels for distances)
        all_regions = np.zeros((num_data, sum(self.sample_sizes), \
            self.num_snps, 2), dtype=np.float32) # two channels
        # initialize labels (one-hot, real/fake)
        all_labels = np.zeros((num_data, 2), dtype=np.float32) # two classes

        # set up parameters for simulations
        sim_params = util.ParamSet()
        sim_params.update(self.param_names, fake_values)

        # simulate each region
        for i in range(num_data):

            # decide real or fake
            fake = np.random.randint(0,2)
            if fake >= 1: # label 1 (sim)
                all_regions[i] = self.simulator(sim_params, self.sample_sizes, \
                    self.num_snps, self.L, prior=self.prior, \
                    weights=self.weights)
            else: # label 0 (real)
                all_regions[i] = real_iterator.real_region(is_train)

            # compute label
            label = np.zeros(2)
            label[fake] = 1
            all_labels[i] = label

        return all_regions, all_labels

def draw_background_rate_from_prior(prior_rates, prob):
    return np.random.choice(prior_rates, p=prob)

def simulate_im(params, sample_sizes, num_snps, L):
    """Note this is a 2 population model"""

    # condense params
    N1 = params.N1.value
    N2 = params.N2.value
    T_split = params.T_split.value
    N_anc = params.N_anc.value
    mig = params.mig.value

    population_configurations = [
        msprime.PopulationConfiguration(sample_size=sample_sizes[0], \
            initial_size = N1),
        msprime.PopulationConfiguration(sample_size=sample_sizes[1], \
            initial_size = N2)]

    # no migration initially
    mig_time = T_split/2

    # directional (pulse)
    if mig >= 0:
        # migration from pop 1 into pop 0 (back in time)
        mig_event = msprime.MassMigration(time = mig_time, source = 1, \
            destination = 0, proportion = abs(mig))
    else:
        # migration from pop 0 into pop 1 (back in time)
        mig_event = msprime.MassMigration(time = mig_time, source = 0, \
            destination = 1, proportion = abs(mig))

    demographic_events = [
        mig_event,
		# move all in deme 1 to deme 0
		msprime.MassMigration(
			time = T_split, source = 1, destination = 0, proportion = 1.0),
        # change to ancestral size
        msprime.PopulationParametersChange(time=T_split, initial_size=N_anc, \
            population_id=0)
	]
    '''dd = msprime.DemographyDebugger(
	       	population_configurations=population_configurations,
        	migration_matrix=mig_matrix,
        	demographic_events=demographic_events)

    dd.print_history()'''

    # simulate tree sequence
    ts = msprime.simulate(
		population_configurations = population_configurations,
		demographic_events = demographic_events,
		mutation_rate = params.mut.value,
		length = L,
		recombination_rate = params.reco.value)

    gt_matrix = ts.genotype_matrix().astype(float)
    snps_total = gt_matrix.shape[0]

    positions = [round(variant.site.position) for variant in ts.variants()]
    assert len(positions) == snps_total
    dist_vec = [0] + [(positions[j+1] - positions[j])/L for j in \
        range(snps_total-1)]

    return util.process_gt_dist(gt_matrix, dist_vec, num_snps)

def simulate_ooa2(params, sample_sizes, num_snps, L, prior=[], weights=[]):
    """Note this is a 2 population model"""

    # sample reco or use value
    if prior != []:
        reco = draw_background_rate_from_prior(prior, weights)
    else:
        reco = params.reco.value

    # condense params
    T1 = params.T1.value
    T2 = params.T2.value
    mig = params.mig.value

    population_configurations = [
        msprime.PopulationConfiguration(sample_size=sample_sizes[0], \
            initial_size = params.N3.value), # YRI is first
        msprime.PopulationConfiguration(sample_size=sample_sizes[1], \
            initial_size = params.N2.value)] # CEU/CHB is second

    # directional (pulse)
    if mig >= 0:
        # migration from pop 1 into pop 0 (back in time)
        mig_event = msprime.MassMigration(time = T2, source = 1, \
            destination = 0, proportion = abs(mig))
    else:
        # migration from pop 0 into pop 1 (back in time)
        mig_event = msprime.MassMigration(time = T2, source = 0, \
            destination = 1, proportion = abs(mig))

    demographic_events = [
        mig_event,
        # change size of EUR
        msprime.PopulationParametersChange(time=T2, \
            initial_size=params.N1.value, population_id=1),
		# move all in deme 1 to deme 0
		msprime.MassMigration(time = T1, source = 1, destination = 0, \
            proportion = 1.0),
        # change to ancestral size
        msprime.PopulationParametersChange(time=T1, \
            initial_size=params.N_anc.value, population_id=0)
	]
    '''dd = msprime.DemographyDebugger(
	       	population_configurations=population_configurations,
        	#migration_matrix=migration_matrix,
        	demographic_events=demographic_events)

    dd.print_history()'''

    ts = msprime.simulate(
		population_configurations = population_configurations,
		demographic_events = demographic_events,
		mutation_rate = params.mut.value,
		length = L,
		recombination_rate = reco)

    gt_matrix = ts.genotype_matrix().astype(float)
    snps_total = gt_matrix.shape[0]

    positions = [round(variant.site.position) for variant in ts.variants()]
    assert len(positions) == snps_total
    dist_vec = [0] + [(positions[j+1] - positions[j])/L for j in \
        range(snps_total-1)]

    return util.process_gt_dist(gt_matrix, dist_vec, num_snps)

def simulate_exp(params, sample_sizes, num_snps, L, prior=[], weights=[]):
    """Note this is a 1 population model"""

    # sample reco or use value
    if prior != []:
        reco = draw_background_rate_from_prior(prior, weights)
    else:
        reco = params.reco.value

    T2 = params.T2.value
    N2 = params.N2.value

    N0 = N2 / math.exp(-params.growth.value * T2)

    demographic_events = [
        msprime.PopulationParametersChange(time=0, initial_size=N0, \
            growth_rate=params.growth.value),
        msprime.PopulationParametersChange(time=T2, initial_size=N2, \
            growth_rate=0),
		msprime.PopulationParametersChange(time=params.T1.value, \
            initial_size=params.N1.value)
	]
    '''dd = msprime.DemographyDebugger(
	       	#population_configurations=population_configurations,
        	#migration_matrix=migration_matrix,
        	demographic_events=demographic_events)

    dd.print_history()'''

    ts = msprime.simulate(sample_size = sum(sample_sizes),
		demographic_events = demographic_events,
		mutation_rate = params.mut.value,
		length = L,
		recombination_rate = reco)

    gt_matrix = ts.genotype_matrix().astype(float)
    snps_total = gt_matrix.shape[0]

    positions = [round(variant.site.position) for variant in ts.variants()]
    assert len(positions) == snps_total
    dist_vec = [0] + [(positions[j+1] - positions[j])/L for j in \
        range(snps_total-1)]

    return util.process_gt_dist(gt_matrix, dist_vec, num_snps, filter=False)


def simulate_const(params, sample_sizes, num_snps, L, prior=[], weights=[]):

    # sample reco or use value
    if prior != []:
        reco = draw_background_rate_from_prior(prior, weights)
    else:
        reco = params.reco.value

    # simulate data
    ts = msprime.simulate(sample_size=sum(sample_sizes), Ne=params.Ne.value, \
        length=L, mutation_rate=params.mut.value, recombination_rate=reco)

    gt_matrix = ts.genotype_matrix().astype(float)
    snps_total = gt_matrix.shape[0]

    positions = [round(variant.site.position) for variant in ts.variants()]
    assert len(positions) == snps_total
    dist_vec = [0] + [(positions[j+1] - positions[j])/L for j in \
        range(snps_total-1)]

    return util.process_gt_dist(gt_matrix, dist_vec, num_snps)

# testing
if __name__ == "__main__":

    batch_size = 100
    S = 36
    R = 50000

    ########
    # SIMS #
    ########

    print("sim ooa")
    ss = [20,20,20]
    sim_ooa = Simulator("sim", simulate_ooa, ["N_A", "N_B"], ss, S, R)
    all_regions, all_labels = sim_ooa.simulate_batch(batch_size, [1000, 4000])
    print("x", all_regions.shape)
    print("y", all_labels.shape)

    print("sim im")
    ss = [20,20]
    sim_im = Simulator("sim", simulate_im, ["N_anc", "T_split"], ss, S, R)
    all_regions, all_labels = sim_im.simulate_batch(batch_size, [16000, 4000])
    print("x", all_regions.shape)
    print("y", all_labels.shape)

    ########
    # REAL #
    ########

    region_file = real_data_random.BIG_DATA + "ceu_s36.npy"
    ss = [0,198,0]
    iterator = real_data_random.RealDataRandomIterator(ss, S, R, region_file, \
        num_test=20)

    print("sim exp")
    sim_im = Simulator("real", simulate_exp, ["N1", "T1"], ss, S, R)
    all_regions, all_labels = sim_im.simulate_batch_real(batch_size, [16000, \
        4000], iterator, True)
    print("x", all_regions.shape)
    print("y", all_labels.shape)

    print("sim const")
    sim_im = Simulator("real", simulate_const, ["Ne"], ss, S, R)
    all_regions, all_labels = sim_im.simulate_batch_real(batch_size, [16000], \
        iterator, True)
    print("x", all_regions.shape)
    print("y", all_labels.shape)

    # TODO fix this one (wrong iterator, need two population)
    print("sim ooa2")
    ss = [20,20]
    sim_im = Simulator("real", simulate_ooa2, ["N_anc", "T1"], ss, S, R)
    all_regions, all_labels = sim_im.simulate_batch_real(batch_size, [16000, \
        4000], iterator, True)
    print("x", all_regions.shape)
    print("y", all_labels.shape)
