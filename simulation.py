"""
Simulate data for training or testing using msprime.
Author: Sara Matheison, Zhanpeng Wang, Jiaping Wang
Date: 2/4/21
"""

# python imports
import math
import msprime
import numpy as np
import random
import sys
import time

from numpy.random import default_rng

# from stdpopsim
import sps.engines
import sps.species
import sps.HomSap

# our imports
import global_vars
import real_data_random
import util

################################################################################
# SIMULATION
################################################################################

class Generator:

    def __init__(self, simulator, param_names, sample_sizes, seed,\
                 mirror_real=False, reco_folder=""):
        self.simulator = simulator
        self.param_names = param_names
        self.sample_sizes = sample_sizes
        self.num_samples = sum(sample_sizes)
        self.rng = default_rng(seed)
        self.curr_params = None

        self.pretraining = False

        # for real data, use HapMap
        if mirror_real and reco_folder != None:
            pop = reco_folder[-4: -1]

            if global_vars.new_data:
                files = [reco_folder + pop + \
                  "_recombination_map_hapmap_format_hg38_chr_" + str(i) + \
                  ".txt" for i in global_vars.CHROM_RANGE]
            else:
                files = [reco_folder + "genetic_map_GRCh37_chr" + str(i) + ".txt" \
                   for i in global_vars.CHROM_RANGE]

            self.prior, self.weights = util.parse_hapmap_empirical_prior(files)

        else:
            self.prior, self.weights = [], []

    def simulate_batch(self, batch_size, params=[], real=False, neg1=True):

        # initialize 4D matrix (two channels for distances)
        regions = np.zeros((batch_size, self.num_samples, global_vars.NUM_SNPS, \
                            2), dtype=np.float32) # two channels

        # set up parameters
        sim_params = util.ParamSet()
        if real:
            pass # keep orig for "fake" real
        elif params == []:
            sim_params.update(self.param_names, self.curr_params)
        else:
            sim_params.update(self.param_names, params)

        # simulate each region
        for i in range(batch_size):
            seed = self.rng.integers(1,high=2**32) # like GAN "noise"

            ts = self.simulator(sim_params, self.sample_sizes, seed, \
                                self.get_reco(sim_params))
            regions[i] = prep_region(ts, neg1)

        return regions

    def real_batch(self, batch_size, is_train): # ignore is_train
        return self.simulate_batch(batch_size, real=True)

    def update_params(self, new_params):
        self.curr_params = new_params

    def get_reco(self, params):
        if self.prior == []:
            reco =  np.random.normal(1e-8, 1e-8)
            reco = params.reco.value if reco <= 0 else reco
            return reco

        return draw_background_rate_from_prior(self.prior, self.weights)
    
def draw_background_rate_from_prior(prior_rates, prob):
    return np.random.choice(prior_rates, p=prob)

def prep_region(ts, neg1):
    """Gets simulated data ready"""
    gt_matrix = ts.genotype_matrix().astype(float)
    snps_total = gt_matrix.shape[0]

    positions = [round(variant.site.position) for variant in ts.variants()]
    assert len(positions) == snps_total
    dist_vec = [0] + [(positions[j+1] - positions[j])/global_vars.L for j in \
        range(snps_total-1)]

    # when mirroring real data
    return util.process_gt_dist(gt_matrix, dist_vec, neg1=neg1)

def simulate_im(params, sample_sizes, seed, reco):
    """Note this is a 2 population model"""
    assert len(sample_sizes) == 2

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

    # simulate tree sequence
    ts = msprime.simulate(
		population_configurations = population_configurations,
		demographic_events = demographic_events,
		mutation_rate = params.mut.value,
		length = global_vars.L,
		recombination_rate = reco,
        random_seed = seed)

    return ts

def simulate_ooa2(params, sample_sizes,seed, reco):
    """Note this is a 2 population model"""
    assert len(sample_sizes) == 2

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

    ts = msprime.simulate(
		population_configurations = population_configurations,
		demographic_events = demographic_events,
		mutation_rate = params.mut.value,
		length =  global_vars.L,
		recombination_rate = reco,
        random_seed = seed)

    return ts

def simulate_postOOA(params, sample_sizes, L, seed, reco):
    """Note this is a 2 population model for CEU/CHB split"""
    assert len(sample_sizes) == 2

    # condense params
    T1 = params.T1.value
    T2 = params.T2.value
    mig = params.mig.value
    #m_EU_AS = params.m_EU_AS.value

    population_configurations = [
        msprime.PopulationConfiguration(sample_size=sample_sizes[0], \
            initial_size = params.N3.value), # CEU is first
        msprime.PopulationConfiguration(sample_size=sample_sizes[1], \
            initial_size = params.N2.value)] # CHB is second

    # symmetric migration
    #migration_matrix=[[0, m_EU_AS],
    #                  [m_EU_AS, 0]]

    # directional (pulse)
    if mig >= 0:
        # migration from pop 1 into pop 0 (back in time)
        mig_event = msprime.MassMigration(time = T2/2, source = 1, \
            destination = 0, proportion = abs(mig))
    else:
        # migration from pop 0 into pop 1 (back in time)
        mig_event = msprime.MassMigration(time = T2/2, source = 0, \
            destination = 1, proportion = abs(mig))

    demographic_events = [
        mig_event,
		# move all in deme 1 to deme 0
		msprime.MassMigration(time = T2, source = 1, destination = 0, \
            proportion = 1.0),
        # set mig rate to zero (need if using migration_matrix)
        #msprime.MigrationRateChange(time=T2, rate=0),
        # ancestral bottleneck
        msprime.PopulationParametersChange(time=T2, \
            initial_size=params.N1.value, population_id=0),
        # ancestral size
        msprime.PopulationParametersChange(time=T1, \
            initial_size=params.N_anc.value, population_id=0)
	]

    ts = msprime.simulate(
		population_configurations = population_configurations,
		demographic_events = demographic_events,
        #migration_matrix = migration_matrix,
		mutation_rate = params.mut.value,
		length =  global_vars.L,
		recombination_rate = reco,
        random_seed = seed)

    return ts

def simulate_exp(params, sample_sizes, seed, reco):
    """Note this is a 1 population model"""
    assert len(sample_sizes) == 1

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

    ts = msprime.simulate(sample_size = sum(sample_sizes),
		demographic_events = demographic_events,
		mutation_rate = params.mut.value,
		length =  global_vars.L,
		recombination_rate = reco,
        random_seed = seed)

    return ts

def simulate_const(params, sample_sizes, seed, reco):
    assert len(sample_sizes) == 1

    # simulate data
    ts = msprime.simulate(sample_size=sum(sample_sizes), Ne=params.Ne.value, \
        length=global_vars.L, mutation_rate=params.mut.value, recombination_rate=reco, \
        random_seed = seed)

    return ts

def simulate_ooa3(params, sample_sizes, seed, reco):
    """From OOA3 as implemented in stdpopsim"""
    assert len(sample_sizes) == 3

    sp = sps.species.get_species("HomSap")

    mult = global_vars.L/141213431 # chr9
    contig = sp.get_contig("chr9",length_multiplier=mult) # TODO vary the chrom

    # 14 params
    N_A = params.N_A.value
    N_B = params.N_B.value
    N_AF = params.N_AF.value
    N_EU0 = params.N_EU0.value
    N_AS0 = params.N_AS0.value
    r_EU = params.r_EU.value
    r_AS = params.r_AS.value
    T_AF = params.T_AF.value
    T_B = params.T_B.value
    T_EU_AS = params.T_EU_AS.value
    m_AF_B = params.m_AF_B .value
    m_AF_EU = params.m_AF_EU.value
    m_AF_AS = params.m_AF_AS.value
    m_EU_AS = params.m_EU_AS.value

    model = sps.HomSap.ooa_3(N_A, N_B, N_AF, N_EU0, N_AS0, r_EU, r_AS, T_AF, \
        T_B, T_EU_AS, m_AF_B, m_AF_EU, m_AF_AS, m_EU_AS)
    samples = model.get_samples(sample_sizes[0], sample_sizes[1], \
        sample_sizes[2]) #['YRI', 'CEU', 'CHB']
    engine = sps.engines.get_engine('msprime')
    ts = engine.simulate(model, contig, samples)

    return ts

# testing
if __name__ == "__main__":

    batch_size = 50
    S = 36
    R = 50000
    SEED = 1833
    params = util.ParamSet()

    # quick test
    print("sim exp")
    generator = Generator(simulate_exp, ["N1", "T1"], [20], S, R, SEED)
    generator.update_params([params.N1.value, params.T1.value])
    mini_batch = generator.simulate_batch(50)
    print("x", mini_batch.shape)
