"""
Simulate data for training or testing using msprime.
Author: Sara Matheison, Zhanpeng Wang, Jiaping Wang, Rebecca Riley
Date: 9/27/22
"""

# python imports
import math
import msprime

# from stdpopsim
import sps.engines
import sps.species
import sps.HomSap

# our imports
import global_vars

################################################################################
# SIMULATION
################################################################################

def im(params, sample_sizes, seed, reco):
    """Note this is a 2 population model"""
    assert len(sample_sizes) == 2

    # condense params
    N1 = params.get("N1")
    N2 = params.get("N2")
    T_split = params.get("T_split")
    N_anc = params.get("N_anc")
    mig = params.get("mig")

    population_configurations = [
        msprime.PopulationConfiguration(sample_size=sample_sizes[0],
            initial_size = N1),
        msprime.PopulationConfiguration(sample_size=sample_sizes[1],
            initial_size = N2)]

    # no migration initially
    mig_time = T_split/2

    # directional (pulse)
    if mig >= 0:
        # migration from pop 1 into pop 0 (back in time)
        mig_event = msprime.MassMigration(time = mig_time, source = 1,
            destination = 0, proportion = abs(mig))
    else:
        # migration from pop 0 into pop 1 (back in time)
        mig_event = msprime.MassMigration(time = mig_time, source = 0,
            destination = 1, proportion = abs(mig))

    demographic_events = [
        mig_event,
		# move all in deme 1 to deme 0
		msprime.MassMigration(
			time = T_split, source = 1, destination = 0, proportion = 1.0),
        # change to ancestral size
        msprime.PopulationParametersChange(time=T_split, initial_size=N_anc,
            population_id=0)
	]

    # simulate tree sequence
    ts = msprime.simulate(
		population_configurations = population_configurations,
		demographic_events = demographic_events,
		mutation_rate = params.get("mut"),
		length = global_vars.L,
		recombination_rate = reco,
        random_seed = seed)

    return ts

def ooa2(params, sample_sizes,seed, reco):
    """Note this is a 2 population model"""
    assert len(sample_sizes) == 2

    # condense params
    T1 = params.get("T1")
    T2 = params.get("T2")
    mig = params.get("mig")

    population_configurations = [
        msprime.PopulationConfiguration(sample_size=sample_sizes[0],
            initial_size = params.get("N3")), # YRI is first
        msprime.PopulationConfiguration(sample_size=sample_sizes[1],
            initial_size = params("N2"))] # CEU/CHB is second

    # directional (pulse)
    if mig >= 0:
        # migration from pop 1 into pop 0 (back in time)
        mig_event = msprime.MassMigration(time = T2, source = 1,
            destination = 0, proportion = abs(mig))
    else:
        # migration from pop 0 into pop 1 (back in time)
        mig_event = msprime.MassMigration(time = T2, source = 0,
            destination = 1, proportion = abs(mig))

    demographic_events = [
        mig_event,
        # change size of EUR
        msprime.PopulationParametersChange(time=T2,
            initial_size=params.get("N1"), population_id=1),
		# move all in deme 1 to deme 0
		msprime.MassMigration(time = T1, source = 1, destination = 0,
            proportion = 1.0),
        # change to ancestral size
        msprime.PopulationParametersChange(time=T1,
            initial_size=params.get("N_anc"), population_id=0)
	]

    ts = msprime.simulate(
		population_configurations = population_configurations,
		demographic_events = demographic_events,
		mutation_rate = params.get("mut"),
		length =  global_vars.L,
		recombination_rate = reco,
        random_seed = seed)

    return ts

def post_ooa(params, sample_sizes, seed, reco):
    """Note this is a 2 population model for CEU/CHB split"""
    assert len(sample_sizes) == 2

    # condense params
    T1 = params.get("T1")
    T2 = params.get("T2")
    mig = params.get("mig")
    #m_EU_AS = params.get("m_EU_AS")

    population_configurations = [
        msprime.PopulationConfiguration(sample_size=sample_sizes[0],
            initial_size = params.get("N3")), # CEU is first
        msprime.PopulationConfiguration(sample_size=sample_sizes[1],
            initial_size = params.get("N2"))] # CHB is second

    # symmetric migration
    #migration_matrix=[[0, m_EU_AS],
    #                  [m_EU_AS, 0]]

    # directional (pulse)
    if mig >= 0:
        # migration from pop 1 into pop 0 (back in time)
        mig_event = msprime.MassMigration(time = T2/2, source = 1,
            destination = 0, proportion = abs(mig))
    else:
        # migration from pop 0 into pop 1 (back in time)
        mig_event = msprime.MassMigration(time = T2/2, source = 0,
            destination = 1, proportion = abs(mig))

    demographic_events = [
        mig_event,
		# move all in deme 1 to deme 0
		msprime.MassMigration(time = T2, source = 1, destination = 0,
            proportion = 1.0),
        # set mig rate to zero (need if using migration_matrix)
        #msprime.MigrationRateChange(time=T2, rate=0),
        # ancestral bottleneck
        msprime.PopulationParametersChange(time=T2,
            initial_size=params.get("N1"), population_id=0),
        # ancestral size
        msprime.PopulationParametersChange(time=T1,
            initial_size=params.get("N_anc"), population_id=0)
	]

    ts = msprime.simulate(
		population_configurations = population_configurations,
		demographic_events = demographic_events,
        #migration_matrix = migration_matrix,
		mutation_rate = params.get("mut"),
		length =  global_vars.L,
		recombination_rate = reco,
        random_seed = seed)

    return ts

def exp(params, sample_sizes, seed, reco):
    """Note this is a 1 population model"""
    assert len(sample_sizes) == 1

    T2 = params.get("T2")
    N2 = params.get("N2")

    N0 = N2 / math.exp(-params.get("growth") * T2)

    demographic_events = [
        msprime.PopulationParametersChange(time=0, initial_size=N0,
            growth_rate=params.get("growth")),
        msprime.PopulationParametersChange(time=T2, initial_size=N2,
            growth_rate=0),
		msprime.PopulationParametersChange(time=params.get("T1"),
            initial_size=params.get("N1"))
	]

    ts = msprime.simulate(sample_size = sum(sample_sizes),
		demographic_events = demographic_events,
		mutation_rate = params.get("mut"),
		length =  global_vars.L,
		recombination_rate = reco,
        random_seed = seed)

    return ts

def const(params, sample_sizes, seed, reco):
    assert len(sample_sizes) == 1

    # simulate data
    ts = msprime.simulate(sample_size=sum(sample_sizes), Ne=params.get("Ne"),
        length=global_vars.L, mutation_rate=params.get("mut"),
        recombination_rate=reco, random_seed = seed)

    return ts

def ooa3(params, sample_sizes, seed, reco):
    """From OOA3 as implemented in stdpopsim"""
    assert len(sample_sizes) == 3

    sp = sps.species.get_species("HomSap")

    mult = global_vars.L/141213431 # chr9
    contig = sp.get_contig("chr9",length_multiplier=mult) # TODO vary the chrom

    # 14 params
    N_A = params.get("N_A")
    N_B = params.get("N_B")
    N_AF = params.get("N_AF")
    N_EU0 = params.get("N_EU0")
    N_AS0 = params.get("N_AS0")
    r_EU = params.get("r_EU")
    r_AS = params.get("r_AS")
    T_AF = params.get("T_AF")
    T_B = params.get("T_B")
    T_EU_AS = params.get("T_EU_AS")
    m_AF_B = params.get("m_AF_B")
    m_AF_EU = params.get("m_AF_EU")
    m_AF_AS = params.get("m_AF_AS")
    m_EU_AS = params.get("m_EU_AS")

    model = sps.HomSap.ooa_3(N_A, N_B, N_AF, N_EU0, N_AS0, r_EU, r_AS, T_AF,
        T_B, T_EU_AS, m_AF_B, m_AF_EU, m_AF_AS, m_EU_AS)
    samples = model.get_samples(sample_sizes[0], sample_sizes[1],
        sample_sizes[2]) #['YRI', 'CEU', 'CHB']
    engine = sps.engines.get_engine('msprime')
    ts = engine.simulate(model, contig, samples, seed=seed)

    return ts
