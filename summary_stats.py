"""
Compare summary statistics from real data with data simulated under the
inferred parameters.
Author: Sara Mathieson, Rebecca Riley
Date: 1/27/23
"""

# python imports
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

# our imports
import global_vars
import ss_helpers
import util

# globals
NUM_TRIAL = 5000
# statistic names
NAMES = [
    "minor allele count (SFS)",
    "inter-SNP distances",
    "distance between SNPs",
    "Tajima's D",
    r'pairwise heterozygosity ($\pi$)',
    "number of haplotypes",
    "Hudson's Fst"]
FST_COLOR = "purple"

# for ooa2 (YRI/CEU) (no longer supported)
#FSC_PARAMS = [21017, 0.0341901, 3105.5, 21954, 33077.5, 2844, 1042]

def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    print("input file", input_file)
    print("output file", output_file)

    if global_vars.OVERWRITE_TRIAL_DATA:
        in_file_data = global_vars.TRIAL_DATA
        value_strs = global_vars.TRIAL_DATA['param_values'].split(',')
        param_values = [float(value_str) for value_str in value_strs]
        assert len(param_values) == len(in_file_data['params'].split(','))
    else:
        param_values, in_file_data = ss_helpers.parse_output(input_file)

    opts, param_values = util.parse_args(in_file_data = in_file_data,
        param_values=param_values)

    generator, iterator, parameters, sample_sizes = util.process_opts(opts,
        summary_stats=True)

    pop_names = opts.data_h5.split("/")[-1].split(".")[0] \
                       if opts.data_h5 is not None else ""
    # sets global_vars.SS_LABELS and global_vars.SS_COLORS
    # overwrite this function in globals.py to change
    global_vars.update_ss_labels(pop_names, num_pops=len(generator.sample_sizes))

    generator.update_params(param_values)
    print("VALUES", param_values)
    print("made it through params")

    # use the parameters we inferred!
    fsc=False
    if opts.model == 'fsc':
        print("\nALERT you are running FSC sim!\n")
        print("FSC PARAMS!", FSC_PARAMS)
        generator.update_params(FSC_PARAMS) # make sure to check the order!
        fsc=True

    '''
    NOTE: for summary stats, use neg1=False to keep hap data as 0/1 (not -1/1)
    NOTE: use region_len=True for Tajima's D (i.e. not all regions have same S)
    '''

    # real
    real_matrices = iterator.real_batch(batch_size=NUM_TRIAL, neg1=False)
    real_matrices_region = iterator.real_batch(batch_size=NUM_TRIAL, neg1=False,
        region_len=True)

    # sim
    sim_matrices = generator.simulate_batch(batch_size=NUM_TRIAL, neg1=False)
    sim_matrices_region = generator.simulate_batch(batch_size=NUM_TRIAL,
        neg1=False, region_len=True)

    num_pop = len(sample_sizes)

    # one pop models
    if num_pop == 1:
        nrows, ncols = 3, 2
        size = (7, 7)
        first_pop, second_pop = [], []

    # two pop models
    elif num_pop == 2:
        nrows, ncols = 4, 4
        size = (14, 10)
        first_pop, second_pop = [0], [1]

    # OOA3
    elif opts.model in ['ooa3']:
        nrows, ncols = 6, 4
        size = (14, 14)
        first_pop, second_pop = [0, 0, 1], [1, 2, 2]

    else:
        print("unsupported number of pops", num_pop)

    # split into individual pops
    real_all, real_region_all, sim_all, sim_region_all = \
        split_matrices(real_matrices, real_matrices_region, sim_matrices,
        sim_matrices_region, sample_sizes)

    # stats for all populations
    real_stats_lst = []
    sim_stats_lst = []
    for p in range(num_pop):
        real_stats_pop = ss_helpers.stats_all(real_all[p], real_region_all[p])
        sim_stats_pop = ss_helpers.stats_all(sim_all[p], sim_region_all[p])
        real_stats_lst.append(real_stats_pop)
        sim_stats_lst.append(sim_stats_pop)

    # Fst over all pairs
    real_fst_lst = []
    sim_fst_lst = []
    for pi in range(len(first_pop)):
        a = first_pop[pi]
        b = second_pop[pi]
        real_ab = np.concatenate((np.array(real_all[a]), np.array(real_all[b])),
            axis=1)
        sim_ab = np.concatenate((np.array(sim_all[a]), np.array(sim_all[b])),
            axis=1)

        # compute Fst
        real_fst = ss_helpers.fst_all(real_ab)
        sim_fst = ss_helpers.fst_all(sim_ab)
        real_fst_lst.append(real_fst)
        sim_fst_lst.append(sim_fst)

    # finall plotting call
    plot_stats_all(nrows, ncols, size, real_stats_lst, sim_stats_lst,
        real_fst_lst, sim_fst_lst, output_file)

def split_matrices(real_matrices, real_matrices_region, sim_matrices,
    sim_matrices_region, sample_sizes):

    # set up empty arrays
    real_all, real_region_all, sim_all, sim_region_all = [], [], [], []

    start_idx = 0
    for s in sample_sizes:
        end_idx = start_idx + s

        # parse real matrices
        real_p = real_matrices[:,start_idx:end_idx,:,:]
        real_region_p = []
        for item in real_matrices_region:
            real_region_p.append(item[start_idx:end_idx,:,:])
        real_all.append(real_p)
        real_region_all.append(real_region_p)

        # parse sim matrices
        sim_p = sim_matrices[:,start_idx:end_idx,:,:]
        sim_region_p = []
        for item in sim_matrices_region:
            sim_region_p.append(item[start_idx:end_idx,:,:])
        sim_all.append(sim_p)
        sim_region_all.append(sim_region_p)

        # last step: update start_idx
        start_idx = end_idx

    return real_all, real_region_all, sim_all, sim_region_all

# one, two, and three pops
def plot_stats_all(nrows, ncols, size, real_stats_lst, sim_stats_lst,
    real_fst_lst, sim_fst_lst, output):
    num_pop = len(real_stats_lst)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=size)

    # labels and colors
    labels = global_vars.SS_LABELS[:num_pop]
    sim_label = global_vars.SS_LABELS[-1]
    colors = global_vars.SS_COLORS[:num_pop]
    sim_color = global_vars.SS_COLORS[-1]

    # plot each population
    rows = [0, 0, 3]
    cols = [0, 2, 2]
    single = True if num_pop == 1 else False
    for p in range(num_pop): # one/two pop won't use last indices
        real_color = colors[p]
        real_label = labels[p]
        real_pop = real_stats_lst[p]
        sim_pop = sim_stats_lst[p]
        plot_population(axes, rows[p], cols[p], real_color, real_label,
            real_pop, sim_color, sim_label, sim_pop, single=single)

    # Fst (all pairs)
    cidx = 0
    first_pop = [0, 0, 1]
    second_pop = [1, 2, 2]
    if num_pop == 2:
        cidx = 1
    for pi in range(len(real_fst_lst)): # pi -> pair index
        pair_label = labels[first_pop[pi]] + "/" + labels[second_pop[pi]]
        ss_helpers.plot_generic(axes[3+pi][cidx], NAMES[6], real_fst_lst[pi],
            sim_fst_lst[pi], FST_COLOR, sim_color, pop=pair_label,
            sim_label=sim_label)

    # overall legend
    if num_pop >= 2:
        for p in range(num_pop):
            p_real = mpatches.Patch(color=colors[p], label=labels[p] + \
                ' real data')
            p_sim = mpatches.Patch(color=sim_color, label=labels[p] + \
                ' sim data')
            if num_pop == 2:
                axes[3][0+3*p].axis('off')
                axes[3][0+3*p].legend(handles=[p_real, p_sim], loc=10,
                    prop={'size': 18})
            if num_pop == 3:
                axes[3+p][1].axis('off')
                axes[3+p][1].legend(handles=[p_real, p_sim], loc=10,
                    prop={'size': 18})

    if num_pop == 2:
        axes[3][2].axis('off')

    plt.tight_layout()
    if output != None:
        plt.savefig(output, dpi=350)
    else:
        plt.show()

def plot_population(axes, i, j, real_color, real_label, real_tuple, sim_color,
    sim_label, sim_tuple, single=False):
    """
    Plot all 6 stats for a single population, starting from the (i,j) subplot.
    """
    for r in range(3):
        for c in range(2):
            idx = 2*r+c
            ss_helpers.plot_generic(axes[i+r][j+c], NAMES[idx], real_tuple[idx],
                sim_tuple[idx], real_color, sim_color, pop=real_label,
                sim_label=sim_label, single=single)

main()
