"""
Compare summary statistics from real data with data simulated under the
inferred parameters.
Author: Sara Mathieson, Rebecca Riley
Date: 9/27/22
"""

# python imports
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import optparse
import seaborn as sns
import sys

# our imports
import global_vars
import ss_helpers
import util

# globals
NUM_TRIAL = 5000
NAMES = ["Tajima's D", r'pairwise heterozygosity ($\pi$)', \
    "number of haplotypes"]

# for ooa2 (YRI/CEU)
FSC_PARAMS = [21017, 0.0341901, 3105.5, 21954, 33077.5, 2844, 1042]

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

    opts, param_values = util.parse_args(in_file_data = in_file_data, param_values = param_values)

    generator, iterator, parameters, _ = util.process_opts(opts, summary_stats = True)

    pop_names = opts.data_h5.split("/")[-1].split(".")[0] \
                       if opts.data_h5 is not None else ""
    # sets global_vars.SS_LABELS and global_vars.SS_COLORS
    # overwrite this function in globals.py to change
    global_vars.update_ss_labels(pop_names)

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
    print("got through real data")

    # sim
    sim_matrices = generator.simulate_batch(batch_size=NUM_TRIAL, neg1=False)
    sim_matrices_region = generator.simulate_batch(batch_size=NUM_TRIAL,
        neg1=False, region_len=True)

    # one pop models
    if opts.model in ['exp', 'const']:
        real_sfs, real_dist, real_ld, real_stats = ss_helpers.stats_all(real_matrices, real_matrices_region)
        sim_sfs, sim_dist, sim_ld, sim_stats = ss_helpers.stats_all(sim_matrices, sim_matrices_region)
        plot_all_stats(real_stats, real_dist, real_sfs, real_ld, sim_stats, sim_dist, sim_sfs, sim_ld, output_file)

    # two pop models
    elif opts.model in ['im', 'ooa2', 'post_ooa', 'msmc', 'fsc']:

        half = real_matrices.shape[1]//2

        # real split
        real_matrices1 = real_matrices[:,:half,:,:]
        real_matrices2 = real_matrices[:,half:,:,:]

        real_matrices_region1 = []
        real_matrices_region2 = []
        for item in real_matrices_region:
            real_matrices_region1.append(item[:half,:,:])
            real_matrices_region2.append(item[half:,:,:])

        # sim split
        sim_matrices1 = sim_matrices[:,:half,:,:]
        sim_matrices2 = sim_matrices[:,half:,:,:]

        sim_matrices_region1 = []
        sim_matrices_region2 = []
        for item in sim_matrices_region:
            sim_matrices_region1.append(item[:half,:,:])
            sim_matrices_region2.append(item[half:,:,:])

        # stats for pop 1
        real_sfs1, real_dist1, real_ld1, real_stats1 = ss_helpers.stats_all(real_matrices1, real_matrices_region1)
        sim_sfs1, sim_dist1, sim_ld1, sim_stats1 = ss_helpers.stats_all(sim_matrices1, sim_matrices_region1)

        # stats for pop 2
        real_sfs2, real_dist2, real_ld2, real_stats2 = ss_helpers.stats_all(real_matrices2, real_matrices_region2)
        sim_sfs2, sim_dist2, sim_ld2, sim_stats2 = ss_helpers.stats_all(sim_matrices2, sim_matrices_region2)

        # two pop stats
        real_fst = ss_helpers.fst_all(real_matrices)
        sim_fst = ss_helpers.fst_all(sim_matrices)

        plot_stats_twopop(real_stats1, real_dist1, real_sfs1, real_ld1, real_stats2, real_dist2, real_sfs2, real_ld2, real_fst, \
            sim_stats1, sim_dist1, sim_sfs1, sim_ld1, sim_stats2, sim_dist2, sim_sfs2, sim_ld2, sim_fst, output_file, fsc=fsc)

    # OOA3
    elif opts.model in ['ooa3']:
        third = real_matrices.shape[1]//3

        # real split
        real_matrices1 = real_matrices[:,:third,:,:]
        real_matrices2 = real_matrices[:,third:third*2,:,:]
        real_matrices3 = real_matrices[:,third*2:,:,:]

        real_matrices_region1 = []
        real_matrices_region2 = []
        real_matrices_region3 = []
        for item in real_matrices_region:
            real_matrices_region1.append(item[:third,:,:])
            real_matrices_region2.append(item[third:third*2,:,:])
            real_matrices_region3.append(item[third*2:,:,:])

        # sim split
        sim_matrices1 = sim_matrices[:,:third,:,:]
        sim_matrices2 = sim_matrices[:,third:third*2,:,:]
        sim_matrices3 = sim_matrices[:,third*2:,:,:]

        sim_matrices_region1 = []
        sim_matrices_region2 = []
        sim_matrices_region3 = []
        for item in sim_matrices_region:
            sim_matrices_region1.append(item[:third,:,:])
            sim_matrices_region2.append(item[third:third*2,:,:])
            sim_matrices_region3.append(item[third*2:,:,:])

        # stats for pop 1
        real_sfs1, real_dist1, real_ld1, real_stats1 = ss_helpers.stats_all(real_matrices1, real_matrices_region1)
        sim_sfs1, sim_dist1, sim_ld1, sim_stats1 = ss_helpers.stats_all(sim_matrices1, sim_matrices_region1)

        # stats for pop 2
        real_sfs2, real_dist2, real_ld2, real_stats2 = ss_helpers.stats_all(real_matrices2, real_matrices_region2)
        sim_sfs2, sim_dist2, sim_ld2, sim_stats2 = ss_helpers.stats_all(sim_matrices2, sim_matrices_region2)

        # stats for pop 3
        real_sfs3, real_dist3, real_ld3, real_stats3 = ss_helpers.stats_all(real_matrices3, real_matrices_region3)
        sim_sfs3, sim_dist3, sim_ld3, sim_stats3 = ss_helpers.stats_all(sim_matrices3, sim_matrices_region3)

        # two pop stats
        real_matrices12 = np.concatenate((np.array(real_matrices1), np.array(real_matrices2)), axis=1)
        real_matrices13 = np.concatenate((np.array(real_matrices1), np.array(real_matrices3)), axis=1)
        real_matrices23 = np.concatenate((np.array(real_matrices2), np.array(real_matrices3)), axis=1)

        sim_matrices12 = np.concatenate((np.array(sim_matrices1), np.array(sim_matrices2)), axis=1)
        sim_matrices13 = np.concatenate((np.array(sim_matrices1), np.array(sim_matrices3)), axis=1)
        sim_matrices23 = np.concatenate((np.array(sim_matrices2), np.array(sim_matrices3)), axis=1)

        print("fst shape", real_matrices12.shape)

        real_fst12 = ss_helpers.fst_all(real_matrices12)
        sim_fst12 = ss_helpers.fst_all(sim_matrices12)
        real_fst13 = ss_helpers.fst_all(real_matrices13)
        sim_fst13 = ss_helpers.fst_all(sim_matrices13)
        real_fst23 = ss_helpers.fst_all(real_matrices23)
        sim_fst23 = ss_helpers.fst_all(sim_matrices23)

        plot_stats_threepop(
            real_stats1, real_dist1, real_sfs1, real_ld1,
            real_stats2, real_dist2, real_sfs2, real_ld2,
            real_stats3, real_dist3, real_sfs3, real_ld3,
            sim_stats1, sim_dist1, sim_sfs1, sim_ld1,
            sim_stats2, sim_dist2, sim_sfs2, sim_ld2,
            sim_stats3, sim_dist3, sim_sfs3, sim_ld3,
            real_fst12, real_fst13, real_fst23,
            sim_fst12, sim_fst13, sim_fst23,
            output_file)

    else:
        print("unsupported", opts.model)

# one pop
def plot_all_stats(real_stats, real_dist, real_sfs, real_ld, sim_stats, sim_dist, sim_sfs, sim_ld, output):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7, 7))

    ss_helpers.plot_sfs(axes.flatten()[0], real_sfs, sim_sfs,
        global_vars.SS_COLORS[0], global_vars.SS_COLORS[1],
        global_vars.SS_LABELS[0], global_vars.SS_LABELS[1], single=True)
    ss_helpers.plot_dist(axes.flatten()[1], real_dist, sim_dist,
        global_vars.SS_COLORS[0], global_vars.SS_COLORS[1],
        global_vars.SS_LABELS[0], global_vars.SS_LABELS[1], single=True)
    ss_helpers.plot_ld(axes.flatten()[2], real_ld, sim_ld,
        global_vars.SS_COLORS[0], global_vars.SS_COLORS[1],
        global_vars.SS_LABELS[0], global_vars.SS_LABELS[1], single=True)

    for i in range(3):
        ss_helpers.plot_generic(axes.flatten()[i+3], NAMES[i], real_stats[i], sim_stats[i],
                                global_vars.SS_COLORS[0], global_vars.SS_COLORS[1],
                                global_vars.SS_LABELS[0], global_vars.SS_LABELS[1], single=True)

    plt.tight_layout()
    if output != None:
        plt.savefig(output, dpi=300)
    else:
        plt.show()

# two pop
def plot_stats_twopop(real_stats1, real_dist1, real_sfs1, real_ld1, real_stats2, real_dist2, real_sfs2, real_ld2, real_fst, \
    sim_stats1, sim_dist1, sim_sfs1, sim_ld1, sim_stats2, sim_dist2, sim_sfs2, sim_ld2, sim_fst, output, fsc=False):

    pop1_label = global_vars.SS_LABELS[0]
    pop2_label = global_vars.SS_LABELS[1]
    sim_label = global_vars.SS_LABELS[-1]

    pop1_color = global_vars.SS_COLORS[0]
    pop2_color = global_vars.SS_COLORS[1]
    sim_color = global_vars.SS_COLORS[-1]

    pop1_real = mpatches.Patch(color=pop1_color, label=pop1_label)
    pop2_real = mpatches.Patch(color=pop2_color, label=pop2_label)
    pop2_sim = mpatches.Patch(color=sim_color, label=sim_label)

    if not fsc:
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(14, 10))
        axes_all = axes.flatten() # TODO don't flatten?

        # row 1
        ss_helpers.plot_sfs(axes_all[0], real_sfs1, sim_sfs1, pop1_color,
            sim_color, pop=pop1_label, sim_label=sim_label)
        ss_helpers.plot_dist(axes_all[1], real_dist1, sim_dist1, pop1_color,
            sim_color, pop=pop1_label, sim_label=sim_label)
        ss_helpers.plot_sfs(axes_all[2], real_sfs2, sim_sfs2, pop2_color,
            sim_color, pop=pop2_label, sim_label=sim_label)
        ss_helpers.plot_dist(axes_all[3], real_dist2, sim_dist2, pop2_color,
            sim_color, pop=pop2_label, sim_label=sim_label)

        # row 2
        ss_helpers.plot_ld(axes_all[4], real_ld1, sim_ld1, pop1_color,
            sim_color, pop=pop1_label, sim_label=sim_label)
        ss_helpers.plot_generic(axes_all[5], NAMES[0], real_stats1[0],
            sim_stats1[0], pop1_color, sim_color, pop=pop1_label, sim_label=sim_label)
        ss_helpers.plot_ld(axes_all[6], real_ld2, sim_ld2, pop2_color,
            sim_color, pop=pop2_label, sim_label=sim_label)
        ss_helpers.plot_generic(axes_all[7], NAMES[0], real_stats2[0],
            sim_stats2[0], pop2_color, sim_color, pop=pop2_label, sim_label=sim_label)

        # row 3
        ss_helpers.plot_generic(axes_all[8], NAMES[1], real_stats1[1],
            sim_stats1[1], pop1_color, sim_color, pop=pop1_label, sim_label=sim_label)
        ss_helpers.plot_generic(axes_all[9], NAMES[2], real_stats1[2],
            sim_stats1[2], pop1_color, sim_color, pop=pop1_label, sim_label=sim_label)
        ss_helpers.plot_generic(axes_all[10], NAMES[1], real_stats2[1],
            sim_stats2[1], pop2_color, sim_color, pop=pop2_label, sim_label=sim_label)
        ss_helpers.plot_generic(axes_all[11], NAMES[2], real_stats2[2],
            sim_stats2[2], pop2_color, sim_color, pop=pop2_label, sim_label=sim_label)

        # row 4
        ss_helpers.plot_fst(axes_all[13], real_fst, sim_fst, pop1_label+"/"+pop2_label, sim_label, "purple", sim_color)
        axes_all[12].axis('off')
        axes_all[14].axis('off')
        axes_all[15].axis('off')

        # overall legend
        pop1_real = mpatches.Patch(color=pop1_color, label=pop1_label + ' real data')
        pop1_sim = mpatches.Patch(color=sim_color, label=pop1_label + ' sim data')
        pop2_real = mpatches.Patch(color=pop2_color, label=pop2_label + ' real data')
        pop2_sim = mpatches.Patch(color=sim_color, label=pop2_label + ' sim data')
        axes_all[12].legend(handles=[pop1_real, pop1_sim], loc=10, prop={'size': 18})
        axes_all[15].legend(handles=[pop2_real, pop2_sim], loc=10, prop={'size': 18})

    # fastsimcoal
    else:

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10.5, 2.5))
        axes_all = axes.flatten() # TODO don't flatten?

        # row 1
        ss_helpers.plot_sfs(axes_all[0], real_sfs1, sim_sfs1, pop1_color, sim_color, pop=pop1_label, sim_label=sim_label)
        ss_helpers.plot_sfs(axes_all[1], real_sfs2, sim_sfs2, pop2_color, sim_color, pop=pop2_label, sim_label=sim_label)
        axes_all[2].axis('off')
        axes_all[2].legend(handles=[pop1_real, pop2_real, pop2_sim], loc=10, prop={'size': 16})

    plt.tight_layout()
    if output != None:
        plt.savefig(output, dpi=300)
    else:
        plt.show()


# three pop
def plot_stats_threepop(
    real_stats1, real_dist1, real_sfs1, real_ld1,
    real_stats2, real_dist2, real_sfs2, real_ld2,
    real_stats3, real_dist3, real_sfs3, real_ld3,
    sim_stats1, sim_dist1, sim_sfs1, sim_ld1,
    sim_stats2, sim_dist2, sim_sfs2, sim_ld2,
    sim_stats3, sim_dist3, sim_sfs3, sim_ld3,
    real_fst12, real_fst13, real_fst23,
    sim_fst12, sim_fst13, sim_fst23,
    output):

    fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(14, 14))

    pop1_label = global_vars.SS_LABELS[0]
    pop2_label = global_vars.SS_LABELS[1]
    pop3_label = global_vars.SS_LABELS[2]
    sim_label = global_vars.SS_LABELS[-1]

    pop1_color = global_vars.SS_COLORS[0]
    pop2_color = global_vars.SS_COLORS[1]
    pop3_color = global_vars.SS_COLORS[2]
    sim_color = global_vars.SS_COLORS[-1]

    # pop 1
    ss_helpers.plot_sfs(axes[0][0], real_sfs1, sim_sfs1, pop1_color,
        sim_color, pop=pop1_label, sim_label=sim_label)
    ss_helpers.plot_dist(axes[0][1], real_dist1, sim_dist1, pop1_color,
        sim_color, pop=pop1_label, sim_label=sim_label)
    ss_helpers.plot_ld(axes[1][0], real_ld1, sim_ld1, pop1_color, sim_color,
        pop=pop1_label, sim_label=sim_label)
    ss_helpers.plot_generic(axes[1][1], NAMES[0], real_stats1[0], sim_stats1[0],
        pop1_color, sim_color, pop=pop1_label, sim_label=sim_label)
    ss_helpers.plot_generic(axes[2][0], NAMES[1], real_stats1[1], sim_stats1[1],
        pop1_color, sim_color, pop=pop1_label, sim_label=sim_label)
    ss_helpers.plot_generic(axes[2][1], NAMES[2], real_stats1[2],
        sim_stats1[2], pop1_color, sim_color, pop=pop1_label, sim_label=sim_label)

    # pop 2
    ss_helpers.plot_sfs(axes[0][2], real_sfs2, sim_sfs2, pop2_color, sim_color,
        pop=pop2_label, sim_label=sim_label)
    ss_helpers.plot_dist(axes[0][3], real_dist2, sim_dist2, pop2_color, sim_color,
        pop=pop2_label, sim_label=sim_label)
    ss_helpers.plot_ld(axes[1][2], real_ld2, sim_ld2, pop2_color, sim_color,
        pop=pop2_label, sim_label=sim_label)
    ss_helpers.plot_generic(axes[1][3], NAMES[0], real_stats2[0], sim_stats2[0],
        pop2_color, sim_color, pop=pop2_label, sim_label=sim_label)
    ss_helpers.plot_generic(axes[2][2], NAMES[1], real_stats2[1], sim_stats2[1],
        pop2_color, sim_color, pop=pop2_label, sim_label=sim_label)
    ss_helpers.plot_generic(axes[2][3], NAMES[2], real_stats2[2], sim_stats2[2],
        pop2_color, sim_color, pop=pop2_label, sim_label=sim_label)

    # pop 3
    ss_helpers.plot_sfs(axes[3][2], real_sfs3, sim_sfs3, pop3_color, sim_color,
        pop=pop3_label, sim_label=sim_label)
    ss_helpers.plot_dist(axes[3][3], real_dist3, sim_dist3, pop3_color, sim_color,
        pop=pop3_label, sim_label=sim_label)
    ss_helpers.plot_ld(axes[4][2], real_ld3, sim_ld3, pop3_color, sim_color,
        pop=pop3_label, sim_label=sim_label)
    ss_helpers.plot_generic(axes[4][3], NAMES[0], real_stats3[0], sim_stats3[0],
        pop3_color, sim_color, pop=pop3_label, sim_label=sim_label)
    ss_helpers.plot_generic(axes[5][2], NAMES[1], real_stats3[1], sim_stats3[1],
        pop3_color, sim_color, pop=pop3_label, sim_label=sim_label)
    ss_helpers.plot_generic(axes[5][3], NAMES[2], real_stats3[2], sim_stats3[2],
        pop3_color, sim_color, pop=pop3_label, sim_label=sim_label)

    # fst 4
    ss_helpers.plot_fst(axes[3][0], real_fst12, sim_fst12,
        real_label=pop1_label+"/"+pop2_label, sim_label=sim_label,
        real_color="purple", sim_color=sim_color)
    ss_helpers.plot_fst(axes[4][0], real_fst13, sim_fst13,
        real_label=pop1_label+"/"+pop3_label, sim_label=sim_label,
        real_color="purple", sim_color=sim_color)
    ss_helpers.plot_fst(axes[5][0], real_fst23, sim_fst23,
        real_label=pop2_label+"/"+pop3_label, sim_label=sim_label,
        real_color="purple", sim_color=sim_color)
    axes[3][1].axis('off')
    axes[4][1].axis('off')
    axes[5][1].axis('off')

    # overall legend
    pop1_real = mpatches.Patch(color=pop1_color, label=pop1_label + ' real data')
    pop1_sim = mpatches.Patch(color=sim_color, label=pop1_label + ' sim data')
    pop2_real = mpatches.Patch(color=pop2_color, label=pop2_label + ' real data')
    pop2_sim = mpatches.Patch(color=sim_color, label=pop2_label + ' sim data')
    pop3_real = mpatches.Patch(color=pop3_color, label=pop3_label + ' real data')
    pop3_sim = mpatches.Patch(color=sim_color, label=pop3_label + ' sim data')
    axes[3][1].legend(handles=[pop1_real, pop1_sim], loc=10, prop={'size': 18})
    axes[4][1].legend(handles=[pop2_real, pop2_sim], loc=10, prop={'size': 18})
    axes[5][1].legend(handles=[pop3_real, pop3_sim], loc=10, prop={'size': 18})

    plt.tight_layout()
    if output != None:
        plt.savefig(output, dpi=300)
    else:
        plt.show()

main()
