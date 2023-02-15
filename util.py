"""
Utility functions and classes (including default parameters).
Author: Sara Mathieson, Rebecca Riley
Date: 9/27/22
"""

# python imports
import numpy as np
import optparse
import sys

# our imports
import generator
import global_vars
import param_set
import real_data_random
import simulation

def parse_params(param_input, simulator):
    """See which params were desired for inference"""
    param_strs = param_input.split(',')
    parameters = []
    for _, p in vars(param_set.ParamSet(simulator)).items():
        if p.name in param_strs:
            parameters.append(p)

    assert len(parameters) == len(param_strs)
    for p in parameters:
        print(p)

    return parameters

def filter_func(x, rate): # currently not used
    """Keep non-singletons. If singleton, filter at given rate"""
    # TODO since we haven't done major/minor yet, might want to add != n-1 too
    if np.sum(x) != 1:
        return True
    return np.random.random() >= rate # keep (1-rate) of singletons

def process_gt_dist(gt_matrix, dist_vec, region_len=False, real=False,
    neg1=True):
    """
    Take in a genotype matrix and vector of inter-SNP distances. Return a 3D
    numpy array of the given n (haps) and S (SNPs) and 2 channels.
    Filter singletons at given rate if filter=True
    """
    og_snps = gt_matrix.shape[0]

    if (real and global_vars.FILTER_REAL_DATA) or (not real and
        global_vars.FILTER_SIMULATED):
        # mask
        singleton_mask = np.array([filter_func(row, global_vars.FILTER_RATE,
            gt_matrix.shape[1] - 1) for row in gt_matrix])

        # reassign
        gt_matrix = gt_matrix[singleton_mask]
        dist_vec = np.array(dist_vec)[singleton_mask]

    num_SNPs = gt_matrix.shape[0] # SNPs x n
    n = gt_matrix.shape[1]

    # double check
    if num_SNPs != len(dist_vec):
        print("gt", num_SNPs, "dist", len(dist_vec))
    assert num_SNPs == len(dist_vec)

    # used for trimming (don't trim if using the entire region)
    S = num_SNPs if region_len else global_vars.NUM_SNPS

    # set up region
    region = np.zeros((n, S, 2), dtype=np.float32)

    mid = num_SNPs//2
    half_S = S//2
    if S % 2 == 1: # odd
        other_half_S = half_S+1
    else:
        other_half_S = half_S

    # enough SNPs, take middle portion
    if mid >= half_S:
        minor = major_minor(gt_matrix[mid-half_S:mid+
            other_half_S,:].transpose(), neg1)
        region[:,:,0] = minor
        distances = np.vstack([np.copy(dist_vec[mid-half_S:mid+other_half_S])
            for k in range(n)])
        region[:,:,1] = distances

    # not enough SNPs, need to center-pad
    else:
        print("NOT ENOUGH SNPS", num_SNPs)
        print(num_SNPs, S, mid, half_S)
        minor = major_minor(gt_matrix.transpose(), neg1)
        region[:,half_S-mid:half_S-mid+num_SNPs,0] = minor
        distances = np.vstack([np.copy(dist_vec) for k in range(n)])
        region[:,half_S-mid:half_S-mid+num_SNPs,1] = distances

    return region # n X SNPs X 2

def major_minor(matrix, neg1):
    """Note that matrix.shape[1] may not be S if we don't have enough SNPs"""
    n = matrix.shape[0]
    for j in range(matrix.shape[1]):
        if np.count_nonzero(matrix[:,j] > 0) > (n/2): # count the 1's
            matrix[:,j] = 1 - matrix[:,j]

    # option to convert from 0/1 to -1/+1
    if neg1:
        matrix[matrix == 0] = -1
    # residual numbers higher than one may remain even though we restricted to
    # biallelic
    #matrix[matrix > 1] = 1 # removing since we filter in VCF
    return matrix

def prep_real(gt, snp_start, snp_end, indv_start, indv_end):
    """Slice out desired region and unravel haplotypes"""
    region = gt[snp_start:snp_end, indv_start:indv_end, :]
    both_haps = np.concatenate((region[:,:,0], region[:,:,1]), axis=1)
    return both_haps

def filter_nonseg(region):
    """Filter out non-segregating sites in this region"""
    nonseg0 = np.all(region == 0, axis=1) # row all 0
    nonseg1 = np.all(region == 1, axis=1) # row all 1
    keep0 = np.invert(nonseg0)
    keep1 = np.invert(nonseg1)
    filter = np.logical_and(keep0, keep1)
    return filter

def parse_args(in_file_data = None, param_values = None):
    """Parse command line arguments."""
    parser = optparse.OptionParser(description='PG-GAN entry point')

    parser.add_option('-m', '--model', type='string',help='exp, im, ooa2, ooa3')
    parser.add_option('-p', '--params', type='string',
        help='comma separated parameter list')
    parser.add_option('-d', '--data_h5', type='string', help='real data file')
    parser.add_option('-b', '--bed', type='string', help='bed file (mask)')
    parser.add_option('-r', '--reco_folder', type='string',
        help='recombination maps')
    parser.add_option('-g', action="store_true", dest="grid",help='grid search')
    parser.add_option('-t', action="store_true", dest="toy", help='toy example')
    parser.add_option('-s', '--seed', type='int', default=1833,
        help='seed for RNG')
    parser.add_option('-n', '--sample_sizes', type='string',
        help='comma separated sample sizes for each population, in haps')
    parser.add_option('-v', '--param_values', type='string',
        help='comma separated values corresponding to params')

    (opts, args) = parser.parse_args()

    '''
    The following section overrides params from the input file with the provided
    args.
    '''

    # note: this series of checks looks like it could be simplified with list
    #       iteration:
    # it can't be, bc the opts object can't be indexed--eg opts['model'] fails
    def param_mismatch(param, og, replacement):
        print("***** WARNING: MISMATCH BETWEEN IN FILE AND CMD ARGS: " + param +
              ", using ARGS (" + str(og) + " -> " + str(replacement) + ")")

    if in_file_data is not None:
        if opts.model is None:
            opts.model = in_file_data['model']
        elif opts.model != in_file_data['model']:
            param_mismatch("MODEL", in_file_data['model'], opts.model)

        if opts.params is None:
            opts.params = in_file_data['params']
        elif opts.params != in_file_data['params']:
            param_mismatch("PARAMS", in_file_data['params'], opts.params)

        if opts.data_h5 is None:
            opts.data_h5 = in_file_data['data_h5']
        elif opts.data_h5 != in_file_data['data_h5']:
            param_mismatch("DATA_H5", in_file_data['data_h5'], opts.data_h5)

        if opts.bed is None:
            opts.bed = in_file_data['bed_file']
        elif opts.bed != in_file_data['bed_file']:
            param_mismatch("BED FILE", in_file_data['bed_file'], opts.bed)

        if opts.sample_sizes is None:
            opts.sample_sizes = in_file_data['sample_sizes']
        elif opts.sample_sizes != in_file_data['sample_sizes']:
            param_mismatch("SAMPLE_SIZES", in_file_data['sample_sizes'],
                opts.sample_sizes)

        if opts.reco_folder is None:
            opts.reco_folder = in_file_data['reco_folder']
        elif opts.reco_folder != in_file_data['reco_folder']:
            param_mismatch("RECO_FOLDER", in_file_data['reco_folder'],
                opts.reco_folder)

        # because we care about the seed from the trial, here in_file_data takes over opts
        if in_file_data['seed'] is not None:
            opts.seed = in_file_data['seed']
            
    if opts.param_values is not None:
        arg_values = [float(val_str) for val_str in
            opts.param_values.split(',')]
        if arg_values != param_values:
            param_mismatch("PARAM_VALUES", param_values, arg_values)
            param_values = arg_values # override at return

    mandatories = ['model','params']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    if param_values is None:
        return opts

    return opts, param_values

def parse_hapmap_empirical_prior(files):
    """
    Parse recombination maps to create a distribution of recombintion rates to
    use for real data simulations. Based on defiNETti software package.
    """
    print("Parsing HapMap recombination rates...")

    # set up weights (probabilities) and reco rates
    weights_all = []
    prior_rates_all = []

    for f in files:
        mat = np.loadtxt(f, skiprows = 1, usecols=(1,2))
        #print(mat.shape)
        mat[:,1] = mat[:,1]*(1.e-8)
        mat = mat[mat[:,1] != 0.0, :] # remove 0s
        weights = mat[1:,0] - mat[:-1,0]
        prior_rates = mat[:-1,1]

        weights_all.extend(weights)
        prior_rates_all.extend(prior_rates)

    # normalize
    prob = weights_all / np.sum(weights_all)

    # make smaller by a factor of 50 (collapse)
    indexes = list(range(len(prior_rates_all)))
    indexes.sort(key=prior_rates_all.__getitem__)

    prior_rates_all = [prior_rates_all[i] for i in indexes]
    prob = [prob[i] for i in indexes]

    new_rates = []
    new_weights = []

    collapse = 50
    for i in range(0,len(prior_rates_all),collapse):
        end = collapse
        if len(prior_rates_all)-i < collapse:
            end = len(prior_rates_all)-i
        new_rates.append(sum(prior_rates_all[i:i+end])/end) # average
        new_weights.append(sum(prob[i:i+end])) # sum

    new_rates = np.array(new_rates)
    new_weights = np.array(new_weights)

    return new_rates, new_weights

def read_demo_file(filename, Ne):
    """Read in a PSMC-like demography"""
    demos = []
    with open(filename, 'r') as demo_file:
        for pop_params in demo_file:
            time, pop = pop_params.strip().split()
            demos.append(msprime.PopulationParametersChange(time=float(time)
                * 4 * Ne, initial_size=float(pop) * Ne))
    return demos

def parse_sample_sizes(n_string):
    """ n_string is e.g. "12,6,8" """
    return [int(n) for n in n_string.split(",")]

def process_opts(opts, summary_stats = False):
    sample_sizes = parse_sample_sizes(opts.sample_sizes)

    real = False
    # if real data provided
    if opts.data_h5 is not None: # h5 is None option at end of func
        real = True
        # frac test isn't currently used in ss

        # if summary_stats:
        #      iterator = real_data_random.RealDataRandomIterator(
        #         opts.data_h5, opts.bed, frac_test =
        #         globals.FRAC_TEST)
        # else:
        # most typical case for real data
        iterator = real_data_random.RealDataRandomIterator(opts.data_h5,
            opts.bed)

    # more flexible way to get the simulator
    simulator = getattr(simulation, opts.model)

    if (global_vars.FILTER_SIMULATED or global_vars.FILTER_REAL_DATA):
        print("FILTERING SINGLETONS")

    # parameter defaults
    parameters = parse_params(opts.params, simulator) # desired params
    param_names = [p.name for p in parameters]

    # generator
    gen = generator.Generator(simulator, param_names, sample_sizes,
        opts.seed, mirror_real=real, reco_folder=opts.reco_folder)

    if opts.data_h5 is None:
        # "real data" is simulated with fixed params
        iterator = generator.Generator(simulator, param_names, sample_sizes,
            opts.seed) # don't need reco_folder

    return gen, iterator, parameters, sample_sizes # last used for disc.

if __name__ == "__main__":
    # test major/minor and post-processing
    global_vars.NUM_SNPS = 4 # make smaller for testing

    a = np.zeros((6,3))
    a[0,0] = 1
    a[1,0] = 1
    a[2,0] = 1
    a[3,0] = 1
    a[0,1] = 1
    a[1,1] = 1
    a[2,1] = 1
    a[4,2] = 1
    dist_vec = [0.3, 0.2, 0.4, 0.5, 0.1, 0.2]

    print(a)
    print(major_minor(a, neg1=True))

    process_gt_dist(a, dist_vec, real=False)
