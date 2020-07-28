"""
Utility functions and classes (including default parameters).
Author: Sara Mathieson
Date: 7/8/20
"""

import numpy as np
import optparse
import sys

from scipy.stats import norm

class Parameter:
    """Holds information about evolutionary parameters to infer"""

    def __init__(self, value, min, max, name):
        self.value = value
        self.min = min
        self.max = max
        self.name = name
        self.proposal_width = (self.max - self.min)/10 # heuristic

    def __str__(self):
        s = '\t'.join(["NAME", "VALUE", "MIN", "MAX"]) + '\n'
        s += '\t'.join([str(self.name), str(self.value), str(self.min), \
            str(self.max)])
        return s

    def start(self):
        # random initialization
        return np.random.uniform(self.min, self.max)

    def start_range(self):
        start_min = np.random.uniform(self.min, self.max)
        start_max = np.random.uniform(self.min, self.max)
        if start_min <= start_max:
            return [start_min, start_max]
        return self.start_range()

    def fit_to_range(self, value):
        value = min(value, self.max)
        return max(value, self.min)

    def proposal(self, curr_value, multiplier):
        # normal around current value (make sure don't go outside bounds)
        new_value = norm(curr_value, self.proposal_width*multiplier).rvs()
        return self.fit_to_range(new_value)

    def proposal_range(self, curr_lst, multiplier):
        new_min = self.fit_to_range(norm(curr_lst[0], self.proposal_width * \
            multiplier).rvs())
        new_max = self.fit_to_range(norm(curr_lst[1], self.proposal_width * \
            multiplier).rvs())
        if new_min <= new_max:
            return [new_min, new_max]
        return self.proposal_range(curr_lst, multiplier) # try again

class ParamSet:

    def __init__(self):
        # ooa
        self.N_A = Parameter(7300, 1000, 20000, "N_A")
        self.N_B = Parameter(2100, 1000, 20000, "N_B")

        # default Ne and reco and mut
        self.Ne = Parameter(10000, 1000, 20000, "Ne")
        self.reco = Parameter(1.25e-8, 1e-9, 1e-7, "reco")
        self.mut = Parameter(1.25e-8, 1e-9, 1e-7, "mut")

        # IM
        self.N_anc = Parameter(15000, 1000, 25000, "N_anc")
        self.T_split = Parameter(2000, 500, 20000, "T_split")
        self.mig = Parameter(0.05, -0.2, 0.2, "mig")

        # IM and exp
        self.N1 = Parameter(9000, 1000, 30000, "N1")
        self.N2 = Parameter(5000, 1000, 30000, "N2")

        # exp
        self.growth = Parameter(0.005, 0.0, 0.05, "growth")

        # ooa2
        self.N3 = Parameter(12000, 1000, 30000, "N3")
        self.T1 = Parameter(2000, 1500, 5000, "T1")
        self.T2 = Parameter(350, 100, 1500, "T2")

        self.all = [self.N_A, self.N_B, self.Ne, self.reco, self.mut, \
            self.N_anc, self.T_split, self.mig, self.N1, self.N2, self.growth, \
            self.N3, self.T1, self.T2]

    def update(self, names, values):
        """Based on generator proposal, update desired param values"""
        assert len(names) == len(values)

        for j in range(len(names)):
            param = names[j]

            # go through all params
            if param == "N_A":
                self.N_A.value = values[j]
            elif param == "N_B":
                self.N_B.value = values[j]
            elif param == "Ne":
                self.Ne.value = values[j]
            elif param == "reco":
                self.reco.value = values[j]
            elif param == "mut":
                self.mut.value = values[j]
            elif param == "N_anc":
                self.N_anc.value = values[j]
            elif param == "T_split":
                self.T_split.value = values[j]
            elif param == "mig":
                self.mig.value = values[j]
            elif param == "N1":
                self.N1.value = values[j]
            elif param == "N2":
                self.N2.value = values[j]
            elif param == "growth":
                self.growth.value = values[j]
            elif param == "N3":
                self.N3.value = values[j]
            elif param == "T1":
                self.T1.value = values[j]
            elif param == "T2":
                self.T2.value = values[j]
            else:
                sys.exit(param + " is not a recognized parameter.")

def parse_params(param_input, all_params):
    """See which params were desired for inference"""
    param_strs = param_input.split(',')
    parameters = []
    for p in all_params.all:
        if p.name in param_strs:
            parameters.append(p)

    assert len(parameters) == len(param_strs)
    for p in parameters:
        print(p)

    return parameters

def filter_func(x, rate):
    """Keep non-singletons. If singleton, filter at given rate"""
    # TODO since we haven't done major/minor yet, might want to add != n-1 too
    if np.sum(x) != 1:
        return True
    return np.random.random() >= rate # keep (1-rate) of singletons

def process_gt_dist(gt_matrix, dist_vec, S, filter=False, rate=None):
    """
    Take in a genotype matrix and vector of inter-SNP distances. Return a 3D
    numpy array of the given n (haps) and S (SNPs) and 2 channels.
    Filter singletons at given rate if filter=True
    """
    if filter:
        # mask
        singleton_mask = np.array([filter_func(row, rate) for row in gt_matrix])

        # reassign
        gt_matrix = gt_matrix[singleton_mask]
        dist_vec = np.array(dist_vec)[singleton_mask]

    n = gt_matrix.shape[1]
    if gt_matrix.shape[0] != len(dist_vec):
        print("gt", gt_matrix.shape[0], "dist", len(dist_vec))
    assert gt_matrix.shape[0] == len(dist_vec)
    region = np.zeros((n, S, 2), dtype=np.float32)
    num_SNPs = gt_matrix.shape[0] # SNPs x n
    mid = num_SNPs//2
    half_S = S//2

    # enough SNPs, take middle portion
    if mid >= half_S:
        minor = major_minor(gt_matrix[mid-half_S:mid+half_S,:].transpose())
        region[:,:,0] = minor
        distances = np.vstack([np.copy(dist_vec[mid-half_S:mid+half_S]) for k \
            in range(n)])
        region[:,:,1] = distances

    # not enough SNPs, need to center-pad
    else:
        minor = major_minor(gt_matrix.transpose())
        region[:,half_S-mid:half_S-mid+num_SNPs,0] = minor
        distances = np.vstack([np.copy(dist_vec) for k in range(n)])
        region[:,half_S-mid:half_S-mid+num_SNPs,1] = distances

    return region # n X SNPs X 2

def major_minor(matrix):
    """Note that matrix.shape[1] may not be S if we don't have enough SNPs"""
    n = matrix.shape[0]
    for j in range(matrix.shape[1]):
        if np.count_nonzero(matrix[:,j] > 0) > (n/2): # count the 1's
            matrix[:,j] = 1 - matrix[:,j]
    # option to convert from 0/1 to -1/+1
    #matrix[matrix == 0] = -1
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

def parse_args():
    """Parse command line arguments."""
    parser = optparse.OptionParser(description='PG-GAN entry point')

    parser.add_option('-m', '--model', type='string', \
        help='ooa, im, yri, ceu, or chb')
    parser.add_option('-s', '--sim_real', type='string', help='sim or real')
    parser.add_option('-p', '--params', type='string', \
        help='comma separated parameter list')
    parser.add_option('-g', '--grid', type='string', help='grid, or sa')
    parser.add_option('-r', '--is_range', type='string', help='range')
    parser.add_option('-i', '--input', type='string', help='input for plotting')
    parser.add_option('-o', '--output', type='string', help='output for plot')
    parser.add_option('-t', action="store_true", dest="toy", help='toy example')

    (opts, args) = parser.parse_args()

    mandatories = ['model','sim_real','params','grid']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts

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
            demos.append(msprime.PopulationParametersChange(time=float(time) \
                * 4 * Ne, initial_size=float(pop) * Ne))
    return demos

if __name__ == "__main__":
    # test major/minor and post-processing
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
    print(major_minor(a))

    process_gt_dist(a, dist_vec, 4, filter=True)
