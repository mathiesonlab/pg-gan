"""
Allows us to read in real data regions randomly, and also use a mask (bed
format) file so we don't use regions that are uncallable.
Author: Sara Mathieson
Date: 7/6/20
"""

# python imports
import allel
import h5py
import numpy as np
import pickle
import random
import sys

# our imports
import util

# fixed globals, TODO change to desired path (more flexibility coming later)
HDF5_PATH = "/homes/smathieson/Public/1000g/hdf5/"
BIG_DATA = "/bigdata/smathieson/amish/1000g/"
POPS = ["YRI", "CEU", "CHB"]
ENDING = ".phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5"
MAX_CHROM = 22
BED_FILE = HDF5_PATH + "20120824_strict_mask.bed"
FRAC_CALLABLE = 0.5

class Region:

    def __init__(self, chrom, start_idx, end_idx, start_sample, end_sample, \
            start_pos, end_pos):
        self.chrom = chrom
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.start_sample = start_sample
        self.end_sample = end_sample
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.region_len = end_pos - start_pos # L

    def __str__(self):
        return str(self.chrom) + ":" + str(self.start_pos) + "-" + \
            str(self.end_pos) + ", samples:" + str(self.start_sample) + "-" + \
            str(self.end_sample) + ", snps:" + str(self.end_idx - \
            self.start_idx) + ", idx:" + str(self.start_idx) + "-" + \
            str(self.end_idx)

    def inside_mask(self, mask_dict):
        mask_lst = mask_dict[self.chrom] # restrict to this chrom
        region_start_idx, start_inside = binary_search(self.start_pos, mask_lst)
        region_end_idx, end_inside = binary_search(self.end_pos, mask_lst)

        # same region index
        if region_start_idx == region_end_idx:
            if start_inside and end_inside: # both inside
                return True
            elif (not start_inside) and (not end_inside): # both outside
                return False
            elif start_inside:
                part_inside = mask_lst[region_start_idx][1] - self.start_pos
            else:
                part_inside = self.end_pos - mask_lst[region_start_idx][0]
            return part_inside/self.region_len >= FRAC_CALLABLE

        # different region index
        part_inside = 0
        # conservatively add at first
        for region_idx in range(region_start_idx+1, region_end_idx):
            part_inside += (mask_lst[region_idx][1] - mask_lst[region_idx][0])

        # add on first if inside
        if start_inside:
            part_inside += (mask_lst[region_start_idx][1] - self.start_pos)
        elif self.start_pos >= mask_lst[region_start_idx][1]:
            # start after closest region, don't add anything
            pass
        else:
            part_inside += (mask_lst[region_start_idx][1] - \
                mask_lst[region_start_idx][0])

        # add on last if inside
        if end_inside:
            part_inside += (self.end_pos - mask_lst[region_end_idx][0])
        elif self.end_pos <= mask_lst[region_start_idx][0]:
            # end before closest region, don't add anything
            pass
        else:
            part_inside += (mask_lst[region_end_idx][1] - \
                mask_lst[region_end_idx][0])

        return part_inside/self.region_len >= FRAC_CALLABLE

class RealDataRandomIterator:

    # if writing data for the first time, set write_regions=True
    # if reading from file, include num_test and filename of saved file
    def __init__(self, sample_size_lst, S, L, filename, write_regions=False, \
            num_test=None):
        """
        sample_size_lst example: [66,66,66] (YRI, CEU, CHB)
        S = number of SNPs to keep (i.e. 36)
        L = region length (i.e. around 50kb)
        This goes through all the files and stores a list of valid regions.
        """
        self.S = S
        self.L = L

        # filter out zero size pops and don't use them (1-3 pops possible)
        self.pops = []
        self.sample_sizes = []
        for i in range(len(sample_size_lst)):
            if sample_size_lst[i] > 0:
                self.pops.append(POPS[i])
                self.sample_sizes.append(sample_size_lst[i])

        if write_regions:
            self.region_lst = []
            self.total_regions = 0

            self.mask_dict = read_mask(BED_FILE) # mask
            for c in range(1, MAX_CHROM+1):
                self._prep_new_chrom(c)

                finished = False
                while not finished:
                    finished = self._process_region(c) # adds to region_lst

                print("num regions", len(self.region_lst))
                print("frac good", len(self.region_lst)/self.total_regions)

            # write to file
            num_regions = len(self.region_lst)
            all_region_data = np.zeros((num_regions, sum(sample_size_lst),S, 2))
            random.shuffle(self.region_lst) # randomize
            print("shuffled")
            for i in range(num_regions):
                if i % 100 == 0:
                    print("frac finished", i/num_regions)
                region = self.region_lst[i]
                data = self._data_from_region(region)
                all_region_data[i] = data

            # save to a file
            np.save(filename, all_region_data)

            # save regions as well (shuffled)
            pickle_filename = filename[:-3] + "pkl"
            with open(pickle_filename, 'wb') as f:
                pickle.dump(self.region_lst, f)

        # read from numpy array file
        else:
            # read from file
            self.all_region_data = np.load(filename)
            print(self.all_region_data.shape)
            self.train_idx = -1
            self.num_test = num_test
            self.test_idx = -1

    def _prep_new_chrom(self, chrom):
        """Prep new chrom from h5 file"""
        print("STARTING NEW CHROM", chrom)

        # source: http://alimanfoo.github.io/2015/09/21/estimating-fst.html
        genotype_all_lst = []
        for pop in self.pops:
            callset = h5py.File(HDF5_PATH + pop + ".chr" + str(chrom) + \
                ENDING, mode='r')
            genotype_all = allel.GenotypeChunkedArray(callset['calldata/GT'])
            genotype_all_lst.append(genotype_all)

        # compute common set of positions (YRI/CEU/CHB same original positions)
        self.positions = allel.SortedIndex(callset['variants/POS'])
        self.start_idx = 0 # start at the SNP at index 0
        self.start_base = self.positions[self.start_idx]
        self.start_sample = 0

        # find cap on number of indvs in one of the pops
        max_desire = max(self.sample_sizes)//2 # i.e. 33
        min_actual = min([arr.shape[1] for arr in genotype_all_lst]) # i.e. 99
        num_blocks = min_actual//max_desire
        self.maxn = num_blocks*max_desire

    def _process_region(self, chrom):
        """Generate a region of simulated data."""

        # set up bounds of region
        end_base = self.start_base + self.L

        # no SNPs in region already, skip it
        if not (self.start_base <= self.positions[self.start_idx] < end_base):
            self.start_base = end_base
            self.start_sample = 0
            return self._process_region(chrom)

        # SNP inside window
        assert self.start_base <= self.positions[self.start_idx] < end_base
        end_idx = self.start_idx
        # TODO technically we should exclude region if we hit first condition
        while end_idx < len(self.positions) and self.positions[end_idx] < \
                end_base:
            end_idx += 1

        # find sample start/end
        end_sample_lst = [self.start_sample + ss//2 for ss in self.sample_sizes]
        end_sample = max(end_sample_lst)

        # prepare each region
        region = Region(chrom, self.start_idx, end_idx, self.start_sample, \
            end_sample, self.start_base, end_base)

        # shift to the next set of samples
        if max(end_sample_lst) != self.maxn:
            self.start_sample = max(end_sample_lst)

        # otherwise shift to next region if we are out of samples
        else:
            self.start_idx = end_idx
            self.start_base = end_base
            self.start_sample = 0

        # if we are off the end of the chromosome, move to next chrom
        if self.start_idx >= len(self.positions) or self.start_base > \
                self.positions[-1]:
            return True

        # normal situation
        else:
            result = region.inside_mask(self.mask_dict)
            if result:
                self.region_lst.append(region)
            self.total_regions += 1 # add to total regions
            return False # not finished yet

    def real_region(self, is_train):

        # choose a random region that is NOT a test region
        if is_train:
            self.train_idx = (self.train_idx + 1) % len(self.all_region_data)
            if self.train_idx == 0:
                self.train_idx += self.num_test # don't use test data

            matrix = self.all_region_data[self.train_idx]
            # option to convert from 0/1 to -1/+1
            #matrix[matrix == 0] = -1 # hopefully none of the distances are 0?
            return matrix

        # go through the test regions
        else:
            self.test_idx = (self.test_idx + 1) % self.num_test
            matrix = self.all_region_data[self.test_idx]
            # option to convert from 0/1 to -1/+1
            #matrix[matrix == 0] = -1 # hopefully none of the distances are 0?
            return matrix

    def _data_from_region(self, region):
        # source: http://alimanfoo.github.io/2015/09/21/estimating-fst.html

        acs_lst = []
        genotype_all_lst = []
        for pop in self.pops:
            callset = h5py.File(HDF5_PATH + pop + ".chr" + str(region.chrom) + \
                ENDING, mode='r')

            portion = callset['calldata/GT'][region.start_idx:region.end_idx, \
                region.start_sample:region.end_sample, :]
            genotype_all = allel.GenotypeChunkedArray(portion)
            # note we don't want "max_allele=1" since it hides tri-allelic that
            # need to be filtered out later
            acs = genotype_all.count_alleles(max_allele=3) # heuristic
            acs_lst.append(np.array(acs))
            genotype_all_lst.append(genotype_all)

        # combine allele counts
        combine = acs_lst[0]
        for i in range(1, len(acs_lst)):
            combine = combine + acs_lst[i]
        acu = allel.AlleleCountsArray(combine)

        # filter out non-seg SNPs and multi-allelic too
        flt = acu.is_segregating() & (acu.max_allele() == 1)

        # reset genotype arrays
        gt_lst = [gt_all.compress(flt, axis=0) for gt_all in genotype_all_lst]

        # compute common set of positions (YRI/CEU/CHB same original positions)
        pos_all = allel.SortedIndex(callset['variants/POS'][region.start_idx: \
            region.end_idx])
        positions = pos_all.compress(flt) # filter on the fly
        assert len(positions) == len(gt_lst[0])

        # put all pops together
        pop_block_lst = [self._unfold(gt) for gt in gt_lst]
        stacked = np.concatenate(pop_block_lst, axis=1)

        dist_vec = [0] + [(positions[j+1] - positions[j])/self.L for j in \
            range(len(positions)-1)]

        after = util.process_gt_dist(stacked, dist_vec, self.S)
        return after

    def _unfold(self, block):
        return np.concatenate((block[:,:,0], block[:,:,1]), axis=1)

    def _augment(self, acs, max_allele):
        """Not used"""
        num_snps = acs.shape[0]
        num_alleles = acs.shape[1]
        to_add = np.zeros((num_snps,max_allele-num_alleles))
        return np.concatenate((acs, to_add), axis=1)

def read_mask(filename):

    mask_dict = {}
    f = open(filename,'r')

    for line in f:
        tokens = line.split()
        chrom_str = tokens[0][3:]
        if chrom_str != 'X' and chrom_str != 'Y':
            chrom = int(chrom_str)
            begin = int(tokens[1])
            end = int(tokens[2])

            if chrom in mask_dict:
                mask_dict[chrom].append([begin,end])
            else:
                mask_dict[chrom] = [[begin,end]]

    f.close()
    return mask_dict

def binary_search(q, lst):
    low = 0
    high = len(lst)-1

    while low <= high:

        mid = (low+high)//2
        if lst[mid][0] <= q <= lst[mid][1]: # inside region
            return mid, True
        elif q < lst[mid][0]:
            high = mid-1
        else:
            low = mid+1

    return mid, False # something close

if __name__ == "__main__":
    # for pre-processing real data (uncomment below to create data)
    S = 36
    L = 50000

    '''
    # YRI
    print("YRI")
    iterator = RealDataRandomIterator([198,0,0], S, L, BIG_DATA + "yri_s36.npy", write_regions=True)

    # CEU
    print("CEU")
    iterator = RealDataRandomIterator([0,198,0], S, L, BIG_DATA + "ceu_s36.npy", write_regions=True)

    # CHB
    print("CHB")
    iterator = RealDataRandomIterator([0,0,198], S, L, BIG_DATA + "chb_s36.npy", write_regions=True)

    # YRI+CEU
    print("YRI/CEU")
    iterator = RealDataRandomIterator([98,98,0], S, L, BIG_DATA + "yri_ceu_s36.npy", write_regions=True)

    # YRI+CHB
    print("YRI/CHB")
    iterator = RealDataRandomIterator([98,0,98], S, L, BIG_DATA + "yri_chb_s36.npy", write_regions=True)

    # CEU+CHB
    print("CEU/CHB")
    iterator = RealDataRandomIterator([0,98,98], S, L, BIG_DATA + "ceu_chb_s36.npy", write_regions=True)
    '''
