"""
Allows us to read in real data regions randomly, and also use a mask (bed
format) file so we don't use regions that are uncallable.
Author: Sara Mathieson
Date: 2/4/21
"""

# python imports
from collections import defaultdict
import h5py
import numpy as np
import random
import sys
import datetime

# our imports
import global_vars
import util

class Region:

    def __init__(self, chrom, start_pos, end_pos):
        self.chrom = chrom
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.region_len = end_pos - start_pos # L

    def __str__(self):
        s = str(self.chrom) + ":" + str(self.start_pos) + "-" +str(self.end_pos)
        return s

    def inside_mask(self, mask_dict, frac_callable = 0.5):
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
            return part_inside/self.region_len >= frac_callable

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

        return part_inside/self.region_len >= frac_callable

def read_mask(filename):
    """Read from bed file"""

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

class RealDataRandomIterator:

    def __init__(self, filename, bed_file, chrom_starts=False):
        callset = h5py.File(filename, mode='r')
        print(list(callset.keys()))
        # output: ['GT'] ['CHROM', 'POS']
        print(list(callset['calldata'].keys()),list(callset['variants'].keys()))

        raw = callset['calldata/GT']
        print("raw", raw.shape)
        newshape = (raw.shape[0], -1)
        self.haps_all = np.reshape(raw, newshape)
        self.pos_all = callset['variants/POS']
        # same length as pos_all, noting chrom for each variant (sorted)
        self.chrom_all = callset['variants/CHROM']
        print("after haps", self.haps_all.shape)
        self.num_samples = self.haps_all.shape[1]

        '''print(self.pos_all.shape)
        print(self.pos_all.chunks)
        print(self.chrom_all.shape)
        print(self.chrom_all.chunks)'''
        self.num_snps = len(self.pos_all) # total for all chroms

        self.mask_dict = read_mask(bed_file) # mask

        # useful for fastsimcoal and msmc
        if chrom_starts:
            self.chrom_counts = defaultdict(int)
            for x in list(self.chrom_all):
                self.chrom_counts[int(x)] += 1

    def find_end(self, start_idx, region_len):
        """
        Based on the given start_idx and the region_len, find the end index
        """
        ln = 0
        chr = self.chrom_all[start_idx]
        i = start_idx
        curr_pos = self.pos_all[start_idx]
        while ln < region_len:

            if len(self.pos_all) <= i+1:
                chr_str = chr.decode("utf-8") if isinstance(chr, bytes) else chr
                print("not enough on chrom", chr_str)
                return -1 # not enough on last chrom

            next_pos = self.pos_all[i+1]
            if self.chrom_all[i+1] == chr:
                diff = next_pos - curr_pos
                ln += diff
            else:
                chr_str = chr.decode("utf-8") if isinstance(chr, bytes) else chr
                print("not enough on chrom", chr_str)
                return -1 # not enough on this chrom
            i += 1
            curr_pos = next_pos

        return i # exclusive

    def real_region(self, neg1, region_len):

        S = global_vars.NUM_SNPS if not global_vars.filter_real_data else global_vars.num_SNPs_adjusted
        start_idx = random.randrange(self.num_snps - S) # inclusive

        # go by region len or by SNPs
        end_idx_len = self.find_end(start_idx, global_vars.L)

        if end_idx_len == -1:
            return self.real_region(neg1, region_len) # try again

        if region_len:
            end_idx_S = end_idx_len
        else:
            end_idx_S = start_idx + S # exclusive

        # make sure we don't span two chroms
        start_chrom = self.chrom_all[start_idx]
        end_chrom = self.chrom_all[end_idx_S-1] # inclusive here

        if start_chrom != end_chrom:
            #print("bad chrom", start_chrom, end_chrom)
            return self.real_region(neg1, region_len) # try again
                                                                            
        hap_data = self.haps_all[start_idx:end_idx_S, :]
        start_base = self.pos_all[start_idx]
        end_base = self.pos_all[end_idx_S]
        end_base_len = self.pos_all[end_idx_len]
        positions_len = self.pos_all[start_idx:end_idx_len]
        positions_S = self.pos_all[start_idx:end_idx_S] # different if !region_len

        chrom_num = int(start_chrom[3:]) if global_vars.new_data else int(start_chrom)
        region = Region(chrom_num, start_base, end_base)
        result = region.inside_mask(self.mask_dict)

        # if we do have an accessible region
        if result:
            dist_vec = [0] + [(positions_S[j+1] - positions_S[j])/global_vars.L for j in \
                range(len(positions_S)-1)]

            after = util.process_gt_dist(hap_data, dist_vec, len(dist_vec), \
                neg1=neg1)
            return after

        # try again if not in accessible region
        return self.real_region(neg1, region_len)

    def real_batch(self, batch_size, neg1=True, region_len=False):
        """Use region_len=True for fixed region length, not by SNPs"""

        if not region_len:
            regions = np.zeros((batch_size, self.num_samples, global_vars.NUM_SNPS, 2), \
                dtype=np.float32)

            for i in range(batch_size):
                regions[i] = self.real_region(neg1, region_len)

        else:
            regions = []
            for i in range(batch_size):
                regions.append(self.real_region(neg1, region_len))

        return regions

    def real_chrom(self, chrom, samples):
        """Mostly used for msmc - gather all data for a given chrom int"""
        start_idx = 0
        for i in range(1, chrom):
            start_idx += self.chrom_counts[i]
        end_idx = start_idx + self.chrom_counts[chrom]
        print(chrom, start_idx, end_idx)
        positions = self.pos_all[start_idx:end_idx]

        assert len(samples) == 2 # two populations
        n = self.haps_all.shape[1]
        half = n//2
        pop1_data = self.haps_all[start_idx:end_idx, 0:samples[0]]
        pop2_data = self.haps_all[start_idx:end_idx, half:half+samples[1]]
        hap_data = np.concatenate((pop1_data, pop2_data), axis=1)
        assert len(hap_data) == len(positions)

        return hap_data.transpose(), positions

if __name__ == "__main__":
    # testing
    S = 36
    L = 50000

    # test file
    filename = sys.argv[1]
    bed_file = sys.argv[2]
    iterator = RealDataRandomIterator(filename, bed_file)

    start_time = datetime.datetime.now()
    for i in range(100):
        region = iterator.real_region(False, False)

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    print("time s:ms", elapsed.seconds,":",elapsed.microseconds)

    # test find_end
    for i in range(10):
        start_idx = random.randrange(iterator.num_snps-global_vars.S)
        iterator.find_end(start_idx, L)
