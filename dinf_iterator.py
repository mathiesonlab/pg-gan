# import dinf
import numpy as np
from numpy.random import default_rng
import sys

import global_vars

class dinf_builder:

    def __init__(self, vcf_path, seed, num_individuals, num_loci=global_vars.NUM_SNPS):
        self.rng = default_rng(seed)

        # NOTE, requires TBI in addition to vcf.gz.
        # TBI file must have same name as vcf but with tbi appened to the end

        self.vcfs = dinf.BagOfVcf([vcf_path])
        ploidy = 2
        self.features = dinf.HaplotypeMatrix(num_individuals=num_individuals,
                                             num_loci=num_loci, ploidy=ploidy, phased=True)
        self.num_samples = num_individuals * ploidy
        self.num_individuals = num_individuals
        self.num_loci = num_loci
        
    def real_region(self, neg1, region_len):
        result = self.features.from_vcf(vb=self.vcfs, sequence_length=global_vars.L,
                                        max_missing_genotypes=10, min_seg_sites=36, rng=self.rng)
        if neg1:
            result[result == 0] = -1

        return result

    def real_batch(self, batch_size = global_vars.BATCH_SIZE, neg1=True,
                   region_len=False):

        if region_len:
            result=[]
        else:
            result = np.zeros((batch_size, self.num_samples, self.num_loci, 2))

        for i in range(batch_size):
            region = self.real_region(neg1, region_len)
            if region_len:
                result.append(region)
            else:
                result[i] = region

        return result

    def save_results(self, outpath):
        N = 5000
        
        results = self.real_batch(batch_size=N, neg1=False, region_len=False)
        np.save(outpath, results)
        
    
class dinf_iterator:

    def __init__(self, path):
        self.batch = np.load(path)

        self.num_samples = 198
        self.num_individuals = 99
        self.sample_sizes = [198]
        
    def real_batch(self, batch_size = global_vars.BATCH_SIZE,
                    neg1=True, region_len=False):
        return self.batch
        
    def simulate_batch(self, batch_size=global_vars.BATCH_SIZE, params=[],
                         region_len=False, real=False, neg1=True):
        return self.real_batch(batch_size, neg1, region_len)

    def update_params(self, new_params):
        pass

    def get_reco(self, params):
        return 1.25e-8


        
    
if __name__ == "__main__":
    vcf_path = sys.argv[1]
    seed = global_vars.DEFAULT_SEED
    num_individuals = 99
    
    iterator = dinf_builder(vcf_path, seed, num_individuals)
    iterator.save_results("dinf_results.npy")
