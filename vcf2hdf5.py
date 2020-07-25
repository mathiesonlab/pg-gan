"""
Script to convert VCF to HDF5 (separate into YRI, CEU, CHB).
Author: Sara Mathieson
Date: 6/12/20
"""

import allel
import numpy as np
import optparse
import os
import sys

# sample sizes (indvs): 108, 99, 103 (x2 for num haps)
POP_LST = ["YRI", "CEU", "CHB"]

# example command line
#python3 vcf2hdf5.py -i ~/Public/1000g/ -s ~/Public/1000g/igsr_samples.tsv -o
#    ~/Public/1000g/hdf5/

def main():
    opts = parse_args()

    files = sorted(os.listdir(opts.vcf_folder))

    for f in files:
        if f.startswith("ALL") and "phase3" in f and f.endswith("gz"):
            convert_vcf(opts.vcf_folder, f, opts.h5_folder, \
                opts.samples_filename, POP_LST)

def parse_args():
    """Parse command line arguments."""
    parser = optparse.OptionParser(description='Convert VCF to HDF5')

    parser.add_option('-i', '--vcf_folder', type='string', \
        help='path to folder of input vcf files')
    parser.add_option('-s', '--samples_filename', type='string', \
        help='path to file of sample info')
    parser.add_option('-o', '--h5_folder', type='string', help='path to output')

    (opts, args) = parser.parse_args()

    mandatories = ['vcf_folder','samples_filename','h5_folder']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts

def read_samples(filename, pop):
    """Find all samples from pop (i.e. CEU)"""
    f = open(filename,'r')
    samples = []

    for line in f:
        tokens = line.split()
        if tokens[3] == pop:
            samples.append(tokens[0])

    f.close()
    return samples


def convert_vcf(vcf_folder, vcf_filename, h5_path, samples_filename, pop_lst):
    """Convert all files in the folder"""

    for pop in pop_lst:
        print(vcf_filename, pop)
        sample_lst = read_samples(samples_filename, pop)
        hdf5_filename = h5_path + pop + vcf_filename[3:-6] + "h5"
        print(hdf5_filename)

        # here we save only GT (genotypes) and POS (SNP positions)
        allel.vcf_to_hdf5(vcf_folder + vcf_filename, hdf5_filename, \
            fields=['GT','POS'], samples=sample_lst)

main()
