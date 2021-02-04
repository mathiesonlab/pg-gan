"""
Script to convert VCF to HDF5. Samples/sites should already be filtered (if
desired) in the VCF.
Author: Sara Mathieson
Date: 2/4/21
"""

import allel
import numpy as np
import optparse
import sys

# example command line
# python3 vcf2hdf5.py -i YRI.vcf.gz -o YRI.h5

def main():
    opts = parse_args()
    convert_vcf(opts.vcf_filename, opts.h5_filename)

def parse_args():
    """Parse command line arguments."""
    parser = optparse.OptionParser(description='Convert VCF to HDF5')

    parser.add_option('-i', '--vcf_filename', type='string', \
        help='path to input VCF file')
    parser.add_option('-o', '--h5_filename', type='string', \
        help='path to output H5 file')

    (opts, args) = parser.parse_args()

    mandatories = ['vcf_filename','h5_filename']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts

def convert_vcf(vcf_filename, h5_filename):
    """Convert vcf_filename"""
    # here we save only CHROM, GT (genotypes) and POS (SNP positions)
    # see: https://scikit-allel.readthedocs.io/en/stable/io.html
    allel.vcf_to_hdf5(vcf_filename, h5_filename, fields=['CHROM','GT','POS'])

if __name__ == "__main__":
    main()
