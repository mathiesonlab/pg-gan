# pg-gan

This software can be used to create realistic simulated data that matches real
population genetic data. It implements a GAN-based algorithm (Generative Adversarial Network)
described in the pre-print [Automatic inference of demographic parameters using Generative Adversarial Networks](https://www.biorxiv.org/content/10.1101/2020.08.05.237834v1).

Python 3.6 (or later) is required, along with the following libraries (these exact versions are likely not necessary, but should be similar):

~~~
msprime==0.7.4
numpy==1.17.2
tensorflow==2.2.0
~~~

See the [msprime documentation](https://msprime.readthedocs.io/en/stable/index.html) and
[tensorflow pip guide](https://www.tensorflow.org/install/pip) for installation instructions.

Dependencies for pre-processing real data (not needed to try out the first two simulation command lines below):

~~~
h5py==2.10.0
~~~

Link: [HDF5 for Python](https://www.h5py.org/)

Dependencies for creating summary statistic plots:

~~~
allel==1.2.1
~~~

Link: [scikit-allel](https://scikit-allel.readthedocs.io/en/stable/)

### Simulated training data

Note that all output is printed to stdout, including current parameter estimates and GAN confusion. If you save this output to a file, it
can then be read in by the summary statistic visualization program.

1. Toy example with simulated training data. `-m im` means use the IM model (isolation with migration) and
`-p` is used to specify the parameters to infer (6 here). `-t` is the "toy" flag, which will run the method without discriminator
pre-training and then for two iterations only. It should take about 5 min with a GPU.

~~~
python3 pg_gan.py -m im -p N1,N2,N_anc,T_split,reco,mig -t
~~~

2. Same example above but without the "toy" flag. This will take several hours to run (likely 5-6 with a GPU, more
    without one).

~~~
python3 pg_gan.py -m im -p N1,N2,N_anc,T_split,reco,mig
~~~

### Real training data

Below is a tutorial that explains how to run `pg-gan` on the 1000 Genomes data. Modifications may be needed for other data, but the process
should generally be similar.

Note: this tutorial will require [bcftools](http://samtools.github.io/bcftools/bcftools.html).

1. Download the Phase 3 `ALL` VCF files from the [1000 Genomes Project](https://www.internationalgenome.org/data) for chromosomes 1-22. Also download
the accessibility mask `20120824_strict_mask.bed`.

2. Identify a set of samples for the population of interest. Here we will use CHB, and a sample file is provided above in `prep_data/CHB_gan.txt`.

3. Create a list of VCF files to use as training data. Here we will use chromosomes 1-22 from CHB, and the list of files is provided in `prep_data/CHB_filelist.txt`.

4. Prepare the VCF files using `bcftools`. This will remove non-segregating sites and multi-allelic sites, as well as retain our desired samples and concatenate chromosomes 1-22 into one VCF file. To do all these operations, use the file `prep_data/prep_vcf.sh`.

~~~
sh prep_data/prep_vcf.sh
~~~

5. Convert this final VCF file into HDF5 format:

~~~
python3 vcf2hdf5.py -i CHB.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz -o CHB.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5
~~~

6. Finally, run `pg-gan` on the data using (for example) a 5-parameter exponential growth model.

~~~
python3 pg_gan.py -m exp -p N1,N2,growth,T1,T2 -d CHB.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5 -b 20120824_strict_mask.bed
~~~

### Additional Options

* Use `-s` to specify a seed (`-s 1833` for example). Note that this seed will create reproducible results if run on the same machine, but not across machines.

* Use `-r` to specify a folder with recombination map files for each chromosome. For example `-r genetic_map/`. This will let cause the generator to sample from this recombination rate distribution when creating simulated data.
