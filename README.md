# pg-gan

This software can be used to create realistic simulated data that matches real
population genetic data. It implements a GAN-based algorithm (Generative Adversarial Network).

Dependencies:

~~~
msprime==0.7.4
numpy==1.17.2
tensorflow==2.2.0
~~~

Dependencies for pre-processing real data:

~~~
allel==1.2.1
h5py==2.10.0
~~~

Links: [scikit-allel](https://scikit-allel.readthedocs.io/en/stable/), [HDF5 for Python](https://www.h5py.org/)

### Example command lines

Note that all output is printed to stdout, including current parameter estimates and GAN confusion.

1. Example with simulated "real" data. `-m im` means use the IM model (isolation with migration), `-s sim` means
use simulated data as the "real" data, `-p` is used to specify the parameters to infer (3 here), and `-g sa` means
use the simulated annealing approach.

~~~
python3 pg_gan.py -m im -s sim -p "N_anc,T_split,mig" -g sa
~~~

2. `-g grid` can be used to run a simple grid search over the parameter of interest (only one parameter supported with this option).

~~~
python3 pg_gan.py -m im -s sim -p "N_anc" -g grid
~~~

3. Example with 1000 Genomes data from the YRI population (note: VCFs need to be pre-processed using `vcf2hdf5.py` and `real_data_random.py`).

~~~
python3 pg_gan.py -m yri -s real -p "N1,N2,growth,T1,T2" -g sa
~~~
