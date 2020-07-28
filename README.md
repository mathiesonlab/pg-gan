# pg-gan

This software can be used to create realistic simulated data that matches real
population genetic data. It implements a GAN-based algorithm (Generative Adversarial Network).

Dependencies:

~~~
msprime==0.7.4
numpy==1.17.2
tensorflow==2.2.0
~~~

See the [msprime documentation](https://msprime.readthedocs.io/en/stable/index.html) and
[tensorflow pip guide](https://www.tensorflow.org/install/pip) for installation instructions.

Dependencies for pre-processing real data (not needed to try out the first two simulation command lines below):

~~~
allel==1.2.1
h5py==2.10.0
~~~

Links: [scikit-allel](https://scikit-allel.readthedocs.io/en/stable/), [HDF5 for Python](https://www.h5py.org/)

### Example command lines

Note that all output is printed to stdout, including current parameter estimates and GAN confusion.

0. Toy example with simulated "real" data. `-m im` means use the IM model (isolation with migration), `-s sim` means
use simulated data as the "real" data, `-p` is used to specify the parameters to infer (3 here), and `-g sa` means
use the simulated annealing approach. `-t` is the "toy" flag, which will run the method for 2 iterations only.

~~~
python3 pg_gan.py -t -m im -s sim -p "N_anc,T_split,mig" -g sa
~~~

1. Same example above but without the "toy" flag. This will take several hours to run (likely 8-10 with a GPU, more
    without one).

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
