Things to fix up:

`vcf2hdf5.py`
X optparse

`util.py`
X ParamSet.all - might be unecessary __dict__ would be suff
X parge_args() uses optparse
    X doesn't use mandatory flags (fixed in argsparse)

`summary_stats.py`
X fix reference to postOOA
X process_opts can be shortened with string access to obj
- significant duplicated code in main() for matrices
- significant duplicated code for all plot_*() functions

`pg_gan.py`
X process_opts() can be shortened with string access to obj

`discriminators.py`
X rename postOOA to post_ooa (naming convention)

`discriminators.py`
X can deprecate the individual pop models
X can make it handle from 1 instead of onepop=0