Things to fix up:

`cdf2hdf5p.py`
- optparse

`util.py`
- ParamSet.all - might be unecessary __dict__ would be suff
- parge_args() uses optparse
    - doesn't use mandatory flags (fixed in argsparse)

`summary_stats.py`
- process_opts can be shortened with string access to obj
- significant duplicated code in main() for matrices
- significant duplicated code for all plot_*() functions

`pg_gan.py`
- process_opts can be shortened with string access to obj

`discriminators.py`
- can deprecate the individual pop models