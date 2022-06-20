'''For collecting global values'''

new_data = False

# general
NUM_SNPS = 36       # number of seg sites, should be divisible by 4
L = 50000           # heuristic to get enough SNPs for simulations (50,000 or fifty-thousand)
CHROM_RANGE = range(1,23)

BATCH_SIZE = 50

# util
DEFAULT_SEED = 1833
DEFAULT_SAMPLE_SIZE = 198

# fitlering
filter_simulated = False
filter_real_data = False
filter_rate = 0.50
num_SNPs_adjusted = NUM_SNPS * 3

# summary stats
frac_test = 0.1
COLOR_DICT = {"YRI": 'darkorange',"CEU": 'blue',"CHB": 'green', "MXL": 'red', "simulation": 'gray'}

ss_labels = []
ss_colors = []
'''Default as CEU so that we get some color

Override by commenting out the function body,
and adding in your definitions. Leave the assert
at the end.
'''
def update_ss_labels(pop_names):    
    # ss_labels is a list of string labels, ex ["CEU", "YRI", "CHB", "simulation"]
    # or ["msprime", "SLiM"]
    ss_labels.extend(pop_names.split("_"))
    ss_labels.append("simulation")


    # colors for plotting, ex ["blue", "darkorange", "green", "gray"] (last is traditionally gray)
    # ss_colors = []
    for label in ss_labels:
        ss_colors.append(COLOR_DICT[label])
    
    assert len(ss_labels) == len(ss_colors)

# to use custom trial data, switch overwrite_trial_data to True and
# change the trial_data dictionary to have the values desired.
# Model, params, and param_values must be defined
overwrite_trial_data = True
trial_data = { 'model': 'const', 'params': 'Ne', 'data_h5': None,
               'bed_file': None, 'reco_folder': None, 'param_values': '10000.'}

'''
Recommended:
{new data: False, filter_simulated = False, filter_real_data = False}

If you're using new data, it is recommended to filter singletons:
{new data: True, filter_simulated = True, filter_real_data = True}
'''

if __name__ == "__main__":
    # testing
    print(ss_labels)
    print(ss_colors)
    update_ss_labels("CEU")
    print(ss_labels)
    print(ss_colors)
        
