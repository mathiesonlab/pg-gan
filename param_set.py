"""
Parameter class for pg-gan, including default values for common models.
Author: Sara Mathieson, Rebecca Riley
Date: 9/27/22
"""

# python imports
import numpy as np
from scipy.stats import norm
import sys

# our imports
import simulation

class Parameter:
    """
    Holds information about evolutionary parameters to infer.
    Note: the value arg is NOT the starting value, just used as a default if
    that parameter is not inferred, or the truth when training data is simulated
    """

    def __init__(self, value, min, max):
        self.value = value
        self.min = min
        self.max = max
        self.proposal_width = (self.max - self.min)/15 # heuristic

    def start(self):
        # random initialization
        return np.random.uniform(self.min, self.max)

    def start_range(self):
        start_min = np.random.uniform(self.min, self.max)
        start_max = np.random.uniform(self.min, self.max)
        if start_min <= start_max:
            return [start_min, start_max]
        return self.start_range()

    def fit_to_range(self, value):
        value = min(value, self.max)
        return max(value, self.min)

    def proposal(self, curr_value, multiplier):
        if multiplier <= 0: # last iter
            return curr_value

        # normal around current value (make sure we don't go outside bounds)
        new_value = norm(curr_value, self.proposal_width*multiplier).rvs()
        new_value = self.fit_to_range(new_value)
        # if the parameter hits the min or max it tends to get stuck
        if new_value == curr_value or new_value == self.min or new_value == \
            self.max:
            return self.proposal(curr_value, multiplier) # recurse
        else:
            return new_value

    def proposal_range(self, curr_lst, multiplier):
        new_min = self.fit_to_range(norm(curr_lst[0], self.proposal_width *
            multiplier).rvs())
        new_max = self.fit_to_range(norm(curr_lst[1], self.proposal_width *
            multiplier).rvs())
        if new_min <= new_max:
            return [new_min, new_max]
        return self.proposal_range(curr_lst, multiplier) # try again

class ParamSet:

    def __init__(self, simulator, iterable_params=[]):
        """Takes in a simulator to determine which params are needed"""

        param_set = {}
        param_set["reco"] = Parameter(1.25e-8, 1e-9, 1e-7)
        param_set["mut"] = Parameter(1.25e-8, 1e-9, 1e-7)

        # constant population size
        if simulator == simulation.const:
            param_set["Ne"] = Parameter(10000, 1000, 30000)

        # exponential growth model
        elif simulator == simulation.exp:
            param_set["N1"] = Parameter(9000, 1000, 30000)
            param_set["N2"] = Parameter(5000, 1000, 30000)
            param_set["T1"] = Parameter(2000, 1500, 5000)
            param_set["T2"] = Parameter(350, 100, 1500)
            param_set["growth"] = Parameter(0.005, 0.0, 0.05)

        # Isolation with Migration model
        elif simulator == simulation.im:
            param_set["N1"] = Parameter(9000, 1000, 30000)
            param_set["N2"] = Parameter(5000, 1000, 30000)
            param_set["N_anc"] = Parameter(15000, 1000, 25000)
            param_set["T_split"] = Parameter(2000, 500, 20000)
            param_set["mig"] = Parameter(0.05, -0.2, 0.2)

        # ooa2
        elif simulator == simulation.ooa2:
            param_set["N1"] = Parameter(9000, 1000, 30000)
            param_set["N2"] = Parameter(5000, 1000, 30000)
            param_set["N3"] = Parameter(12000, 1000, 30000)
            param_set["N_anc"] = Parameter(15000, 1000, 25000)
            param_set["T1"] = Parameter(2000, 1500, 5000)
            param_set["T2"] = Parameter(350, 100, 1500)
            param_set["mig"] = Parameter(0.05, -0.2, 0.2)

        # post ooa
        elif simulator == simulation.post_ooa:
            param_set["N1"] = Parameter(9000, 1000, 30000)
            param_set["N2"] = Parameter(5000, 1000, 30000)
            param_set["N3"] = Parameter(12000, 1000, 30000)
            param_set["N_anc"] = Parameter(15000, 1000, 25000)
            param_set["T1"] = Parameter(2000, 1500, 5000)
            param_set["T2"] = Parameter(350, 100, 1500)
            param_set["mig"] = Parameter(0.05, -0.2, 0.2)

        # ooa3 (dadi)
        elif simulator == simulation.ooa3:
            param_set["N_A"] = Parameter(7300, 1000, 30000)
            param_set["N_B"] = Parameter(2100, 1000, 20000)
            param_set["N_AF"] = Parameter(12300, 1000, 40000)
            param_set["N_EU0"] = Parameter(1000, 100, 20000)
            param_set["N_AS0"] = Parameter(510, 100, 20000)
            param_set["r_EU"] = Parameter(0.004, 0.0, 0.05)
            param_set["r_AS"] = Parameter(0.0055, 0.0, 0.05)
            param_set["T_AF"] = Parameter(8800, 8000, 15000)
            param_set["T_B"] = Parameter(5600, 2000, 8000)
            param_set["T_EU_AS"] = Parameter(848, 100, 2000)
            param_set["m_AF_B"] = Parameter(25e-5, 0.0, 0.01)
            param_set["m_AF_EU"] = Parameter(3e-5, 0.0,  0.01)
            param_set["m_AF_AS"] = Parameter(1.9e-5, 0.0, 0.01)
            param_set["m_EU_AS"] = Parameter(9.6e-5, 0.0, 0.01)

        else:
            sys.exit(str(simulator) + " not supported")

        if iterable_params == []:
            params = param_set
        else:
            params = {}
            names = param_set.keys()
            for param_name in names:
                if param_name in iterable_params:
                    params[param_name] = param_set[param_name] # add to new set

        self.iterable_params = iterable_params
        self.param_set = params
        self.simulator = simulator

    def to_list(self):
        if self.iterable_params == []:
            param_range = self.param_set.keys()
        else:
            param_range = self.iterable_params

        result = []
        for param_name in param_range:
            param = self.param_set[param_name]
            result.append(param.value)
        return result

    def __str__(self):
        if self.iterable_params == []:
            param_range = self.param_set.keys()
        else:
            param_range = self.iterable_params

        s = '\t'.join(["NAME", "VALUE", "MIN", "MAX"]) + '\n'
        for param_name in param_range:
            param = self.param_set[param_name]
            s += '\t'.join([param_name, str(param.value), str(param.min), \
                str(param.max)])
        return s

    def clone(self, start=False):
        # make the object but reset it
        new_param_set = ParamSet(self.simulator, self.iterable_params)

        # copy over current values
        for param_name in self.param_set:
            new_param_set.param_set[param_name].value = \
                new_param_set.param_set[param_name].start() if start \
                else self.get(param_name)

        return new_param_set

    def propose_param(self, param_name, value, multiplier):
        proposal = self.param_set[param_name].proposal(value, multiplier)
        self.param_set[param_name].value = proposal

    def proposal_all(self, multiplier, value_dict=None):
        if value_dict is None:
            value_dict = self.param_set

        for key in self.param_set:
            self.propose_param(key, value_dict.get(key), multiplier)

    def update(self, iterable_params):
        new_params = iterable_params.param_set

        for key in new_params:
            self.param_set[key] = new_params[key]

    def get(self, param_name):
        return self.param_set[param_name].value
