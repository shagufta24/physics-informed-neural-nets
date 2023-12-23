import numpy as np
import itertools

# Global variables

S = 1206  # number of samples
input_var = 2  # number of input variables
output_var = 3540  # number of output variables
output_var_clipped = 3306
L = 1206  # number of hidden layer nodes
wt_lower = -1  # lower limit of range for weight
wt_upper = 1  # upper limit of range for weight 

alpha_list = np.array([i for i in np.arange(-10, 10.1, 0.1)])  # alpha values
mach_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # mach values
list1 = [mach_list, alpha_list]
pairs = np.array([p for p in itertools.product(*list1)])  # combinations of mach and alpha AKA cartesian product (creating samples)