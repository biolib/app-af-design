import os 
import joblib
from joblib import delayed
import biolib

# Load AF Design tool
af_design = biolib.load('protein-tools/af-design-dev')

# Make dir to save the parameter sweep results
output_dir = "4mzk_param_sweep"
os.mkdir(output_dir)

# Define fixed parameters:
input_file = "/home/ubuntu/app-af-design/4mzk.pdb"
chain = "A"
design_func = "3"
protocol = "binder"

fixed_args = [ "--pdb", input_file, "--design", design_func, "--protocol", protocol, "--chain", chain]


def run_afdesign(args, name):
    design_results = af_design.cli([*fixed_args, *args])
    design_results.save_files(name)

def sweep_parameter(parameter, param_range, save_dir_name, number_of_jobs=3):
    joblib.Parallel(n_jobs=number_of_jobs)(delayed(run_afdesign)([f'--{parameter}', str(param_value)], f'{save_dir_name}_{param_value}') for param_value in param_range)


# Parameter sweep
# Ranges for binder_len
start = 15
end = 245
step = 10


"""
Sweeps the parameters in range for specified argument.
parameter: The parameter for which to search through values
param_range: The range of values to search through
save_dir_name: Naming scheme for the dirs where results will be saved. Param value is appended to the end of the name
number_of_jobs: Number of jobs running at a time
"""
# Example with binder_length. Runs will be saved as `binder_len_{param_value}` eg `binder_len_5`
sweep_parameter(
    parameter='binder-len',
    param_range=range(start, end, step),
    save_dir_name=f'{output_dir}/binder_len',
    number_of_jobs=3
)

