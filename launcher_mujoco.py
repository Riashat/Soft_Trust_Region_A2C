import os
import itertools
import numpy as np
import subprocess
import argparse
parser = argparse.ArgumentParser()

def grid_search(args_vals):
    """ arg_vals: a list of lists, each one of format (argument, list of possible values) """
    lists = []
    for arg_vals in args_vals:
        arg, vals = arg_vals
        ll = []
        for val in vals:
            ll.append("-" + arg + " " + str(val) + " ")
        lists.append(ll)
    return ["".join(item) for item in itertools.product(*lists)]


parser = argparse.ArgumentParser()

parser.add_argument('--experiments', type=int, default=5)
parser.add_argument('--algo', type=str, default="a2c")
parser.add_argument('--env_name', default='HalfCheetah-v1', help='environment to train on (default: HalfCheetah-v1)')
parser.add_argument('--actor_lambda',type=float,default=0.01)
# parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--num_processes', type=int, default=16, help='how many training CPU processes to use (default: 16)')
# parser.add_argument('-g',  type=str, default='0', help=['specify GPU'])
parser.add_argument('--num_frames', type=int, default=10e6, help='number of frames to train (default: 10e6)')

parser.add_argument('-f', type=str, default="./results/")          # Folder to save results in

locals().update(parser.parse_args().__dict__)    


job_prefix = "python "
exp_script = './main_a2c.py ' 
job_prefix += exp_script

args = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.g

experiments = args.experiments
env_name = args.env_name
seed = args.seed
algo = args.algo
num_processes=args.num_processes
num_frames = args.num_frames


folder = args.f


"""
Specify other hyperparameters to search over here- 
"""
grid = [] 
grid += [['-env_name', [env_name]]]
grid += [['-algo', [algo]]]
# grid += [['-num_frames', [num_frames]]]
# grid += [['-seed', [seed]]]
grid += [['-f', [folder]]]
grid += [['-num_processes', [num_processes]]]
#grid += [['-gpu', [gpu]]]

job_strs = []
for settings in grid_search(grid):
    for e in range(experiments):    
        job_str = job_prefix + settings
        job_strs.append(job_str)
print("njobs", len(job_strs))

for job_str in job_strs:
    os.system(job_str)