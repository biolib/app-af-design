import sys
import numpy as np
sys.path.append('/')
from design import mk_design_model, clear_mem

import argparse
from argparse import RawTextHelpFormatter
import tensorflow as tf


# read inputs provided by user
parser = argparse.ArgumentParser(
    prog="AF-Dessign", formatter_class=RawTextHelpFormatter)

protocol = parser.add_argument_group('Define Protocol options')
protocol.add_argument("--protocol", dest="protocol",  help="Protocol for AF-design. Default: binder.\n\tbinder: Generate a protein binder sequence that AF thinks that will bind to the target structure.\n\tfixbb: Generate a new protein sequence that AF thinks that folds into that conformation.",
                      choices=["binder", "fixbb"], default="binder", required=True)
protocol.add_argument("--design", dest="design", help="Different functions for the design. Logits, soft, hard and for complex topologies: \n\t2 stages: soft-->hard\n\t3 stages: logits --> soft --> hard",
                      default="3", required=True, choices=["2", "3", "logits", "soft", "hard"])

# Prep input arguments:
prep_input = parser.add_argument_group('Define Input options')
prep_input.add_argument('--pdb', dest='pdb',
                        help="PDB input file", required=True)
prep_input.add_argument('--chain', dest="chain",
                        help="Which chain in the PDB, default: A", default="A")
prep_input.add_argument('--binder-len', dest='binder_len',
                        help="Binder length, max: 256 amino acids", type=int)


# General arguments:
mk_design = parser.add_argument_group('Define global options')
mk_design.add_argument('--dropout', dest="dropout",
                       help="Use dropout during design, helps jump out of local minima.", default="True")
mk_design.add_argument('--dropout-scale', dest="dropout_scale",
                       help="Dropout scale value, default: 1.0", default=1.0, type=float)

mk_design.add_argument("--soft-mode", dest="soft_mode", help="Soft mode options for sequence representation: softmax, softmax_gumbel, logits",
                       default="logits", choices=["softmax", "softmax_gumbel", "logits"])
mk_design.add_argument("--hard", dest="hard",
                       help="Hard sequence representation. Default: True", default="True")
mk_design.add_argument('--num-seq', dest="num_seq",
                       help="Number of sequences to predict, default: 1", default=1, type=int)
mk_design.add_argument('--num-models', dest="num_models",
                       help="Number of model parameters to use, default: 1", default=1, type=int)
mk_design.add_argument('--model-mode', dest="model_mode",
                       help="How to run the models, default: sample.\n\tsample: at each iteration, randomly select one model param to use.\n\tparallel: run num_models in parallel, average the gradients.", choices=["sample", "parallel"], default="sample")
mk_design.add_argument('--num-recycles', dest="num_recycles",
                       help="Max number of recycles to use during design. For de novo proteins, 0 is usually enough. Default: 0", default=0, type=int)
mk_design.add_argument('--recycle-mode', dest="recycle_mode", help="How to run recycles, default: sample.\n\tsample: at each iteration, randomly select a number of recycles to use, recommended.\n\tnadd_prev: add prediction logits across all recycles, stable but slow and requires memory.\n\tlast: only use gradients from last recycle.\n\tbackprop: use outputs from last recycle, but backprop through all recycles.",
                       default="sample", choices=["sample", "add_prev", "last", "backprop"])

# iters specific arguments
iters = parser.add_argument_group('Iterations')
iters.add_argument("--iters", dest="iters",
                   help="General iterations for design: logits, soft and hard. Default: 100", default=100,  type=int)
iters.add_argument("--iters-soft", dest="iters_soft",
                   help="Iterations for complex topologies. Soft iters. Default: 2stage=100, 3stage=300", type=int)
iters.add_argument("--iters-temp", dest="iters_temp",
                   help="Iterations for complex topologies. Temp iters. Default: 2stage=100, 3stage=100", type=int)
iters.add_argument("--iters-hard", dest="iters_hard",
                   help="Iterations for complex topologies. Hard iters. Default: 2stage=50, 3stage=50", type=int)
iters.add_argument("--temp", dest="temp",
                   help="Temperature factor, default: 1.0", default=1.0, type=float)

# Weights
weights = parser.add_argument_group("Weights")
weights.add_argument("--con", dest="con",
                     help="Maximize number of contacts within binder", default=0.5, type=float)
weights.add_argument("--icon", dest="i_con",
                     help="Maximize number of contacts with the interface of the protein", default=0.05, type=float)
weights.add_argument("--ipae", dest="i_pae",
                     help="Minimize Predicted Alignment Error (PAE) interface of the protein", default=0.01, type=float)
weights.add_argument("--ibkg", dest="i_bkg",
                     help="Background KL loss", default=0.0, type=float)
weights.add_argument("--dgram-cce", dest="dgram_cce",
                     help="Distogram log loss. Minimizes the categorical-cross-entropy between predicted distogram and pdb.\nOnly for fixbb protocol. Default: 1.0", default=1.0, type=float)
weights.add_argument("--fape", dest="fape",
                     help="Frame Aligned Point error (FAPE) loss. Minimize the difference between coordinates.\nOnly for fixbb protocol. Default: 0.0", default=0.0, type=float)
weights.add_argument("--rmsd", dest="rmsd",
                     help="Root Mean Squared Deviation (RMSD). Only for fixbb protocol. Default: 0.0", default=0.0, type=float)

args = parser.parse_args()

clear_mem()

# Create weights dictionary:
if args.protocol == "binder":
    if args.con is not None: con = args.con
    else: con = 0.5
    if args.i_pae is not None: i_pae = args.i_pae
    else: i_pae = 0.01
    if args.i_con is not None: i_con = args.i_con
    else: i_con = 0.05
    if args.i_bkg is not None: i_bkg = args.i_bkg
    else: args.i_bkg

    weights = {"con": con, "i_pae":i_pae, "i_con":i_con, "i_bkg":i_bkg}

elif args.protocol == "fixbb":
    if args.dgram_cce is not None: dgram_cce = args.dgram_cce
    else: dgram_cce = 1.0
    if args.fape is not None: fape = args.fape
    else: fape = 0.0
    if args.rmsd is not None: rmsd = args.rmsd 
    else: rmsd = 0.0
    if args.con is not None: con = args.con
    else: con = 0.5
    if args.i_pae is not None: i_pae = args.i_pae
    else: i_pae = 0.01
    if args.i_con is not None: i_con = args.i_con
    else: i_con = 0.05
    if args.i_bkg is not None: i_bkg = args.i_bkg
    else: args.i_bkg

    weights = {"dgram_cce": dgram_cce, "fape": fape, "rmsd": rmsd, "con": con, "i_pae":i_pae, "i_con":i_con, "i_bkg":i_bkg}

# BINDER HALLUCINATION
print("Create a model...")

if args.model_mode is not None:  model_mode = args.model_mode
else: model_mode = "sample"

model = mk_design_model(protocol=args.protocol, num_models=args.num_models, num_seq=args.num_seq,
                        model_mode=model_mode, num_recycles=args.num_recycles, recycle_mode=args.recycle_mode, weights = weights)
# Set opt
if args.dropout == "True": dropout = True
elif args.dropout == "False": dropout = False


model._default_opt["temp"] = args.temp
model._default_opt["dropout"] = dropout
model._default_opt["dropout_scale"] = args.dropout_scale


# TODO SOFT, HARD, GUMBEL...

# Set weights 
# Talk to Mads TODO
print("Read and prepare inputs...")
model.prep_inputs(pdb_filename=args.pdb, chain=args.chain,
                  binder_len=args.binder_len)

# Check the iterations values:
if args.design == "3":
    if args.iters_soft is not None:
        iters_soft = args.iters_soft
    else:
        iters_soft = 100

    if args.iters_temp is not None:
        iters_temp = args.iters_temp
    else:
        iters_temp = 100

    if args.iters_hard is not None:
        iters_hard = args.iters_hard
    else:
        iters_hard = 50
    print("Design 3 stage binder...")
    model.design_3stage(soft_iters=iters_soft,
                        temp_iters=iters_temp, hard_iters=iters_hard, temp=args.temp, dropout=dropout)

elif args.design == "2":
    if args.iters_soft is not None:
        iters_soft = args.iters_soft
    else:
        iters_soft = 300

    if args.iters_temp is not None:
        iters_temp = args.iters_temp
    else:
        iters_temp = 100

    if args.iters_hard is not None:
        iters_hard = args.iters_hard
    else:
        iters_hard = 50
    print("Design 2 stage binder...")
    model.design_2stage(soft_iters=iters_soft,
                        temp_iters=iters_temp, hard_iters=iters_hard, temp=args.temp, dropout=dropout)
elif args.design == "logits":
    model.design_logits(iters=args.iters, temp=args.temp, dropout=dropout)
elif args.design == "soft":
    model.design_soft(iters=args.iters, temp=args.temp, dropout=dropout)
elif args.design == "hard":
    model.design_hard(iters=args.iters, temp=args.temp, dropout=dropout)

model.save_pdb(filename=f"output/{args.protocol}_{args.pdb}")
seqs = model.get_seqs()

if args.protocol == "fixbb":
    model.plot_traj(filename=f"output/{args.protocol}_{args.pdb.split('.')[0]}_traj.png",dpi=150)

with open("output/output.md", "w") as out:
    out.write(f"# AF-design {args.protocol} predicted sequences\n\n")
    out.write(f"**Target protein**: {args.pdb} **and chain** {args.chain}\n\n")
    out.write(f"**Binder length**: {args.binder_len}\n\n")
    out.write(f"**Design function**: {args.design}\n\n\n")   
    out.write(f"**Predicted binder sequences:**\n\n```\n")
    for seq in seqs:
        print("Predicted binder sequence:",seq)
        out.write(seq+"\n")
    out.write(f"```\n")
    if args.protocol == "fixbb":
        out.write(f"![Trajectories plot](output/{args.protocol}_{args.pdb.split('.')[0]}_traj.png")


