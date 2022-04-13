import numpy as np
from design import mk_design_model, clear_mem
import sys


# STEP 1: FIXED BACKBONE DESIGN
model = mk_design_model(protocol="fixbb")
model.prep_inputs(pdb_filename="1TEN.pdb", chain="A")
model.design_3stage()


# STEP 2: HALLUCINATION
model = mk_design_model(protocol="hallucination")
model.prep_inputs(length=100, seq_init="gumbel")
model.design_2stage()


# STEP 3: BINDER HALLUCINATION
model = mk_design_model(protocol="binder")
model.prep_inputs(pdb_filename="4MZK.pdb", chain="A", binder_len=19)
model.design_3stage(soft_iters=100, temp_iters=100, hard_iters=10)

