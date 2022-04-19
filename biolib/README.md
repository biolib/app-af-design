# AlphaFold Design

AlphaFold design backpropagates through AlphaFold to generate new protein sequences. This tool includes two protocols:

- Fix backbone design. For a given protein backbone, generate/design a new sequence that AlphaFold thinks folds into that conformation.
 - Binder design. For a given protein target and protein binder length, generate/hallucinate a protein binder sequence AlphaFold thinks will bind to the target structure. 

### Run AlphaFold Design in Python 

```
import biolib

af_design = biolib.load('Protein_Tools/af_design')
af_design_results = af_design.cli(args='--pdb example.pdb --chain A --protocol binder --design 2 --binder-len 20 --num-seq 10')
af_design_results.save_files("biolib_results/")
```

We have also created a [Google Colaboratory notebook](https://colab.research.google.com/drive/1iCGh-qpgkyPSQT2Hrrr3rgvMaGgcMrbm?usp=sharing) to see how to run AF Design step by step. 

