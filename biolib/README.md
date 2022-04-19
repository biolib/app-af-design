# AlphaFold Design



### Run AlphaFold Design in Python 

```
import biolib
af_design = biolib.load('Protein_Tools/af_design')
print(af_design.cli(args='--pdb example.pdb --chain A --protocol binder --design 2 --'))
```

