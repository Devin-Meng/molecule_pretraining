from rdkit import Chem
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--output_file')
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file

suppl = Chem.SDMolSupplier(input_file)
mol_list = []
for mol in suppl:
    if not mol: continue
    mol_list.append(mol)

print('delete those molecules that are not mol')
with Chem.SDWriter(output_file) as w:
    for mol in mol_list:
        w.write(mol)

print('done!!!')
