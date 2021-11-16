from rdkit import Chem
import numpy as np
from tqdm import tqdm

input_path = '/home/Zhaoxu/molecule_pretraining/data/input_dict.npy'
save_path = '/home/Zhaoxu/molecule_pretraining/data/new_input.npy'
input_dict = np.load(input_path, allow_pickle=True)
print(f'the original length is {len(input_dict)}')

new_input = []
j = 0
for i in tqdm(input_dict):
    mol = Chem.MolFromSmiles(i['smi'])
    if mol:
        new_input.append(i)
    else: j = j+1

print(f'the processed length is {len(new_input)}')
print(f'the record is {j}')
np.save(save_path, new_input)