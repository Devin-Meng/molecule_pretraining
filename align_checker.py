from rdkit import Chem
import numpy as np
from tqdm import tqdm

input_file = '/home/Zhaoxu/molecule_pretraining/data/input_dict.npy'
save_path = '/home/Zhaoxu/molecule_pretraining/data/new_dict.npy'
print('start to load')
input_dict = np.load(input_file, allow_pickle=True)
print(f'original length is {len(input_dict)}')

new_list = []
j = 0
for i in range(len(input_dict)):
    mol = Chem.MolFromSmiles(input_dict[i]['smi'])
    if len(mol.GetAtoms()) == len(input_dict[i]['charges']):
        new_list.append(input_dict[i])
    else:
        print(f'number {i} not match')
        j = j + 1
print(f'find {j} bad examples')
print(f'new dict length is {len(new_list)}')

new_dict = np.array(new_list)

np.save(save_path, new_dict)
print('done!!!')
