from tqdm import tqdm
import numpy as np

def smilesreader(input_path):
    print('extracting smiles')
    smile_id = []
    dict_smi = {}
    with open(input_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            str1, str2 = line.split()[0], line.split()[1]
            dict_smi[str2] = str1
        f.close()
        return dict_smi

def chargereader(label_path):
    print('reading charges')
    dict_charge = {}

    with open(label_path, 'r') as f:
        lines = f.readlines()
        f.close()

    for i in tqdm(lines):
        if i.startswith('Z'):
            val_name = i[:-1]
        else:
            number = [float(j) for j in i.strip().split(' ')]
            dict_charge[val_name] = number
    return dict_charge

def nameassigner(dict_smi, dict_charge):
    data_list = []
    for i in tqdm(dict_smi.keys()):
        data_dict = {}
        data_dict['smi']=dict_smi[i]
        data_dict['charges']=dict_charge[i]
        data_list.append(data_dict)
    return data_list

input_path = 'data/smiles_all.smi'
label_path = 'data/charges_all.txt'
save_path = 'data/input_dict.npy'

dict_smi = smilesreader(input_path)
dict_charge = chargereader(label_path)
print('assign charges')
data_list = nameassigner(dict_smi, dict_charge)

data_list = np.array(data_list)

np.save(save_path, data_list)
print('done!!!')