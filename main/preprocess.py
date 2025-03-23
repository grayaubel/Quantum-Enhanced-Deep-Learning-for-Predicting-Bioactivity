import numpy as np
import rdkit # type: ignore

from rdkit import Chem, DataStructs # type: ignore
from rdkit.Chem import AllChem # type: ignore

import sys
import torch # type: ignore


def smiles_to_ecfp(mol, radius=3, n_bits=2048):
    
    morgan_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    ecfp = morgan_gen.GetCountFingerprint(mol)
    
    ecfp_array = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(ecfp, ecfp_array)
    
    return ecfp_array


def split_dataset(dataset, ratio):
    """Shuffle and split a dataset."""
    np.random.seed(1234)  # fix the seed for shuffle.
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]


def create_datasets(dataset, device):

    dir_dataset = '../data/' + dataset + '/'

    def create_dataset(filename):

        print(filename)

        """Load a dataset."""
        with open(dir_dataset + filename, 'r') as f:
            data_original = f.read().strip().split('\n')

        """Exclude the data contains '.' in its smiles."""
        data_original = [data for data in data_original
                         if '.' not in data.split()[0]]
        
        dataset = []

        for data in data_original:

            smiles, homo, lumo, gap, property = data.strip().split()

            """Create each data with the above defined functions."""
            mol = Chem.MolFromSmiles(smiles)
            fingerprints = smiles_to_ecfp(mol)

            """Transform the aboce each data of numpy
            to pytorch tensor on a device (i.e., CPU or GPU).
            """
            fingerprints = torch.LongTensor(fingerprints).to(device)
            quantum = torch.FloatTensor(np.array([float(homo), float(lumo), float(gap)])).to(device)
            property = torch.FloatTensor(([float(property)])).to(device)

            dataset.append((fingerprints, quantum, property))

        return dataset
    
    '''Split data train,test,val : 70:15:15'''
    dataset_train = create_dataset('dataset.txt')
    dataset_train, dataset_test = split_dataset(dataset_train, 0.7)
    dataset_test, dataset_val = split_dataset(dataset_test, 0.5)

    return dataset_train, dataset_test, dataset_val