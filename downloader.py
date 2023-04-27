
import torch
import os
import numpy as np
from torch.utils.data import Dataset

"""
Attention
--> unsqueeze ou non (charly = non)
--> transpose ou non (charly = non)
"""
class SpectroDataset(Dataset):
    """ A custom dataset to load spectrograms """
    def __init__(self, datapath):
        self.path = datapath

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        link = self.path[idx]
        spectro = np.load(link)
        spectro = np.transpose(spectro)
        spectro = torch.from_numpy(spectro)
        return spectro
    

class SpectroDataset4D(SpectroDataset):
    def __init__(self, datapath):
        self.path = datapath
        #super(SpectroDataset4D,self).__init__()
        

    def __getitem__(self, idx):
        link = self.path[idx]
        spectro = np.load(link)
        #spectro = np.transpose(spectro)
        spectro = np.expand_dims(spectro, axis=0)
        spectro = torch.from_numpy(spectro)
        return spectro




def make_path(folder_path):
    all_wav = []
    folders_locuteur = os.listdir(folder_path)
    for folder in folders_locuteur:
        wav_files = os.listdir(folder_path + folder)
        for wav in wav_files:
            all_wav.append(folder_path + folder + "/" + wav)
    return all_wav