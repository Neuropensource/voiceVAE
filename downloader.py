
import torch
import os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
"""
Attention
--> unsqueeze ou non (charly = non)
--> transpose ou non (charly = non)
"""
#ATTENTION IL FAUT AJOUTER UN ROOTPATH pour qu'il soit appelable depuis n'importe où
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
        #self.path = datapath
        super(SpectroDataset4D,self).__init__(datapath)
        
    def __getitem__(self, idx):
        link = self.path[idx]
        spectro = np.load(link)
        #spectro = np.transpose(spectro)
        spectro = np.expand_dims(spectro, axis=0)
        spectro = torch.from_numpy(spectro)
        return spectro

'''
#on ne veut plus passer par locuteurID puisque toute l'info est contenu dans le datapath 
# class conditionalDataset(SpectroDataset):
#     def __init__(self, datapath, locuteurID):
#         super(conditionalDataset,self).__init__(datapath)
#         self.labels = locuteurID

#     def __getitem__(self, idx):
#         link = self.path[idx]
#         spectro = np.load(link)
#         spectro = np.expand_dims(spectro, axis=0)
#         spectro = torch.from_numpy(spectro)
#         label = self.labels[idx]
#         #label = torch.tensor(label, dtype=torch.long)   #on convertit en index tensor
#         #label = torch.nn.functional.one_hot(label, num_classes=163)  # il faut donner l'embedding en one hot (ou alors c'est le modèle qui aurait pu le faire...)
#         return spectro, label
'''   


#TODO rootpath jsuqu'au labels.xlsx
class conditionalDataset(SpectroDataset):
    def __init__(self, datapath, labFile):
        super(conditionalDataset,self).__init__(datapath)
        self.labFile = labFile 

    def __getitem__(self, idx):
        link = self.path[idx]
        soundID = self.get_soundID(link) #quand on fait ça, bien vérifier que rien ne change l'objet dans la fonction
        speakerID = self.get_speaker_identity(soundID,self.labFile)

        spectro = np.load(link)
        spectro = np.expand_dims(spectro, axis=0)
        spectro = torch.from_numpy(spectro)
        #speakerID = torch.tensor(speakerID, dtype=torch.long)   #on convertit en index tensor
        #speakerID = torch.nn.functional.one_hot(speakerID, num_classes=163)  # il faut donner l'embedding en one hot (ou alors c'est le modèle qui aurait pu le faire...)
        return spectro, speakerID
    
    def get_soundID(self,soundpath):
        return soundpath.split('/')[-1].split('.')[0]
    
    def get_speaker_identity(self,soundID,labFile):
        return labFile[labFile['sound_id'] == soundID]['client_id'].values[0]



def make_path(folder_path):
    all_wav = []
    folders_locuteur = os.listdir(folder_path)
    for folder in folders_locuteur:
        wav_files = os.listdir(folder_path + folder)
        for wav in wav_files:
            all_wav.append(folder_path + folder + "/" + wav)
    return all_wav


def get_labels(labels_path):
    labFile = pd.read_excel(labels_path)
    list_loc = list(labFile['client_id'].unique())
    labFile['client_id'] = labFile['client_id'].apply(lambda x: list_loc.index(x))
    return labFile