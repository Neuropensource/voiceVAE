# -*- coding: utf-8 -*- 

'''
This file shoudl by used as a long experiment script. 
It should run various training and testing experiments
'''

#---general imports---
import os 
import numpy as np
from tqdm import tqdm
import argparse
import time
import yaml
import subprocess as sp
import sys 

#---deep learning imports---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

#---local imports---
from downloader import SpectroDataset4D, conditionalDataset, make_path ,get_labels
from models.modules.bottleneck import vanillaBottleneck
from models.modules.CNNdecoder import vanillaDecoder, conditionalDecoder
from models.modules.CNNencoder import vanillaEncoder
from models.vanillaVAE import vanillaVAE
from models.cVAE import conditionalVAE


#---config file modifier---
from config.configuration import ConfigFileModifier
#utilisation de gridsearchCV ? rayTune ? Optuna ? 
from sklearn.model_selection import GridSearchCV





def expe_unit(config_file):

    #CONFIG  --> les arguments qui viennent de config files
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

    #All informations from config files are stored in the following variables 
    TRAINING_CONFIG = config['Training']
    MODEL_CONFIG = config['Model']
    DATASET_CONFIG = config['Dataset']


    start = time.time()
    np.random.seed(1234)
    folder_path = DATASET_CONFIG["datapath"] 
    pathLabel = DATASET_CONFIG["pathLabel"]
    labFile = get_labels(pathLabel)
    all_wav = make_path(folder_path)
    np.random.shuffle(all_wav)

    #split
    nb_train = int(len(all_wav)*DATASET_CONFIG['prop_train'])
    train_wav, eval_wav = all_wav[:nb_train], all_wav[nb_train:]
    train_spectros= conditionalDataset(train_wav,labFile) 
    eval_spectros = conditionalDataset(eval_wav,labFile) 
    train_loader = DataLoader(train_spectros,batch_size=32,shuffle=True)
    eval_loader = DataLoader(eval_spectros,batch_size=32,shuffle=True)


    #MODEL INITIALIZATION
    
    encoder = vanillaEncoder(z_dim= MODEL_CONFIG['Z_DIM'])
    bottleneck = vanillaBottleneck(z_dim= MODEL_CONFIG['Z_DIM'])
    decoder = conditionalDecoder(z_dim= MODEL_CONFIG['Z_DIM'], ydim=MODEL_CONFIG['ydim']) 
    model = conditionalVAE(encoder, bottleneck, decoder, beta= MODEL_CONFIG['beta'])
    #TODO PARAMETRISER D'AVANTAGE LES MODELES POUR GRIDSEARCH
    

    # TRAINING INITIALIZATION
    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('used device =', device)
    print('nb GPU available : ',torch.cuda.device_count())
    if torch.cuda.device_count() > 0:
        print('nom du GPU utilisé :' , torch.cuda.get_device_name(0))
    #model
    model.to(device)
    num_epochs = TRAINING_CONFIG['epochs'] 
    print("number of epochs : ", num_epochs)
    optimizer = torch.optim.Adam(model.parameters(), betas=[0.5, 0.999], lr=TRAINING_CONFIG['lr'])
    #monitoring
    writer = SummaryWriter()
    outputs = []


    #TRAINING LOOP
    for epoch in range(num_epochs):
        ### TRAIN ###
        #init
        print("Starting epoch {}".format(epoch+1))
        model.train()
        cumloss = 0
        #reweighting
        for batch in train_loader:
            batchSpectro, batchLoc = batch[0].to(device), batch[1].to(device)     
            results = model(batchSpectro, batchLoc)
            recon = results['logits']
            loss, details = model.minibatch_loss(batch=batchSpectro, device= device, y= batchLoc) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #monitoring   
            cumloss += loss.item()  
        print(f'Epoch:{epoch+1}, Loss:{cumloss/len(train_loader)}') 
        writer.add_scalar("loss/train loss",  cumloss/len(train_loader),epoch)
        outputs.append((epoch, batchSpectro, recon))
        ### EVAL ###
        #init
        model.eval()
        cumloss = 0
        #eval
        with torch.no_grad():
            for batch in eval_loader: #TODO : particulier pour cVAE
                batchSpectro, batchLoc = batch[0].to(device), batch[1].to(device)  #TODO il y a une erreur dans le trainingVAE à corriger "recon" c'est pour ça qu'on avait loss bizarre
                recon = model(batchSpectro, batchLoc)['logits']
                loss, _ = model.minibatch_loss(recon,device,y=batchLoc) 
        #monitoring
                cumloss += loss.item()
        print(f'[epoch={epoch+1}] val loss: {cumloss/len(eval_loader)}')    
        writer.add_scalar("loss/val loss", cumloss/len(eval_loader), epoch)

    #RAMENEZ LES TENSEURS AU CPU AVANT DE LES SAUVER
    model.to('cpu')
    #SAVING MODEL
  
    #on verifie que le dossier existe
    if not os.path.exists("modelsParam/cVAE"):
        os.makedirs("modelsParam/cVAE")
    torch.save(model.state_dict(), "modelsParam/cVAE/cVAEep{}zdim{}ydim{}.pth".format(epoch+1,MODEL_CONFIG['Z_DIM'], MODEL_CONFIG['ydim']))
    
    stop = time.time()
    print("time elapsed: ", stop-start)

#TODO rajouter un print ou un save de tous les paramètres de l'entrainement qui ont été utilisés
#--> et l'utiliser pour gérer les folders dans lesquels on save
if __name__ == "__main__":
    # grid = {"Training": { "lr": [0.0001,0.0005,0.001,0.005,0.01]},"Model": {"Z_DIM": [8,16,32,64,128], "beta": [0.1,0.5,1,5,10], "ydim": [2,4,8,16,32]}}
    grid = { "Z_DIM": [64,128], "ydim": [0,10,100]}
    
    config_file = "config/expe.yml"
    # config = ConfigFileModifier()   #TODO : écriture avec une classe à enrichir ensuite
    # while config.list_conf:
    #     config.modify_config_file(config_file)

    for zdim in grid['Z_DIM']:
        for ydim in grid['ydim']:
        

            with open(config_file, 'r') as file:
                config = yaml.load(file, Loader=yaml.FullLoader)

            config['Model']['Z_DIM'] = zdim
            config['Model']['ydim'] = ydim
            
            with open(config_file, 'w') as file:
                yaml.dump(config, file)

            expe_unit(config_file)
        
        
    
    
    

    
  

