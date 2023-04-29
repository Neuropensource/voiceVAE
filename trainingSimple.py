# -*- coding: utf-8 -*- ???

#--config--
tboard = False
modelType = 'AE2D'

#---general imports---
import os 
import numpy as np
import tqdm
import argparse
import time

#---deep learning imports---

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
if tboard:
    from torch.utils.tensorboard import SummaryWriter

#---local imports---
from downloader import SpectroDataset, SpectroDataset4D, make_path
from models.VAEcharly import VAECharly
from models.modules.bottleneck import VariationalBottleneck
from models.modules.CNNdecoder import ConvolutionalDecoder
from models.modules.CNNencoder import ConvolutionalEncoder
from models.simpleAE import AE1D, AE2D

#PARSER
parser = argparse.ArgumentParser(description='general parser')
parser.add_argument("--device", type=str, default="local", help="device to use ('local' or 'cluster')")
parser.add_argument("--partial", type=bool, default=False, help="use only a part of the dataset")  #required=False
#recuperation des arguments
args = parser.parse_args()
print(args.partial)

#on local machine or on clusterSMS
if args.device == 'local':
    datapath = "../Data/charly/spectrograms_win800_fra200_rFalse_nfft800_nmelsNone_amptodbTrue/"
elif args.device == 'cluster':
    datapath = "/data1/data/expes/hippolyte.dreyfus/charly/spectrograms_win800_fra200_rFalse_nfft800_nmelsNone_amptodbTrue/"
else:
    raise AssertionError # à revoir comment on utilise ça (les assert et raise)



#All informations from config files are stored in the following variables
TRAINING_CONFIG = {}
MODEL_CONFIG = {}
DATA_CONFIG = {"DATAPATH" : datapath}
#DATA_CONFIG = {"DATAPATH" : "/data1/data/expes/hippolyte.dreyfus/charly/spectrograms_win800_fra200_rFalse_nfft800_nmelsNone_amptodbTrue/"}




if __name__ == "__main__":
    start = time.time()
    #RECORD FROM THE CLUSTER
    #tensorboard, start session, a configurer
    #autres methodes ?

    #DATA INITIALIZATION
    #separation train/test
    folder_path = DATA_CONFIG["DATAPATH"]
    all_wav = make_path(folder_path)
    np.random.seed(1234)
    np.random.shuffle(all_wav)
    train_wav, test_wav = np.array_split(all_wav,2) # TO DO PROPERLY, on split la liste des chemins en deux listes train et test
    
    if args.partial:
        train_wav = train_wav[:1000]
        test_wav = test_wav[:100]

    if modelType == 'AE2D':
        Train_spectros = SpectroDataset4D(train_wav) 
        Test_spectros = SpectroDataset4D(test_wav)
    elif modelType == 'AE1D':
        Train_spectros = SpectroDataset(train_wav) 
        Test_spectros = SpectroDataset(test_wav)
    else:
        raise AssertionError
    
    train_loader = DataLoader(Train_spectros,batch_size=32,shuffle=True)
    test_loader = DataLoader(Test_spectros,batch_size=32,shuffle=True)


    #MODEL INITIALIZATION
    z_dim = 64
    if modelType == 'AE2D':
        modelAE = AE2D(z_dim)
    elif modelType == 'AE1D':
        modelAE = AE1D(z_dim)
    else:
        raise AssertionError
    

    
    # TRAINING INITIALIZATION
    #print('nb GPU available : ',torch.cuda.device_count())
    #print('nom du GPU utilisé' : torch.cuda.get_device_name(0))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('used device =', device)
    modelAE.to(device)

    num_epochs = 5
    optimizer = torch.optim.Adam(modelAE.parameters(), betas=[0.5, 0.999], lr=0.00005)
    #optimizer = torch.optim.Adam(modelAE.parameters(),lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    outputs = []
    
    
    #TENSORBOARD INITIALIZATION
    if tboard:
        writer = SummaryWriter()
    
    #TRAINING LOOP
    for epoch in range(num_epochs):

        #train
        print("Starting epoch {}".format(epoch+1))
        modelAE.train()
        num_batch = 0
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            recon = modelAE(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_batch += 1
            train_loss += loss.item()
        #monitoring
            #print('batch number : ', num_batch, 'over', len(train_loader))
        train_loss = train_loss/len(train_loader)
        if tboard:  
            writer.add_scalar("loss/train loss",  train_loss,epoch)
        print(f'[Epoch:{epoch+1}], train loss:{train_loss}')
        #print('torch is using', torch.cuda.device_count(), 'GPUs')
        
        
        #eval
        modelAE.eval()
        val_losses = []
        with torch.no_grad():
            for i,batch in enumerate(test_loader):
                if i < 50:
                    batch = batch.to(device)
                    loss = criterion(modelAE(batch), batch)
                    val_losses.append(loss)
        val_loss = torch.mean(torch.stack(val_losses))
        #monitoring
        print(f'[Epoch={epoch+1}] val loss: {val_loss.item()}')
        outputs.append((epoch, train_loss, val_loss))
        if tboard:
            writer.add_scalar("loss/val loss", val_loss, epoch)

        #SAVE MODEL
    try:
        torch.save(modelAE.state_dict(), "modelsParam/{}/ep{}AE1D.pth".format(modelType ,epoch+1))
    except FileNotFoundError:
        os.mkdir("modelsParam/{}".format(modelType))
        torch.save(modelAE.state_dict(), "modelsParam/{}/ep{}AE1D.pth".format(modelType, epoch+1))
    #save les paths utilisés pour le train et le test
    np.save("TrainValTest/trainSet.npy",train_wav)
    np.save("TrainValTest/testSet.npy",test_wav)
    np.save("log/{}outputs.npy".format(modelType),outputs)
    
    stop = time.time()
    print("time elapsed: ", stop-start)
    print("goodbye world")
    
  

