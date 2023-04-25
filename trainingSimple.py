# -*- coding: utf-8 -*- ???

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

#---local imports---
from downloader import SpectroDataset, make_path
from models.VAEcharly import VAECharly
from models.modules.bottleneck import VariationalBottleneck
from models.modules.CNNdecoder import ConvolutionalDecoder
from models.modules.CNNencoder import ConvolutionalEncoder
from models.simpleAE import AE1D

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
    print("cluster hello-world") 
    #tensorboard, start session, a configurer
    #autres methodes ?

    #DATA INITIALIZATION
    np.random.seed(1234)
    folder_path = DATA_CONFIG["DATAPATH"]
    all_wav = make_path(folder_path)
    if args.partial:
        all_wav = all_wav[:2000]
    np.random.shuffle(all_wav)
    spectros = SpectroDataset(all_wav) 
    data_loader = DataLoader(spectros,batch_size=32,shuffle=True)


    #MODEL INITIALIZATION
    # encoder = ConvolutionalEncoder(z_dim=16)
    # bottleneck = VariationalBottleneck(z_dim=16)
    # decoder = ConvolutionalDecoder(z_dim=16)
    # model = VAECharly(encoder, bottleneck, decoder, beta=0, metric_name='mse', init=None)
    modelAE = AE1D(32)
    

    
    # TRAINING INITIALIZATION
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modelAE.to(device)
    num_epochs = 5
    optimizer = torch.optim.Adam(modelAE.parameters(), betas=[0.5, 0.999], lr=0.00005)
    #optimizer = torch.optim.Adam(modelAE.parameters(),lr=1e-3, weight_decay=1e-5)
    outputs = []
    criterion = nn.MSELoss()
    
    
    
    #TRAINING LOOP
    for epoch in range(num_epochs):
        #train
        print("Starting epoch {}".format(epoch+1))
        modelAE.train()
        num_batch = 0
        for batch in data_loader:
            batch = batch.to(device)
            recon = modelAE(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_batch += 1
            # if num_batch % 100 == 0:
            #     print(num_batch, 'over', len(data_loader), 'batches done')
                

        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        outputs.append((epoch, batch, recon))
        print('torch is using', torch.cuda.device_count(), 'GPUs')
        

        #eval
        modelAE.eval()
        val_losses = []
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(device)
                loss = criterion(modelAE(batch), batch)
                val_losses.append(loss)
        val_loss = torch.mean(torch.stack(val_losses))
        print(f'[epoch={epoch+1}] val loss: {val_loss.item()}')

        #SAVE MODEL
        torch.save(modelAE.state_dict(), "modelsParam/ep{}modelAE.pth".format(epoch+1))
    
    stop = time.time()
    print("time elapsed: ", stop-start)
    print("goodbye world")
    
  

