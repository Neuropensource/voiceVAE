# -*- coding: utf-8 -*- 

#---general imports---
import os 
import numpy as np
from tqdm import tqdm
import argparse
import time
import yaml

#---deep learning imports---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

#---local imports---
from downloader import SpectroDataset4D, make_path
from models.modules.bottleneck import vanillaBottleneck
from models.modules.CNNdecoder import vanillaDecoder
from models.modules.CNNencoder import vanillaEncoder
from models.vanillaVAE import vanillaVAE



#PARSER  --> les arguments qui viennent de la ligne de commande
parser = argparse.ArgumentParser(description='general parser')
parser.add_argument("--device", type=str, default="local", help="device to use ('local' or 'cluster')")
parser.add_argument("--partial", type=bool, default=False, help="use only a part of the dataset")  #required=False
#recuperation des arguments
args = parser.parse_args() # à retravailler pour que le partial soit correct

#on local machine or on clusterSMS
if args.device == 'local':
    datapath = "../Data/charly/spectrograms_win800_fra200_rFalse_nfft800_nmelsNone_amptodbTrue/"
elif args.device == 'cluster':
    datapath = "/data1/data/expes/hippolyte.dreyfus/charly/spectrograms_win800_fra200_rFalse_nfft800_nmelsNone_amptodbTrue/"
else:
    raise AssertionError # à revoir comment on utilise ça (les assert et raise)


#CONFIG  --> les arguments qui viennent de config files
config = yaml.load(open("config/expe.yml", "r"), Loader=yaml.FullLoader)
print(config)

#All informations from config files are stored in the following variables 
# --> il ne doit plus avoir de constante ici mais que des variables venant d'un yml
# TRAINING_CONFIG = {'prop_train' : 0.8 ,'optimizer' : config['Training']['optimizer'] , 
#                     'loss' :config['Training']['loss'], 'batch_size' : config['batch_size'], 
#                     'epochs' :config['epochs'],} #  etc.
# MODEL_CONFIG = {'Z_DIM' : config['Z_DIM'], 'BETA' : config['BETA'], }
TRAINING_CONFIG = config['Training']
MODEL_CONFIG = config['Model']
DATA_CONFIG = {"DATAPATH" : datapath}




if __name__ == "__main__":
    start = time.time()
    #RECORD FROM THE CLUSTER
    print("cluster hello-world") 
    #tensorboard, start session, a configurer
    #autres methodes ?

    #DATA INITIALIZATION
    np.random.seed(1234)
    #torch.manual_seed(1234)
    folder_path = DATA_CONFIG["DATAPATH"]
    all_wav = make_path(folder_path)
    if args.partial:
        all_wav = all_wav[:1000]
    np.random.shuffle(all_wav)
    spectros = SpectroDataset4D(all_wav)
    #comprendre pourquoi avec le prop en paramètre ça ne marche pas
    # prop = TRAINING_CONFIG['prop_train']
    # train_spectros, eval_spectros= random_split(spectros, [int(len(spectros)*prop) , int(len(spectros)*float(1-prop))])
    train_spectros, eval_spectros= random_split(spectros, [int(len(spectros)*0.8) , int(len(spectros)*0.2)])

    data_loader = DataLoader(train_spectros,batch_size=32,shuffle=True)
    eval_loader = DataLoader(eval_spectros,batch_size=32,shuffle=True)


    #MODEL INITIALIZATION
    '''IMPORTANT
    soit tous les modèles doivent prendre les mêmes arguments, 
    soit il doit y avoir des if pour les modèles qui prennent des params en plus
    soit (BEST SOLUTION) il faut utiliser args and kwargs
    '''
    encoder = vanillaEncoder(z_dim= MODEL_CONFIG['Z_DIM'])
    bottleneck = vanillaBottleneck(z_dim= MODEL_CONFIG['Z_DIM'])
    decoder = vanillaDecoder(z_dim= MODEL_CONFIG['Z_DIM'])
    model = vanillaVAE(encoder, bottleneck, decoder, beta= MODEL_CONFIG['beta'])
   
    

    # TRAINING INITIALIZATION
    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('used device =', device)
    print('nb GPU available : ',torch.cuda.device_count())
    if torch.cuda.device_count() > 0:
        print('nom du GPU utilisé :' , torch.cuda.get_device_name(0))
    
    #model
    model.to(device)
    num_epochs = 5
    optimizer = torch.optim.Adam(model.parameters(), betas=[0.5, 0.999], lr=0.00005)
    if False: #si le model n'a pas de loss définie
        criterion = nn.MSELoss()
    
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
        for batch in tqdm(data_loader):
            batch = batch.to(device)
            results = model(batch)
            recon = results['logits']
            loss, details = model.minibatch_loss(batch=batch, device= device)
            # loss = criterion(recon, batch) --> pour un AE dont la loss peut ne pas être définie dans le modèle
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #monitoring   
            cumloss += loss.item()  
        print(f'Epoch:{epoch+1}, Loss:{cumloss/len(data_loader)}') 
        writer.add_scalar("loss/train loss",  cumloss/len(data_loader),epoch)
        outputs.append((epoch, batch, recon))
        
        ### EVAL ###
        #init
        model.eval()
        cumloss = 0
        #eval
        with torch.no_grad():
            for batch in eval_loader:
                batch = batch.to(device)
                loss, _ = model.minibatch_loss(recon,device)
        #monitoring
                cumloss += loss.item()
        print(f'[epoch={epoch+1}] val loss: {cumloss/len(eval_loader)}')    
        writer.add_scalar("loss/val loss", cumloss/len(eval_loader), epoch)

    #SAVING MODEL
    if args.partial:
        pass
    else:
        torch.save(model.state_dict(), "modelsParam/ep{}vanillaVAE.pth".format(epoch+1))
    
    stop = time.time()
    print("time elapsed: ", stop-start)
    print("goodbye world")
    
  

