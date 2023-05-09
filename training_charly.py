import torch
import os 
import numpy as np
from downloader import SpectroDataset, make_path
from torch.utils.data import DataLoader
import tqdm

from models.VAEcharly import VAECharly
from models.modules.bottleneck import VariationalBottleneck
from models.modules.CNNdecoder import ConvolutionalDecoder
from models.modules.CNNencoder import ConvolutionalEncoder


if __name__ == "__main__":

    np.random.seed(1234)
    folder_path = "../Data/charly/spectrograms_win800_fra200_rFalse_nfft800_nmelsNone_amptodbTrue/"

    all_wav = make_path(folder_path)

    np.random.shuffle(all_wav)

    #si l'on veut lancer un petit entrainenement

    spectros = SpectroDataset(all_wav) 

    data_loader = DataLoader(spectros,batch_size=16,shuffle=True)

    #initialisation du modèle
    encoder = ConvolutionalEncoder(z_dim=16)
    bottleneck = VariationalBottleneck(z_dim=16)
    decoder = ConvolutionalDecoder(z_dim=16)
    model = VAECharly(encoder, bottleneck, decoder, beta=0, metric_name='mse', init=None)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Device
    # if torch.cuda.is_available():
    #     torch.backends.cudnn.deterministic = True
    # model, stream, scaler = load_model_and_data_from_config(device)

    nepoch = 1000
    optimizer = torch.optim.Adam(model.parameters(), betas=[0.5, 0.999], lr=0.00005)

    # Uncomment if needed
    #optimizer = torch.optim.Adam(model.parameters(), betas=[0.5, 0.999], lr=0.005)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : 0.95 ** epoch)
    #cmap = gen_spectrogram_cmap()
    #early_stopping_counts = 0
    #early_stopping_threshold = 5   
    previous_val_loss = 1000000

    # Training loop
    for epoch in range(nepoch):
        model.train()
        train_losses = []
        for batch in tqdm(data_loader, desc=str(epoch), leave=False):
            #batch_labels = stream.dataset.retreive_labels_from_sound_ids(batch['sound_id']) # To get the labels of the current batch sounds in a pandas dataframe
            #gender_labels = stream.dataset.retreive_labels_data_from_sound_ids('gender', batch['sound_id'], device) # To get the speaker gender labels of the current batch sounds in a cuda tensor
            optimizer.zero_grad()
            loss, details = model.minibatch_loss(batch, device, scaler=scaler)
            loss.backward()
            optimizer.step()
            train_losses.append(loss)
        train_loss = torch.mean(torch.stack(train_losses))
        print(f'[epoch={epoch+1}] train loss: {train_loss.item()}')
       

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=str(epoch), leave=False):
                loss, _ = model.minibatch_loss(batch, device, scaler)
                val_losses.append(loss)
        val_loss = torch.mean(torch.stack(val_losses))
        print(f'[epoch={epoch+1}] val loss: {val_loss.item()}')
    
        #early_stopping_counts, previous_val_loss = check_early_stopping(val_loss, previous_val_loss, early_stopping_counts, early_stopping_threshold, epoch, model)
        #plot_val_spectrograms(model, stream, scaler, device, cmap, epoch)

        # Uncomment if needed
        #scheduler.step()





