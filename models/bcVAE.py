# importations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt




""" 
on veut un autoencoder qui résume des sons (présentés sous forme de spectrogramme ?)
l'espace latent est contraint par IRM
on ne se contente pas de convolution puisque l'input est une TS ?
on peut utiliser des RNN pour encoder et decoder ?
""" 
class bcVAE(nn.Module):
    """
    Paramétrages du bcVAE : taille du LS, rapport reconstruction/kld,
    type d'encoder, type de bottleneck, type de decoder,
    ... quoi d'autres ???
    """
    def __init__(self,encoder, bottleneck, decoder):
        super().__init__()        
        # N, 1, 21, 401
        self.encoder = encoder 
        self.bottleneck = bottleneck
        self.decoder = decoder
        

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def generate(self, x):
        return self.decoder(x)
    

    def forward(self, x):
        distributions = self.encoder(x)
        bottleneck_results = self.bottleneck(distributions)
        logits = self.reconstructor(bottleneck_results['z'])
        bottleneck_results.update({'logits': logits})
        return bottleneck_results


