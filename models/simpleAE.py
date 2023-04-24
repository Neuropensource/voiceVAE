import torch 
import torch.nn as nn

'''
même classe que le AE dans generativeNN,
 mais transformé pour avoir la bonne taille d'input
'''
class AE1D(nn.Module):
    def __init__(self,dimLS):
        super().__init__()        
        # N, 1, 28, 28 --> on veut que ce soit du 20,401
        self.encoder = nn.Sequential(
            nn.Conv1d(21, 21, 3, stride=2, padding=1), # -> N, 16, 14, 14
            nn.ReLU(),
            nn.Conv1d(21, 32, 3, stride=2, padding=1), # -> N, 32, 7, 7
            nn.ReLU(),
            nn.Conv1d(32, 64, 7), # -> N, 64, 1, 1
            #on réduit encore la taille du LS
            nn.ReLU(),
            nn.Conv1d(64, 32, 1), # -> N, 32, 1, 1
            nn.ReLU(),
            nn.Conv1d(32, dimLS, 1) # -> N, dimLS, 1, 1
   
        )
        
        # N , 2, 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(dimLS, 32, 1), # -> N, 32, 1, 1
            nn.ReLU(),
            nn.ConvTranspose1d(32, 64, 1), # -> N, 64, 1, 1
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 7), # -> N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose1d(32, 21, 3, stride=2, padding=1, output_padding=0), # N, 16, 14, 14 (N,16,13,13 without output_padding)
            nn.ReLU(),
            nn.ConvTranspose1d(21, 21, 3, stride=2, padding=1, output_padding=0), # N, 1, 28, 28  (N,1,27,27)
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self,x):
        return self.encoder(x)
    
    def generate(self,x):
        return self.decoder(x)
        