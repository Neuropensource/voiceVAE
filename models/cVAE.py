import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable





class conditionalVAE(nn.Module):

    def __init__(self, encoder, bottleneck, reconstructor, beta):
        super(conditionalVAE, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.reconstructor = reconstructor
        self.beta = beta

    
    #remplacer le nom de variable bottleneck_results par all_results (le forward retourne logits mais aussi z (mu et logvar))
    def forward(self, x, y):
        
        distributions = self.encoder(x)
        all_results = self.bottleneck(distributions)
        logits = self.reconstructor(all_results['z'], y) # y on rajoute en input du reconstructor
        all_results.update({'logits': logits})
        return all_results

    
    def loss(self, forward_data, targets):
        logits = forward_data['logits']
        bottleneck_data = self.bottleneck.kl_divergence(forward_data['mu'], forward_data['logvar'])

        batch_size = logits.shape[0]
        recon_loss = F.mse_loss(logits, targets, reduction='sum').div(batch_size)
        loss = recon_loss + self.beta * bottleneck_data['total_kld']

        results = {'loss': loss, 'mse': recon_loss, 'total_kld': bottleneck_data['total_kld'],
            'mean_kld': bottleneck_data['mean_kld']}

        return results
    
    #normalement chez charly c'est un tenseur deja 
    def minibatch_loss(self, batch, device,y): # KWARGS A UTILISER POUR POUVOIR CREER UN MODELE GENERIQUE
        
        targets = Variable(batch.to(device)).float()
        forward_data = self(targets,y)
        details = self.loss(forward_data,targets)

        if 'loss' in details:
            loss = details['loss']
        elif self.metric_name in details:
            loss = details[self.metric_name]
        else:
            loss = details[list(details.keys())[0]]

        details = {key:value if type(value) is dict else value.item() for key, value in details.items()}
        return loss, details
    
   