import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def batch_norm_init(m):
    m.weight.data.fill_(1)
    if m.bias.data is not None:
        m.bias.data.zero_()

def kaiming_uniform_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        batch_norm_init(m)




class VAECharly(nn.Module):

    def __init__(self, encoder, bottleneck,
        reconstructor, beta, metric_name='mse', init=None):

        super(VAECharly, self).__init__()

        self.encoder = encoder
        self.bottleneck = bottleneck
        self.reconstructor = reconstructor
        self.metric_name = metric_name
        self.beta = beta
        self.apply(kaiming_uniform_init)
      

    def forward(self, x):
        distributions = self.encoder(x)
        bottleneck_results = self.bottleneck(distributions)
        logits = self.reconstructor(bottleneck_results['z'])
        bottleneck_results.update({'logits': logits})
        return bottleneck_results

    '''
    La loss prend en entrée (d'une part), le résultat du modèle pour la reconstruction pour mu et pour sigma,
    (d'autre part) les targets.
    '''
    def loss(self, forward_data, targets):
        logits = forward_data['logits']
        bottleneck_data = self.bottleneck.kl_divergence(forward_data['mu'], forward_data['logvar'])

        batch_size = logits.shape[0]
        recon_loss = F.mse_loss(logits, targets, reduction='sum').div(batch_size)
        loss = recon_loss + self.beta * bottleneck_data['total_kld']

        results = {'loss': loss, 'mse': recon_loss, 'total_kld': bottleneck_data['total_kld'],
            'mean_kld': bottleneck_data['mean_kld']}

        return results
    
    '''
    La minibatch loss pour utiliser le scaler et donc la batchnorm ? 
    '''
    def minibatch_loss(self, batch, device, scaler=None):
        targets = Variable(batch['sound'].unsqueeze(1).to(device)).float()

        if scaler:
            scaled_targets = scaler.transform(targets)
            forward_data = self(scaled_targets)
        else:
            forward_data = self(targets)
        details = self.loss(forward_data, scaled_targets if scaler else targets)

        if 'loss' in details:
            loss = details['loss']
        elif self.metric_name in details:
            loss = details[self.metric_name]
        else:
            loss = details[list(details.keys())[0]]

        details = {key:value if type(value) is dict else value.item() for key, value in details.items()}
        return loss, details


