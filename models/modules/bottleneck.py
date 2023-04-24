import torch.nn as nn
import torch
from torch.autograd import Variable


class VariationalBottleneck(nn.Module):

    def __init__(self, z_dim):
        super(VariationalBottleneck, self).__init__()
        self._z_dim = z_dim

    @property    
    def z_dim(self):
        return self._z_dim

    def forward(self, distributions):
        mu = distributions[:, :self._z_dim]
        logvar = distributions[:, self._z_dim:]
        z = self.reparametrize(mu, logvar)

        return {'z': z, 'latent_variables': z, 'mu': mu, 'logvar': logvar}

    def reparametrize(self, mu, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.
        :param mu: torch.Tensor
                Mean of the normal distribution. Shape (batch_size, latent_dim)
        :param logvar: torch.Tensor
                Diagonal log variance of the normal distribution. Shape (batch_size,
                latent_dim)
        :return:
        """
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

    def kl_divergence(self, mu, logvar):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return {'total_kld': total_kld, 'dimension_wise_kld': dimension_wise_kld, 'mean_kld': mean_kld}

    def sample(self, num_samples, original_shape, reconstructor, device):
        x, y = original_shape if len(original_shape) == 2 else original_shape[1:]
        z = torch.randn(num_samples, x, y).to(device)
        return reconstructor(z)
