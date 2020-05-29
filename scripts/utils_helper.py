import numpy as np
import pandas as pd
import scipy
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score as NMI
from torch.utils.data import DataLoader, Dataset

class GeneDataset(Dataset):    
    def __init__(self, X, labels, batch, batch_cont):
        self.X = X
        self.label = labels
        self.batch = batch
        self.batch_cont = batch_cont
        self.nb_genes = X.shape[1]
        self.n_batches = len(batch.unique())
        self.n_labels = len(labels.unique())
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        if type(self.X) is np.ndarray:
            X = self.X[index].squeeze()    
        else:
            X = self.X[index].toarray().squeeze()
        label = self.label.iloc[index]
        batch = self.batch.iloc[index]
        batch_cont = self.batch_cont.iloc[index]

        return X, label, batch, batch_cont
    
def entropy_batch_mixing(latent_space, batches, n_neighbors=50, n_pools=50, n_samples_per_pool=100):
    
    def entropy(hist_data):
        n_batches = len(np.unique(hist_data))
        counts = pd.Series(hist_data[:,0]).value_counts()
        freqs = counts/counts.sum()
        return sum([-f*np.log(f) if f!=0 else 0 for f in freqs])

    n_neighbors = min(n_neighbors, len(latent_space) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(latent_space)
    kmatrix = nne.kneighbors_graph(latent_space) - scipy.sparse.identity(latent_space.shape[0])

    score = 0
    for t in range(n_pools):
        indices = np.random.choice(np.arange(latent_space.shape[0]), size=n_samples_per_pool)
        score += np.mean(
            [
                entropy(
                    batches[
                        kmatrix[indices[i]].nonzero()[1]
                    ]
                )
                for i in range(n_samples_per_pool)
            ]
        )
    return score / float(n_pools)

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import adjusted_rand_score as ARI

def clustering_scores(n_labels, labels, latent, prediction_algorithm="knn"):
    if n_labels > 1:
        if prediction_algorithm == "knn":
            labels_pred = KMeans(n_labels, n_init=200).fit_predict(latent)  # n_jobs>1 ?
        elif prediction_algorithm == "gmm":
            gmm = GMM(n_labels)
            gmm.fit(latent)
            labels_pred = gmm.predict(latent)

        ari_score = ARI(labels, labels_pred)
        return ari_score



    
# -*- coding: utf-8 -*-
"""Main module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl

from scvi.models.log_likelihood import log_zinb_positive, log_nb_positive
from scvi.models.utils import one_hot

torch.backends.cudnn.benchmark = True



import collections
from typing import Iterable, List

import torch
from torch import nn as nn
from torch.distributions import Normal
from torch.nn import ModuleList

from scvi.models.utils import one_hot


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


class FCLayers(nn.Module):

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in + sum(self.n_cat_list), n_out),
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.ReLU(),
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def forward(self, x: torch.Tensor, *cat_list: int, instance_id: int = 0):
        one_hot_cat_list = []  # for generality in this list many indices useless.
        assert len(self.n_cat_list) <= len(cat_list), "nb. categorical args provided doesn't match init. params."
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            assert not (n_cat and cat is None), "cat not provided while n_cat != 0 in init. params."
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear):
                            if x.dim() == 3:
                                one_hot_cat_list = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            x = torch.cat((x, *one_hot_cat_list), dim=-1)
                        x = layer(x)
        return x


# Encoder
class Encoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


# Decoder
class Decoder(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), 
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, dispersion: str, z: torch.Tensor, *cat_list: int):
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)        
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_r, px_rate


    
# VAE model
class VAE(nn.Module):

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        reconstruction_loss: str = "nb",
        kl_weight = 0.1,
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.kl_weight = kl_weight

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        else:  # gene-cell
            pass

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = Decoder(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            n_layers=n_layers,
            n_hidden=n_hidden,
        )
        

    def sample_from_posterior_z(self, x, y=None, give_mean=False):
        if self.log_variational:
            x = torch.log(1 + x)
        qz_m, qz_v, z = self.z_encoder(x, y)  # y only used in VAEC
        if give_mean:
            z = qz_m
        return z


    def get_reconstruction_loss(self, x, px_rate, px_r):
        # Reconstruction Loss
        if self.reconstruction_loss == "nb":
            reconst_loss = -log_nb_positive(x, px_rate, px_r)
        elif self.reconstruction_loss == "mse":
            x_ = x
            if self.log_variational:
                x_ = torch.log(1 + x_)

            reconst_loss = F.mse_loss(px_rate, x_)
        return reconst_loss

    def inference(self, x, batch_index=None, y=None, n_samples=1):
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        qz_m, qz_v, z = self.z_encoder(x_, y)
        px_r, px_rate = self.decoder(self.dispersion, z, batch_index, y)
        
        if self.dispersion == "gene-label":
            px_r = F.linear(one_hot(y, self.n_labels), self.px_r)  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_rate = torch.exp(px_rate)
        px_r = torch.exp(px_r)

        return dict(            
            px_r=px_r,
            px_rate=px_rate,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
        )

    def forward(self, x, batch_index=None, y=None):

        # Parameters for z latent distribution
        outputs = self.inference(x, batch_index, y)
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]
        z = outputs['z']
        
        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)
        kl_divergence = kl_divergence_z

        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r)

        return reconst_loss , kl_divergence, z  

from torch import nn as nn
class Discriminator(nn.Module):
    def __init__(self, in_features, layers, n_class):
        super(Discriminator, self).__init__()
        
        modules = []
        last_layer_size = in_features
        for l in layers:
            modules.append(nn.Linear(last_layer_size, l))
            modules.append(nn.ReLU())
            last_layer_size = l
        
        modules.append(nn.Linear(last_layer_size, n_class))
        modules.append(nn.LogSoftmax(dim=1))
        self.main = nn.Sequential(*modules)
        
    def forward(self, input):
        return self.main(input)

    
class Regressor(nn.Module):
    def __init__(self, in_features, layers, n_class):
        super(Regressor, self).__init__()
        
        modules = []
        last_layer_size = in_features
        for l in layers:
            modules.append(nn.Linear(last_layer_size, l))
            modules.append(nn.ReLU())
            last_layer_size = l
        
        modules.append(nn.Linear(last_layer_size, 1)) 
        self.main = nn.Sequential(*modules)
        
    def forward(self, input):
        return self.main(input)
    
from scvi.inference import UnsupervisedTrainer
import time
import logging
import sys
import time
from tqdm import trange
from scvi.inference.posterior import Posterior
logger = logging.getLogger(__name__)


class GANTrainer():
    def __init__(self, batch_type, model, disc, dataset, device, batch_size=128):
        self.batch_type = batch_type # 'discrete' or 'continuous'
        self.model = model
        self.disc = disc
        self.dataset = dataset
        self.device = device
        self.dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    def get_latent(self, give_mean=True):
        latent = []
        batch = []
        labels = []
        for sample_batch, label, batch_index, batch_cont in self.dataset_loader:
            sample_batch = sample_batch.to(self.device)
            latent += [self.model.sample_from_posterior_z(sample_batch, give_mean=give_mean)]
            batch += [batch_index]
            labels += [label]
        latent = torch.cat(latent)
        batch = torch.cat(batch).reshape((-1,1))
        labels = torch.cat(labels)
        return latent, labels, batch
    
        
    def train(self, n_epochs=20, lr=1e-3, eps=0.01, params=None, enc_lr=1e-3, disc_lr=1e-3):
        begin = time.time()
        self.model.train()
        self.disc.train()

        if params is None:
            params = filter(lambda p: p.requires_grad, self.model.parameters())

        optimizer = self.optimizer = torch.optim.Adam(params, lr=lr, eps=eps)
        optimizerE = self.optimizerE = torch.optim.Adam(self.model.z_encoder.parameters(), lr = enc_lr)
        optimizerD = self.optimizerD = torch.optim.Adam(self.disc.parameters(), lr = disc_lr)
        
        
        self.n_epochs = n_epochs

        nll_loss = nn.NLLLoss(reduction='none')
        kl_loss = nn.KLDivLoss()
        mse_loss = nn.MSELoss()

        with trange(n_epochs, desc="training", file=sys.stdout) as pbar:
            vae_loss_list, E_loss_list, D_loss_list = [], [], []
            for self.epoch in pbar:
                vae_loss_list_epoch, E_loss_list_epoch, D_loss_list_epoch = [], [], []
                
                pbar.update(1)
    
                for sample_batch, _, batch_dis, batch_cont in self.dataset_loader:                    

                    batch_dis = batch_dis.reshape((-1,1))
                    sample_batch = sample_batch.to(self.device)
                    batch_dis = batch_dis.to(self.device)
                    batch_cont = batch_cont.to(self.device)
                    ############################
                    # (1) Update VAE network
                    ###########################                    
                    self.model.zero_grad()
                    reconst_loss, kl_divergence, z = self.model(sample_batch, batch_dis)
                    loss = torch.mean(reconst_loss + self.model.kl_weight * kl_divergence)
                    vae_loss_list_epoch.append(loss.item())
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    ############################
                    # (1) Update D Net
                    ###########################     
                    for disc_iter in range(10):
                        self.disc.zero_grad()
                        batch_pred = self.disc(z)
                        if self.batch_type is 'continuous':
                            D_loss = mse_loss(batch_pred.squeeze(), batch_cont)
                        elif self.batch_type is 'discrete':
                            D_loss = nll_loss(batch_pred, batch_dis.view(-1))
                            D_loss = torch.mean(D_loss) 
                            
                        D_loss_list_epoch.append(D_loss.item())
                        D_loss.backward(retain_graph=True)
                        optimizerD.step()
                    ############################
                    # (1) Update E Net
                    ########################### 
                    self.model.z_encoder.zero_grad()
                    E_loss = -1 * D_loss
                    E_loss_list_epoch.append(E_loss.item())
                    E_loss.backward(retain_graph=True)
                    optimizerE.step()
                    
                vae_loss_list.append(sum(vae_loss_list_epoch)/len(vae_loss_list_epoch))
                D_loss_list.append(sum(D_loss_list_epoch)/len(D_loss_list_epoch))
                E_loss_list.append(sum(E_loss_list_epoch)/len(E_loss_list_epoch))
        
        self.model.eval()
        self.disc.eval()
        return vae_loss_list, D_loss_list, E_loss_list
