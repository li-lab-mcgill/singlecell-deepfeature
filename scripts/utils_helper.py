import numpy as np
import pandas as pd
import scipy
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score as NMI

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

#         asw_score = silhouette_score(latent, labels)
#         nmi_score = NMI(labels, labels_pred)
        ari_score = ARI(labels, labels_pred)
#         uca_score = unsupervised_clustering_accuracy(labels, labels_pred)[0]
#         logger.debug(
#             "Clustering Scores:\nSilhouette: %.4f\nNMI: %.4f\nARI: %.4f\nUCA: %.4f"
#             % (asw_score, nmi_score, ari_score, uca_score)
#         )
#         return asw_score, nmi_score, ari_score, uca_score
        return ari_score

    
    
    
    
    
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl

from scvi.models.log_likelihood import log_zinb_positive, log_nb_positive
from scvi.models.utils import one_hot

torch.backends.cudnn.benchmark = True


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
    ):
        super().__init__()
        self.n_latent = n_latent
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels

        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )


        self.decoder = Decoder(
            n_latent,
            n_input,
            n_cat_list=[n_batch],
            n_layers=n_layers,
            n_hidden=n_hidden,
        )
        

    def get_latents(self, x, y=None):
        return [self.sample_from_posterior_z(x, y)]

    def sample_from_posterior_z(self, x, y=None, give_mean=False):
        
        x = torch.log(1 + x)
        qz_m, qz_v, z = self.z_encoder(x, y) 
        if give_mean:
            z = qz_m
        return z


    def get_reconstruction_loss(self, x, px_rate, px_r):
        reconst_loss = -log_nb_positive(x, px_rate, px_r)
        return reconst_loss

    def inference(self, x, batch_index=None, y=None):
        x_ = x
        
        x_ = torch.log(1 + x_)

        qz_m, qz_v, z = self.z_encoder(x_, y)
        px_r, px_rate = self.decoder(z, batch_index, y)
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

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )
        kl_divergence = kl_divergence_z

        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r)

        return reconst_loss , kl_divergence
#         return reconst_loss , 0.0, z  # kl_divergence / x.size(1)




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
        one_hot_cat_list = [] 
        
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat > 1:  
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat 
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
        
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent



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

        self.px_rate_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.ReLU()# nn.Softmax(dim=-1)
        )

        self.px_r_decoder = nn.Linear(n_hidden, n_output)


    def forward(self, z: torch.Tensor, *cat_list: int):
        px = self.px_decoder(z, *cat_list)
        px_rate = self.px_rate_decoder(px)
        px_r = self.px_r_decoder(px)
        return px_r, px_rate




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
        
        modules.append(nn.Linear(last_layer_size, 1)) # changed
#         modules.append(nn.LogSoftmax(dim=1))
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

class GANTrainer(UnsupervisedTrainer):
    def __init__(
        self,
        model,
        disc,
        gene_dataset,
        train_size, test_size,
        **kwargs
    ):
        self.disc = disc
                 
        super().__init__(model, gene_dataset, train_size=train_size, test_size=test_size, **kwargs)
        if type(self) is GANTrainer:
            self.train_set, self.test_set, self.validation_set = self.train_test_validation(
                model, gene_dataset, train_size, test_size
            )
        
    def train(self, n_epochs=20, lr=1e-3, eps=0.01, params=None, enc_lr=1e-3, disc_lr=1e-3):
        begin = time.time()
        self.model.train()
        self.disc.train()

        if params is None:
            params = filter(lambda p: p.requires_grad, self.model.parameters())

        optimizer = self.optimizer = torch.optim.Adam(params, lr=lr, eps=eps, weight_decay=self.weight_decay)
        optimizerE = self.optimizerE = torch.optim.Adam(self.model.z_encoder.parameters(), lr = enc_lr, weight_decay=self.weight_decay)
        optimizerD = self.optimizerD = torch.optim.Adam(self.disc.parameters(), lr = disc_lr, weight_decay=self.weight_decay)
        
        self.n_epochs = n_epochs
        nll_loss = nn.NLLLoss(reduction='none') 
        kl_loss = nn.KLDivLoss()
        mse_loss = nn.MSELoss()

        with trange(n_epochs, desc="training", file=sys.stdout, disable=not self.show_progbar) as pbar:
            vae_loss_list, E_loss_list, D_loss_list = [], [], []
            for self.epoch in pbar:
                vae_loss_list_epoch, E_loss_list_epoch, D_loss_list_epoch = [], [], []
                
                pbar.update(1)
                self.on_epoch_begin()

    
                for tensors_list in self.data_loaders_loop():
                    if tensors_list[0][0].shape[0] < 3:
                        continue
                    
                    sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors_list[0]  
                    ############################
                    # (1) Update VAE network
                    ###########################                    
                    self.model.zero_grad()
                        
                    reconst_loss, kl_divergence, z = self.model(sample_batch, local_l_mean, local_l_var, batch_index)
                    loss = torch.mean(reconst_loss + self.kl_weight * kl_divergence)
                    
                    vae_loss_list_epoch.append(loss.item())
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    ############################
                    # (1) Update D Net
                    ###########################     
                    for disc_iter in range(10):
                        self.disc.zero_grad()

                        batch_pred = self.disc(z)
#                         D_loss = nll_loss(batch_pred, batch_index.view(-1)) 
#                         D_loss = torch.mean(D_loss) # todo
                        D_loss = mse_loss(batch_pred, batch_index.view(-1))
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
        return vae_loss_list, D_loss_list, E_loss_list

    
import logging
import operator
import os
from functools import reduce

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from scvi.dataset.dataset import DownloadableDataset, GeneExpressionDataset

logger = logging.getLogger(__name__)


class AnnDatasetFromAnnData(GeneExpressionDataset):
    """Forms a ``GeneExpressionDataset`` from a ``anndata.AnnData`` object.

    :param ad: ``anndata.AnnData`` instance.
    """

    def __init__(self, ad: anndata.AnnData):
        super().__init__()
        (
            X,
            batch_indices,
            labels,
            gene_names,
            cell_types,
            self.obs,
            self.obsm,
            self.var,
            self.varm,
            self.uns,
        ) = extract_data_from_anndata(ad)
        
        
        self.populate_from_data(
            X=X,
            batch_indices=batch_indices,
            gene_names=gene_names,
            cell_types=cell_types,
            labels = labels
        )
        
        self.filter_cells_by_count()
        

def extract_data_from_anndata(ad: anndata.AnnData):
    data, labels, batch_indices, gene_names, cell_types = None, None, None, None, None

    # treat all possible cases according to anndata doc
    if isinstance(ad.X, np.ndarray):
        data = ad.X.copy()
    if isinstance(ad.X, pd.DataFrame):
        data = ad.X.values
    if isinstance(ad.X, csr_matrix):
        # keep sparsity above 1 Gb in dense form
        if reduce(operator.mul, ad.X.shape) * ad.X.dtype.itemsize < 1e9:
            logger.info("Dense size under 1Gb, casting to dense format (np.ndarray).")
            data = ad.X.toarray()
        else:
            data = ad.X.copy()

    gene_names = np.asarray(ad.var.index.values, dtype=str)

    if "batch_indices" in ad.obs.columns:
        batch_indices = ad.obs["batch_indices"].values

    if "cell_types" in ad.obs.columns:
        cell_types = ad.obs["cell_types"]
        labels = cell_types.rank(method="dense").values.astype("int")
        cell_types = cell_types.drop_duplicates().values.astype("str")

    if "labels" in ad.obs.columns:
        labels = ad.obs["labels"]

    return (
        data,
        batch_indices,
        labels,
        gene_names,
        cell_types,
        ad.obs,
        ad.obsm,
        ad.var,
        ad.varm,
        ad.uns,
    )
