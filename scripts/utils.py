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



from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import to_categorical

def classification_acc_measure(latent, labels):
    
    cell_types = pd.Series(labels).unique()
    latent_dim = latent.shape[1]
    
    #Model
    inputs = Input(shape=(latent_dim,))
    # x = Dense(10, activation='relu')(inputs)
    x = Dense(len(cell_types), activation='softmax')(inputs)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    labels = to_categorical(labels,num_classes=len(cell_types))
    history = model.fit(latent, labels, epochs=20, validation_split=0.1)
    
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']

    return val_acc[-1]
    