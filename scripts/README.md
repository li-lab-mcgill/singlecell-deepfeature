## File name conventions:
File names show both the dataset and the model for latent space inference: `Dataset_model.ipynb`


# Datasets:
## scVI built-in datasets:
scVI python package includes seven built-in datasets to make it easier to reproduce results from the scVI paper. here are the two we have used:
* **RETINA:** 27,499 mouse retinal bipolar neurons, profiled in two batches using the Drop-Seq technology
* **PBMC:** 12,039 human peripheral blood mononuclear cells profiled with 10x

# Models
## scVI
scVI model uses a VAE to embedd single-cell samples and batch-effect correction
There is a good [documentation](https://scvi.readthedocs.io/en/stable/index.html#) and [package](https://github.com/YosefLab/scVI) for python.

## DPFE (Deep Private Feature Extraction)
In our proposed method we use the [Deep Private-Feature Extraction](https://arxiv.org/abs/1802.03151) framework for batch-effect correction in single-cell RNA-seq data.
In our method we treat the batch-labels as sensitive private features and minimize it's **Mutual Information** with the latent variables witch contain the biological information of single-cell data.
