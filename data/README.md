# Single-cell RNA-Seq Datasets:

## MDD Brain Cells Dataset:
The data contains more than 80k single cells from 34 patients.

The raw files for all patients are available in
`/home/mcb/li_lab/jszymb/data/turecki_suicide/data/`

The raw files are aggregated (except for 2 paitents) and stored as a single ready-to-use `.h5ad` format file available in `/home/mcb/users/mbahra5/project/data/turecki_types_all.h5ad` which can easily be read using the [scanpy](https://anndata.readthedocs.io/en/stable/) library like this:

```python
import scanpy as sc
adata = sc.read_h5ad(path_of_h5ad_file)
```
The script for loading the raw data and aggregation can be found in `/home/mcb/users/mbahra5/project/joseph/Turecki_data_aggregation.ipynb`



## Mouse pancreas single-cell RNA-seq dataset (GSE84133):
The 1,886 cells in 13 cell types after the exclusion of hybrid cells. The dataset is available in [GSE84133](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133)

In python you can download the datasets from the GEO website using the [GEOparse](https://pypi.org/project/GEOparse/) package:

```python
import GEOparse
import pandas as pd
data_path = 'path where you want the GEOparse to download and save the data to'

gse = GEOparse.get_GEO(geo='GSE84133', destdir=data_path)
supp = gse.download_supplementary_files(directory=data_path, download_sra=False)

data = []
for k in tqdm(supp.keys()):
    for v in supp[k].values():
        if 'mouse' in v:
            data.append(pd.read_csv(v))
df = pd.concat(data)
```

Other Datasets from GEO can be downloaded similarly.


[A number of other datasets](https://nbviewer.jupyter.org/github/YosefLab/scVI/blob/master/tests/notebooks/data_loading.ipynb) are also available through the `scvi` library. 

## Human pancreatic islet cells through 4 technologies :
Human pancreatic islet cell datasets produced across four technologies, CelSeq (GSE81076) CelSeq2 (GSE85241), Fluidigm C1 (GSE86469), and SMART-Seq2 (E-MTAB-5061) are all aggregated and stored in `/home/mcb/users/mbahra5/project/data/panc8/panc4tech.h5ad` and can be read using scanpy library like before:
```python
import scanpy as sc
adata = sc.read_h5ad('/home/mcb/users/mbahra5/project/data/panc8/panc4tech.h5ad')
```

<!---
## Pre-Frontal Cortex Starmap Dataset :
3,722 mouse Cortex cells profiled using the STARmap technology in 3 batches. You can find more details about the data in the [original publication](https://www.ncbi.nlm.nih.gov/pubmed/29930089).
 
The dataset can be loaded easily using the scvi library like the [RETINA dataset](#RETINA).
```python
from scvi.dataset import PreFrontalCortexStarmapDataset
dataset = PreFrontalCortexStarmapDataset(save_path=save_path)
```
-->

<!---
## RETINA:
27,499 mouse retinal bipolar neurons, profiled in two batches using the Drop-Seq technology

The dataset can be downloaded and manipulated through the [scvi](https://github.com/YosefLab/scVI) python library:
```python
from scvi.dataset import RetinaDataset
save_path = "where you wan to save the data"
dataset = RetinaDataset(save_path=save_path)
```
-->





















