# Datasets:
## Turecki Dataset:
The data contains more than 80k single cells from 34 patients.

The raw files for all patients are available in
`/home/mcb/li_lab/jszymb/data/turecki_suicide/data/`

The raw files are aggregated (except for 2 paitents) and stored as a single ready-to-use `.h5ad` format file available in `/home/mcb/users/mbahra5/project/data/turecki_types_all.h5ad` which can easily be read using the scanpy library like this:

```python
import scanpy as sc
adata = sc.read_h5ad(path_of_h5ad_file)
```
The script for loading the raw data and aggregation can be found in `project/joseph/Turecki_data_aggregation.ipynb`
## 
