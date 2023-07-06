# scGAL

scGAL is a single-cell multi-omics data clustering method based on generative adversarial network and autoencoder. scGAL is able to correlate gene copy number and gene expression data of the same tumor cell line across modalities to learn more information about ITH and thus achieve accurate clustering of tumor cells.



## Requirements

* Python 3.8+.

# Installation
## Clone repository
First, download scGAL from github and change to the directory:
```bash
git clone https://github.com/zhyu-lab/scgal
cd scgal
```

## Create conda environment (optional)
Create a new environment named "scgal":
```bash
conda create --name scgal python=3.8.13
```

Then activate it:
```bash
conda activate scgal
```

## Install requirements
Use pip to install the requirements:
```bash
python -m pip install -r requirements.txt
```

Now you are ready to run **scGAL**!

## Usage

scGAL uses single-cell multi-omics data to aggregate tumor cells into distinct subpopulations.

Example:

```
python run_scgal_sa501.py
```

## Input Files

The input data should contain DNA copy number data and gene expression data for the same cell line. Rows=cells, Cols=Genes. Authentic labels of DNA cells, if present, should also be entered into the method.

We use the true labels of cells to calculate ARI and V-measure to assess the quality of clustering. If the data set does not have the true labels of the cells, the label file may not be entered. The silhouette coefficients are used to assess the quality of clustering.

## Output Files

### Low-dimensional representations

The low-dimensional representations are written to a file with name "latent.txt".

### Cell labels

The cell-to-cluster assignments are written to a file with name "labels.txt".
## Arguments

* `--dna <filename>` Replace \<filename\> with the file containing the  copy number matrix.
* `--rna <filename>` Replace \<filename\> with the file containing the expression matrix.
* `--label <filename>` Replace \<filename\> with the file containing the real label of cells.


## Optional arguments

Parameter | Description | Possible values
---- | ----- | ------
--epochs | number of epoches to train the scGAL | Ex: epochs=200  default:180
--batch_size | batch size | Ex: batch_size=32  default:64
--lr | learning rate | Ex: lr=0.0001  default:0.0008
--Kmax | maximum number of clusters to consider | Ex: max_k=20  default:30
--latent_dim | latent dimension | Ex: latent_dim=4  default:3
--lambda_A | a weight factor to balance reconstruction loss and adversarial loss | Ex: lambda_A=10  default:5
--pool_size | the size of rna buffer that stores previously generated rnas | Ex: pool_size=32  default:64
--seed | random seed | Ex: seed=0  default:6


## Contact

If you have any questions, please contact lrx102@stu.nxu.edu.cn.