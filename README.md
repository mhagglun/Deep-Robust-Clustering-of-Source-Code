# Deep-Robust-Clustering-of-Source-Code
Implementation of the Deep Robust Clustering technique as presented in the paper ![Deep Robust Clustering by Contrastive Learning](https://arxiv.org/abs/2008.03030), but adapted for clustering source code.

## Prerequisites

- python3.5+
- pytorch

```
pip install -r requirements.txt
```

## Usage

Extract and augment methods from a directory of source code.
```
python extract_and_augment_methods.py -d <path to directory>
```

Run preprocessing to build vocabularies and datasets
```
./preprocess.sh
```

Edit config.py and then train the model

```
python train.py
```

Extract features from test set and visualize clusters
```
python feature_extraction.py
```