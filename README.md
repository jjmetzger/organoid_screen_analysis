# Deep learning analysis of micropattern-based organoid screens

This repository contains the necessary files to run
classification and unsupervised dimensionality reduction
for biological imaging data, in particular organoid data. A convolutional autoencoder
is used for unsupervised learning, its latent space can be interpreted as a phenotypic
space.

A minimal dataset for testing both classification and latent space extraction is provided
in the `data` directory, weights for a pretrained autoencoder for this dataset are located in the
`ae_results/ae_weights.pth`.

The code is written in Python (3.7 or higher required) and is based on the PyTorch and Fastai libraries (tested with PyTorch 1.0.0,
torchvision 0.2.2, Fastai 1.0.60) and also requires pandas (>=1.0.5).


### Classification

An example classification is demonstrated in `Classification_resnet.ipynb`.

### Phenotypic space extraction

The autoencoder that encodes the phenotypic space can be trained using the following syntax

```
python autoencoder.py data/ --num_epochs 100 --output_folder ae_results
```
For full help on how to use the autoencoder, run 
```
python autoencoder.py --help
```

An example of how to extract the latent space from the autoencoder is given in
`Extract_latent_vectors.ipynb`.

