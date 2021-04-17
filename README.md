# Compressive Neural Representations of Volumetric Scalar Fields

![alt text](https://github.com/matthewberger/neurcomp/blob/c42d2ec69736fb80dcaeb70a7d15a6dc198ae390/images/teaser_small.jpg "Overview")

[Yuzhe Lu](), [Kairong Jiang](), [Joshua A. Levine](https://jalevine.bitbucket.io/), [Matthew Berger](https://matthewberger.github.io/)

[arxiv link](https://arxiv.org/pdf/2104.04523.pdf)

This project contains code, based on the above paper, that addresses the compression of volumetric scalar fields using coordinate-based multilayer perceptrons.

## Installation

We have included a slim `requirements.txt` file for relevant libraries.

## Training a model

To train a model, please see the `train.py` script. The only required option is the volume itself, which we expect to be a 3D numpy array. We have included a small volume for testing purposes, located at `volumes/test_vol.npy`:

```
python train.py --volume volumes/test_vol.npy
```

By default, CUDA is disabled, but can be enabled via the flag `--cuda`. There are a number of other options for training, the two most relevant are: (1) `--compression_ratio`, the ratio of volume resolution to network size, thus defining the number of network parameters, and (2) `--grad_lambda`, for specifying a value for gradient regularization.

Once training has complete, two files are written: the network weights, and the network configuration. By default, these are written to files `thenet.pth` and `thenet.json`, but can be changed via specifying options for `--network` and `--config`.

Furthermore, if you would like to obtain the volume represented by the network once training has completed, then enable the flag `--enable-vol-debug`. This will write out, both, the input volume, and network-predicted volume, as vti files that you can load in external visualization tools, e.g. ParaView.

## Compressing the network

To quantize the weights of the network, please see the `net_compress.py` script. To run:

```
python net_compress.py --net thenet.pth --config thenet.json --compressed compressed_out
```

The `--compressed` option specifies the binary file to write to for the compressed representation.


## Decompressing the network

To decompress the network, please see the `net_decompress.py` script. To run:

```
python net_decompress.py --volume volumes/test_vol.npy --compressed compressed_out
```

The above assumes that we have access to the original volume for comparison purposes, though this is simple enough to modify.
