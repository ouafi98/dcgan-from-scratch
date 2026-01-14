# DCGAN from Scratch

From-scratch PyTorch implementation of a Deep Convolutional GAN (DCGAN) on MNIST dataset,
with detailed logging, diagnostics, and reproducible training.

## Features
- "Correct" DCGAN architecture
- Stable training with BCEWithLogitsLoss
- Fixed-latent visualization
- Loss and discriminator confidence plots
- Reproducible experiments (seeded)

## Dataset
- MNIST (28×28)

## Generated Samples

Samples generated from a fixed latent vector during training:

![Epoch 1](output/samples/epoch_001.png)
![Epoch 5](output/samples/epoch_005.png)
![Epoch 10](output/samples/epoch_010.png)

## Training Curves

### GAN Losses
![Training losses](training_losses.png)

### Discriminator confidence
![Discriminator confidence](discriminator_confidence.png)

## Notes
GAN losses do not directly correlate with image quality.
Training progress is assessed primarily via fixed-latent image grids.