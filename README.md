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


## Fixed Latent Samples

Below is the evolution of the generator output using a fixed latent vector:

| Epoch 1 | Epoch 5 | Epoch 10 |
|--------|----------|----------|
| ![](output/samples/epoch_001.png) | ![](output/samples/epoch_005.png) | ![](output/samples/epoch_010.png) |

## Training Curves

### GAN Losses
![Training losses](training_losses.png)

### Discriminator confidence
![Discriminator confidence](discriminator_confidence.png)
