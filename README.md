# WaveNet-NameSynth

This project implements a WaveNet-inspired neural network designed for generating character-level names. The model is trained on a dataset of names and learns to produce realistic name-like outputs by predicting the next character in a sequence based on the context of preceding characters.

## Table of Contents
- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Sampling](#sampling)
- [Results](#results)

## Project Overview

The WaveNet-based name generator is built to showcase the power of deep learning in generating text sequences, particularly names. The model leverages character-level embeddings and deep neural network layers to capture the nuances of name structures and produce novel name-like sequences.

This project demonstrates advanced deep learning concepts, including:
- Embedding layers
- Sequential model stacking
- Batch normalization
- Custom linear layer
- Gradient-based optimization

## Model Architecture

The architecture of the model is composed of several key components:

1. **Embeddings Layer**: Maps input characters to dense vector representations.
2. **FlattenConsecutive**: Custom layer to flatten consecutive numbers for subsequent layers.
3. **Linear Layers**: Fully connected layers that learn complex relationships in the data.
4. **BatchNorm1d**: Batch normalization layers to stabilize and accelerate training.
5. **Tanh Activation**: Non-linear activation function to introduce non-linearity into the model.

The model consists of **500k** parameters and is structured as a sequence of linear transformations and non-linear activations, ensuring a rich and expressive representation of the input data.

## Installation

To get started with the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sudhan-Dahake/WaveNet-NameSynth.git
   cd WaveNet-NameSynth
   ```

2. **Install Dependencies**: Ensure you have Python 3.7 or higher installed. Then, install the required Python packages using pip.
   ```bash
   pip install torch
   ```

## Usage

The project includes scripts for both training the model and generating names. Below is a quick guide on how to use these scripts:

### Training the Model

The **train.py** script is used to train the model on the provided dataset of names.
```bash
   python train.py
```

This will train the model and save the trained model state to **model_state.pth**.


### Sampling from the Model

Once the model is trained, you can generate new names using the **sampling.py** script.
```bash
   python sampling.py
```

This will load the trained model and generate a list of names based on the learned patterns.


## Training

The training process involves optimizing the model parameters using the cross-entropy loss function. The model is trained for 200k steps with a learning rate schedule that starts at 0.1 and decays to 0.01 after 100k steps. The model is trained on 80% of the dataset, with the remaining 20% split between validation and test sets.


## Sampling

The sampling process involves generating names by predicting the next character in a sequence until the end-of-sequence character ('.') is reached. The model's output probabilities are sampled to generate realistic names.


## Results

After training, the model achieved a validation loss of **1.99**. The generated names show a strong resemblance to human-like names, demonstrating the model's ability to learn complex patterns in character sequences.
