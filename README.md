
# Transformer-based Neural Machine Translation (NMT)

This project is a PyTorch-based implementation of a Neural Machine Translation (NMT) model inspired by the architecture from the paper *Attention is All You Need*. The model utilizes a transformer architecture to translate sequences between two languages efficiently, with a robust multi-head self-attention mechanism that captures long-range dependencies in text.

## Project Overview

The NMT model in this repository has been implemented using:
- **Encoder-Decoder Transformer Blocks**: Each block uses multi-head self-attention to understand context across the sequence.
- **Multi-Head Attention**: Captures multiple dimensions of dependencies and language relationships simultaneously.
- **Feed-Forward Layers and Residual Connections**: Each layer is followed by a fully connected layer and normalized with residual connections to improve convergence.
- **Positional Encoding**: Maintains word order within sentences.

This implementation draws on the PyTorch framework, allowing for efficient training and model extensibility.

## Key Components

### Files
1. **dataset.py** - Defines the `BilingualDataset` class, which formats data for training and handles tokenization.
2. **model.py** - Defines the complete transformer model architecture, including encoder and decoder blocks.
3. **train.py** - Contains the training and validation routines.
4. **translate.py** - Enables inference and translation of sentences using a pretrained model.

### Core Highlights

- **Attention Mechanism**: Inspired by the *Attention is All You Need* paper, the multi-head attention in both the encoder and decoder allows the model to capture complex dependencies across words in a sequence.
- **Positional Encoding**: Since transformer models are non-recurrent, positional encodings are added to give the model an understanding of word order.
- **Layer Normalization and Residual Connections**: For enhanced performance and training stability, each layer uses residual connections and normalization.

### Building and Running the Model

#### Prerequisites
- PyTorch
- Hugging Face Datasets and Tokenizers
- TorchMetrics (for evaluation)

Install dependencies with:
```bash
pip install torch datasets tokenizers torchmetrics
```

#### Training the Model
To train the model:
```bash
python train.py
```

The `train.py` script handles the model's full training process, including data loading, tokenization, and checkpointing.

#### Translation
To translate a sentence:
```bash
python translate.py "Your text here"
```

### Attention Mechanism: *Attention is All You Need*
The attention mechanism allows the model to focus selectively on different parts of a sequence, crucial for translations where word dependencies vary. In our model:
1. **Multi-Head Attention** allows the model to jointly attend to information from different representation subspaces at different positions.
2. **Scaled Dot-Product Attention** is used within each head, enhancing focus on relevant portions of the input sequence.

## Future Enhancements
- **Language Expansion**: Support for more language pairs.
- **Interactive Web Interface**: An interface to test translations easily.

