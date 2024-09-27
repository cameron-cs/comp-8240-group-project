# comp-8240-group-project
A C-LSTM Neural Network for Text Classification paper

## Overview

This project implements a Convolutional Long Short-Term Memory (C-LSTM) neural network for text classification. The architecture combines the strengths of Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks, enabling the model to capture both local phrase features (via CNN) and long-term dependencies (via LSTM). The embedding layer is initialised with pre-trained GloVe word embeddings, which helps improve performance by leveraging external knowledge from massive text corpora.

## Model Architecture

The architecture consists of the following components:

1. **Embedding Layer**: Pre-trained GloVe embeddings are used to initialise the embedding layer. These embeddings are fine-tuned during training to capture task-specific representations.
   
2. **Convolutional Layers**: Several 2D convolution layers with varying filter sizes are used to capture different n-gram features from the input sentence.

3. **LSTM Layer**: After the CNN layers, the output is passed to an LSTM layer, which models the sequential dependencies in the extracted features.

4. **Dropout Layer**: A dropout layer is applied to reduce overfitting during training.

5. **Fully Connected Layer**: The final output from the LSTM layer is passed through a fully connected layer with softmax activation for classification.

## Setup Instructions

To use the CLSTM model with pre-trained GloVe embeddings:

1. Prepare your dataset in a compatible format, ensuring each data point has corresponding labels.
   
2. Load the GloVe embedding matrix into the model to initialise the embedding layer.

3. Configure the model with appropriate parameters for your specific task, including sequence length, number of classes, vocabulary size, embedding size, filter sizes for convolution layers, and LSTM settings.

4. Train the model using your dataset and validate the performance using a validation set.

5. Evaluate the model on a test set to determine the accuracy and performance of the classifier.

## Configuration

The model configuration can be controlled by adjusting the following parameters:

- `max_length`: Maximum length of input sequences (e.g., sentences or documents).
- `num_classes`: Number of output classes for classification.
- `vocab_size`: Size of the vocabulary, which corresponds to the total number of unique tokens in your dataset.
- `embedding_size`: Dimensionality of the GloVe embeddings (e.g., 300 for GloVe 300D).
- `filter_sizes`: List of filter sizes for the convolutional layers, typically representing n-grams like 3, 4, or 5.
- `num_filters`: Number of filters per convolutional layer, determining how many features are extracted from each n-gram.
- `num_layers`: Number of LSTM layers to stack after the convolutional layers.
- `l2_reg_lambda`: Strength of L2 regularisation to prevent overfitting.
- `keep_prob`: Dropout rate for regularisation during training.

## Example Workflow

1. Load the pre-trained GloVe embeddings corresponding to the words in your dataset.
   
2. Initialise the model with appropriate filter sizes, LSTM hidden units, and dropout rates based on your dataset characteristics.

3. Train the model using a chosen optimiser and monitor the training process using a validation set.

4. Once training is complete, evaluate the model performance using a test set to ensure generalisation.

5. Fine-tune hyperparameters such as the number of filters, LSTM layers, and dropout rate to optimise performance on your dataset.

## References

- Zhou, Chunting, et al. "A C-LSTM Neural Network for Text Classification." arXiv preprint arXiv:1511.08630 (2015). [Link to Paper](https://arxiv.org/abs/1511.08630)
- GloVe: Global Vectors for Word Representation. Available at [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
