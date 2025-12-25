# ateliers-deep-learning

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Lab 1: Deep Learning with PyTorch

## Objective
The main purpose of this lab was to get familiar with the PyTorch library by establishing Deep Neural Network (DNN) architectures for both Classification and Regression tasks.

## Work Summary

### Part 1: Regression (NYSE Stock Prices)
* **Data Processing:** Loaded the NYSE dataset, normalized the stock prices using `MinMaxScaler`, and created sliding window sequences (Time Series) for the input.
* **Model Architecture:** Built a standard Multi-Layer Perceptron (MLP) using `torch.nn` to predict future stock prices.
* **Hyperparameter Tuning:** Implemented a GridSearch simulation to optimize Learning Rate, Epochs, and Hidden Layer size.
* **Results:** Visualized the Loss (MSE) and compared Predicted vs. Actual stock prices.
* **Regularization:** Improved the model using Dropout layers and L2 Regularization (Weight Decay) to prevent overfitting.

### Part 2: Multi-class Classification (Predictive Maintenance)
* **Data Processing:** Cleaned the "Predictive Maintenance" dataset (AI4I 2020) and encoded categorical features (`Type`, `Failure Type`) using Label Encoding.
* **Handling Imbalance:** Addressed the severe class imbalance in failure types using **SMOTE** (Synthetic Minority Over-sampling Technique).
* **Model Architecture:** Constructed a Deep Neural Network for multi-class classification.
* **Evaluation:** Calculated Accuracy, Confusion Matrix, and F1-Score to evaluate performance across all failure types.

## Tools Used
* **Language:** Python
* **Libraries:** PyTorch, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
* **Environment:** Kaggle Notebooks

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Lab 2: Computer Vision with PyTorch (CNN, Faster R-CNN, ViT)

## Objective
The main goal of this lab was to master building various neural architectures for Computer Vision using PyTorch. [cite_start]This included establishing CNNs from scratch, adapting Object Detection models (Faster R-CNN), applying Transfer Learning (VGG16), and building Vision Transformers (ViT) from scratch. [cite: 67]

## Work Summary

### Part 1: CNN & Object Detection
* [cite_start]**CNN Classifier:** Built a custom Convolutional Neural Network (CNN) with 2 convolutional layers, MaxPooling, and Batch Normalization. achieved high accuracy (~99%) on the MNIST dataset efficiently. [cite: 71]
* **Faster R-CNN:** Adapted the `Faster R-CNN` object detection model to classify MNIST digits by treating the entire image as a bounding box. [cite_start]This demonstrated how to adapt detection architectures for classification, though it was computationally heavier than a simple CNN. [cite: 72]
* **Transfer Learning (VGG16):** Utilized a pre-trained VGG16 model. Since VGG16 expects 3-channel RGB images of size 224x224, the MNIST data was resized and normalized. [cite_start]Fine-tuning the final layer resulted in high accuracy but required significantly more training time due to the larger input size. [cite: 73]

### Part 2: Vision Transformer (ViT)
* [cite_start]**ViT Implementation:** Implemented a Vision Transformer from scratch based on the "Attention is All You Need" and "ViT" papers. [cite: 75, 77]
* **Architecture:** Created custom classes for `PatchEmbedding`, `MultiHeadAttention`, and `TransformerBlock`. The model splits images into fixed-size patches, linearly embeds them, adds position embeddings, and feeds them into a Transformer encoder.
* [cite_start]**Results:** The ViT model successfully classified MNIST digits, demonstrating that pure attention-based mechanisms can replace traditional CNNs for image classification tasks. [cite: 78]

## Synthesis & Conclusion
* **Standard CNNs** remain the most efficient choice for simple tasks like MNIST (fast training, high accuracy).
* **Faster R-CNN** is powerful for detection but is overkill for simple classification.
* **Transfer Learning** is excellent when data is scarce, but the input requirements (image size) can slow down training for small image datasets like MNIST.
* **Vision Transformers** offer a modern alternative to CNNs, capturing global context through self-attention, though they often require more data or regularization to outperform CNNs on small datasets.

## Tools Used
* **Language:** Python
* **Libraries:** PyTorch, Torchvision, NumPy, Matplotlib
* **Environment:** Kaggle (GPU Accelerated)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Lab 3: NLP with Sequence Models & Transformers

## Objective
The objective of this lab was to apply Deep Learning techniques to Natural Language Processing (NLP). This involved collecting Arabic text data, preprocessing it, building Recurrent Neural Networks (RNN, LSTM, GRU) for scoring relevance, and exploring Transformers (GPT-2) for text generation.

## Work Summary

### Part 1: Classification/Regression with RNNs
* **Data Collection:** Created a dataset of Arabic text with associated "relevance scores" (0-10).
* **Preprocessing:** Implemented an NLP pipeline using `NLTK`, including Tokenization, Stop-Word removal, and Stemming (ISRI Stemmer) to prepare Arabic text for modeling.
* **Modeling:** Built a flexible PyTorch `SequenceRegressor` class capable of instantiating:
    * **Simple RNN:** Basic recurrent units.
    * **Bidirectional RNN:** Processing sequence in both directions to capture context better.
    * **GRU (Gated Recurrent Unit):** Efficient gating to handle long-term dependencies.
    * **LSTM (Long Short-Term Memory):** Robust handling of vanishing gradients.
* **Comparison:** Trained all four models and compared their MSE Loss. LSTM and GRU generally outperformed the simple RNN in stability and final error.

### Part 2: Transformers (GPT-2)
* **Architecture:** Utilized the `GPT-2` architecture via the Hugging Face `transformers` library.
* **Fine-Tuning:** Prepared a text dataset and fine-tuned the pre-trained Language Model (LM) to adapt to the specific domain data.
* **Generation:** Used the fine-tuned model to generate new paragraphs based on input prompts, demonstrating the capabilities of Attention-based mechanisms over traditional RNNs.

## Tools Used
* **Language:** Python
* **Libraries:** PyTorch, NLTK, Transformers (Hugging Face), BeautifulSoup, Pandas
* **Environment:** Kaggle (GPU Accelerated)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Lab 3: NLP with Sequence Models & Transformers

## Objective
The objective of this lab was to apply Deep Learning techniques to Natural Language Processing (NLP). This involved collecting Arabic text data, preprocessing it, building Recurrent Neural Networks (RNN, LSTM, GRU) for scoring relevance, and exploring Transformers (GPT-2) for text generation.

## Work Summary

### Part 1: Classification/Regression with RNNs
* **Data Collection:** Created a dataset of Arabic text with associated "relevance scores" (0-10).
* **Preprocessing:** Implemented an NLP pipeline using `NLTK`, including Tokenization, Stop-Word removal, and Stemming (ISRI Stemmer) to prepare Arabic text for modeling.
* **Modeling:** Built a flexible PyTorch `SequenceRegressor` class capable of instantiating:
    * **Simple RNN:** Basic recurrent units.
    * **Bidirectional RNN:** Processing sequence in both directions to capture context better.
    * **GRU (Gated Recurrent Unit):** Efficient gating to handle long-term dependencies.
    * **LSTM (Long Short-Term Memory):** Robust handling of vanishing gradients.
* **Comparison:** Trained all four models and compared their MSE Loss. LSTM and GRU generally outperformed the simple RNN in stability and final error.

### Part 2: Transformers (GPT-2)
* **Architecture:** Utilized the `GPT-2` architecture via the Hugging Face `transformers` library.
* **Fine-Tuning:** Prepared a text dataset and fine-tuned the pre-trained Language Model (LM) to adapt to the specific domain data.
* **Generation:** Used the fine-tuned model to generate new paragraphs based on input prompts, demonstrating the capabilities of Attention-based mechanisms over traditional RNNs.

## Tools Used
* **Language:** Python
* **Libraries:** PyTorch, NLTK, Transformers (Hugging Face), BeautifulSoup, Pandas
* **Environment:** Kaggle (GPU Accelerated)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
