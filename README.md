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
