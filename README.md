# Pneumonia Classifier using CNN (PneumoniaMNIST)

This repository contains code to train a CNN model to classify chest X-rays as Pneumonia or Normal using the PneumoniaMNIST dataset.

## 📂 Dataset
- Download: https://www.kaggle.com/datasets/rijulshr/pneumoniamnist/data
- Upload `pneumoniamnist.npz` into `/content` if using Colab.

## ⚙️ How to Run (in Google Colab)
1. Upload the dataset to Colab.
2. Open `train_and_eval.ipynb`.
3. Run all cells to train and evaluate the model.
4. Test accuracy and training plots will be shown.

## 🧪 Evaluation
- Accuracy, Loss
- Early stopping used
- Class imbalance handled with weighted loss

## 🧵 Hyperparameters
- Epochs: 70
- Batch Size: 64
- Learning Rate: 0.0001