# Pneumonia Classification using ResNet-50 🫁

This project uses **transfer learning** with ResNet-50 to classify chest X-ray images from the **PneumoniaMNIST** dataset into pneumonia or normal classes.

## 🧪 Dataset
The dataset used is `pneumoniamnist.npz` from [MedMNIST](https://medmnist.com/).

## 🛠️ Requirements
Install dependencies using:

```bash
pip install -r requirements.txt
```

## 🚀 Run Training
To train and evaluate the model:

```bash
python train_and_eval.py
```

Make sure `pneumoniamnist.npz` is in the same directory or update the path accordingly.

## 📈 Output
- Plots training & validation accuracy/loss
- Prints test accuracy

## ✅ Model
- Base: `ResNet-50` (ImageNet weights)
- Layers: GlobalAveragePooling + Dropout + Dense
- Loss: Binary Crossentropy
- Optimizer: Adam (lr=1e-4)
