# Exploring VGG6 on CIFAR-10 with Different Configurations

### CS6886W – System Engineering for Deep Learning  
**Department of Computer Science and Engineering**  
**Indian Institute of Technology Madras**

---

## 📘 Overview
This repository contains the implementation and experiments for **Assignment 1** of the CS6886W course.  
The objective is to explore the performance of a **VGG6 Convolutional Neural Network** on the **CIFAR-10** dataset under different configurations — varying activation functions, optimizers, learning rates, and batch sizes.  

The project is implemented in **Python (PyTorch)** and supports both **local execution** and **Weights & Biases (W&B)** integration for automated experiment tracking and hyperparameter sweeps.

---

## 🧩 Repository Structure
```
├── VGG6_main.py            # Main driver script
├── train.py                # Training and evaluation loop
├── model.py                # VGG6 model architecture
├── image_transform.py      # Data augmentation and normalization transforms
├── utils.py                # Helper functions
├── sweep_config.yaml       # Configuration file for W&B sweep experiments
├── Charts/                 # Folder containing training and validation plots
└── README.md               # Project documentation (this file)
└── weights.pth             # Best Model Weights
└── requirements.txt        # requirement.txt to create python env
```

---

## ⚙️ Requirements

### Python Version
- Python 3.10 or higher

### Dependencies
Install all required dependencies with:
```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not provided, you can manually install the following key packages:
```bash
pip install torch torchvision torchaudio matplotlib numpy tqdm pyyaml wandb
```

---

## 🚀 Running the Code

### **Option 1: Run Locally (Without W&B)**
You can train and evaluate the VGG6 model directly without using W&B:
```bash
python VGG6_main.py --activation=GELU --batch_size=256 --epochs=100 --learning_rate=0.01 --optimizer=Nadam --mode=train --weight_file_path=WEIGHT_FILE_PATH
```
This will:
- Train the model on CIFAR-10  with provided command line parameters 
- Display training progress in the console  
- Save the model weights in the WEIGHT_FILE_PATH provided

---

---

## 🚀 Running the Code

### **Option 1: Run Locally (Without W&B)**
You can only evaluate the VGG6 model directly with already trained weight file
```bash
python VGG6_main.py --activation=GELU --batch_size=256 --epochs=100 --learning_rate=0.01 --optimizer=Nadam --mode=val --weight_file_path=WEIGHT_FILE_PATH
```
This will:
- Validate the model on CIFAR-10  with provided command line parameters 
- Display the accuracy on Validation and Test data set 

---

### **Option 2: Run with W&B (Tracking Enabled)**
To enable experiment tracking using **Weights & Biases**:
```bash
/usr/bin/env python VGG6_main.py --activation=GELU --batch_size=256 --epochs=100 --learning_rate=0.01 --mode=train --optimizer=Nadam --wandb_mode=wandb_standalone
```
This will:
- Log training and validation metrics to your W&B account  
- Automatically generate loss, accuracy, and performance plots  
Note: before executing above user needs to login into wandb account once.

---

### **Option 3: Run W&B Sweep (Hyperparameter Search)**
To run a full hyperparameter sweep as defined in `sweep_config.yaml`:
```bash
wandb sweep sweep_config.yaml
wandb agent <SWEEP_ID>
```
This performs multiple training runs with varying configurations (e.g., optimizer, activation, learning rate) and identifies the best-performing setup automatically.

---

## 🧪 Experiment Details

- **Dataset:** CIFAR-10  
- **Model:** Custom 6-layer VGG (VGG6)  
- **Optimizers Tested:** SGD, Nesterov-SGD, Adam, Adagrad, RMSprop, Nadam  
- **Activations Tested:** ReLU, Sigmoid, Tanh, SiLU, GELU  
- **Batch Sizes:** 32, 64, 128,256,512 
- **Learning Rates:** 0.01, 0.001  
- **Epochs:** 20,40,60,80,100

---

## 📊 Results Summary

| Configuration | Activation | Optimizer | LR | Batch Size | epochs|Best Val Accuracy | Test Accuracy |
|----------------|-------------|------------|----|-------------|-------------------|----------------|
| Best Model | GELU | Nadam | 0.01 | 256 |100| **88.4%** | **90.4%** |

The GELU + NADAM configuration provided the best Results.

---

## ♻️ Reproducibility

- Random seeds are fixed in all scripts for deterministic results.  
- The training pipeline and data loading are fully reproducible.  
- The environment setup and dependency versions are provided for consistency.  

---

## 📁 Outputs
- W&B runs are automatically logged if enabled  
- Trained model weights are saved in the local directory 
- Charts are generated in W&B dashboard if running with W&B 

---

## 🔗 GitHub Repository
**Repository Link:** [https://github.com/cs24m542/VGG6-on-CIFAR-10/tree/main]

---



---

## 🧑‍💻 Author
**Name:** [– Vijay Kumar Agrawal]  
**Roll Number:** [– CS24M542]  
**Course:** CS6886W – System Engineering for Deep Learning  
**Institution:** IIT Madras  
