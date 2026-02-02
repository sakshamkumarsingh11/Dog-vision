# Dog - Vison – Image Classification with Deep Learning

This repository contains my implementation of an **unstructured data machine learning project** focused on **image classification using deep learning**.  
The project is part of my learning journey in machine learning and deep learning, where the emphasis is on working with **image data** using **TensorFlow and Convolutional Neural Networks (CNNs)**.

Unlike structured datasets (tables, CSVs), image data is unstructured and requires neural networks to automatically learn meaningful features.

---

## 📌 Project Overview

**Goal:**  
Build an end-to-end deep learning pipeline that can **classify images into multiple categories (dog breeds)**.

**Type of problem:**  
Multi-class image classification

**Core idea:**  
Train a neural network to learn visual patterns from images and predict the correct class label.

---

## 🧠 What This Project Covers

This project demonstrates a complete deep learning workflow:

- Loading and understanding image datasets
- Preprocessing and scaling image data
- Data augmentation to improve generalization
- Building CNN models using TensorFlow/Keras
- Applying **transfer learning** with pre-trained models
- Training, evaluating, and improving model performance
- Making predictions on unseen images

---

## 📂 Files in This Project

| File | Description |
|-----|-------------|
| `Dog_vision.ipynb` | Main notebook containing the full project workflow |
| Other `.ipynb` files | Supporting or experimental notebooks related to the same project |

---

## ⚙️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **TensorFlow Hub** (for transfer learning)
- **NumPy**
- **Matplotlib**
- **Google Colab**

---

## 🧪 Model & Approach

### 1. Data Preparation
- Images are resized to a fixed shape
- Pixel values are normalized
- Data is split into training and validation sets
- Data augmentation is applied to reduce overfitting

### 2. Model Building
- Convolutional Neural Networks (CNNs) are used for feature extraction
- Pre-trained models are leveraged using transfer learning
- Custom classification layers are added on top

### 3. Training
- Loss function: Categorical Cross-Entropy
- Optimizer: Adam
- Metrics: Accuracy
- Training is done in multiple phases to improve performance

### 4. Evaluation
- Accuracy and loss curves are visualized
- Predictions are tested on unseen images
- Model performance is analyzed to understand strengths and weaknesses

---

## 🚀 How to Run the Project

### Option 1: Run on Google Colab (Recommended)
1. Open the notebook in Google Colab
2. Enable GPU  
   `Runtime → Change runtime type → GPU`
3. Run all cells sequentially

### Option 2: Run Locally
1. Clone the repository
   ```bash
   [git clone <your-repo-url>](https://github.com/sakshamkumarsingh11/Dog-vision.git)
   pip install tensorflow tensorflow-hub numpy matplotlib
   jupyter notebook

