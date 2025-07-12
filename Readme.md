# 🦋 Butterfly Species Classification with Transfer Learning

This project focuses on building an image classification model using **Transfer Learning (MobileNetV2)** to identify butterfly species from images. The dataset used is sourced from Kaggle and contains high-quality images across multiple butterfly species.

## 📁 Dataset

- **Source**: [Butterfly Image Classification - Kaggle](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)
- **License**: CC0: Public Domain
- **Structure**:
  - `train/`: Training images categorized by species
  - `test/`: Testing images (without labels)
  - `Training_set.csv`: Filename–label mapping for training images
  - `Testing_set.csv`: Filename list for test images

You can automatically download the dataset using `kagglehub`:
```python
import kagglehub
path = kagglehub.dataset_download("phucthaiv02/butterfly-image-classification")

🧠 Project Workflow
1. 📦 Data Loading and Augmentation
Read the CSV metadata using pandas

Load and preprocess the images with ImageDataGenerator

Augmentation: rotation, zoom, horizontal flip, and rescaling

2. 🔧 Model Architecture
Base Model: MobileNetV2 (pretrained on ImageNet, frozen)

Top Layers:

Global Average Pooling

Dense(128, ReLU)

Dropout(0.5)

Output Layer (Softmax for classification)

3. 🏋️ Training
Optimizer: Adam

Loss: Categorical Crossentropy

Metric: Accuracy

Early Stopping: Stop if val_loss doesn't improve for 5 epochs

Model saved to: butterfly_model.h5

4. 📈 Evaluation
Plot training and validation accuracy/loss

Visual inspection of predicted labels on sample test images

5. 🔍 Prediction
Predict test set using the trained model

Output is a new label column appended to test_df

Visualize sample predictions in a grid format

🧪 Results

| Metric               | Value                      |
| -------------------- | -------------------------- |
| Final Train Accuracy | \~81,60%                      |
| Final Val Accuracy   | \~86,22%                      |
| Model File Size      | \~10.96MB (MobileNetV2-based) |


📌 Dependencies
Python 3.8+

TensorFlow 2.x

Pandas, NumPy, Matplotlib, Seaborn

KaggleHub (for auto-downloading the dataset)

Install requirements:

pip install tensorflow pandas numpy matplotlib seaborn kagglehub

📂 File Structure
.
├── notebook.ipynb             
├── butterfly_model.h5         
├── README.md

🚀 How to Run
Clone this repo

Install dependencies

Run notebook.ipynb step by step

Optionally re-train or load saved model

Visualize predictions at the end

