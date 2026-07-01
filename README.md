# Skin Cancer Detection 🩺🔍

A deep learning project focused on detecting skin lesions using the state-of-the-art YOLOv8 object detection model. This project aims to accurately identify skin cancer and other lesions, leveraging advanced computer vision techniques for medical image analysis.

## 📌 Overview

Skin cancer is one of the most common types of cancer globally. Early and accurate detection is crucial for effective treatment. This project utilizes the **YOLOv8** (You Only Look Once version 8) architecture by Ultralytics to detect skin lesions from medical images.

The model is trained on a comprehensive dataset (`skin-lesion-segmentation-classification`) and predicts bounding boxes to precisely localize lesions within images.

## 📂 Project Structure

This project follows an industrial standard Data Science folder structure:

```
.
├── data/               # Datasets used in the project
│   ├── raw/            # The original, immutable data dump
│   └── processed/      # The final, canonical data sets for modeling
├── docs/               # Documentation files
├── models/             # Trained and serialized models, model predictions, or model summaries
├── notebooks/          # Jupyter notebooks. Naming convention is a number (for ordering),
│                       # the creator's initials, and a short `-` delimited description
│   └── Lesion_Detection.ipynb
├── src/                # Source code extracted from the notebook
│   ├── data_setup.py   # Script to download and prepare the dataset
│   └── train.py        # Script to train the YOLOv8 model
├── .gitignore          # Specifies intentionally untracked files to ignore
├── README.md           # The top-level README for developers using this project
└── requirements.txt    # The requirements file for reproducing the analysis environment
```

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/HammadIsmail/-Skin-Cancer-Detection.git
cd -Skin-Cancer-Detection
```

### 2. Set up the environment

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the Project

You can choose to either run the Jupyter Notebook interactively or use the automated Python scripts in the `src/` folder.

**Option A: Using Python Scripts (Recommended)**
First, download and configure the dataset:
```bash
python src/data_setup.py
```
Then, start the model training:
```bash
python src/train.py
```

**Option B: Using Jupyter Notebook**
Launch Jupyter Notebook or JupyterLab to explore the implementation interactively:

```bash
jupyter notebook notebooks/Lesion_Detection.ipynb
```

The scripts and notebook will automatically download the dataset from Hugging Face, format the `data.yaml`, and initiate the training process using YOLOv8.

## 📊 Dataset

The dataset used is `skin-lesion-segmentation-classification` from Hugging Face.
- **Train Images:** 6,675
- **Validation Images:** 1,911
- **Test Images:** 961

The script automatically downloads and extracts the data. No manual downloading is required!

## 🧠 Model Training

The project utilizes the `yolov8n.pt` (nano) model as the starting point and trains on the skin lesion dataset for 50 epochs with an image size of 640x640.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="skin_lesion_yolo",
    name="yolov8_detection"
)
```

Trained weights and validation metrics will be saved in the `runs/` directory (or `skin_lesion_yolo/yolov8_detection/` inside the project root).



