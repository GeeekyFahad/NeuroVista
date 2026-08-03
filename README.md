# NeuroVista 🧠

NeuroVista is a deep learning pipeline for classifying the stage of Alzheimer's disease from brain MRI scans. It uses a fine-tuned ResNet18 to categorize scans into four dementia severity levels, with a custom preprocessing pipeline built for MRI-specific challenges like skull artifacts and contrast variation.

## Classes

The model predicts one of four stages:

| Label | Class |
|---|---|
| 0 | Non Demented |
| 1 | Very Mild Dementia |
| 2 | Mild Dementia |
| 3 | Moderate Dementia |

## Features

- **Transfer learning** with a pretrained ResNet18 backbone, fine-tuned for 4-class classification
- **MRI-specific preprocessing**: auto-cropping, approximate 2D skull-stripping, and CLAHE contrast enhancement
- **Data augmentation**: random horizontal flip, rotation, and color jitter for the training set
- **Full training loop** with early stopping and per-epoch accuracy, precision, recall, F1, and confusion matrix reporting
- **Simple CLI inference** for predicting on a single image

## Project Structure

```
NeuroVista/
├── model/
│   ├── dataset.py      # Custom PyTorch Dataset for loading MRI images by class folder
│   ├── model.py         # ResNet18-based classifier
│   ├── preprocess.py    # MRI preprocessing utilities (CLAHE, auto-crop, skull-strip)
│   ├── train.py         # Training loop with metrics and early stopping
│   └── predict.py       # CLI script for single-image inference
└── notebooks/
    └── demo.ipynb        # Notebook for exploration/demo
```

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/GeeekyFahad/NeuroVista.git
cd NeuroVista/model
```

### 2. Install dependencies

```bash
pip install torch torchvision opencv-python numpy pillow scikit-learn
```

### 3. Prepare the dataset

Organize your MRI images into the following folder structure inside a `data/` directory at the project root:

```
data/
├── Non Demented/
├── Very mild Dementia/
├── Mild Dementia/
└── Moderate Dementia/
```

Each folder should contain the corresponding `.jpg`, `.jpeg`, or `.png` images.

### 4. Train the model

```bash
python train.py
```

This trains for up to 20 epochs with early stopping (patience of 5), and saves the best-performing model as `best_model.pth`.

### 5. Run inference

```bash
python predict.py <path_to_image>
```

Example output:

```
Prediction: Very mild Dementia
Confidence: 0.874
```

## Tech Stack

- Python
- PyTorch / torchvision
- OpenCV
- scikit-learn (metrics)
- Pillow

## Roadmap

- [ ] Integrate the `preprocess.py` pipeline (CLAHE, auto-crop, skull-strip) into the training/inference flow
- [ ] Add a `requirements.txt`
- [ ] Add Grad-CAM or similar visualization for model interpretability
- [ ] Flesh out `notebooks/demo.ipynb` with an end-to-end walkthrough
- [ ] Add a small evaluation report / benchmark results section

## Disclaimer

This project is for educational and research purposes only. It is **not** a medical diagnostic tool and should not be used for clinical decision-making.

## Author

Built by [Fahad](https://github.com/GeeekyFahad)
