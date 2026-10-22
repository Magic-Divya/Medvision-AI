
# MedVision-AI
## Deep Learning for Chest X-ray Image Classification

MedVision-AI is a final-year deep learning project that uses DenseNet121 and PyTorch to classify chest X-ray images into two dataset classes: NORMAL and PNEUMONIA.

**Important:** This is an educational research prototype, not a clinically validated diagnostic tool. Do not use its predictions for medical decisions.

## 1. Problem Statement

Chest X-ray classification is a computer vision task that uses deep learning to identify visual patterns in medical images. This project investigates transfer learning for classifying images labeled NORMAL or PNEUMONIA.

## 2. Objectives

- Prepare and inspect the dataset.
- Apply image preprocessing and data augmentation.
- Implement a DenseNet121 classifier.
- Train the model and select a checkpoint using validation loss.
- Evaluate performance using classification metrics and ROC-AUC.
- Develop an interactive Streamlit application.
- Analyze errors and document limitations.

## 3. Technology Stack

- Python
- PyTorch and Torchvision
- DenseNet121
- NumPy and Pandas
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit and Pillow

## 4. Dataset and Preprocessing

The dataset contains `train`, `val`, and `test` directories, each organized into `NORMAL` and `PNEUMONIA` classes.

The current data-loading pipeline uses the training directory to create a stratified training/validation split and uses the separate test directory for evaluation.

Preprocessing includes:
- Resizing images to 224 × 224 pixels
- Converting images to RGB
- Random horizontal flipping during training
- Converting images to tensors
- ImageNet normalization

The raw dataset is excluded from Git. Obtain the dataset from its authorized source and place it in the expected directory before training.

## 5. Model and Training

The project uses DenseNet121 with a modified final layer for two-class classification.

The training configuration includes:
- Optimizer: AdamW
- Learning rate: 0.0001
- Loss: Class-weighted cross-entropy
- Validation-based checkpoint selection

The trained checkpoint is stored at `artifacts/checkpoints/best_model.pth`.

## 6. Test Results

The current recorded evaluation results are:

| Metric | Result |
|---|---:|
| Test images | 624 |
| Accuracy | 85.42% |
| Weighted precision | 87.51% |
| Weighted recall | 85.42% |
| Weighted F1-score | 84.55% |
| ROC-AUC | 94.49% |

Confusion matrix (rows are actual classes, columns are predicted classes):

| Actual / Predicted | NORMAL | PNEUMONIA |
|---|---:|---:|
| NORMAL | 147 | 87 |
| PNEUMONIA | 4 | 386 |

The model correctly classified many PNEUMONIA-labeled images, but it also incorrectly predicted PNEUMONIA for a substantial number of NORMAL images. These results do not establish clinical safety or performance on other populations.

## 7. Project Structure

- `app/` — Streamlit application
- `artifacts/checkpoints/` — model checkpoints
- `configs/` — project configuration
- `data/` — local dataset, excluded from Git
- `reports/` — evaluation and dataset analysis
- `src/medvision/data/` — data loading and preprocessing
- `src/medvision/models/` — model definition
- `src/medvision/training/` — training utilities
- `train.py` — training entry point
- `evaluate.py` — evaluation entry point
- `predict.py` — image prediction utility
- `requirements.txt` — Python dependencies

## 8. Run Locally

Activate your virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Ensure the dataset and trained checkpoint are available in the expected locations.

Start the website:

```powershell
python -m streamlit run app/app.py
```

Train the model:

```powershell
python train.py
```

Evaluate the model:

```powershell
python evaluate.py
```

## 9. Evaluation and Reports

The project includes evaluation metrics, a confusion matrix, dataset quality checks, image distribution analysis, and misclassification inspection scripts. Results are stored under `reports/`.

## 10. Limitations and Responsible Use

- This is an educational prototype, not a medical device.
- Dataset labels, image quality, and class distribution may affect results.
- Performance may vary across hospitals, equipment, and populations.
- Model confidence is not clinical certainty.
- Exact duplicate checks do not prove patient-level independence.
- The results must not be interpreted as evidence of real-world diagnostic performance.

## 11. Future Work

- Assess patient-level separation when reliable patient identifiers are available.
- Investigate class-specific errors and decision thresholds.
- Add training curves and ROC visualizations.
- Evaluate reproducibility across random seeds.
- Add automated tests and deployment documentation.

## 12. Academic Project

MedVision-AI demonstrates dataset preparation, transfer learning, model evaluation, and interactive image inference using deep learning.
