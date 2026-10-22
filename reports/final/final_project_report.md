\# MedVision-AI: Deep Learning for Chest X-ray Image Classification



\## Abstract



MedVision-AI is a final-year deep learning project that investigates binary classification of chest X-ray images into the dataset categories NORMAL and PNEUMONIA. The project uses transfer learning with DenseNet121, implemented in PyTorch, and provides an interactive Streamlit interface for demonstrating model predictions. The workflow includes dataset inspection, image preprocessing, model training, checkpoint selection, quantitative evaluation, and error analysis. The recorded evaluation achieved 85.42% accuracy and a ROC-AUC of 0.9449 on 624 test images. These results describe performance on the evaluated dataset and do not establish clinical usefulness or safety.



\## 1. Introduction



Medical image classification is an important computer vision research area. Deep learning models can learn image features from labeled examples and use them to make predictions on previously unseen images. Chest X-ray classification provides an opportunity to study image preprocessing, transfer learning, model evaluation, and responsible presentation of AI outputs.



MedVision-AI focuses on classifying images into two dataset labels: NORMAL and PNEUMONIA. It is an educational prototype and is not intended for medical diagnosis or treatment decisions.



\## 2. Problem Statement



The project investigates whether a DenseNet121 image classifier can learn patterns associated with the NORMAL and PNEUMONIA labels in a labeled chest X-ray dataset.



\## 3. Objectives



\- Inspect the dataset structure and image distribution.

\- Preprocess images into a consistent model input format.

\- Apply training-time data augmentation.

\- Implement and train a DenseNet121 classifier.

\- Select a model checkpoint using validation loss.

\- Evaluate performance using classification metrics and ROC-AUC.

\- Examine misclassified examples and dataset quality.

\- Develop an interactive Streamlit application.

\- Document limitations and possible improvements.



\## 4. Dataset Description



The dataset is organized into training, validation, and test directories, with NORMAL and PNEUMONIA class folders. The training pipeline uses images from the training directory and creates a stratified validation split. The separate test directory is used for final evaluation.



The recorded test set contains 624 images: 234 labeled NORMAL and 390 labeled PNEUMONIA. The class distribution is therefore imbalanced.



Dataset quality checks identified exact duplicate image files within some individual splits. The audit found no byte-for-byte identical images shared across the checked train, validation, and test splits. This does not prove patient-level independence because image hashes cannot establish whether different images belong to the same patient.



The dataset should be obtained from an authorized source. Raw dataset files are excluded from the Git repository.



\## 5. Data Preprocessing



Images are resized to 224 × 224 pixels and converted to RGB. They are converted into tensors and normalized using ImageNet channel statistics:



\- Mean: (0.485, 0.456, 0.406)

\- Standard deviation: (0.229, 0.224, 0.225)



Random horizontal flipping with probability 0.5 is applied during training. Evaluation transformations do not include random flipping.



A stratified split with a fixed random seed is used to create training and validation subsets from the training directory. Stratification helps preserve class proportions, but it does not guarantee separation by patient.



\## 6. Model Architecture



The project uses DenseNet121, a convolutional neural network architecture with densely connected layers. Dense connections allow feature information to be reused across network layers.



The original classification layer is replaced with a fully connected layer containing two outputs, corresponding to NORMAL and PNEUMONIA.



The implementation uses PyTorch and Torchvision. The architecture is initialized using the available pretrained-weight option during model setup, while evaluation loads the saved checkpoint without downloading pretrained weights again.



\## 7. Training Methodology



The training configuration includes:



\- Optimizer: AdamW

\- Learning rate: 0.0001

\- Loss function: class-weighted cross-entropy

\- Input resolution: 224 × 224

\- Checkpoint selection: lowest validation loss

\- Execution device: CPU in the current environment



The best checkpoint is saved at `artifacts/checkpoints/best\_model.pth`.



Training duration and hardware can affect model performance. The final reproducible report should record the exact number of epochs, batch size, random seed, package versions, and checkpoint used for evaluation from the actual training logs and configuration.



\## 8. Evaluation Metrics



The model was evaluated on 624 test images. The recorded results are:



| Metric | Result |

|---|---:|

| Test images | 624 |

| Accuracy | 85.42% |

| Weighted precision | 87.51% |

| Weighted recall | 85.42% |

| Weighted F1-score | 84.55% |

| ROC-AUC | 94.49% |

| Test loss | 0.6058 |



Accuracy measures the fraction of correctly classified images. Precision measures how often predictions for a class are correct. Recall measures how many examples of a class are identified. The F1-score combines precision and recall. ROC-AUC measures ranking discrimination across thresholds and does not, by itself, establish a clinically appropriate operating threshold.



\### 8.1 Confusion Matrix



Rows represent actual classes and columns represent predicted classes.



| Actual / Predicted | NORMAL | PNEUMONIA |

|---|---:|---:|

| NORMAL | 147 | 87 |

| PNEUMONIA | 4 | 386 |



\### 8.2 Class-Specific Results



| Class | Precision | Recall | F1-score | Support |

|---|---:|---:|---:|---:|

| NORMAL | 97.35% | 62.82% | 76.36% | 234 |

| PNEUMONIA | 81.61% | 98.97% | 89.46% | 390 |



\## 9. Results and Discussion



The recorded accuracy is 85.42%, while the ROC-AUC is 94.49%. These metrics measure different aspects of model behavior and should not be interpreted as interchangeable.



The model has high recall for PNEUMONIA-labeled images in this test set, but its recall for NORMAL images is lower. Of the 234 NORMAL-labeled images, 87 were predicted as PNEUMONIA. This corresponds to a NORMAL-class recall of approximately 62.82%.



This error pattern indicates that the classifier tends to assign the PNEUMONIA label to many images belonging to the NORMAL class. Accuracy alone would not communicate this limitation, which is why class-specific metrics and the confusion matrix are important.



The results apply only to the current dataset, split, checkpoint, and evaluation procedure. They cannot establish performance on other hospitals, imaging systems, or patient populations.



\## 10. Error Analysis and Dataset Quality



The project includes scripts for dataset inspection, duplicate checks, image distribution analysis, prediction diagnostics, and misclassification review. Supporting outputs are stored in the reports directory.



Exact duplicate detection identifies byte-identical image files. It cannot determine patient identity or detect every visually similar image. Patient-level splitting requires trustworthy patient identifiers and should be considered in future work.



The large number of NORMAL images predicted as PNEUMONIA should be investigated further by reviewing image quality, labels, preprocessing, class balance, model calibration, and decision thresholds. Threshold changes should be evaluated on validation data rather than selected solely on the test set.



\## 11. Streamlit Application



The project provides a Streamlit web interface that allows a user to upload a supported image, run inference using the saved model, and view the predicted class and class probabilities.



The interface includes a warning that the application is intended only for educational research. A displayed confidence value is the model's estimated probability and is not a measure of clinical certainty.



\## 12. Limitations



\- The model is not clinically validated and must not be used for medical decisions.

\- Dataset labels and image quality may introduce errors or bias.

\- Class imbalance can affect predictions and aggregate metrics.

\- Exact duplicate checks do not prove patient-level separation.

\- Results from a single test set may not generalize to external data.

\- Model confidence does not guarantee calibration or correctness.

\- CPU training and limited experimentation may constrain the experiments performed.

\- The reported metrics should be reproduced against the exact saved checkpoint before final academic submission.



\## 13. Future Enhancements



\- Use patient-level train, validation, and test separation when reliable identifiers are available.

\- Repeat training across multiple random seeds.

\- Compare DenseNet121 with suitable baseline architectures.

\- Plot training and validation loss and performance curves.

\- Add ROC and precision-recall curves to the evaluation report.

\- Investigate misclassified examples and assess probability calibration.

\- Add automated tests and reproducible environment configuration.

\- Evaluate external datasets before making any claims about generalization.



\## 14. Conclusion



MedVision-AI demonstrates an end-to-end educational deep learning workflow for chest X-ray image classification. The project combines dataset preprocessing, DenseNet121 transfer learning, checkpoint selection, quantitative evaluation, error inspection, and a Streamlit inference interface.



The current recorded test results show 85.42% accuracy and 94.49% ROC-AUC. Class-specific analysis reveals that the model misclassifies a substantial number of NORMAL-labeled images as PNEUMONIA. This highlights the importance of examining confusion matrices and class-level metrics rather than relying on accuracy alone.



The project provides a foundation for further experimentation in reproducibility, patient-level data separation, error analysis, and responsible AI deployment. It remains an educational prototype and is not a clinical diagnostic system.



\## 15. Project Files



\- `train.py` — model training

\- `evaluate.py` — model evaluation

\- `predict.py` — single-image prediction utility

\- `src/medvision/data/dataset.py` — dataset loading and preproscessing

\- `src/medvision/models/densenet.py` — model architecture

\- `app/app.py` — Streamlit interface

\- `reports/evaluation/` — recorded evaluation outputs

\- `reports/dataset\_quality\_audit.md` — dataset audit



