# 🩻 Multiclass Chest X-Ray Classification using SoftMatch

<p align="center">
  <strong>Semi-Supervised Learning · SoftMatch · DenseNet-121</strong>
</p>

<p align="center">
  An experimental deep learning project for multiclass chest X-ray classification under limited labeled data conditions.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Model-DenseNet--121-2E8B57" alt="DenseNet-121">
  <img src="https://img.shields.io/badge/Learning-Semi--Supervised-7B61FF" alt="Semi-Supervised Learning">
  <img src="https://img.shields.io/badge/Environment-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white" alt="Google Colab">
</p>

---

## 📌 Overview

Deep learning models for medical image classification typically require large amounts of labeled data. However, annotating medical images requires specialized expertise, time, and resources.

This project explores **SoftMatch-based Semi-Supervised Learning (SSL)** with **DenseNet-121** to classify chest X-ray (CXR) images into three classes:

- 🫁 **Normal**
- 🦠 **COVID-19**
- 🩺 **Pneumonia**

The main objective is to investigate how effectively a model can learn from unlabeled images when only a small proportion of the training data has labels.

The experiments also examine the contribution of **Uniform Alignment (UA)** in addressing prediction bias under limited and imbalanced labeled-data conditions.

## 🎯 Research Objectives

- Implement SoftMatch with a pretrained DenseNet-121 backbone for multiclass CXR classification.
- Evaluate model performance under limited labeled-data scenarios.
- Compare SoftMatch configurations with and without Uniform Alignment.
- Investigate the impact of labeled-data proportions on classification performance.
- Analyze model predictions using evaluation metrics and Grad-CAM visualizations.

## 🧠 Methodology

The research follows an experimental and comparative approach.

### 1. Dataset

The study uses the *COVID-19-Pneumonia-Normal Chest X-Ray Images* dataset from Mendeley Data.

| Dataset characteristic | Description |
|---|---|
| Total images | 5,228 |
| Image format | PNG |
| Original image dimensions | 256 × 256 pixels |
| Input channels | RGB |
| Classification classes | Normal, COVID-19, Pneumonia |

The dataset is divided using stratified splitting to preserve class proportions across the training, validation, and test subsets.

### 2. Data Split

| Subset | Proportion | Number of images |
|---|---:|---:|
| Training pool | 70% | 3,659 |
| Validation | 10% | 523 |
| Testing | 20% | 1,046 |
| **Total** | **100%** | **5,228** |

The split uses a fixed random seed of 42 to support reproducibility.

### 3. Semi-Supervised Learning

SoftMatch uses model predictions on unlabeled images to generate pseudo-labels. Instead of relying exclusively on a rigid confidence threshold, it applies adaptive soft weighting to determine the contribution of unlabeled samples to the training objective.

The experiments investigate three labeled-data proportions:

- **5% labeled data** — extreme label scarcity.
- **10% labeled data** — low-label setting.
- **20% labeled data** — increased label availability.

The labeled subset also incorporates a controlled class-retention strategy to simulate imbalance during training.

### 4. Model Architecture

**DenseNet-121** is used as the feature-extraction backbone, initialized with pretrained ImageNet weights. Its final classification layer is modified to produce three output classes.

The training process combines:

- Supervised loss from labeled images.
- Unsupervised loss from pseudo-labeled images.
- Weak and strong image augmentation.
- SoftMatch confidence-based weighting.
- Optional Uniform Alignment.

### 5. Uniform Alignment Ablation

Two configurations are compared:

| Configuration | Description |
|---|---|
| SoftMatch + UA | Uses Uniform Alignment when calculating confidence-based sample weights. |
| SoftMatch without UA | Calculates sample weights from the original prediction confidence. |

Both configurations are evaluated under the same labeled-data scenarios to investigate the effect of Uniform Alignment.

### 6. Image Augmentation

The preprocessing pipeline resizes images to 224 × 224 pixels and normalizes them using ImageNet statistics.

The training pipeline uses weak and strong augmentation to support consistency learning while keeping transformations relatively moderate for medical images.

### 7. Evaluation

Model performance is assessed using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Class-specific recall
- Confusion Matrix
- Multiclass ROC-AUC using One-vs-Rest
- Grad-CAM for visual interpretability

## 📊 Experimental Results

The following results are reported in the research paper.

| Labeled data | Uniform Alignment | Accuracy | Macro F1-score | Pneumonia Recall |
|---|---|---:|---:|---:|
| 5% | Enabled | 91.68% | 91.72% | 78.06% |
| 5% | Disabled | 90.73% | 90.82% | 75.28% |
| 10% | Enabled | 94.46% | 94.58% | 88.61% |
| 10% | Disabled | 94.17% | 94.31% | 84.72% |
| 20% | Enabled | 93.31% | 93.37% | 85.28% |
| 20% | Disabled | **95.79%** | **95.89%** | **91.94%** |

### Key Findings

- **Best overall performance:** The 20% labeled-data scenario without Uniform Alignment achieved 95.79% accuracy and a 95.89% macro F1-score.
- **Best UA configuration:** Uniform Alignment achieved its strongest overall results at 10% labeled data, with 94.46% accuracy and a 94.58% macro F1-score.
- **Benefits under label scarcity:** At 5% labeled data, enabling UA improved accuracy, macro F1-score, and Pneumonia recall compared with the configuration without UA.
- **Context-dependent effectiveness:** Uniform Alignment did not consistently improve performance as labeled-data availability increased.

These findings highlight the importance of evaluating semi-supervised learning methods across multiple label regimes rather than relying on a single experimental setting.

## 🛠️ Technology Stack

- **Python** — implementation and experimentation
- **PyTorch** — deep learning framework
- **DenseNet-121** — pretrained classification backbone
- **Google Colab** — notebook-based development and execution
- **Grad-CAM** — visual model interpretability
- **Scikit-learn** — classification evaluation metrics

The exact dependency versions should be documented based on the environment used to run the notebooks.

## 📂 Repository Structure

```text
Klasifikasi-Citra-C-Xray-Mengunakan-Metode-Softmatch/
│
├── sofmatch xdensent121 fase 2.ipynb
├── baseline densent121
├── README.md
└── .gitignore
```

The repository may also contain experiment outputs or additional files, depending on which research artifacts are published.

The chest X-ray dataset is not included in this repository.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/MNabildawami/Klasifikasi-Citra-C-Xray-Mengunakan-Metode-Softmatch.git
cd Klasifikasi-Citra-C-Xray-Mengunakan-Metode-Softmatch
```

### 2. Open the Notebook

Open the notebook in Google Colab:

[**Open in Google Colab**](https://colab.research.google.com/)

Alternatively, open the `.ipynb` file locally using Jupyter Notebook.

### 3. Prepare the Dataset

Obtain the dataset from its original source on Mendeley Data and configure the dataset path according to the notebook.

The dataset must be available in the expected directory structure before training or evaluation can run.

### 4. Run the Experiments

Execute the notebook cells in order, including:

1. Environment setup and imports.
2. Dataset loading and preprocessing.
3. Stratified data splitting.
4. Labeled and unlabeled subset preparation.
5. Model initialization.
6. SoftMatch training.
7. Evaluation and result visualization.

Exact execution requirements depend on the code and available compute resources.

## 🔬 Research Contribution

This project focuses on the intersection of semi-supervised learning and medical image classification, particularly the challenge of learning from limited labeled data.

Its experimental contributions include:

- A SoftMatch–DenseNet-121 implementation for three-class chest X-ray classification.
- Comparative experiments across 5%, 10%, and 20% labeled-data scenarios.
- An ablation study of Uniform Alignment.
- Evaluation focused on both aggregate performance and minority-class recall.
- Grad-CAM-based visual analysis of model predictions.

## ⚠️ Limitations and Disclaimer

This repository is intended for **academic research and experimentation**.

- The reported metrics represent results from the experimental dataset and evaluation protocol described in the accompanying research.
- Performance on other datasets or clinical populations may differ.
- Grad-CAM visualizations provide model interpretability cues and do not establish clinical correctness.
- This project is not a certified medical device and must not be used as a substitute for professional medical diagnosis.

## 👨‍💻 Author

**M. Nabil Dawami**

Informatics Engineering  
Faculty of Science and Technology  
Universitas Islam Negeri Sultan Syarif Kasim Riau, Indonesia

**Research topic:** Semi-Supervised Learning for Multiclass Chest X-Ray Classification

## 📄 Publication and Research Context

This repository accompanies research on multiclass chest X-ray classification using SoftMatch under limited labeled-data conditions.

The manuscript reports the experimental methodology, ablation results, classification metrics, and Grad-CAM analysis. Publication metadata and a final publication link can be added when the official publication details are available.

---

<p align="center">
  <strong>Exploring robust medical image classification with limited labeled data.</strong>
</p>
