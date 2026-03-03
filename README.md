# Hierarchical Feature Representation with Multi-level Semantic Alignment for Medical Report Generation

This is the official repository for the paper: **"Hierarchical Feature Representation with Multi-level Semantic Alignment for Medical Report Generation"**.

📢 **Status: Under Review**

The source code and pre-trained models are currently **private** as the paper is undergoing the peer-review process.
* **Code Release**: We are committed to open science. All implementation details, including the model architecture, training scripts, and evaluation protocols, will be made publicly available once the paper is officially accepted for publication.
* **Update Notifications**: You can **Star** or **Watch** this repository to receive notifications regarding the code release.

---

## 📖 Abstract

Medical report generation requires a precise mapping between visual findings and linguistic descriptions. In this work, we propose a novel framework that leverages:
* **Hierarchical Feature Representation**: To capture multi-scale visual information from medical images, ranging from fine-grained lesions to global anatomical structures.
* **Multi-level Semantic Alignment**: To bridge the semantic gap between visual features and medical terminology through a progressive alignment mechanism.

Our approach demonstrates superior performance on benchmark datasets (e.g., IU-Xray and MIMIC-CXR) compared to existing state-of-the-art methods.

---

## 🛠️ Installation & Preparation

### 1. Pre-trained Models
To run this project, please prepare the following pre-trained weights:

* **BioBERT**: Download [BioBERT-v1.1](https://github.com/dmis-lab/biobert) and place it under the `models/` directory.
    * Path: `models/biobert-v1.1/`
* **MedCLIP**: Download the **MedCLIP-ResNet** weights and place them under the `pretrained/` directory.
    * Path: `pretrained/medclip-resnet/`

### 2. Datasets
Please download the datasets and place them in the `data/` folder:
* **IU-Xray**: Available [here](https://iuhealth.org/find-medical-services/x-rays).
* **MIMIC-CXR**: Available [here](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).

---

## 📂 Repository Roadmap

Upon acceptance, this repository will be updated with:
- [ ] **Core Architecture**: Full PyTorch/TensorFlow implementation of the proposed model.
- [ ] **Pre-processing**: Scripts for data cleaning and feature extraction.
- [ ] **Checkpoints**: Pre-trained weights for reproducible results.
- [ ] **Documentation**: A comprehensive guide on environment setup and how to run the training/inference pipeline.

---

## 📧 Contact

For academic inquiries or further information regarding this research, please feel free to:
* Open a **GitHub Issue**.
* Contact us via email: **yuanw0638@gmail.com**
* Or contact the authors directly as listed in the manuscript.
