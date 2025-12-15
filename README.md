

# An Effective Multimodal Framework for Sickle Cell Anemia Detection using Transfer Learning and Explainable AI


![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![AI](https://img.shields.io/badge/AI-Explainable-green)

---

## 🩸 Overview

This project presents a multimodal, explainable AI framework for detecting **Sickle Cell Anemia (SCA)** using both **clinical blood report data** and **blood smear images**. It leverages **transfer learning** (EfficientNet-B0), **Support Vector Machine (SVM)** classifiers, and **Explainable AI** tools like **Grad-CAM** and **LIME** to provide transparent and accurate diagnosis.

> 🔍 The framework is designed for clinicians and researchers to achieve faster, more interpretable, and scalable diagnosis of SCA—especially in low-resource settings.

---

<img width="2752" height="1536" alt="unnamed" src="https://github.com/user-attachments/assets/8fe55895-5559-45cf-895c-53fb8234de66" />

---

## 🎯 Objectives

* ✅ Apply **image augmentation** techniques to improve model generalization.
* ✅ Develop multimodal models:

  * Image-based CNN using **EfficientNet-B0**
  * Text-based SVM for hematological data
* ✅ Integrate **XAI methods**: Grad-CAM & LIME
* ✅ Create an interactive **Streamlit-based UI** supporting:

  * Text-only input
  * Image-only input
  * Combined (multimodal) input

---

## 🧠 Methodology

### 🔬 1. Image-Based Pipeline

* **Dataset**:

  * **AneRBC Dataset**: 12,000 RBC images
  * **Sickle-Specific Dataset**: 422 positive, 147 negative
* **Transfer Learning Models**:

  * ResNet-18, DenseNet-121, EfficientNet-B0, Custom CNN
* **Best Performer**: `EfficientNet-B0` (99.1% accuracy)
* **Explainability**: `Grad-CAM` visualizations overlaid on input images

### 📊 2. Text-Based Pipeline

* **Clinical Features**:

  * RBC, PCV, MCV, MCH, MCHC, RDW, TLC, Platelet Count, HGB
* **Model**: `SVM`

  * With **SMOTE** oversampling for class balance
  * Achieved 96% accuracy
* **Explainability**: `LIME`-based feature attribution per prediction

### ⚙️ 3. Multimodal Ensemble

* Combines image and text predictions via **soft voting**
* Provides **combined diagnostic result**
* UI supports all 3 modes (Text, Image, Both)

---

## 🖥️ User Interface (Streamlit)

| Input Type   | Models Used        | Explainability  |
| ------------ | ------------------ | --------------- |
| Text Only    | SVM                | LIME            |
| Image Only   | EfficientNet-B0    | Grad-CAM        |
| Text + Image | SVM + EfficientNet | LIME + Grad-CAM |

---

## 🧪 Results

| Model           | Accuracy | F1-Score | Recall | Remarks              |
| --------------- | -------- | -------- | ------ | -------------------- |
| EfficientNet-B0 | 99.1%    | 0.99     | 1.00   | Best image model     |
| SVM             | 96.0%    | 0.96     | 0.94   | Best text model      |
| DenseNet-121    | 88.5%    | 0.88     | 0.87   | Moderate performance |
| Custom CNN      | 81.0%    | 0.79     | 0.75   | Poor specificity     |

---

## 🧰 Tech Stack

| Layer          | Technologies Used         |
| -------------- | ------------------------- |
| UI             | `Python`, `Streamlit`     |
| ML Models      | `scikit-learn`, `PyTorch` |
| Explainability | `LIME`, `grad-cam`        |
| Image I/O      | `OpenCV`, `Pillow`        |
| Packaging      | `venv`, `pip`, `joblib`   |

---

## 📁 Folder Structure

```
.
├── app.py                      # Streamlit UI
├── models/
│   ├── efficientnet_b0.pth     # Image model
│   ├── svm_model.pkl           # Text model
│   └── scaler.pkl              # Feature scaler
├── utils/
│   ├── predict.py              # Prediction logic
│   ├── lime_explainer.py       # LIME integration
│   └── gradcam_utils.py        # Grad-CAM logic
├── requirements.txt
└── README.md
```

---

## 🔄 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sickle-cell-detection-ai.git
cd sickle-cell-detection-ai
```

### 2. Create Environment & Install Dependencies

```bash
python -m venv env
source env/bin/activate   # or env\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Launch the Web App

```bash
streamlit run app.py
```

---

## 🧪 Testing

* **Unit tests** included for model predictions
* UI tested with:

  * Real blood smear images
  * Synthetic numerical inputs
* Verified LIME/Grad-CAM explanations for multiple edge cases

---

## 🔮 Future Work

* 🧬 Add genomic and patient history data for richer diagnostics
* 📱 Real-time mobile deployment
* 🔁 Integration with hospital EHR systems
* 📈 Use SHAP for global feature attribution
* 🗣️ Add multilingual voice-based UI for broader accessibility
  
---

For better understanding : https://code2tutorial.com/tutorial/c4807d57-fbff-4330-8626-53eb61e409a2/index.md
