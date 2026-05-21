# ☀️ Solar Panel Defect Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-orange)
![Deep Learning](https://img.shields.io/badge/DeepLearning-green)
![Model](https://img.shields.io/badge/MobileNetV2-lightgrey)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

Python • TensorFlow • Deep Learning • MobileNetV2 • Streamlit

---

## 📌 About

AI-powered Solar Panel Defect Detection system built using MobileNetV2 and Streamlit. The model classifies solar panel images into multiple defect categories with confidence scores. This project demonstrates an end-to-end deep learning pipeline, from data preprocessing and training to deployment.

---

## 🚀 Features

* 📤 Upload solar panel image
* 🔍 Predict defect type
* 🖱️ Predict button

---

## 🧠 Model Details

* **Model:** MobileNetV2
* **Type:** Transfer Learning
* **Framework:** TensorFlow / Keras
* **Input Size:** 224 × 224

### Classes:

* Bird Drop Panel
* Clean Panel
* Dust Panel
* Electrical Damage
* Physical Damage
* Snow Covered

---

## 📊 Model Performance

* ✅ Training Accuracy: ~91%
* 🔥 Validation Accuracy: ~85.6%

---

## 📸 Screenshots

### 🔹 Main Dashboard

![Main Dashboard](https://github.com/Joe-in-absentia/Solar_Panel_Defects_detection/blob/6b68673f7b15ad83e68fbbc0a37f060a05edb784/dashboard.png)

### 🔹 Prediction Output

![Main Dashboard](https://github.com/Joe-in-absentia/Solar_Panel_Defects_detection/blob/c3931e351ad14156c0bea110eb8b6023e120bbbe/main.png)

---

## 💡 Tech Stack

* Python
* TensorFlow
* Streamlit
* NumPy
* PIL

---

## 🏁 Conclusion

This project demonstrates how deep learning can automate solar panel defect detection efficiently. By using MobileNetV2 with transfer learning, the model achieves strong accuracy while remaining lightweight.

The Streamlit interface makes the system interactive and user-friendly, showcasing a complete end-to-end AI solution with real-world relevance.

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&pause=1000&color=F5A623&center=true&vCenter=true&width=700&lines=☀️+Solar+Panel+Defect+Detection;AI-Powered+Visual+Inspection+System" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Models](https://img.shields.io/badge/Models-MobileNetV2%20%7C%20ResNet50%20%7C%20EfficientNetB0-lightgrey?style=flat-square)]()
[![Best Accuracy](https://img.shields.io/badge/Best%20Accuracy-85%25-success?style=flat-square)]()

</div>

---

## 📖 Overview

**Solar Panel Defect Detection** is a production-ready deep learning pipeline that automates visual quality inspection of solar panels using **multi-class image classification**. The system benchmarks three state-of-the-art CNN architectures via transfer learning, selects the best-performing model, and serves real-time predictions through an interactive **Streamlit** web application.

> Designed for scalability in real-world solar farm inspection workflows, reducing manual inspection overhead and improving defect detection reliability.

---

## 🎯 Problem Statement

Manual inspection of solar panels is:
- **Time-consuming** — large farms contain thousands of panels
- **Error-prone** — subtle defects like dust and electrical damage are easy to miss
- **Expensive** — requires trained technicians on-site

This system automates defect identification, enabling rapid, accurate, and cost-effective inspection at scale.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📤 **Image Upload** | Upload any solar panel image via the web interface |
| 🔍 **Defect Classification** | Identifies 6 distinct panel conditions |
| 📊 **Confidence Scoring** | Returns per-class probability distribution |
| 🧠 **Multi-Model Study** | Benchmarks MobileNetV2, ResNet50, EfficientNetB0 |
| ⚡ **Efficient Deployment** | Lightweight inference via Streamlit |
| 🖥️ **Interactive Dashboard** | Clean, minimal UI for real-time use |

---

## 🗂️ Defect Classes

```
solar-panel-defects/
├── 🐦 Bird-drop          — Soiling from bird droppings
├── ✅ Clean              — No defects detected
├── 🌫️ Dusty             — Dust accumulation reducing efficiency
├── ⚡ Electrical-damage  — Burnt cells or damaged wiring
├── 💥 Physical-Damage    — Cracks, chips, or structural damage
└── ❄️ Snow-Covered       — Panel covered by snow
```

---

## 🧠 Model Architecture & Training

### Architectures Compared

| Model | Type | Parameters | Status |
|---|---|---|---|
| MobileNetV2 | Lightweight CNN | ~3.4M | Baseline |
| ResNet50 | Deep Residual CNN | ~25.6M | Underperformed |
| **EfficientNetB0** | **Compound Scaled CNN** | **~5.3M** | **✅ Best Model** |

### Training Configuration

```yaml
Framework:        TensorFlow / Keras
Pretrained Weights: ImageNet
Input Resolution: 224 × 224 × 3
Batch Size:       32
Epochs:           18
Optimizer:        Adam
Loss Function:    Categorical Crossentropy
Augmentation:     ImageDataGenerator (flip, zoom, rotation, shear)
```

### Transfer Learning Strategy

```
Pretrained Base (Frozen)
        │
        ▼
  Global Average Pooling
        │
        ▼
  Dense Layer (ReLU)
        │
        ▼
  Dropout (Regularization)
        │
        ▼
  Output Layer — Softmax (6 classes)
```

---

## 📊 Results & Performance

### Accuracy Comparison

```
MobileNetV2   ████████████████░░░░░   76%
ResNet50      ██████████░░░░░░░░░░░   52%
EfficientNetB0 █████████████████░░░░   85% 🏆
```

### EfficientNetB0 — Detailed Metrics (Best Model)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Bird-drop | High | High | Strong |
| Clean | High | High | Strong |
| Dusty | High | High | Strong |
| Electrical-damage | High | High | Strong |
| Physical-Damage | Moderate | ~62% ⚠️ | Moderate |
| Snow-Covered | High | High | Strong |
| **Overall Accuracy** | | | **85%** |

> ⚠️ **Known Limitation:** Physical Damage recall (~62%) is lower, likely due to class imbalance. Addressed in [Future Work](#-future-work).

---

## 🖥️ Screenshots

### Main Dashboard
![Dashboard](https://github.com/Joe-in-absentia/Solar_Panel_Defects_detection/blob/6b68673f7b15ad83e68fbbc0a37f060a05edb784/dashboard.png)

### Prediction Output
![Prediction](https://github.com/Joe-in-absentia/Solar_Panel_Defects_detection/blob/c3931e351ad14156c0bea110eb8b6023e120bbbe/main.png)

---

## ⚙️ How It Works

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Raw Image   │────▶│  Preprocessing   │────▶│   Feature Extraction │
│  (Upload)    │     │  Resize 224×224  │     │   EfficientNetB0     │
└──────────────┘     │  Normalize [0,1] │     │   (ImageNet Weights) │
                     └──────────────────┘     └──────────┬───────────┘
                                                         │
                     ┌──────────────────┐     ┌──────────▼───────────┐
                     │  Output Result   │◀────│  Classification Head │
                     │  Class + Score   │     │  Dense → Softmax     │
                     └──────────────────┘     └──────────────────────┘
```

1. **Preprocessing** — Images resized to 224×224 and normalized; augmentation applied during training
2. **Feature Extraction** — Pretrained EfficientNetB0 base extracts rich visual features
3. **Classification** — Custom dense head maps features to 6 defect classes
4. **Evaluation** — Precision, Recall, F1-Score computed per class; best model selected
5. **Deployment** — EfficientNetB0 served via Streamlit for real-time inference

---

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.10
TensorFlow >= 2.x
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Joe-in-absentia/Solar_Panel_Defects_detection.git
cd Solar_Panel_Defects_detection

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
Solar_Panel_Defects_detection/
│
├── 📂 dataset/                  # Training & validation images
│   ├── Bird-drop/
│   ├── Clean/
│   ├── Dusty/
│   ├── Electrical-damage/
│   ├── Physical-Damage/
│   └── Snow-Covered/
│
├── 📂 models/                   # Saved trained models
│   ├── mobilenetv2.h5
│   ├── resnet50.h5
│   └── efficientnetb0.h5        # ← Deployed model
│
├── 📂 notebooks/                # Training & evaluation notebooks
│   ├── training.ipynb
│   └── model_comparison.ipynb
│
├── app.py                       # Streamlit application
├── requirements.txt
└── README.md
```

---

## 💻 Tech Stack

<div align="center">

| Category | Technology |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow, Keras |
| Model Architectures | MobileNetV2, ResNet50, EfficientNetB0 |
| Data Augmentation | ImageDataGenerator |
| Image Processing | PIL / Pillow |
| Numerical Computing | NumPy |
| Web Application | Streamlit |

</div>

---

## 🔮 Future Work

- [ ] **Fine-tune pretrained layers** — Unfreeze top layers for domain-specific adaptation
- [ ] **Handle class imbalance** — Apply SMOTE or class-weighted loss for Physical Damage
- [ ] **Confusion matrix visualization** — Add in-app analytics dashboard
- [ ] **Grad-CAM heatmaps** — Explainability layer to visualize model attention
- [ ] **Cloud deployment** — Package for AWS / GCP / Azure inference endpoints
- [ ] **Drone integration** — Real-time detection from aerial footage via API

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

<div align="center">

**Joe**

[![GitHub](https://img.shields.io/badge/GitHub-Joe--in--absentia-181717?style=for-the-badge&logo=github)](https://github.com/Joe-in-absentia)

*If you found this project useful, please consider giving it a ⭐ — it helps others discover the work!*

</div>

---

<div align="center">

**Built with ☀️ and deep learning**

</div>


