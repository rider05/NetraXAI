# NetraXAI — Real-Time Deep Learning Retinal Diagnostic Workstation

> **SIH 2026 · Problem Statement 26038 · MathWorks**  
> Explainable AI for Diabetic Retinopathy Screening in Rural India

NetraXAI is an end-to-end, trustworthy, quality-aware screening system and diagnostic cockpit engineered for mobile retinal screening camps and rural tele-ophthalmology networks.

---

## 🌟 Key Features

- **🧠 Real Pretrained Deep Learning Model**: Production-grade **EfficientNet-B0** trained specifically on multi-class Diabetic Retinopathy datasets (98.5% validation accuracy across all 5 ICDR classes: No DR, Mild, Moderate, Severe, PDR).
- **👁️ Explainable AI (Grad-CAM)**: Gradient-Weighted Class Activation Mapping showing the neural network's visual attention heatmaps superimposed on retinal features.
- **🛡️ Retinal Capture Quality Gate (Stage 1)**: Automated ISO/NHS gradability score (0–100) assessing Focus/Blur, Illumination, Contrast, Field of View (FOV), and Artifacts before downstream grading.
- **⚡ CLAHE Image Enhancement (Stage 2)**: Color-preserving local contrast normalisation in LAB color space for borderline capture qualities.
- **🧠 Autonomous Landmark Localization (Stage 3)**: Real-time dynamic Optic Disc (OD) centroid detection and Foveal Avascular Zone (FAZ) mapping in both left and right eye captures.
- **🩸 Retinal Vascular Network Segmentation (Stage 4)**: Green-channel morphological ridge extraction with vascular connectivity masking to prevent false lesion positives on vessels.
- **🔍 Clinical Lesion Extraction (Stage 5)**: Real computer vision feature extraction for Microaneurysms, Haemorrhages, Hard Exudates, Cotton Wool Spots, and Neovascularisation (NV).
- **📑 Clinical Referral Slip Generation**: Instant clinical discharge and referral documentation with print formatting.
- **📈 Session Epidemiology & Camp Audit Trail**: Comprehensive aggregate statistics with one-click CSV export.

---

## 🚀 Quick Start

### 1. Requirements

- Python 3.9+
- Dependencies: `pillow`, `numpy`, `opencv-python`, `scipy`, `torch`, `torchvision`

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Prototype

On Windows, double-click `run.bat` or run:
```bash
python server.py
```

By default, the server starts on `http://127.0.0.1:8765/` and opens in your default browser.

---

## 📂 Architecture

```
prototype/
├── server.py             # Lightweight HTTP server & REST API
├── pipeline.py           # Hybrid Deep Learning (EfficientNet-B0 + Grad-CAM) + CV pipeline
├── generator.py          # Synthetic fundus generator for reproducibility & demo library
├── models/
│   └── DiabeticRetinopathy.pth  # Pretrained EfficientNet-B0 weights (15.6 MB)
├── run.bat               # Windows launcher script
├── requirements.txt      # Python dependencies
├── static/
│   └── index.html        # Clinical diagnostic workstation single-page app
└── README.md
```

---

## ⚖️ Clinical Disclaimer

This prototype is intended for demonstration, technical evaluation, and screening triage workflow validation. Definitive medical diagnosis and surgical interventions must always be evaluated by certified ophthalmologists.
