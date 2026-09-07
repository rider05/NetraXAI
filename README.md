# NetraXAI — Explainable Retinal AI Diagnostic Workstation

> **SIH 2026 · Problem Statement 26038 · MathWorks**  
> Explainable AI for Diabetic Retinopathy Screening in Rural India

NetraXAI is an end-to-end, trustworthy, quality-aware screening system and diagnostic cockpit engineered for mobile retinal screening camps and rural tele-ophthalmology networks.

---

## 🌟 Key Features

- **🛡️ Retinal Capture Quality Gate (Stage 1)**: Automated gradability score (0–100) assessing Focus/Blur, Illumination, Contrast, Field of View (FOV), and Artifacts before downstream grading.
- **⚡ CLAHE-Style Image Enhancement (Stage 2)**: Color-preserving local contrast normalisation for borderline capture qualities.
- **🔍 Retinal Biomarker & Lesion Extraction (Stage 3)**: Automatic localization and counting of Microaneurysms, Haemorrhages, Hard Exudates, and Neovascularisation (NV).
- **📊 Calibrated ICDR Severity Classification (Stage 4)**: 5-level clinical classification (No DR, Mild NPDR, Moderate NPDR, Severe NPDR, Proliferative DR) with calibrated confidence metrics.
- **👁️ Evidence Overlay & Explainability (Stage 5)**: High-resolution visual evidence overlays and interactive split before/after wipe slider.
- **📑 Clinical Referral Slip Generation**: Instant clinical discharge and referral documentation with print formatting.
- **📈 Session Epidemiology & Camp Audit Trail**: Comprehensive aggregate statistics with one-click CSV export.

---

## 🚀 Quick Start

### 1. Requirements

- Python 3.9+
- Dependencies: `pillow`, `numpy`

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
├── pipeline.py           # 5-stage explainable screening pipeline (pure NumPy/PIL)
├── generator.py          # Synthetic fundus generator for reproducibility
├── run.bat               # Windows launcher script
├── requirements.txt      # Python dependencies
├── static/
│   └── index.html        # Clinical diagnostic workstation single-page app
└── README.md
```

---

## ⚖️ Clinical Disclaimer

This prototype is intended for demonstration, technical evaluation, and screening triage workflow validation. Definitive medical diagnosis and surgical interventions must always be evaluated by certified ophthalmologists.
