# NetraXAI Test Sample Library

This directory contains real clinical fundus photographs and out-of-distribution (OOD) test images for evaluating NetraXAI.

## 1. Real Retinal Fundus Scans (`01_retinal_fundus_scans/`)
Upload these images directly in the NetraXAI web interface to test AI grading, Grad-CAM attention heatmaps, and biomarker detection:

| Filename | Expected Diagnosis | Clinical Characteristics |
|---|---|---|
| `retina_normal_healthy_EDA06.jpg` | **Level 0 (No DR)** | Clear macula, sharp optic nerve head, healthy vascular tree |
| `retina_normal_macula_OD.jpg` | **Level 0 (No DR)** | Normal right eye fundus photograph (National Eye Institute) |
| `retina_normal_healthy_OS.jpg` | **Level 0 (No DR)** | Normal left eye fundus photograph |
| `retina_normal_field_scan.jpg` | **Level 0 (No DR)** | Healthy posterior pole with normal vessel branching |
| `retina_dr_mild_early_EDA03.jpg` | **Level 1 (Mild NPDR)** | Scattered microaneurysms and early dot/blot hemorrhages |
| `retina_dr_moderate_maculopathy.png` | **Level 2 (Moderate NPDR)** | Lipid exudate deposits, intraretinal hemorrhages, maculopathy |
| `retina_dr_severe_cotton_wool.png` | **Level 3 (Severe NPDR)** | Prominent cotton wool spots (nerve fiber infarcts) |
| `retina_dr_proliferative_pdr_EDA01.jpg` | **Level 4 (Proliferative DR)** | Neovascularization at disc (NVD), abnormal new fragile vessels |

## 2. Non-Retinal Alert Testing (`02_non_retinal_alert_tests/`)
Upload these negative samples to test the **Out-of-Distribution Quality Gate Alert Dialog**:

| Filename | Content Type | Expected NetraXAI Response |
|---|---|---|
| `non_retina_chest_radiograph_xray.png` | PA Chest X-ray | ⚠️ Rejection Alert (Grayscale / monochrome scan) |
| `non_retina_medical_prescription_doc.png` | Scanned Paper Document | ⚠️ Rejection Alert (Monochrome document / scan) |
| `non_retina_skin_selfie_sample.jpg` | Human Skin / Selfie | ⚠️ Rejection Alert (No retinal vessels or optic disc) |
| `non_retina_blue_sky_nature.jpg` | Outdoor Landscape / Sky | ⚠️ Rejection Alert (Incompatible blue color spectrum) |
| `non_retina_anterior_cataract_eye.png` | Anterior Eye / Cataract | ⚠️ Rejection Alert (Not an ocular fundus photograph) |
