"""Download and organize real clinical retinal fundus photos and non-retinal alert test images."""
import os
import sys
import urllib.request
import urllib.parse
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "test_samples")
FUNDUS_DIR = os.path.join(SAMPLES_DIR, "01_retinal_fundus_scans")
NON_RETINA_DIR = os.path.join(SAMPLES_DIR, "02_non_retinal_alert_tests")

os.makedirs(FUNDUS_DIR, exist_ok=True)
os.makedirs(NON_RETINA_DIR, exist_ok=True)

# Wikimedia items to download
DOWNLOAD_MANIFEST = [
    # --- Retinal Fundus Images ---
    {
        "target_name": "retina_normal_healthy_EDA06.jpg",
        "wiki_title": "File:Fundus photograph-normal retina EDA06.JPG",
        "folder": FUNDUS_DIR,
        "description": "Normal healthy human retina (National Eye Institute / NIH). Grade 0: No Diabetic Retinopathy."
    },
    {
        "target_name": "retina_normal_macula_OD.jpg",
        "wiki_title": "File:Fundus photo Retina OD.jpg",
        "folder": FUNDUS_DIR,
        "description": "Normal fundus photo of right eye showing clear optic disc and fovea. Grade 0: No DR."
    },
    {
        "target_name": "retina_normal_healthy_OS.jpg",
        "wiki_title": "File:Fundus photograph Retina OS.jpg",
        "folder": FUNDUS_DIR,
        "description": "Normal left eye retinal fundus photograph showing vascular tree branching."
    },
    {
        "target_name": "retina_normal_field_scan.jpg",
        "wiki_title": "File:Right eye fundus photograph.jpg",
        "folder": FUNDUS_DIR,
        "description": "Clear clinical fundus scan of the posterior pole with crisp retinal vessels."
    },
    {
        "target_name": "retina_dr_mild_early_EDA03.jpg",
        "wiki_title": "File:Fundus retinopathy EDA03.JPG",
        "folder": FUNDUS_DIR,
        "description": "Early Non-Proliferative Diabetic Retinopathy (NPDR) with scattered microaneurysms and dot hemorrhages."
    },
    {
        "target_name": "retina_dr_moderate_maculopathy.png",
        "wiki_title": "File:Fundus - diabetic retinopathy.png",
        "folder": FUNDUS_DIR,
        "description": "Diabetic Retinopathy with exudates, blot hemorrhages, and macular edema signs."
    },
    {
        "target_name": "retina_dr_severe_cotton_wool.png",
        "wiki_title": "File:A sample retinal image with cotton wool spots and hemorrhages.png",
        "folder": FUNDUS_DIR,
        "description": "Severe NPDR with prominent cotton wool spots (nerve fiber layer infarcts) and hemorrhages."
    },
    {
        "target_name": "retina_dr_proliferative_pdr_EDA01.jpg",
        "wiki_title": "File:Fundus Proliferative retinopathy EDA01.JPG",
        "folder": FUNDUS_DIR,
        "description": "Proliferative Diabetic Retinopathy (PDR) with active neovascularization at the disc and fibrous proliferation."
    },

    # --- Non-Retinal Images for Alert Testing ---
    {
        "target_name": "non_retina_chest_radiograph_xray.png",
        "wiki_title": "File:Chest Xray PA 3-8-2010.png",
        "folder": NON_RETINA_DIR,
        "description": "Grayscale PA chest radiograph. Tests Stage 1 monochrome/grayscale rejection alert."
    },
    {
        "target_name": "non_retina_anterior_cataract_eye.png",
        "wiki_title": "File:Cataract in human eye.png",
        "folder": NON_RETINA_DIR,
        "description": "External anterior eye photo of cataract lens (not a fundus photo). Tests non-retina rejection."
    },
]

def fetch_wiki_urls(titles):
    params = urllib.parse.urlencode({
        "action": "query",
        "titles": "|".join(titles),
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json"
    })
    url = f"https://commons.wikimedia.org/w/api.php?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "NetraXAISampleDownloader/1.0 (medical project)"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
        urls = {}
        pages = data.get("query", {}).get("pages", {})
        for pid, page in pages.items():
            t = page.get("title", "")
            ii = page.get("imageinfo", [])
            if ii:
                urls[t] = ii[0]["url"]
        return urls

def download_image(url, out_path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = resp.read()
    with open(out_path, "wb") as f:
        f.write(data)
    # Validate and resize if extremely huge to optimize testing load speed (max dimension 1200)
    try:
        with Image.open(out_path) as im:
            im = im.convert("RGB")
            if max(im.size) > 1200:
                im.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                im.save(out_path, quality=92)
    except Exception as e:
        print(f"Warning normalizing {out_path}: {e}")

def create_synthetic_test_cases():
    """Create additional non-retinal testing samples (prescription doc, skin, landscape)."""
    # 1. Scanned Medical Prescription / Invoice Document
    doc_path = os.path.join(NON_RETINA_DIR, "non_retina_medical_prescription_doc.png")
    doc_img = Image.new("RGB", (700, 900), (250, 250, 252))
    d = ImageDraw.Draw(doc_img)
    # Draw clinic letterhead
    d.rectangle([30, 30, 670, 90], fill=(235, 240, 250), outline=(200, 210, 230))
    d.text((50, 45), "METRO CARE EYE CLINIC & RESEARCH CENTER", fill=(20, 50, 100))
    d.text((50, 65), "Prescription & Consultation Summary · Patient #MC-8819", fill=(100, 110, 130))
    # Simulated lines of text
    y = 120
    for i in range(18):
        d.line([50, y, 650, y], fill=(230, 230, 235), width=1)
        w_len = np.random.randint(200, 580)
        d.rectangle([50, y + 6, 50 + w_len, y + 16], fill=(160, 165, 175))
        y += 35
    doc_img.save(doc_path)
    print("Created synthetic test doc:", doc_path)

    # 2. Human Skin / Selfie Photo Sample
    skin_path = os.path.join(NON_RETINA_DIR, "non_retina_skin_selfie_sample.jpg")
    skin_arr = np.zeros((600, 600, 3), dtype=np.uint8)
    for y_idx in range(600):
        factor = 1.0 + (y_idx / 1200.0)
        skin_arr[y_idx, :, 0] = min(255, int(235 * factor))  # R
        skin_arr[y_idx, :, 1] = min(255, int(185 * factor))  # G
        skin_arr[y_idx, :, 2] = min(255, int(155 * factor))  # B
    noise = np.random.normal(0, 3, (600, 600, 3))
    skin_arr = np.clip(skin_arr + noise, 0, 255).astype(np.uint8)
    skin_img = Image.fromarray(skin_arr)
    skin_img.save(skin_path, quality=95)
    print("Created synthetic skin sample:", skin_path)

    # 3. Nature / Blue Sky Photo
    sky_path = os.path.join(NON_RETINA_DIR, "non_retina_blue_sky_nature.jpg")
    sky_arr = np.zeros((600, 600, 3), dtype=np.uint8)
    for y_idx in range(600):
        r_val = int(40 + (y_idx / 600) * 80)
        g_val = int(120 + (y_idx / 600) * 70)
        b_val = int(220 + (y_idx / 600) * 30)
        sky_arr[y_idx, :] = [r_val, g_val, b_val]
    sky_img = Image.fromarray(sky_arr)
    sky_img.save(sky_path, quality=95)
    print("Created synthetic blue sky sample:", sky_path)

def main():
    print("Starting download of test samples...")
    titles = [item["wiki_title"] for item in DOWNLOAD_MANIFEST]
    urls = fetch_wiki_urls(titles)

    downloaded = 0
    for item in DOWNLOAD_MANIFEST:
        t = item["wiki_title"]
        target_path = os.path.join(item["folder"], item["target_name"])
        if t in urls:
            url = urls[t]
            print(f"Downloading {item['target_name']} from {url[:60]}...")
            try:
                download_image(url, target_path)
                downloaded += 1
                print(f"  [OK] Saved to {item['target_name']}")
            except Exception as e:
                print(f"  [FAIL] Failed to download {item['target_name']}: {e}")
        else:
            print(f"  [FAIL] URL not found for {t}")

    print("Creating additional non-retina synthetic test files...")
    create_synthetic_test_cases()

    # Generate Catalog README
    readme_path = os.path.join(SAMPLES_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# NetraXAI Test Sample Library\n\n")
        f.write("This directory contains real clinical fundus photographs and out-of-distribution (OOD) test images for evaluating NetraXAI.\n\n")
        f.write("## 1. Real Retinal Fundus Scans (`01_retinal_fundus_scans/`)\n")
        f.write("Upload these images directly in the NetraXAI web interface to test AI grading, Grad-CAM attention heatmaps, and biomarker detection:\n\n")
        f.write("| Filename | Expected Diagnosis | Clinical Characteristics |\n")
        f.write("|---|---|---|\n")
        f.write("| `retina_normal_healthy_EDA06.jpg` | **Level 0 (No DR)** | Clear macula, sharp optic nerve head, healthy vascular tree |\n")
        f.write("| `retina_normal_macula_OD.jpg` | **Level 0 (No DR)** | Normal right eye fundus photograph (National Eye Institute) |\n")
        f.write("| `retina_normal_healthy_OS.jpg` | **Level 0 (No DR)** | Normal left eye fundus photograph |\n")
        f.write("| `retina_normal_field_scan.jpg` | **Level 0 (No DR)** | Healthy posterior pole with normal vessel branching |\n")
        f.write("| `retina_dr_mild_early_EDA03.jpg` | **Level 1 (Mild NPDR)** | Scattered microaneurysms and early dot/blot hemorrhages |\n")
        f.write("| `retina_dr_moderate_maculopathy.png` | **Level 2 (Moderate NPDR)** | Lipid exudate deposits, intraretinal hemorrhages, maculopathy |\n")
        f.write("| `retina_dr_severe_cotton_wool.png` | **Level 3 (Severe NPDR)** | Prominent cotton wool spots (nerve fiber infarcts) |\n")
        f.write("| `retina_dr_proliferative_pdr_EDA01.jpg` | **Level 4 (Proliferative DR)** | Neovascularization at disc (NVD), abnormal new fragile vessels |\n\n")
        f.write("## 2. Non-Retinal Alert Testing (`02_non_retinal_alert_tests/`)\n")
        f.write("Upload these negative samples to test the **Out-of-Distribution Quality Gate Alert Dialog**:\n\n")
        f.write("| Filename | Content Type | Expected NetraXAI Response |\n")
        f.write("|---|---|---|\n")
        f.write("| `non_retina_chest_radiograph_xray.png` | PA Chest X-ray | ⚠️ Rejection Alert (Grayscale / monochrome scan) |\n")
        f.write("| `non_retina_medical_prescription_doc.png` | Scanned Paper Document | ⚠️ Rejection Alert (Monochrome document / scan) |\n")
        f.write("| `non_retina_skin_selfie_sample.jpg` | Human Skin / Selfie | ⚠️ Rejection Alert (No retinal vessels or optic disc) |\n")
        f.write("| `non_retina_blue_sky_nature.jpg` | Outdoor Landscape / Sky | ⚠️ Rejection Alert (Incompatible blue color spectrum) |\n")
        f.write("| `non_retina_anterior_cataract_eye.png` | Anterior Eye / Cataract | ⚠️ Rejection Alert (Not an ocular fundus photograph) |\n")

    print(f"\nCompleted! Downloaded {downloaded} real images + created synthetic tests and README catalog.")

if __name__ == "__main__":
    main()
