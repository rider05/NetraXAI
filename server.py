"""NetraXAI prototype server — pure stdlib (http.server + PIL + numpy).

Endpoints
    GET  /api/samples             list of demo sample cards (with thumbnails)
    GET  /api/image/<sample_id>   full-resolution sample fundus PNG
    POST /api/analyze             run the full pipeline (quality→drive grading→evidence)
    GET  /                        single-page application
"""
import base64
import io
import json
import os
import re
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

from PIL import Image

from generator import make_fundus
from pipeline import run_pipeline

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")
GRADE_LABELS = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
QUALITY_NAMES = {0: "good", 1: "borderline", 2: "poor"}

NAMES = {
    (0, 0): ("RH-1021", "Sunita Devi"),
    (0, 1): ("RH-1022", "Ramesh Yadav"),
    (0, 2): ("RH-1023", "Meena Kumari"),
    (1, 0): ("RH-1024", "Dinesh Kumar"),
    (1, 1): ("RH-1025", "Sarita Devi"),
    (1, 2): ("RH-1026", "Ashok Sharma"),
    (2, 0): ("RH-1027", "Geeta Rani"),
    (2, 1): ("RH-1028", "Mohan Lal"),
    (2, 2): ("RH-1029", "Kavita Singh"),
    (3, 0): ("RH-1030", "Rajesh Gupta"),
    (3, 1): ("RH-1031", "Poonam Yadav"),
    (3, 2): ("RH-1032", "Vijay Pal"),
    (4, 0): ("RH-1033", "Anita Kumari"),
    (4, 1): ("RH-1034", "Suresh Chandra"),
    (4, 2): ("RH-1035", "Nirmala Devi"),
}


def _png_b64(img: Image.Image, max_dim: int) -> str:
    img = img.copy()
    img.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class _Library:
    """Demo sample library (built once at startup, immutable after)."""

    def __init__(self):
        self._lock = Lock()
        self._samples = {}
        self._cards = []
        for grade in range(5):
            for q in range(3):
                sid = "s{}_{}".format(grade, q)
                img = make_fundus(grade, q, seed=42)
                pid, name = NAMES[(grade, q)]
                thumb = Image.new("RGB", (60, 60), (16, 32, 56))
                try:
                    thumb = img.copy()
                    thumb.thumbnail((240, 240))
                    tbuf = io.BytesIO()
                    thumb.save(tbuf, "PNG")
                    thumb_b64 = "data:image/png;base64," + base64.b64encode(tbuf.getvalue()).decode("ascii")
                except Exception:
                    thumb_b64 = ""
                entry = {
                    "id": sid,
                    "grade": grade,
                    "grade_label": GRADE_LABELS[grade],
                    "quality": q,
                    "quality_label": QUALITY_NAMES[q].capitalize(),
                    "patient_id": pid,
                    "name": name,
                    "thumb": thumb_b64,
                    "image": img,
                }
                self._samples[sid] = entry
                self._cards.append({
                    "id": sid,
                    "grade": grade,
                    "grade_label": GRADE_LABELS[grade],
                    "quality": q,
                    "quality_label": QUALITY_NAMES[q].capitalize(),
                    "patient_id": pid,
                    "name": name,
                    "thumb": thumb_b64,
                })

    def cards(self):
        return self._cards

    def get(self, sid):
        return self._samples.get(sid)

    def size(self):
        return len(self._samples)


LIB = _Library()


def _serve_analysis(img: Image.Image, patient_id: str, name: str, source: str) -> dict:
    t0 = time.time()
    res = run_pipeline(img, patient_id, name)
    elapsed = round((time.time() - t0) * 1000)

    images = {
        "original": _png_b64(img, 760),
        "explanation": _png_b64(res["explanation"], 760),
    }
    if res["enhanced"] is not None:
        images["enhanced"] = _png_b64(res["enhanced"], 760)

    rpt = res["report"]
    return {
        "source": source,
        "patient_id": patient_id,
        "name": name,
        "quality": res["quality"],
        "rejected": res["rejected"],
        "enhanced": res["enhanced"] is not None,
        "grade": res["grade"],
        "lesions": res["lesions"],
        "images": images,
        "report": rpt,
        "latency_ms": elapsed,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _bad(self, msg):
        self._send(400, {"error": msg})

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self._send(200, {"ok": True, "app": "NetraXAI", "samples": LIB.size()})
            return

        if path == "/api/samples":
            self._send(200, {"samples": LIB.cards()})
            return

        m = re.match(r"^/api/image/([A-Za-z0-9_]+)$", path)
        if m:
            entry = LIB.get(m.group(1))
            if entry is None:
                self._bad("unknown sample id")
                return
            img = entry["image"].copy()
            img.thumbnail((900, 900))
            buf = io.BytesIO()
            img.save(buf, "PNG")
            data = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/" or path == "/index.html":
            self._serve_static("index.html")
            return

        self._send(404, {"error": "not found"})

    def _serve_static(self, name):
        fp = os.path.join(STATIC, name)
        if not os.path.isfile(fp):
            self._send(404, {"error": "missing static"})
            return
        with open(fp, "rb") as fh:
            data = fh.read()
        ctype = "text/html; charset=utf-8" if name.endswith(".html") else "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path != "/api/analyze":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            self._bad("invalid JSON body")
            return

        source = body.get("source", "sample")

        if source == "sample":
            sid = body.get("id", "")
            entry = LIB.get(sid)
            if entry is None:
                self._bad("unknown sample id")
                return
            with LIB._lock:
                res = _serve_analysis(entry["image"], entry["patient_id"], entry["name"], "sample")
                res["sample_id"] = sid
            self._send(200, res)
            return

        if source == "upload":
            b64 = body.get("image_b64", "")
            try:
                payload = b64.split(",", 1)[1] if "," in b64 else b64
                img = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
            except Exception:
                self._bad("could not decode uploaded image")
                return
            res = _serve_analysis(img, body.get("patient_id", "UPL-001"), body.get("name", "Uploaded image"), "upload")
            self._send(200, res)
            return

        self._bad("unknown source")


def main():
    import sys
    port = 8765
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    handler = Handler
    server = None
    for p in range(port, port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), handler)
            port = p
            break
        except OSError:
            continue
    if server is None:
        print("No free port found (tried {}-{})".format(port, port + 20))
        return
    url = "http://127.0.0.1:{}/".format(port)
    print("NetraXAI prototype listening on", url)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()