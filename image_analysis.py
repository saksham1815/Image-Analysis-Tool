import streamlit as st
import cv2
import numpy as np
import json
import mediapipe as mp
import easyocr
import face_recognition
import torch
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image, ImageDraw
from datetime import datetime
from skimage.segmentation import slic
from skimage.color import label2rgb

# -----------------------------------------------------------------------------
# APP CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Advanced Image Analysis", layout="wide")
st.sidebar.title("🔧 Tools & Upload")

# -----------------------------------------------------------------------------
# GLOBALS & INITIALIZATIONS
# -----------------------------------------------------------------------------
reader = easyocr.Reader(['en'], gpu=False)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').eval()
report_log = []

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def log_report(action: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_log.append(f"[{ts}] {action}")

def load_image(uploaded) -> np.ndarray:
    data = uploaded.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def parse_json(uploaded) -> list:
    text = uploaded.read().decode('utf-8')
    return json.loads(text)

# -----------------------------------------------------------------------------
# IMAGE PROCESSING
# -----------------------------------------------------------------------------
def basic_processing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    out = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    log_report("Basic Processing: histogram equalization")
    return out

def semantic_segmentation(img):
    segments = slic(img, n_segments=100, compactness=10, sigma=1)
    seg = label2rgb(segments, img, kind='avg')
    out = (seg * 255).astype(np.uint8)
    log_report("Semantic Segmentation (SLIC)")
    return out

def instance_segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    out = img.copy()
    out[mask == 255] = [0, 255, 0]
    log_report("Instance Segmentation (threshold proxy)")
    return out

def human_segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = img.copy()
    out[gray < 100] = [0, 0, 255]
    log_report("Human Segmentation (threshold proxy)")
    return out

# -----------------------------------------------------------------------------
# FACE ANALYSIS
# -----------------------------------------------------------------------------
def detect_faces(img):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4)
    out = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    log_report(f"Face Detection: {len(faces)} found")
    return out, faces

def extract_landmarks(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    landmarks = face_recognition.face_landmarks(rgb)
    log_report(f"Extracted landmarks for {len(landmarks)} face(s)")
    return landmarks

def save_landmarks_to_json(landmarks, fname="landmarks.json"):
    with open(fname, 'w') as f:
        json.dump(landmarks, f)
    log_report("Landmarks saved to JSON")
    return fname

def sketch_from_json(img, landmarks_json):
    """Draws a sketch from landmarks. Handles both:
      - landmarks_json = [ {feature: [(x,y),...]}, … ]
      - landmarks_json = [ [(x,y),…], … ] or a flat list of pt dicts.
    """
    h, w = img.shape[:2]
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)

    for face in landmarks_json:
        # Case A: dict-of-lists
        if isinstance(face, dict):
            for pts in face.values():
                # pts is a list of (x,y) or [x,y]
                pts2 = [(int(x), int(y)) for x, y in pts]
                if len(pts2) > 1:
                    draw.line(pts2, fill="black", width=1)
                for x, y in pts2:
                    draw.ellipse((x-1, y-1, x+1, y+1), fill="black")
        # Case B: flat list of dicts or lists
        elif isinstance(face, list):
            # detect if elements are dicts with 'x','y'
            if face and isinstance(face[0], dict) and 'x' in face[0]:
                pts2 = [(int(pt['x'] * w), int(pt['y'] * h)) for pt in face]
            else:
                # assume list of [x,y]
                pts2 = [(int(x), int(y)) for x, y in face]
            if len(pts2) > 1:
                draw.line(pts2, fill="black", width=1)
            for x, y in pts2:
                draw.ellipse((x-1, y-1, x+1, y+1), fill="black")
    log_report("Sketch generated from JSON landmarks")
    return np.array(canvas)

def match_and_zoom(main_img, query_img, threshold=0.6):
    rgb1 = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
    rgb2 = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    enc_main = face_recognition.face_encodings(rgb1)
    enc_q = face_recognition.face_encodings(rgb2)
    if not enc_main or not enc_q:
        log_report("Face Match: encoding failed")
        return [], False
    q_enc = enc_q[0]
    locs = face_recognition.face_locations(rgb1)
    encs = face_recognition.face_encodings(rgb1, locs)
    matches = []
    for loc, enc in zip(locs, encs):
        dist = face_recognition.face_distance([q_enc], enc)[0]
        if dist < threshold:
            top, right, bottom, left = loc
            crop = main_img[top:bottom, left:right]
            matches.append((crop, dist))
    log_report(f"Face Match: {len(matches)} match(es) found")
    return matches, bool(matches)

# -----------------------------------------------------------------------------
# TEXT & CLASSIFICATION
# -----------------------------------------------------------------------------
def extract_text(img):
    res = reader.readtext(img)
    txt = "\n".join([f"{r[1]} ({r[2]:.2f})" for r in res])
    log_report("Text Extraction (OCR)")
    return txt

def classify_vit(img):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inputs = feature_extractor(pil, return_tensors="pt")
    outputs = vit_model(**inputs)
    idx = outputs.logits.argmax(-1).item()
    label = vit_model.config.id2label[idx]
    log_report(f"Classification (ViT): {label}")
    return label

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
uploaded = st.sidebar.file_uploader("Upload image", type=["jpg","png","jpeg"])
category = st.sidebar.selectbox("Category", [
    "Basic Processing",
    "Segmentation",
    "Face Analysis",
    "Sketch from JSON",
    "Text Extraction",
    "Classification",
    "Download Report"
])

sub = None
if category == "Segmentation":
    sub = st.sidebar.selectbox("Method", ["Semantic", "Instance", "Human"])
elif category == "Face Analysis":
    sub = st.sidebar.selectbox("Tool", [
        "Detect Faces",
        "Extract & Download Landmarks",
        "Face Match & Zoom",
        "Sketch from Detected Faces"
    ])

# -----------------------------------------------------------------------------
# MAIN UI
# -----------------------------------------------------------------------------
st.title("🔍 Advanced Image Analysis")

if not uploaded:
    st.info("Please upload an image to get started.")
    st.stop()

img = load_image(uploaded)
st.image(img, caption="Input Image", channels="BGR", use_column_width=True)

if category == "Basic Processing":
    out = basic_processing(img)
    st.image(out, caption="Processed Image", channels="BGR")

elif category == "Segmentation":
    if sub == "Semantic":
        out = semantic_segmentation(img)
    elif sub == "Instance":
        out = instance_segmentation(img)
    else:
        out = human_segmentation(img)
    st.image(out, caption=f"{sub} Segmentation", channels="BGR")

elif category == "Face Analysis":
    if sub == "Detect Faces":
        out, faces = detect_faces(img)
        st.image(out, caption="Detected Faces", channels="BGR")
    elif sub == "Extract & Download Landmarks":
        lm = extract_landmarks(img)
        fname = save_landmarks_to_json(lm)
        st.write(lm)
        with open(fname, "r") as f:
            st.download_button("Download Landmarks JSON", f, file_name="landmarks.json")
    elif sub == "Face Match & Zoom":
        qf = st.sidebar.file_uploader("Upload query face", type=["jpg","png","jpeg"], key="qm")
        if qf:
            qimg = load_image(qf)
            matches, ok = match_and_zoom(img, qimg)
            if matches:
                for i, (crop, dist) in enumerate(matches):
                    st.image(crop, caption=f"Match {i+1} (dist={dist:.2f})", channels="BGR")
            else:
                st.warning("No matches found.")
    else:  # Sketch from detected faces
        out, faces = detect_faces(img)
        st.image(out, caption="Detected Faces", channels="BGR")
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            sk = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            st.image(sk, caption="Sketch", channels="BGR")
        log_report("Sketches generated from detected faces")

elif category == "Sketch from JSON":
    jf = st.sidebar.file_uploader("Upload landmarks JSON", type=["json"])
    if jf:
        lm_json = parse_json(jf)
        sk = sketch_from_json(img, lm_json)
        st.image(sk, caption="Sketch from JSON", use_column_width=True)

elif category == "Text Extraction":
    txt = extract_text(img)
    st.text_area("Extracted Text", txt, height=200)

elif category == "Classification":
    lbl = classify_vit(img)
    st.success(f"Predicted Label: {lbl}")

else:  # Download Report
    if report_log:
        st.download_button("Download Report Log", "\n".join(report_log), file_name="report_log.txt")
    else:
        st.info("No operations performed yet.")
