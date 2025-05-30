# 🧠 Advanced Image Analysis Web App

This Streamlit-powered web application integrates various image processing, face analysis, text recognition, and AI-based classification tools into a single, easy-to-use interface. Users can upload images, select from multiple functionalities via sidebar navigation, and download analysis results.

---

## 🚀 Features

### 🖼️ Basic Image Processing
- Histogram equalization
- Noise reduction (coming soon)

### ✂️ Segmentation
- **Semantic Segmentation** (SLIC)
- **Instance Segmentation** (mock implementation via thresholding)
- **Human Segmentation** (mock implementation)

### 😊 Face Analysis
- Face detection (Haar cascades)
- Facial landmark extraction and export to JSON
- Face matching and zoom using `face_recognition`
- Sketch generation from detected landmarks

### ✍️ Sketch from JSON
- Generate face sketches using uploaded JSON landmark files (either dict-based or flat list format)

### 🔍 Text Extraction (OCR)
- Extract text from images using `EasyOCR`

### 🤖 Image Classification
- Classify images using Vision Transformer (ViT by Google)

### 📥 Reporting
- Downloadable activity log for all operations performed

---

## 📂 Installation

### 🔧 Prerequisites
- Python 3.8 or later
- pip

### ⬇️ Install Dependencies

```bash
pip install -r requirements.txt
