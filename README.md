# 🚗 AI License Plate Recognition System

An end-to-end **Automatic Number Plate Recognition (ANPR)** system built using **YOLOv8, EasyOCR, OpenCV, and Streamlit**.
The application allows users to upload a vehicle video, automatically detect license plates, extract the plate numbers using OCR, and display the results through an interactive web dashboard.

---

# 📌 Features

* 🚘 **License Plate Detection** using YOLOv8
* 🔎 **Text Recognition** using EasyOCR
* ⚡ **Optimized Processing Pipeline** for faster inference
* 📊 **Interactive Dashboard** built with Streamlit
* 🧾 **Detected Plate Table with Timestamps**
* 🖼 **Plate Image Gallery**
* 📥 **Download Detection Results as CSV**
* 📈 **Plate Frequency Analytics Chart**

---

# 🧠 Tech Stack

| Component        | Technology           |
| ---------------- | -------------------- |
| Detection Model  | YOLOv8 (Ultralytics) |
| OCR Engine       | EasyOCR              |
| Image Processing | OpenCV               |
| Backend          | Python               |
| Frontend         | Streamlit            |
| Data Handling    | Pandas               |

---

# 📂 Project Structure

```
plate-recognition-app
│
├── app.py
├── license_plate_backend.py
├── license_plate_best.pt
├── requirements.txt
├── packages.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/license-plate-recognition-streamlit.git
cd license-plate-recognition-streamlit
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

# 📤 How It Works

1. Upload a vehicle video.
2. YOLOv8 detects license plates in each frame.
3. Detected plates are cropped and preprocessed.
4. EasyOCR extracts the plate characters.
5. Regex validation filters valid license plates.
6. The Streamlit dashboard displays:

   * Detected plate numbers
   * Detection timestamps
   * Plate image gallery
   * Frequency analytics
   * CSV download option

---

# 📦 Requirements

```
streamlit
opencv-python-headless
ultralytics
easyocr
numpy
pandas
torch
torchvision
```

---

# 📊 Example Output

The system produces:

* Detected license plate numbers
* Timestamp of detection
* Cropped plate images
* Plate vision time analytics

Example detection output:

```
LP11LJI
IL84OCX
NL64XXX
```

---

# 📈 Future Improvements

* Real-time webcam detection
* Vehicle tracking integration
* Multi-camera traffic monitoring
* Plate database integration
* Faster inference with optimized models

---

# 👨‍💻 Author

**Viraj Agrawal**

AI/ML Project – Automatic License Plate Recognition System
