# 🚦 TrafficShield AI
## Automated Traffic Violation Detection System
### Gridlock Hackathon 2.0 — Flipkart × HackerEarth | Prototype Round 2

---

## 🎯 Problem Statement
Automated Photo Identification and Classification for Traffic Violations Using Computer Vision

## 🔍 What It Detects
| Violation | Severity |
|-----------|----------|
| 🪖 Helmet Non-Compliance | HIGH |
| 🏍️ Triple Riding | HIGH |
| 🔒 Seatbelt Non-Compliance | MEDIUM |
| 🚦 Red Light Violation | CRITICAL |
| ↔️ Wrong-Side Driving | HIGH |
| 📦 Vehicle Overloading | MEDIUM |
| 🅿️ Illegal Parking | LOW |

---

## ⚙️ Tech Stack
- **YOLOv8** (Ultralytics) — Object detection
- **OpenCV** — Image processing & annotation
- **EasyOCR** — License plate recognition
- **Flask** — Web application
- **Chart.js** — Analytics dashboard
- **Python 3.10+**

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the app
```bash
python app.py
```

### 3. Open browser
```
http://localhost:5000
```

---

## 📁 Project Structure
```
trafficvio/
├── app.py                  # Flask application
├── requirements.txt
├── utils/
│   └── detector.py         # YOLOv8 detection engine
├── templates/
│   ├── index.html          # Main UI (upload + detect)
│   └── dashboard.html      # Analytics dashboard
└── static/
    ├── uploads/            # Uploaded images
    └── results/            # Annotated output images
```

---

## 🏙️ Real-World Application
Designed for Bengaluru's traffic ecosystem:
- Integrates with existing CCTV infrastructure
- 24/7 automated monitoring
- Reduces manual patrol by ~80%
- Generates court-admissible evidence with timestamps

---

## 👤 Team
**Sureshkumar (Suresh Tak)**

B.Tech CSE (AI/ML), CSMSS University
GitHub: github.com/Suru12415
