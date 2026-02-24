# 🎬 CampusBinge – Sentiment Analysis API

CampusBinge is a FastAPI-based backend service that performs sentiment analysis on text input using HuggingFace Transformers and PyTorch.  

The API supports both single text prediction and batch predictions, and provides interactive API documentation via Swagger UI.

---

## 🚀 Features

- Single text sentiment prediction
- Batch sentiment prediction
- FastAPI backend
- Interactive API docs (Swagger & ReDoc)
- Clean modular project structure
- Ready for deployment

---

## 🛠 Tech Stack

- Python 3.11
- FastAPI
- Uvicorn
- HuggingFace Transformers
- PyTorch

---

## 📁 Project Structure

```
CampusBinge/
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── ...
│   ├── model/
│   ├── graph/
│
├── tests/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Omee-Yehme/CampusBinge.git
cd CampusBinge
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the API

```bash
uvicorn src.api.app:app --reload
```

API will start at:

```
http://localhost:8000
```

---

## 📖 API Documentation

Swagger UI:
```
http://localhost:8000/docs
```

ReDoc:
```
http://localhost:8000/redoc
```

---

## 🔍 API Endpoints

### 🔹 POST `/predict`

Performs sentiment analysis on a single text.

Request:
```json
{
  "text": "I love this project!"
}
```

Response:
```json
{
  "label": "POSITIVE",
  "score": 0.98
}
```

---

### 🔹 POST `/batch-predict`

Performs sentiment analysis on multiple texts.

Request:
```json
{
  "texts": [
    "I love this",
    "This is terrible"
  ]
}
```

Response:
```json
[
  {
    "label": "POSITIVE",
    "score": 0.97
  },
  {
    "label": "NEGATIVE",
    "score": 0.99
  }
]
```

---

## 🧪 Running Tests

```bash
pytest
```

---

## 🌍 Deployment (Optional)

The application can be deployed using:
- Render
- Railway
- AWS EC2
- Docker

---

## 📌 Notes

- Model inference is powered by HuggingFace Transformers.
- Ensure Python 3.11 is installed.
- Designed for backend and ML evaluation purposes.

---

## 👨‍💻 Author

Om Sunil Ingale  
B.Tech – Artificial Intelligence & Data Science  
Nagpur, Maharashtra
