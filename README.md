# Lung Abnormality Identification System with Explainable AI

This project is a comprehensive AI framework for detecting lung abnormalities from Chest CT scans and X-Ray images. It includes a Python FastAPI backend with a Deep Learning model (ResNet50) and a mock Grad-CAM implementation for X-AI, and an Angular frontend for the user interface.

## Project Structure

- `backend/`: FastAPI application, AI Model, and Logic.
- `frontend/`: Angular Web Application.

## Prerequisites

- **Python 3.8+**
- **Node.js 18+** and **npm**

## Setup & Running

### 1. Backend (Python)

Navigate to the `backend` directory and install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
python main.py
```

> The server will start on `http://localhost:8000`.

### 2. Frontend (Angular)

Navigate to the `frontend` directory and install dependencies:

```bash
cd frontend
npm install
```

Start the Angular development server:

```bash
npx ng serve
```

> The application will be available at `http://localhost:4200`.

## Features

- **Drag & Drop Upload**: Easy interface to analyze medical images.
- **Deep Learning Model**: Uses a ResNet50 architecture for classification.
- **Explainable AI (Grad-CAM)**: Visualizes the regions of the image that contributed to the prediction.
- **Real-time Results**: Instant feedback with prediction confidence scores.
- **Secure Handling**: Images are processed in memory and not permanently stored.

## Disclaimer

This is a **demonstration prototype**. The AI model uses pre-trained ImageNet weights and a simulated heatmap for demonstration purposes. For clinical use, the model must be trained on a validated medical dataset (e.g., COVID-19 Radiography Database).
