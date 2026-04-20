# Spectacular

Spectacular is an AI-powered eyewear recommendation and Augmented Reality (AR) try-on platform developed as a Computer Science graduation project. The system leverages advanced computer vision and machine learning techniques to analyze facial structures, recommend optimal frame styles, and allow users to visualize eyewear in real-time through an interactive 3D AR interface.

---

## Features

**Facial Structure Analysis**
Utilizes facial landmark detection to precisely extract structural proportions and determine user face shapes.

**Personalized AI Recommendations**
Integrates a Hybrid Neural Matrix Factorization (NeuMF) deep learning model to generate tailored eyewear recommendations based on user features and frame attributes.

**Real-Time AR Try-On**
Implements responsive 3D model normalization and rendering to seamlessly overlay selected frames onto the user's face in real-time.

**Decoupled Architecture**
Features a high-performance REST API backend designed for heavy machine learning inference, integrated with a responsive modern web frontend.

---

## Technology Stack

### Frontend
| Technology | Purpose |
|---|---|
| Next.js & React | UI framework |
| TypeScript | Type-safe application logic |
| Tailwind CSS | Styling |
| React Three Fiber | 3D rendering |

### Backend & Machine Learning
| Technology | Purpose |
|---|---|
| Python 3.10+ | Core runtime |
| FastAPI | REST API framework |
| PyTorch & PyTorch Lightning | Model training and inference |
| MediaPipe | Facial landmark detection |
| SQLite / Local JSON | Database |

---

## Repository Structure

```
.
├── client/                  # Next.js frontend application
│   ├── api/                 # API client configurations (Axios)
│   ├── public/              # Static assets and 3D models
│   ├── src/                 # React components, pages, and application logic
│   ├── types/               # TypeScript interfaces (e.g., landmark definitions)
│   └── utils/               # Geometry and helper functions
│
├── server/                  # FastAPI and PyTorch backend
│   ├── database/            # Database configurations and mock data
│   ├── env/                 # Conda environment specifications (ml-env.yml)
│   ├── models/              # PyTorch model definitions
│   ├── scripts/             # Data processing and training scripts (NeuMF)
│   ├── services/            # Core business logic and ML classifiers
│   └── main.py              # FastAPI application entry point
│
└── .gitignore               # Root git ignore configuration
```

---

## Prerequisites

Ensure the following are installed on your local development environment before proceeding:

- Node.js v18.0.0 or higher
- pnpm
- Python 3.10 or higher
- Conda (optional, but recommended for managing the ML environment)

---

## Installation and Setup

### 1. Client

Navigate to the client directory and install the necessary dependencies:

```bash
cd client
pnpm install
```

Start the frontend development server:

```bash
pnpm run dev
```

The client application will be accessible at `http://localhost:3000`.

### 2. Server

Navigate to the server directory. It is strongly recommended to use the provided Conda environment file to ensure all machine learning dependencies are correctly configured.

```bash
cd server
conda env create -f env/ml-env.yml
conda activate <environment_name>
```

Start the backend API server:

```bash
uvicorn main:app --reload
```

The FastAPI application will be accessible at `http://localhost:8000`. Interactive API documentation is available at `http://localhost:8000/docs`.

---

## Model Training

To retrain the recommendation engine or face classification models, the training scripts are located in the `server/scripts/` directory:

```bash
cd server
python scripts/train_recommendation.py
```

Model checkpoints and TensorBoard logs will be saved automatically to the designated `lightning_logs` directories.

---

## License

This project was developed for academic purposes as a final year graduation project. Please review local institutional guidelines regarding the distribution and reproduction of academic coursework.
