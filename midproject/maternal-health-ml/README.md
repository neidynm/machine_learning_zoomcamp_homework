# Maternal Health Risk Prediction - Streamlit App

A machine learning web application for predicting maternal health risk levels using XGBoost classification.

## 📋 Project Overview

This application predicts maternal health risk levels (Low Risk, Mid Risk, High Risk) based on various health indicators including:
- Age
- Blood Pressure (Systolic and Diastolic)
- Blood Sugar levels
- Body Temperature
- Heart Rate
- Calculated features (Pulse Pressure, Mean Arterial Pressure)

## 🚀 Quick Start Guide

### Prerequisites

Before you begin, ensure you have the following installed:
- **Docker Desktop** (recommended) or Docker Engine
- **Python 3.11+** (if running locally without Docker)
- **Git** (optional, for version control)

### Installation Options

You can run this app in three ways:
1. **Using Docker Compose** (Recommended - Easiest)
2. **Using Docker CLI**
3. **Running Locally** (Without Docker)

---

## 🐳 Option 1: Using Docker Compose (Recommended)

This is the easiest method for deployment.

### Step 1: Prepare Your Files

Create a project directory and add all files:

```bash
mkdir maternal-health-app
cd maternal-health-app
```

Ensure you have these files in your directory:
- `app.py` - Streamlit application
- `Dockerfile` - Docker configuration
- `docker-compose.yml` - Docker Compose configuration
- `requirements.txt` - Python dependencies
- `train_model.py` - Model training script
- `model.pkl` - Trained model (you'll create this)
- `label_encoder.pkl` - Label encoder (you'll create this)

### Step 2: Train and Save Your Model

First, install the required Python packages to train your model:

```bash
pip install ucimlrepo pandas numpy scikit-learn xgboost
```

Then run the training script:

```bash
python train_model.py
```

This will:
- Download the maternal health dataset
- Train the XGBoost model
- Save `model.pkl` and `label_encoder.pkl` files

### Step 3: Build and Run with Docker Compose

```bash
# Build and start the container
docker-compose up --build

# Or run in detached mode (background)
docker-compose up -d --build
```

### Step 4: Access Your App

Open your browser and navigate to:
```
http://localhost:8501
```

### Step 5: Stop the App

```bash
# If running in foreground, press Ctrl+C, then:
docker-compose down

# If running in background:
docker-compose down
```

---

## 🐋 Option 2: Using Docker CLI

If you prefer using Docker commands directly:

### Step 1: Train Your Model

```bash
pip install ucimlrepo pandas numpy scikit-learn xgboost
python train_model.py
```

### Step 2: Build Docker Image

```bash
docker build -t maternal-health-predictor .
```

### Step 3: Run Docker Container

```bash
docker run -p 8501:8501 maternal-health-predictor
```

### Step 4: Access Your App

Navigate to `http://localhost:8501` in your browser.

### Step 5: Stop the Container

```bash
# Find the container ID
docker ps

# Stop the container
docker stop <container_id>

# Remove the container (optional)
docker rm <container_id>
```

---

## 💻 Option 3: Running Locally (Without Docker)

### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
pip install ucimlrepo  # For training
```

### Step 3: Train Your Model

```bash
python train_model.py
```

### Step 4: Run Streamlit App

```bash
streamlit run app.py
```

### Step 5: Access Your App

The terminal will show you the URL (typically `http://localhost:8501`)

---

## 📦 Project Structure

```
maternal-health-app/
├── app.py                    # Main Streamlit application
├── train_model.py            # Model training script
├── model.pkl                 # Trained XGBoost model (generated)
├── label_encoder.pkl         # Label encoder (generated)
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container configuration
├── docker-compose.yml        # Docker Compose configuration
├── .dockerignore            # Files to exclude from Docker
└── README.md                # This file
```

---

## 🛠️ Docker Commands Reference

### Useful Docker Commands

```bash
# View running containers
docker ps

# View all containers (including stopped)
docker ps -a

# View Docker images
docker images

# Stop a running container
docker stop <container_id>

# Remove a container
docker rm <container_id>

# Remove an image
docker rmi <image_name>

# View container logs
docker logs <container_id>

# Access container shell
docker exec -it <container_id> /bin/bash
```

### Docker Compose Commands

```bash
# Start services
docker-compose up

# Start services in background
docker-compose up -d

# Build images
docker-compose build

# Build and start
docker-compose up --build

# Stop services
docker-compose down

# View logs
docker-compose logs

# View logs for specific service
docker-compose logs streamlit-app
```

---

## 🔧 Configuration

### Changing the Port

**Docker Compose Method:**
Edit `docker-compose.yml`:
```yaml
ports:
  - "8502:8501"  # Change 8502 to your desired port
```

**Docker CLI Method:**
```bash
docker run -p 8502:8501 maternal-health-predictor
```

### Environment Variables

You can customize Streamlit behavior by setting environment variables in `docker-compose.yml`:

```yaml
environment:
  - STREAMLIT_SERVER_PORT=8501
  - STREAMLIT_SERVER_ADDRESS=0.0.0.0
  - STREAMLIT_THEME_BASE=light
  - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

---

## 🧪 Testing the App

Once the app is running, you can test it with sample patient data:

**Example 1: Low Risk Patient**
- Age: 25
- Systolic BP: 120
- Diastolic BP: 80
- Blood Sugar: 7.0
- Body Temperature: 98.0
- Heart Rate: 70

**Example 2: High Risk Patient**
- Age: 42
- Systolic BP: 160
- Diastolic BP: 100
- Blood Sugar: 15.0
- Body Temperature: 102.0
- Heart Rate: 100

---

## 📊 Model Information

- **Algorithm**: XGBoost Classifier
- **Dataset**: UCI Maternal Health Risk Dataset
- **Features**: 8 (6 original + 2 engineered)
- **Classes**: Low Risk, Mid Risk, High Risk
- **Evaluation**: ~90%+ accuracy on test set

---

## 🐛 Troubleshooting

### Port Already in Use

If port 8501 is already in use:
```bash
# Find what's using the port (Linux/Mac)
lsof -i :8501

# Kill the process
kill -9 <PID>

# Or use a different port
docker run -p 8502:8501 maternal-health-predictor
```

### Model Files Not Found

Make sure you've run `train_model.py` before building the Docker image:
```bash
python train_model.py
```

Verify the files exist:
```bash
ls -la model.pkl label_encoder.pkl
```

### Docker Build Fails

Clear Docker cache and rebuild:
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Permission Denied (Linux)

If you get permission errors with Docker on Linux:
```bash
sudo usermod -aG docker $USER
newgrp docker
```

---

## 🚀 Deployment to Cloud

### Deploy to AWS EC2

1. Launch an EC2 instance with Docker installed
2. Copy your files to the instance
3. Run: `docker-compose up -d`
4. Configure security groups to allow port 8501

### Deploy to Google Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/maternal-health-app

# Deploy to Cloud Run
gcloud run deploy maternal-health-app \
  --image gcr.io/PROJECT_ID/maternal-health-app \
  --platform managed \
  --port 8501
```

### Deploy to Heroku

1. Install Heroku CLI
2. Create `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT
   ```
3. Deploy:
   ```bash
   heroku create maternal-health-app
   git push heroku main
   ```

---

## 📝 Notes

- This is a demonstration application for educational purposes
- Not intended for actual medical diagnosis
- Always consult healthcare professionals for medical advice
- Model performance may vary based on the training data

