# Online Recruitment Fraud Detection System
### AASTU Machine Learning Course Project — Group 4
**Addis Ababa Science and Technology University**  
*Department of Software Engineering*

---

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![React 18](https://img.shields.io/badge/react-18.x-cyan.svg)](https://react.dev/)
[![Flask API](https://img.shields.io/badge/flask-REST_API-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Project Overview
This repository hosts our final course project: an end-to-end Machine Learning (ML) and Natural Language Processing (NLP) pipeline designed to classify online job advertisements as either legitimate or fraudulent. Utilizing the Greek Employment Scam Aegean Dataset (EMSCAD), the system addresses severe class imbalance (~4.8% fraud) through custom cost-sensitive learning algorithms, threshold optimization, and advanced NLP preprocessing. 

The project features a **fully deployed environment** combining a cached Python Flask REST API backend with a high-fidelity, dark-theme React Single-Page Application (SPA) frontend. It serves as a real-time **decision-support utility** for job board moderators and candidates.

---

## 🛠️ System Architecture

Our implementation is divided into three core stages: preprocessing/feature engineering, cost-sensitive classification training, and real-time inference serving.

```mermaid
flowchart TD
    subgraph Data Pipeline
        DF[Raw Dataset] --> FE[7 Custom Red-Flag Features]
        FE --> SPLIT[Stratified 70/15/15 Split]
    end
    
    subgraph Preprocessing & Vectorization
        SPLIT --> TEXT[Text Concatenator]
        SPLIT --> CAT[Ordinal Encoder]
        SPLIT --> NUM[Numeric Imputer]
        TEXT --> TFIDF[TF-IDF Vectorizer <br/> ngram: 1,2 | Sublinear TF]
    end

    subgraph Cost-Sensitive ML Model Training
        TFIDF & CAT & NUM --> COMBO[Merged Feature Matrix]
        COMBO --> RF[Random Forest <br/> Balanced Weight]
        COMBO --> XGB[XGBoost <br/> Scale Pos Weight: 20]
        COMBO --> MNB[Multinomial NB <br/> Text Only]
    end

    subgraph Real-Time API Inference
        RF --> PKL[joblib Pipeline Export]
        PKL --> API[Flask REST API <br/> In-Memory Model Cache]
        API <-->|JSON REST API| UI[React / Vite Premium Frontend]
    end
```

---

## 📂 Repository Structure

The codebase is organized cleanly to separate raw source logic, serialized model binary exports, academic research notebooks, and the web app frontend:

```
project_root/
├── data/
│   └── fake_job_postings.csv             # Raw EMSCAD Job Postings Dataset
├── jupyter/                              # Academic Verification Notebooks
│   ├── 01_import_libraries.ipynb
│   ├── 02_load_dataset.ipynb
│   ├── 03_data_understanding.ipynb
│   ├── 04_exploratory_data_analysis.ipynb # Visualizations (Missing heatmaps, clouds)
│   ├── 05_preprocessing.ipynb
│   ├── 06_feature_engineering.ipynb
│   ├── 07_model_training.ipynb
│   ├── 08_evaluation.ipynb              # Classification matrices comparison
│   ├── 09_threshold_tuning.ipynb        # Validation F1-score sweep curves
│   ├── 10_error_analysis.ipynb
│   ├── 11_save_model.ipynb
│   └── 12_app_demo.ipynb
├── src/                                  # Modular Source Code
│   ├── __init__.py
│   ├── preprocessing.py                  # TextConcatenator, ColumnTransformer, Splits
│   ├── feature_engineering.py            # 7 Custom Domain-Specific Indicators
│   ├── training.py                       # Cost-Sensitive Classifier Initialization
│   ├── evaluation.py                     # Metric calculation & plotting
│   └── inference.py                      # Real-Time Predictor & Risk Signal Generator
├── models/                               # Serialized Artifacts & Performance Plots
│   ├── final_pipeline.joblib             # Compressed Random Forest Pipeline
│   ├── threshold.joblib                  # Calibrated Decision Boundary Metadata
│   ├── threshold_curve.png               # Validation Sweep Graph
│   ├── confusion_matrices.png            # Multi-Model Subplot Comparisons
│   └── model_comparison.csv              # Exported Metric Dataframe
├── frontend/                             # Single-Page Web Application
│   ├── src/
│   │   ├── App.jsx                       # React State Machine & Form handler
│   │   ├── index.css                     # Premium Dark-Theme Styling
│   │   └── main.jsx
│   ├── package.json                      # Node Package Manifest
│   └── vite.config.js                    # Vite Development Bundler config
├── app.py                                # CORS-enabled Flask REST API
├── requirements.txt                      # Python Virtual Environment Dependencies
└── README.md                             # This Project README
```

---

## 📈 Model Performance Summary
The models were trained on the $70\%$ split and evaluated on the $15\%$ validation split. Since the dataset is heavily skewed, models were evaluated using the **calibrated probability decision threshold ($0.74$)** to maximize minority-class F1-score.

| Classifier Model | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Random Forest** (Selected) | **0.9417** | **0.8692** | **0.9040** | **0.9768** | **0.9750** |
| Logistic Regression | 0.9055 | 0.8846 | 0.8949 | 0.9506 | 0.9673 |
| Gradient Boosting (XGBoost) | 0.8212 | 0.9538 | 0.8826 | 0.9672 | 0.9666 |
| Multinomial Naive Bayes | 1.0000 | 0.7615 | 0.8646 | 0.9757 | 0.9723 |
| Calibrated Linear SVC | 0.6195 | 0.5385 | 0.5761 | 0.6714 | 0.6558 |
| Dummy Stratified (Baseline) | 0.4651 | 0.4615 | 0.4633 | 0.4839 | 0.4654 |

### Key Experimental Findings:
1. **Random Forest Ensemble** delivered the optimal balance, achieving an **F1-score of 90.40%** at our optimized decision threshold of $0.74$.
2. **XGBoost** reached the highest recall ($95.38\%$), proving highly aggressive at catching fraud, but suffered from more false positives.
3. **Multinomial Naive Bayes** achieved **100% Precision**, meaning it produced zero false positives, making it highly reliable but conservative (missing ~24% of actual scams).

---

## 🚀 Installation & Running Guide

Ensure you have **Python 3.11+** and **Node.js 18+** installed on your machine.

### Step 1: Environment Setup
1. Clone the repository and navigate into the root directory:
   ```bash
   git clone https://github.com/Almichot-1/Fraud-Detection-in-Job-Postings.git
   cd Fraud-Detection-in-Job-Postings
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Step 2: Running the Machine Learning Pipeline
To retrain the models, execute threshold sweeps, generate evaluation plots, and serialize the optimal pipeline, run:
```bash
python run_pipeline.py
```
This script saves the serialized model pipeline directly to `models/final_pipeline.joblib`.

### Step 3: Launching the Backend REST API
Start the Flask prediction server (runs on `http://localhost:5000`):
```bash
python app.py
```
*Note: The model pipeline will be loaded and cached in memory upon server initialization to keep response latencies under 0.1 seconds.*

### Step 4: Running the React Frontend
Open a new terminal window, navigate to the `frontend/` directory, and start the Vite dev server (runs on `http://localhost:5173`):
```bash
cd frontend
npm install
npm run dev
```

Open **`http://localhost:5173`** in your browser to interact with the system.

---

## 🔬 Domain-Specific Features (7 Red Flags)
Our model evaluates seven custom domain-specific indicators to identify recruitment fraud:
* **Company Profile Absence**: Binary flag showing if the organization omitted their profile description.
* **Salary Range Non-disclosure**: Flags whether the structured salary range field is blank.
* **External Contact Redirection**: Uses regular expressions to scan the job description for direct contact details (unverified email addresses or external URLs), flagging postings trying to bypass standard application flows.
* **Suspicious Keyword Count**: Scans the text for common recruitment scam phrases (e.g., *deposit*, *commission*, *guaranteed*, *work from home*, *no experience required*, *wire transfer*).
* **Length Anomaly**: Detects abnormally brief descriptions (under 200 characters).
* **Company Logo Absence**: Flags whether the company logo is missing.
* **Screening Questions Absence**: Flags whether the recruiter omitted applicant screening questions.

---

## ⚖️ Ethical & Privacy Notice
> [!IMPORTANT]
> **Decision-Support Framing**: This application is a statistical decision-support utility. Its predictions are based on historical probability distributions and must not be used as absolute moral, legal, or authoritative judgments to blacklist employers.
>
> **Privacy Compliance**: To guarantee user data protection, the inference engine operates entirely on-demand. No user input text is stored, logged, or transmitted by either the React client or the Flask prediction server.

---

## 👥 Contributors (AASTU Group 4)
* **Team Member 1** - *Lead ML Engineer*
* **Team Member 2** - *Data Scientist / NLP Specialist*
* **Team Member 3** - *Backend API Developer*
* **Team Member 4** - *Frontend Engineer / UI Designer*

---

## 📚 References
1. S. Bansal, "Real / Fake Job Posting Prediction," Kaggle, 2020.
2. Samos Aegean University, "Employment Scam Aegean Dataset (EMSCAD)", Greek Research Portal.
3. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, 2011.
4. Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System," *KDD*, 2016.
