# 🛰️ Kṣatra – AI-Powered Satellite Collision Prediction System

Kṣatra is a modern ML-powered orbital conjunction prediction system.  
It computes **Time of Closest Approach (TCA)**, **relative speed**, **miss distance**, and predicts **collision risk** between two satellites using a trained **XGBoost model**.

A full **interactive Streamlit dashboard** is included for real-time TLE input, 3D orbit visualization, and ML-based risk scoring.

---

## 🚀 Features

### 🔭 Real Orbital Mechanics
- SGP4 propagation  
- TCA (Time of Closest Approach) search  
- Miss distance and relative velocity  
- r–t–n reference frame feature extraction  

### 🤖 Machine Learning
- XGBoost binary classifier  
- Trained on:
  - Real screening snapshots  
  - Synthetic near-miss events  
  - SMOTE & oversampled balanced dataset  
- Predicts probability of collision in real time  

### 🎨 UI & Visualization
- Streamlit dashboard  
- 3D orbit visualization (Plotly)  
- Real-time risk band output (Low/Medium/High)  
- Clean, polished UI with custom CSS  

---

## 📦 Tech Stack

| Component | Technology |
|----------|-------------|
| Orbit propagation | **SGP4** |
| Closest approach | Custom physics-based search |
| Machine Learning | **XGBoost** |
| Model serving | Streamlit |
| UI / 3D plots | Plotly |
| Training pipeline | Python, pandas, NumPy, SMOTE |
| Deployment | Streamlit Cloud |

---


---

## ⚙️ How It Works

### **1️⃣ SGP4 Physics-Based Orbital Propagation**
- TLEs → Satrec object  
- Propagate over time to estimate relative orbital geometry  
- Identify coarse conjunction candidates  

### **2️⃣ Closest-Approach Refinement**
Using a fine window (±10 minutes):

- Compute precise **TCA (JD)**
- Miss distance (km)
- Relative velocity (km/s)
- Relative geometry in RTN frame

### **3️⃣ Feature Engineering**
Model uses:

- Radial / tangential / normal miss components  
- Miss distance norm  
- Relative velocity  
- Closing rate  
- Orbital radius norms  
- Additional metadata: TCA JD, vrel_kms, miss_km  

### **4️⃣ Machine Learning – XGBoost**
We train a gradient-boosted binary classifier using:

- Real TLE snapshots
- Synthetic near-miss generator
- Oversampling + SMOTE

Achieves **~99–100% ROC-AUC** on balanced data.

### **5️⃣ Streamlit Dashboard**
Features:

- Live TLE input
- Automatic TLE validation
- ML-powered collision probability
- Orbit visualization in 3D
- Modern UI with dark theme

---

## 🧠 ML Concepts Used

- Feature engineering  
- Binary Classification  
- Gradient Boosting (XGBoost)  
- Synthetic data augmentation  
- Random oversampling  
- SMOTE  
- Train–test split & stratification  
- ROC-AUC evaluation  
- Model serialization (joblib)  

---

## 🛰️ Why This Project Matters

Kṣatra helps:

- Assess conjunction risk for **new satellite launches**
- Predict dangerous approaches for **LEO/MEO spacecraft**
- Improve **situational awareness** for mission planning  
- Prototype **Space Traffic Management (STM)** systems  
- Provide training tools for universities & research labs  

### **Future Scope**
- 🚀 Auto-fetch TLEs from Space-Track API  
- 🚀 Batch prediction for 1-vs-many satellites  
- 🚀 REST API backend for mission control  
- 🚀 Reinforcement Learning for collision avoidance  
- 🚀 Daily risk reports & historical analytics  

---


