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

## 📁 Project Structure

