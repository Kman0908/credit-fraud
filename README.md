---
title: FraudSense
emoji: 🔍
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# 🔍 FraudSense: Credit Card Fraud Detection

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face-blue?logo=huggingface)](https://huggingface.co/spaces/Kman0908/FraudSense)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/Package_Manager-uv-purple)](https://github.com/astral-sh/uv)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)](https://streamlit.io/)

FraudSense is an end-to-end Machine Learning web application deployed via Docker. It detects fraudulent credit card transactions in real-time, handling severe class imbalance (0.17% fraud rate) using advanced sampling techniques and gradient boosting.

## 🧠 The Challenge
Credit card fraud datasets are notoriously imbalanced. In this specific dataset, out of **284,807 transactions**, only **492 were fraudulent**. A naive model predicting "Legitimate" every single time would achieve 99.8% accuracy but catch zero fraud. This project focuses on high recall and precision for the minority class.

## 🛠️ Tech Stack
* **Frontend/Deployment:** Streamlit, Docker, Hugging Face Spaces
* **Environment Management:** `uv` (lightning-fast Rust-based package manager)
* **Model:** `CatBoostClassifier`
* **Data Processing:** `pandas`, `scikit-learn`
* **Imbalance Handling:** `SMOTE` (Synthetic Minority Over-sampling Technique)
* **Visualization:** `matplotlib`, `seaborn`

## 🚀 Live Demo
Try the application live here: [FraudSense on Hugging Face](https://huggingface.co/spaces/Kman0908/FraudSense)

## 💻 Run it Locally

This project uses `uv` for strict dependency management. 

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Kman0908/credit-fraud.git](https://github.com/Kman0908/credit-fraud.git)
   cd credit-fraud