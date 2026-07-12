# 🛡️ ChurnShield AI: Explainable Churn Prediction Dashboard

> **ChurnShield AI** is an end-to-end, production-ready prototype that predicts customer churn and explains *why*.  
> It’s designed for non-technical business users — transforming a complex *black box* model into an **intuitive, interactive dashboard** for actionable retention insights.

---

## ✨ Key Features

### 📈 Accurate Churn Prediction
- Powered by a **hyperparameter-tuned LightGBM** model.  
- Classifies customers into **Low**, **Moderate**, and **High** risk categories.  
- Achieves **AUC > 0.88** on validation data.

### 🧠 Explainable AI (XAI)
- Integrated with **SHAP (SHapley Additive exPlanations)**.  
- Generates clear, human-readable reasons for each prediction.  
- Visualizes which features increased or decreased a customer's churn risk.

### 🔬 Individual "What-If" Simulator
- Choose a high-risk customer and **modify key business levers** (e.g., annual premium).  
- Observe **real-time updates** in the churn score for decision experimentation.

### 🌍 Global Policy Simulator
- Conduct **strategic-level experiments** (e.g., 10% premium reduction for all customers).  
- See how policy changes impact **overall churn risk distribution**.

### 📊 Interactive Dashboard
- Built with **React**, **Tailwind CSS**, and **Recharts**.  
- Features a clean, responsive design with intuitive charts and data tables.

### ✅ Proven Accuracy
- Validation script ensures model fairness and reliability on **imbalanced data**.  
- Uses key metrics like **ROC-AUC**, **Balanced Accuracy**, and **F1-score**.

---

## 🛠️ Tech Stack

| Category | Technology |
|-----------|-------------|
| **Frontend** | React.js, Tailwind CSS, Recharts, Lucide-React |
| **Backend** | Python, Flask |
| **Machine Learning** | Scikit-learn, LightGBM, Pandas, NumPy |
| **Explainability** | SHAP (SHapley Additive exPlanations) |

---

## 🚀 Setup and Running Instructions

### 🔧 Prerequisites
- **Python:** ≥ 3.8  
- **Node.js:** ≥ 16  
- **Dataset:** Place your raw CSV files (`customer.csv`, `demographic.csv`, etc.) inside a folder named `archive/` in the project root.

---

### 🧩 1. Backend Setup (Flask API)

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Preprocess the raw data
python preprocessing.py
# → Creates `processed_features.csv` and `processed_target.csv`

# 5. Train the model
python train.py
# → Saves `churn_model.pkl`

# 6. Run the backend server
python app.py
# → Flask API available at: http://127.0.0.1:5000
```

Keep this terminal open — the backend must remain running.

---

### 💻 2. Frontend Setup (React UI)

Open a **new terminal** in the same project root.

```bash
# 1. Install required Node.js packages
npm install

# 2. Start the React development server
npm start
```

The app should automatically open in your browser at:  
👉 **http://localhost:3000**

---

## 📁 Project Workflow

### 🧹 **1. Preprocessing — `preprocessing.py`**
- Loads, merges, and cleans data from `archive/`.  
- Outputs:  
  - `processed_features.csv`  
  - `processed_target.csv`

### 🧠 **2. Training — `train.py`**
- Trains and tunes a **LightGBM** classifier.  
- Exports the final model as `churn_model.pkl`.

### ⚙️ **3. Backend — `app.py`**
- Loads the trained model and processed data.  
- Exposes API endpoints:
  - `/analyze` → Get churn prediction + SHAP explanations  
  - `/simulate` → Run “What-If” and policy simulations

### 🖥️ **4. Frontend — `ChurnPredictionUI.js`**
- Fetches data from backend APIs.  
- Displays:
  - Summary metrics  
  - Risk-level charts  
  - Interactive customer tables  
  - Simulators for scenario exploration

---

## 📊 Example Dashboard Preview

| Feature | Description |
|----------|-------------|
| **Summary Cards** | Display churn distribution and model performance |
| **Customer Table** | Filter by churn risk or demographic attributes |
| **SHAP Explanations** | Visualize feature influence per customer |
| **Simulators** | Adjust variables to see impact on churn score |

---

## 🧪 Validation Metrics

| Metric | Description | Value |
|--------|--------------|-------|
| **ROC-AUC** | Measures discrimination ability | > 0.88 |
| **Balanced Accuracy** | Handles imbalanced data | High |
| **F1-Score** | Balance between precision & recall | Consistent |

---



