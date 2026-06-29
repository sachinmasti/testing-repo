# 🌻 Teen Mental Health — Academic Performance Predictor

A machine learning project that predicts **academic performance** of teenagers based on their social media habits, sleep patterns, mental health indicators, and lifestyle factors.

---

## 📁 Project Structure

```
testing-repo/
├── apps/
│   ├── app.py        # FastAPI application
│   └── schema.py     # Pydantic v2 input validation
├── dataset/
│   └── Teen_Mental_Health_Dataset.csv
├── model/
│   └── sgd_regressor.joblib
├── main.ipynb        # EDA, training, evaluation notebook
└── model.py          # Model training script
```

---

## 🧠 Problem Statement

Predict a teenager's **academic performance score** using features like daily social media usage, sleep hours, stress level, anxiety level, and addiction level.

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Data Processing | Pandas, NumPy |
| Visualization | Seaborn, Matplotlib, mplcyberpunk |
| ML & Preprocessing | Scikit-learn, feature-engine |
| Model | SGDRegressor, Ridge, Lasso, ElasticNet, RandomForest |
| API | FastAPI, Pydantic v2 |
| Serialization | Joblib |

---

## 🔄 ML Pipeline

- **Categorical Encoding** — OneHotEncoder for `gender`, `platform_usage`
- **Ordinal Encoding** — OrdinalEncoder for `social_interaction_level` (low → medium → high)
- **Scaling** — StandardScaler for numerical features
- **Validation** — Pydantic v2 with field validators

---

## 📊 Model Evaluation

- Learning curve analysis
- Residual plot (random scatter around zero ✅)
- KDE distribution of residuals
- Cross-validation (5-fold) across multiple models

---

## 🚀 API Usage

**Run the server:**
```bash
cd apps
uvicorn app:app --reload
```

**Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| GET | `/home` | Health check |
| GET | `/model_load` | Check if model loaded |
| POST | `/predict` | Get prediction |

**Sample POST request to `/predict`:**
```json
{
  "age": 16,
  "gender": "male",
  "daily_social_media_hours": 4.5,
  "platform_usage": "Instagram",
  "sleep_hours": 6.5,
  "screen_time_before_sleep": 2.0,
  "physical_activity": 1.5,
  "social_interaction_level": "medium",
  "stress_level": 6,
  "anxiety_level": 5,
  "addiction_level": 4,
  "depression_label": "no"
}
```

---

## 📦 Installation

```bash
pip install fastapi uvicorn scikit-learn pandas joblib pydantic feature-engine
```

---

## 👤 Author

**Sachin** — [GitHub](https://github.com/sachinmasti) | [LinkedIn](https://linkedin.com/in/sachin-masti) | [Medium](https://medium.com/@Sachinmasti)
