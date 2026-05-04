# SkPro — Probabilistic ML Dashboard

> The unified framework for probabilistic supervised machine learning in Python.
> Full-stack app: FastAPI backend + React frontend.

---

## 📁 Project Structure

```
skpro-app/
├── backend/
│   ├── main.py          ← FastAPI app (all API endpoints)
│   ├── models.py        ← 4 probabilistic models implementation
│   └── requirements.txt
├── frontend/
│   ├── public/index.html
│   ├── src/
│   │   ├── App.js       ← Full React UI
│   │   ├── index.js
│   │   └── index.css
│   └── package.json
└── sample_datasets/
    ├── house_prices.csv
    ├── energy_consumption.csv
    ├── medical_data.csv
    └── salary_data.csv
```

---

## 🚀 Setup & Run

### Step 1 — Backend (Python API)

```bash
cd backend

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn main:app --reload --port 8000
```

Open your browser to http://localhost:8000/docs for the interactive API docs.

### Step 2 — Frontend (React)

Open a **new terminal**:

```bash
cd frontend

# Install Node dependencies
npm install

# Start the React app
npm start
```

Frontend opens at http://localhost:3000

---

## 🔌 API Endpoints

| Method | Endpoint    | Description                                      |
|--------|-------------|--------------------------------------------------|
| GET    | /models     | List all available probabilistic model types     |
| POST   | /upload     | Upload a CSV dataset (multipart/form-data)       |
| POST   | /predict    | Fit model, return predictions + uncertainty      |
| POST   | /compare    | Compare 2+ models side-by-side                  |

### POST /upload
```
Body: multipart/form-data with `file` field (CSV)
Returns: { dataset_id, shape, numeric_columns, preview, missing_values }
```

### POST /predict
```json
{
  "dataset_id": "house_prices",
  "target_column": "price_usd",
  "feature_columns": ["size_sqft", "bedrooms", "age_years"],
  "model_type": "bayesian_ridge",
  "test_size": 0.2
}
```
Returns:
```json
{
  "predictions": [{ "index": 0, "x": 0, "y_true": 245000, "y_pred": 238500, "y_lower": 210000, "y_upper": 265000 }],
  "model_summary": { "r2_score": 0.82, "mae": 15200, "rmse": 21000, "training_time_seconds": 0.08, ... }
}
```

### POST /compare
```json
{
  "dataset_id": "house_prices",
  "target_column": "price_usd",
  "feature_columns": ["size_sqft", "bedrooms"],
  "model_types": ["bayesian_ridge", "gaussian_process"]
}
```

### GET /models
```json
{
  "models": [
    { "id": "bayesian_ridge", "name": "Bayesian Ridge Regression", "category": "Linear", ... },
    { "id": "gaussian_process", "name": "Gaussian Process Regressor", "category": "Non-parametric", ... },
    { "id": "quantile_regression", "name": "Quantile Regression", "category": "Linear", ... },
    { "id": "gradient_boosting", "name": "Gradient Boosting with Intervals", "category": "Ensemble", ... }
  ]
}
```

---

## 🧠 Supported Models

| Model ID              | What it does                                    | Uncertainty source        |
|-----------------------|-------------------------------------------------|---------------------------|
| `bayesian_ridge`      | Bayesian linear regression                      | Posterior std × 1.645     |
| `gaussian_process`    | GP with Matérn kernel                           | GP posterior variance     |
| `quantile_regression` | Linear models at 10th/50th/90th quantile        | 10th–90th percentile band |
| `gradient_boosting`   | Gradient Boosting at 10th/50th/90th quantile    | 10th–90th percentile band |

---

## 📊 Sample Datasets

| File                    | Rows | Target          | Features                              |
|-------------------------|------|-----------------|---------------------------------------|
| house_prices.csv        | 200  | price_usd       | size_sqft, bedrooms, age_years, ...   |
| energy_consumption.csv  | 150  | energy_kwh      | temperature_c, humidity_pct, ...      |
| medical_data.csv        | 180  | cholesterol     | age, bmi, blood_pressure, glucose     |
| salary_data.csv         | 160  | salary_usd      | years_experience, education_years, ...  |

---

## ⚠️ Edge Cases Handled

- **Missing values**: Rows with NaN in selected columns are dropped before fitting
- **Non-numeric columns**: Rejected with a clear error message
- **Small datasets**: Rejected if < 10 rows remain after cleaning
- **GP on large data**: Auto-subsampled to 300 rows (O(n³) complexity)
- **Quantile crossing**: `y_lower` is capped to never exceed `y_pred`

---

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, scikit-learn, pandas, numpy
- **Frontend**: React 18, Recharts (charts), Axios (HTTP)
- **Models**: scikit-learn BayesianRidge, GaussianProcessRegressor, QuantileRegressor, GradientBoostingRegressor
