# ⚽ 23-24 Soccer Match Outcome Predictor

This project builds a machine learning model to predict soccer match outcomes for the 2023–2024 season. Using a Random Forest Classifier and hyperparameter tuning via `RandomizedSearchCV`, the model aims to classify whether the home team wins or not based on match statistics.

---

## 📊 Dataset

- **File**: `23-24Season.csv`
- **Source**: Custom dataset with match stats for the 2023–2024 season
- **Target Variable**: `Target`  
  - `1` → Home team wins  
  - `0` → Otherwise (loss or draw)

---

## 🧠 Model Pipeline

1. **Preprocessing**  
   - Drop missing values  
   - Feature scaling (if needed)  
   - Encode categorical columns (if any)

2. **Train-Test Split**  
   - 80% training, 20% testing

3. **Model Selection**  
   - `RandomForestClassifier`

4. **Hyperparameter Tuning**  
   - `RandomizedSearchCV` over:
     - `n_estimators`
     - `max_depth`
     - `min_samples_split`
     - `min_samples_leaf`
     - `max_features`
     - `class_weight`

5. **Evaluation Metrics**  
   - Accuracy  
   - Precision, Recall, F1-Score  
   - Confusion Matrix

---

## 🏆 Best Model Performance (Latest)

| Metric       | Value |
|--------------|-------|
| Accuracy     | 58%   |
| F1 Score     | ~0.57 |
| Best Params  | (from `RandomizedSearchCV`) |

> ⚠️ Class imbalance observed: consider using SMOTE or adjusting thresholds in future iterations.

---

## 🔧 Requirements

- Python 3.8+
- pandas
- scikit-learn
- numpy
- matplotlib / seaborn (for optional visualization)

Install all with:
```bash
pip install -r requirements.txt
