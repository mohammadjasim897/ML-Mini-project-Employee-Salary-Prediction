"""
Employee Salary Prediction — ML Training Pipeline
===================================================
Trains multiple regression models on the employee salary dataset,
evaluates them, selects the best, and saves model artifacts as pickle files.
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'employee_salary_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'salary_model.pkl')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'preprocessor.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'model_metadata.pkl')

# ── Features ─────────────────────────────────────────────────────────
NUMERICAL_FEATURES = [
    'age', 'experience_years', 'skills_count',
    'certifications', 'performance_rating'
]

CATEGORICAL_FEATURES = [
    'gender', 'education_level', 'domain',
    'job_title', 'company_size', 'city_tier'
]

TARGET = 'salary_inr'


def load_data():
    """Load and validate the employee dataset."""
    if not os.path.exists(DATA_PATH):
        print("[!!] Dataset not found!")
        print("     Run 'python data/generate_dataset.py' first.")
        raise SystemExit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"[OK] Loaded {len(df)} records from dataset")
    print(f"     Features: {len(df.columns)} columns")
    return df


def build_preprocessor():
    """Build the sklearn ColumnTransformer for feature preprocessing."""
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, NUMERICAL_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )

    return preprocessor


def train_and_evaluate():
    """Train multiple models, compare, and save the best one."""

    # ── Load Data ────────────────────────────────────────────────────
    df = load_data()

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # ── Train/Test Split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[OK] Train/Test split: {len(X_train)} / {len(X_test)}")

    # ── Build Preprocessor ───────────────────────────────────────────
    preprocessor = build_preprocessor()

    # ── Define Models ────────────────────────────────────────────────
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42
        ),
    }

    # ── Train & Evaluate All Models ──────────────────────────────────
    print("\n" + "=" * 65)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 65)

    results = {}
    best_model_name = None
    best_r2 = -np.inf

    for name, model in models.items():
        print(f"\n▸ Training: {name} ...")

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name] = {
            'pipeline': pipeline,
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
        }

        print(f"  R² Score:  {r2:.4f}")
        print(f"  MAE:       ₹{mae:,.0f}")
        print(f"  RMSE:      ₹{rmse:,.0f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    # ── Results Summary ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"\n{'Model':<25} {'R²':>8} {'MAE (₹)':>15} {'RMSE (₹)':>15}")
    print("-" * 65)
    for name, res in results.items():
        marker = " ⭐ BEST" if name == best_model_name else ""
        print(f"{name:<25} {res['r2']:>8.4f} {res['mae']:>15,.0f} {res['rmse']:>15,.0f}{marker}")

    # ── Save Best Model ──────────────────────────────────────────────
    best_pipeline = results[best_model_name]['pipeline']
    best_result = results[best_model_name]

    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   R² = {best_result['r2']:.4f}, MAE = ₹{best_result['mae']:,.0f}")

    # Save the full pipeline (preprocessor + model)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_pipeline, f)
    print(f"\n💾 Model saved: {MODEL_PATH}")

    # Save preprocessor separately for inspection
    with open(PREPROCESSOR_PATH, 'wb') as f:
        pickle.dump(best_pipeline.named_steps['preprocessor'], f)
    print(f"💾 Preprocessor saved: {PREPROCESSOR_PATH}")

    # ── Feature Importance (if tree-based) ───────────────────────────
    feature_importance = None
    best_model = best_pipeline.named_steps['model']

    if hasattr(best_model, 'feature_importances_'):
        fitted_preprocessor = best_pipeline.named_steps['preprocessor']
        feature_names = fitted_preprocessor.get_feature_names_out()

        importances = best_model.feature_importances_
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\n📊 Top 15 Feature Importances:")
        for _, row in feat_imp.head(15).iterrows():
            bar = "█" * int(row['importance'] * 100)
            print(f"   {row['feature']:<40} {row['importance']:.4f}  {bar}")

        feature_importance = feat_imp.to_dict('records')

    # ── Save Metadata ────────────────────────────────────────────────
    # Extract unique values for the web app dropdowns
    df_meta = pd.read_csv(DATA_PATH)

    # Build domain → job_titles mapping
    domain_jobs = {}
    for domain in df_meta['domain'].unique():
        domain_jobs[domain] = sorted(df_meta[df_meta['domain'] == domain]['job_title'].unique().tolist())

    metadata = {
        'model_name': best_model_name,
        'r2_score': best_result['r2'],
        'mae': best_result['mae'],
        'rmse': best_result['rmse'],
        'all_results': {k: {kk: vv for kk, vv in v.items() if kk != 'pipeline'}
                        for k, v in results.items()},
        'feature_importance': feature_importance,
        'domains': sorted(df_meta['domain'].unique().tolist()),
        'domain_jobs': domain_jobs,
        'education_levels': sorted(df_meta['education_level'].unique().tolist()),
        'company_sizes': ['Startup', 'SME', 'MNC', 'Enterprise'],
        'city_tiers': ['Tier 1', 'Tier 2', 'Tier 3'],
        'genders': ['Male', 'Female', 'Other'],
        'numerical_features': NUMERICAL_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
    }

    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"💾 Metadata saved: {METADATA_PATH}")

    print("\n✅ Training complete!")
    return results, best_model_name


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train_and_evaluate()
