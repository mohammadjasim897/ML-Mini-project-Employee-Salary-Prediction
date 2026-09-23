"""
Employee Salary Prediction — Flask Web Application
====================================================
Serves the salary prediction API and the premium frontend UI.
Generates natural-language employee summaries alongside predictions.
"""

import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load Model Artifacts ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

model_path = os.path.join(MODEL_DIR, 'salary_model.pkl')
metadata_path = os.path.join(MODEL_DIR, 'model_metadata.pkl')

if not os.path.exists(model_path) or not os.path.exists(metadata_path):
    print("[!!] Model files not found!")
    print("     Run the following commands first:")
    print("       python data/generate_dataset.py")
    print("       python model/train_model.py")
    raise SystemExit(1)

with open(model_path, 'rb') as f:
    model_pipeline = pickle.load(f)

with open(metadata_path, 'rb') as f:
    metadata = pickle.load(f)

print(f"[OK] Loaded model: {metadata['model_name']}")
print(f"     R² = {metadata['r2_score']:.4f}, MAE = ₹{metadata['mae']:,.0f}")


# ── Employee Summary Generator ───────────────────────────────────────

def format_salary_inr(amount):
    """Format salary in Indian numbering system (Lakhs/Crores)."""
    if amount >= 10000000:
        return f"₹{amount/10000000:.2f} Cr"
    elif amount >= 100000:
        return f"₹{amount/100000:.2f} Lakhs"
    else:
        return f"₹{amount:,.0f}"


def format_salary_indian(amount):
    """Format number with Indian comma system (e.g., 12,45,000)."""
    s = str(int(amount))
    if len(s) <= 3:
        return f"₹{s}"
    result = s[-3:]
    s = s[:-3]
    while s:
        result = s[-2:] + ',' + result
        s = s[:-2]
    return f"₹{result}"


def generate_employee_summary(data, predicted_salary):
    """Generate a natural-language summary of the employee profile."""
    name = data.get('name', 'The employee')
    age = data.get('age', 'N/A')
    gender = data.get('gender', '')
    job_title = data.get('job_title', 'professional')
    domain = data.get('domain', 'their')
    experience = data.get('experience_years', 0)
    education = data.get('education_level', '')
    company_size = data.get('company_size', '')
    city_tier = data.get('city_tier', '')
    performance = data.get('performance_rating', 0)
    skills = data.get('skills_count', 0)
    certs = data.get('certifications', 0)

    # Gender pronoun
    if gender == 'Male':
        pronoun = 'He'
        possessive = 'his'
    elif gender == 'Female':
        pronoun = 'She'
        possessive = 'her'
    else:
        pronoun = 'They'
        possessive = 'their'

    # Experience description
    if experience < 2:
        exp_desc = f"a fresher with {experience} year{'s' if experience != 1 else ''} of experience"
    elif experience < 5:
        exp_desc = f"an early-career professional with {experience} years of experience"
    elif experience < 10:
        exp_desc = f"a mid-level professional with {experience} years of experience"
    elif experience < 20:
        exp_desc = f"a senior professional with {experience} years of experience"
    else:
        exp_desc = f"a highly experienced veteran with {experience} years of experience"

    # Performance description
    if performance >= 4.5:
        perf_desc = "an outstanding"
    elif performance >= 3.5:
        perf_desc = "a strong"
    elif performance >= 2.5:
        perf_desc = "an average"
    else:
        perf_desc = "a developing"

    # Build summary
    salary_formatted = format_salary_indian(predicted_salary)
    salary_lakhs = format_salary_inr(predicted_salary)

    summary = (
        f"{name} is a {age}-year-old {job_title} working in the {domain} domain. "
        f"{pronoun} is {exp_desc}, holding a {education} degree. "
        f"Currently employed at {'an' if company_size in ['Enterprise', 'MNC'] else 'a'} "
        f"{company_size} company in a {city_tier} city, "
        f"{name.split()[0]} demonstrates {perf_desc} performance rating of {performance}/5.0. "
        f"With {skills} technical skills and {certs} certification{'s' if certs != 1 else ''}, "
        f"{possessive} predicted annual salary is {salary_formatted} ({salary_lakhs}) per annum."
    )

    # Key highlights
    highlights = []
    if experience >= 10:
        highlights.append("🏅 Highly experienced professional")
    if performance >= 4.0:
        highlights.append("⭐ Top performer")
    if education == "PhD":
        highlights.append("🎓 Doctorate holder")
    elif education == "Master's":
        highlights.append("🎓 Post-graduate degree")
    if certs >= 3:
        highlights.append("📜 Well-certified professional")
    if skills >= 10:
        highlights.append("🛠️ Highly skilled (10+ skills)")
    if company_size in ['MNC', 'Enterprise']:
        highlights.append(f"🏢 Working at {company_size}")
    if city_tier == 'Tier 1':
        highlights.append("🌆 Metro city professional")

    return {
        'text': summary,
        'highlights': highlights,
    }


# ── Routes ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main prediction page."""
    return render_template('index.html')


@app.route('/api/metadata')
def get_metadata():
    """Return dropdown options and model info for the frontend."""
    return jsonify({
        'domains': metadata['domains'],
        'domain_jobs': metadata['domain_jobs'],
        'education_levels': metadata['education_levels'],
        'company_sizes': metadata['company_sizes'],
        'city_tiers': metadata['city_tiers'],
        'genders': metadata['genders'],
        'model_name': metadata['model_name'],
        'r2_score': round(metadata['r2_score'], 4),
        'mae': round(metadata['mae']),
        'feature_importance': metadata.get('feature_importance', [])[:10],
    })


@app.route('/api/predict', methods=['POST'])
def predict_salary():
    """Predict salary for given employee features."""
    data = request.get_json()

    # Validate required fields
    required = [
        'age', 'gender', 'education_level', 'domain', 'job_title',
        'experience_years', 'skills_count', 'certifications',
        'company_size', 'city_tier', 'performance_rating'
    ]

    for field in required:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400

    try:
        # Build input DataFrame matching training features
        input_data = pd.DataFrame([{
            'age': int(data['age']),
            'experience_years': float(data['experience_years']),
            'skills_count': int(data['skills_count']),
            'certifications': int(data['certifications']),
            'performance_rating': float(data['performance_rating']),
            'gender': data['gender'],
            'education_level': data['education_level'],
            'domain': data['domain'],
            'job_title': data['job_title'],
            'company_size': data['company_size'],
            'city_tier': data['city_tier'],
        }])

        # Predict
        predicted_salary = model_pipeline.predict(input_data)[0]
        predicted_salary = max(predicted_salary, 150000)  # Floor
        predicted_salary = round(predicted_salary / 1000) * 1000  # Round to nearest 1000

        # Generate summary
        summary = generate_employee_summary(data, predicted_salary)

        return jsonify({
            'predicted_salary': int(predicted_salary),
            'salary_formatted': format_salary_indian(predicted_salary),
            'salary_lakhs': format_salary_inr(predicted_salary),
            'monthly_salary': format_salary_indian(predicted_salary / 12),
            'summary': summary,
            'model_used': metadata['model_name'],
            'model_accuracy': round(metadata['r2_score'] * 100, 1),
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    print("\n  💰 Employee Salary Prediction System")
    print(f"  Open http://localhost:{port} in your browser\n")
    app.run(debug=debug, host='0.0.0.0', port=port)
