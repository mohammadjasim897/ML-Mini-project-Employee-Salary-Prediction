"""
SalaryIQ — Streamlit Employee Salary Prediction App
=====================================================
Premium UI for predicting employee salaries in ₹ INR
with an AI-generated employee summary.
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="SalaryIQ — AI Salary Prediction",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Load Model Artifacts ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR = os.path.join(BASE_DIR, 'data')

model_path = os.path.join(MODEL_DIR, 'salary_model.pkl')
metadata_path = os.path.join(MODEL_DIR, 'model_metadata.pkl')
dataset_path = os.path.join(DATA_DIR, 'employee_salary_dataset.csv')

# Check if model files exist, if not, generate and train
if not os.path.exists(model_path) or not os.path.exists(metadata_path):
    with st.spinner("🔧 First run — generating dataset & training model..."):
        import subprocess, sys
        # Generate dataset
        if not os.path.exists(dataset_path):
            subprocess.run([sys.executable, os.path.join(DATA_DIR, 'generate_dataset.py')], check=True)
        # Train model
        subprocess.run([sys.executable, os.path.join(MODEL_DIR, 'train_model.py')], check=True)
    st.rerun()


@st.cache_resource
def load_model():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(metadata_path, 'rb') as f:
        meta = pickle.load(f)
    return model, meta


model_pipeline, metadata = load_model()


# ── Helper Functions ──────────────────────────────────────────────────

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


def format_salary_lakhs(amount):
    """Format salary in Lakhs/Crores."""
    if amount >= 10000000:
        return f"₹{amount / 10000000:.2f} Cr"
    elif amount >= 100000:
        return f"₹{amount / 100000:.2f} Lakhs"
    else:
        return f"₹{amount:,.0f}"


def generate_summary(data, salary):
    """Generate natural-language employee summary."""
    name = data['name']
    age = data['age']
    gender = data['gender']
    job_title = data['job_title']
    domain = data['domain']
    experience = data['experience_years']
    education = data['education_level']
    company_size = data['company_size']
    city_tier = data['city_tier']
    performance = data['performance_rating']
    skills = data['skills_count']
    certs = data['certifications']

    if gender == 'Male':
        pronoun, possessive = 'He', 'his'
    elif gender == 'Female':
        pronoun, possessive = 'She', 'her'
    else:
        pronoun, possessive = 'They', 'their'

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

    if performance >= 4.5:
        perf_desc = "an outstanding"
    elif performance >= 3.5:
        perf_desc = "a strong"
    elif performance >= 2.5:
        perf_desc = "an average"
    else:
        perf_desc = "a developing"

    salary_fmt = format_salary_indian(salary)
    salary_lkh = format_salary_lakhs(salary)

    summary = (
        f"{name} is a {age}-year-old {job_title} working in the {domain} domain. "
        f"{pronoun} is {exp_desc}, holding a {education} degree. "
        f"Currently employed at {'an' if company_size in ['Enterprise', 'MNC'] else 'a'} "
        f"{company_size} company in a {city_tier} city, "
        f"{name.split()[0]} demonstrates {perf_desc} performance rating of {performance}/5.0. "
        f"With {skills} technical skills and {certs} certification{'s' if certs != 1 else ''}, "
        f"{possessive} predicted annual salary is **{salary_fmt}** ({salary_lkh}) per annum."
    )

    highlights = []
    if experience >= 10:
        highlights.append("🏅 Highly experienced")
    if performance >= 4.0:
        highlights.append("⭐ Top performer")
    if education == "PhD":
        highlights.append("🎓 Doctorate holder")
    elif education == "Master's":
        highlights.append("🎓 Post-graduate")
    if certs >= 3:
        highlights.append("📜 Well-certified")
    if skills >= 10:
        highlights.append("🛠️ 10+ skills")
    if company_size in ['MNC', 'Enterprise']:
        highlights.append(f"🏢 {company_size}")
    if city_tier == 'Tier 1':
        highlights.append("🌆 Metro city")

    return summary, highlights


# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #12121a 50%, #0d0d18 100%);
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
    }
    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7c3aed, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        letter-spacing: -0.03em;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        font-family: 'Inter', sans-serif;
    }

    /* Stats row */
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 1rem 0 1.5rem;
        flex-wrap: wrap;
    }
    .stat-item {
        text-align: center;
    }
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7c3aed, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 500;
    }

    /* Salary display */
    .salary-display {
        text-align: center;
        background: rgba(18, 18, 30, 0.65);
        border: 1px solid rgba(124, 58, 237, 0.25);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(16px);
        position: relative;
        overflow: hidden;
    }
    .salary-display::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(135deg, #7c3aed, #06b6d4);
    }
    .salary-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .salary-amount {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.75rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .salary-sub {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.25rem;
    }
    .salary-monthly {
        display: inline-block;
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        padding: 0.4rem 1rem;
        margin-top: 0.5rem;
        color: #64748b;
        font-size: 0.85rem;
    }
    .salary-monthly strong {
        color: #06b6d4;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Summary card */
    .summary-card {
        background: rgba(18, 18, 30, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(16px);
    }
    .summary-text {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.8;
    }

    /* Highlight tags */
    .highlight-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.75rem;
    }
    .highlight-tag {
        padding: 0.3rem 0.7rem;
        background: rgba(124, 58, 237, 0.1);
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 8px;
        font-size: 0.8rem;
        color: #06b6d4;
        font-weight: 500;
    }

    /* Model badge */
    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.75rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 100px;
        font-size: 0.75rem;
        color: #10b981;
        font-weight: 500;
    }
    .model-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #10b981;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Feature importance bar */
    .imp-bar-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.4rem;
    }
    .imp-bar-label {
        font-size: 0.7rem;
        color: #94a3b8;
        min-width: 120px;
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .imp-bar-track {
        flex: 1;
        height: 8px;
        background: rgba(255,255,255,0.05);
        border-radius: 4px;
        overflow: hidden;
    }
    .imp-bar-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(135deg, #7c3aed, #06b6d4);
    }
    .imp-bar-value {
        font-size: 0.65rem;
        font-family: 'JetBrains Mono', monospace;
        color: #64748b;
        min-width: 40px;
    }

    /* Override Streamlit defaults */
    .stSelectbox label, .stSlider label, .stTextInput label {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em !important;
    }

    div[data-testid="stForm"] {
        background: rgba(18, 18, 30, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        backdrop-filter: blur(16px);
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #7c3aed, #06b6d4) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.75rem !important;
        border-radius: 12px !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(124, 58, 237, 0.35) !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #64748b;
        font-size: 0.8rem;
        border-top: 1px solid rgba(148, 163, 184, 0.1);
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>💰 SalaryIQ</h1>
    <p>AI-Powered Employee Salary Prediction in ₹ INR</p>
</div>
""", unsafe_allow_html=True)

# Model badge
st.markdown(f"""
<div style="text-align: center; margin-bottom: 0.5rem;">
    <span class="model-badge">
        <span class="model-dot"></span>
        {metadata['model_name']} • R² {metadata['r2_score']:.4f}
    </span>
</div>
""", unsafe_allow_html=True)

# Stats row
st.markdown(f"""
<div class="stats-row">
    <div class="stat-item">
        <div class="stat-value">10</div>
        <div class="stat-label">Domains</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">5000+</div>
        <div class="stat-label">Training Records</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">{metadata['r2_score']*100:.1f}%</div>
        <div class="stat-label">Accuracy (R²)</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Prediction Form ──────────────────────────────────────────────────
with st.form("prediction_form"):
    st.markdown("### 👤 Employee Details")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", placeholder="e.g., Rahul Sharma")
    with col2:
        age = st.slider("Age", 22, 60, 28)

    col3, col4 = st.columns(2)
    with col3:
        gender = st.selectbox("Gender", metadata['genders'])
    with col4:
        education = st.selectbox("Education Level", metadata['education_levels'])

    col5, col6 = st.columns(2)
    with col5:
        domain = st.selectbox("Industry Domain", metadata['domains'])
    with col6:
        job_titles = metadata['domain_jobs'].get(domain, [])
        job_title = st.selectbox("Job Title", job_titles)

    col7, col8 = st.columns(2)
    with col7:
        experience = st.slider("Experience (Years)", 0.0, 35.0, 5.0, step=0.5)
    with col8:
        skills = st.slider("Skills Count", 1, 15, 5)

    col9, col10 = st.columns(2)
    with col9:
        certifications = st.slider("Certifications", 0, 5, 1)
    with col10:
        performance = st.slider("Performance Rating", 1.0, 5.0, 3.5, step=0.1)

    col11, col12 = st.columns(2)
    with col11:
        company_size = st.selectbox("Company Size", metadata['company_sizes'])
    with col12:
        city_tier = st.selectbox("City Tier", metadata['city_tiers'])

    submitted = st.form_submit_button("🔮 Predict Salary")

# ── Results ───────────────────────────────────────────────────────────
if submitted:
    if not name.strip():
        st.error("Please enter the employee's name.")
    else:
        with st.spinner("🧠 Analyzing employee profile..."):
            input_data = pd.DataFrame([{
                'age': age,
                'experience_years': experience,
                'skills_count': skills,
                'certifications': certifications,
                'performance_rating': performance,
                'gender': gender,
                'education_level': education,
                'domain': domain,
                'job_title': job_title,
                'company_size': company_size,
                'city_tier': city_tier,
            }])

            predicted_salary = model_pipeline.predict(input_data)[0]
            predicted_salary = max(predicted_salary, 150000)
            predicted_salary = round(predicted_salary / 1000) * 1000

        salary_fmt = format_salary_indian(predicted_salary)
        salary_lkh = format_salary_lakhs(predicted_salary)
        monthly = format_salary_indian(predicted_salary / 12)

        # Salary display
        st.markdown(f"""
        <div class="salary-display">
            <div class="salary-label">Predicted Annual Salary</div>
            <div class="salary-amount">{salary_fmt}</div>
            <div class="salary-sub">{salary_lkh} per annum</div>
            <div class="salary-monthly">Monthly: <strong>{monthly}</strong></div>
        </div>
        """, unsafe_allow_html=True)

        # Summary
        emp_data = {
            'name': name.strip(), 'age': age, 'gender': gender,
            'education_level': education, 'domain': domain,
            'job_title': job_title, 'experience_years': experience,
            'skills_count': skills, 'certifications': certifications,
            'company_size': company_size, 'city_tier': city_tier,
            'performance_rating': performance,
        }
        summary_text, highlights = generate_summary(emp_data, predicted_salary)

        gender_emoji = '👨‍💼' if gender == 'Male' else ('👩‍💼' if gender == 'Female' else '🧑‍💼')

        st.markdown("### 📝 Employee Summary")
        st.markdown(f"""
        <div class="summary-card">
            <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1rem; padding-bottom:0.75rem; border-bottom: 1px solid rgba(148,163,184,0.1);">
                <div style="width:48px;height:48px;border-radius:12px;background:linear-gradient(135deg,#7c3aed,#06b6d4);display:flex;align-items:center;justify-content:center;font-size:1.4rem;">{gender_emoji}</div>
                <div>
                    <div style="font-weight:700;font-size:1.1rem;color:#f1f5f9;">{name.strip()}</div>
                    <div style="font-size:0.8rem;color:#64748b;">{job_title} • {domain}</div>
                </div>
            </div>
            <div class="summary-text">{summary_text}</div>
            <div class="highlight-tags">
                {''.join(f'<span class="highlight-tag">{h}</span>' for h in highlights)}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature Importance
        feat_imp = metadata.get('feature_importance', [])
        if feat_imp:
            st.markdown("### 📊 What Influences Salary Most?")
            max_imp = max(f['importance'] for f in feat_imp[:10])
            chart_html = ""
            for f in feat_imp[:10]:
                pct = (f['importance'] / max_imp * 100)
                label = f['feature'].replace('num__', '').replace('cat__', '').replace('_', ' ').title()
                chart_html += f"""
                <div class="imp-bar-container">
                    <span class="imp-bar-label" title="{f['feature']}">{label}</span>
                    <div class="imp-bar-track">
                        <div class="imp-bar-fill" style="width:{pct:.1f}%"></div>
                    </div>
                    <span class="imp-bar-value">{f['importance']*100:.1f}%</span>
                </div>
                """
            st.markdown(f'<div class="summary-card">{chart_html}</div>', unsafe_allow_html=True)

        st.balloons()

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with 🤖 Machine Learning & Streamlit • ML Mini Project<br>
    Employee Salary Prediction System — Salaries in ₹ INR
</div>
""", unsafe_allow_html=True)
