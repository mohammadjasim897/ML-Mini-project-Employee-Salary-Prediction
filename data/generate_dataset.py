"""
Employee Salary Dataset Generator
==================================
Generates a realistic synthetic dataset of 5,000+ Indian employees
across 10 industry domains with salaries in INR (₹).
"""

import os
import random
import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

NUM_RECORDS = 5000

# ── Domain → Job Titles Mapping ─────────────────────────────────────
DOMAIN_JOBS = {
    'IT': [
        'Software Developer', 'Data Scientist', 'DevOps Engineer',
        'Frontend Developer', 'Backend Developer', 'Full Stack Developer',
        'QA Engineer', 'Cloud Architect', 'ML Engineer', 'Cybersecurity Analyst'
    ],
    'Finance': [
        'Financial Analyst', 'Accountant', 'Investment Banker',
        'Risk Analyst', 'Auditor', 'Tax Consultant',
        'Portfolio Manager', 'Credit Analyst', 'Compliance Officer', 'Treasury Analyst'
    ],
    'Healthcare': [
        'Doctor', 'Nurse', 'Pharmacist', 'Medical Lab Technician',
        'Hospital Administrator', 'Physiotherapist', 'Radiologist',
        'Dentist', 'Clinical Research Associate', 'Public Health Specialist'
    ],
    'Marketing': [
        'Marketing Manager', 'SEO Specialist', 'Content Writer',
        'Brand Manager', 'Digital Marketing Analyst', 'Social Media Manager',
        'Market Research Analyst', 'PR Specialist', 'Growth Hacker', 'Campaign Manager'
    ],
    'HR': [
        'HR Manager', 'Recruiter', 'Training Coordinator',
        'Compensation Analyst', 'HR Business Partner', 'Talent Acquisition Lead',
        'Employee Relations Specialist', 'Payroll Specialist', 'HR Generalist', 'HRIS Analyst'
    ],
    'Education': [
        'Professor', 'Lecturer', 'Research Associate',
        'Academic Counselor', 'Curriculum Designer', 'Education Consultant',
        'Lab Instructor', 'Teaching Assistant', 'Dean', 'Instructional Designer'
    ],
    'Engineering': [
        'Mechanical Engineer', 'Civil Engineer', 'Electrical Engineer',
        'Chemical Engineer', 'Structural Engineer', 'Project Manager',
        'Design Engineer', 'Quality Engineer', 'Process Engineer', 'Site Engineer'
    ],
    'Legal': [
        'Corporate Lawyer', 'Legal Advisor', 'Paralegal',
        'Compliance Manager', 'Patent Attorney', 'Legal Analyst',
        'Contract Specialist', 'Litigation Associate', 'In-House Counsel', 'Legal Consultant'
    ],
    'Sales': [
        'Sales Manager', 'Account Executive', 'Business Development Manager',
        'Sales Engineer', 'Key Account Manager', 'Inside Sales Representative',
        'Regional Sales Head', 'Territory Manager', 'Pre-Sales Consultant', 'Channel Partner Manager'
    ],
    'Operations': [
        'Operations Manager', 'Supply Chain Analyst', 'Logistics Coordinator',
        'Procurement Manager', 'Warehouse Manager', 'Process Improvement Analyst',
        'Inventory Analyst', 'Fleet Manager', 'Operations Analyst', 'Facility Manager'
    ],
}

# ── Base Salary Ranges by Domain (annual, INR) ──────────────────────
DOMAIN_BASE_SALARY = {
    'IT':           (400000, 2500000),
    'Finance':      (350000, 2200000),
    'Healthcare':   (300000, 3000000),
    'Marketing':    (300000, 1800000),
    'HR':           (280000, 1600000),
    'Education':    (250000, 1500000),
    'Engineering':  (350000, 2000000),
    'Legal':        (400000, 2800000),
    'Sales':        (280000, 2000000),
    'Operations':   (300000, 1800000),
}

# ── Multipliers ──────────────────────────────────────────────────────
EDUCATION_MULTIPLIER = {
    "Bachelor's": 1.0,
    "Master's":   1.20,
    "PhD":        1.45,
}

COMPANY_SIZE_MULTIPLIER = {
    'Startup':    0.85,
    'SME':        0.95,
    'MNC':        1.15,
    'Enterprise': 1.25,
}

CITY_TIER_MULTIPLIER = {
    'Tier 1': 1.20,
    'Tier 2': 1.00,
    'Tier 3': 0.80,
}

TIER_1_CITIES = ['Mumbai', 'Bangalore', 'Delhi', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune']
TIER_2_CITIES = ['Jaipur', 'Lucknow', 'Chandigarh', 'Ahmedabad', 'Kochi', 'Indore', 'Nagpur']
TIER_3_CITIES = ['Bhopal', 'Patna', 'Ranchi', 'Dehradun', 'Raipur', 'Guwahati', 'Mysuru']

GENDER_OPTIONS = ['Male', 'Female', 'Other']

INDIAN_MALE_NAMES = [
    'Aarav', 'Vivaan', 'Aditya', 'Vihaan', 'Arjun', 'Sai', 'Reyansh',
    'Ayaan', 'Krishna', 'Ishaan', 'Rohan', 'Rahul', 'Amit', 'Vikram',
    'Karan', 'Mohit', 'Nikhil', 'Pranav', 'Rajesh', 'Suresh',
    'Deepak', 'Ankit', 'Harsh', 'Gaurav', 'Manish',
]

INDIAN_FEMALE_NAMES = [
    'Ananya', 'Aadhya', 'Aanya', 'Diya', 'Myra', 'Sara', 'Ira',
    'Priya', 'Neha', 'Pooja', 'Shreya', 'Kavya', 'Riya', 'Meera',
    'Nisha', 'Sneha', 'Anjali', 'Divya', 'Swati', 'Tanvi',
    'Sakshi', 'Isha', 'Kritika', 'Simran', 'Bhavna',
]

INDIAN_SURNAMES = [
    'Sharma', 'Verma', 'Gupta', 'Singh', 'Kumar', 'Patel', 'Reddy',
    'Nair', 'Iyer', 'Joshi', 'Mehta', 'Chatterjee', 'Banerjee', 'Das',
    'Malik', 'Rao', 'Pillai', 'Thakur', 'Mishra', 'Chopra',
    'Bhat', 'Agarwal', 'Saxena', 'Kapoor', 'Deshpande',
]


def generate_name(gender):
    """Generate a realistic Indian name based on gender."""
    if gender == 'Male':
        first = random.choice(INDIAN_MALE_NAMES)
    elif gender == 'Female':
        first = random.choice(INDIAN_FEMALE_NAMES)
    else:
        first = random.choice(INDIAN_MALE_NAMES + INDIAN_FEMALE_NAMES)
    surname = random.choice(INDIAN_SURNAMES)
    return f"{first} {surname}"


def compute_salary(domain, education, experience, company_size, city_tier,
                   performance, skills_count, certifications):
    """
    Compute a realistic salary based on multiple factors.
    Returns salary in INR (annual).
    """
    base_min, base_max = DOMAIN_BASE_SALARY[domain]

    # Experience-based progression (non-linear)
    exp_factor = 1.0 + (experience ** 0.7) * 0.12

    # Education boost
    edu_factor = EDUCATION_MULTIPLIER[education]

    # Company size impact
    comp_factor = COMPANY_SIZE_MULTIPLIER[company_size]

    # City tier impact
    city_factor = CITY_TIER_MULTIPLIER[city_tier]

    # Performance boost (rating 1-5 → 0.85 to 1.25)
    perf_factor = 0.75 + (performance / 5.0) * 0.50

    # Skills and certifications minor boost
    skill_factor = 1.0 + (skills_count * 0.015) + (certifications * 0.03)

    # Calculate base salary within domain range
    base_salary = base_min + (base_max - base_min) * min(experience / 25.0, 1.0) * 0.6

    # Apply all multipliers
    salary = base_salary * exp_factor * edu_factor * comp_factor * city_factor * perf_factor * skill_factor

    # Add random noise (±8%)
    noise = np.random.normal(1.0, 0.08)
    salary *= max(noise, 0.75)

    # Clamp to realistic range
    salary = max(salary, 200000)
    salary = min(salary, 7500000)

    # Round to nearest 1000
    return round(salary / 1000) * 1000


def generate_dataset(n=NUM_RECORDS):
    """Generate the complete employee dataset."""
    records = []

    domains = list(DOMAIN_JOBS.keys())
    education_levels = list(EDUCATION_MULTIPLIER.keys())
    company_sizes = list(COMPANY_SIZE_MULTIPLIER.keys())
    city_tiers = list(CITY_TIER_MULTIPLIER.keys())

    for i in range(n):
        # Basic demographics
        gender = random.choices(GENDER_OPTIONS, weights=[0.55, 0.40, 0.05])[0]
        name = generate_name(gender)
        age = random.randint(22, 58)

        # Education (weighted: Bachelor's more common)
        education = random.choices(education_levels, weights=[0.55, 0.35, 0.10])[0]

        # Domain and job title
        domain = random.choice(domains)
        job_title = random.choice(DOMAIN_JOBS[domain])

        # Experience (correlated with age, with some randomness)
        min_exp = max(0, age - 26) if education == "Bachelor's" else max(0, age - 28)
        max_exp = max(0, age - 22)
        experience = round(random.uniform(max(0, min_exp * 0.5), max_exp), 1)

        # Skills and certifications
        skills_count = random.randint(2, min(15, 3 + int(experience * 0.5)))
        certifications = random.randint(0, min(5, 1 + int(experience * 0.15)))

        # Company size (weighted)
        company_size = random.choices(company_sizes, weights=[0.20, 0.30, 0.35, 0.15])[0]

        # City tier (weighted)
        city_tier = random.choices(city_tiers, weights=[0.45, 0.35, 0.20])[0]
        if city_tier == 'Tier 1':
            city = random.choice(TIER_1_CITIES)
        elif city_tier == 'Tier 2':
            city = random.choice(TIER_2_CITIES)
        else:
            city = random.choice(TIER_3_CITIES)

        # Performance rating
        performance = round(random.uniform(1.5, 5.0), 1)

        # Compute salary
        salary = compute_salary(
            domain, education, experience, company_size,
            city_tier, performance, skills_count, certifications
        )

        records.append({
            'employee_id': f'EMP{i+1:05d}',
            'name': name,
            'age': age,
            'gender': gender,
            'education_level': education,
            'domain': domain,
            'job_title': job_title,
            'experience_years': experience,
            'skills_count': skills_count,
            'certifications': certifications,
            'company_size': company_size,
            'city_tier': city_tier,
            'city': city,
            'performance_rating': performance,
            'salary_inr': salary,
        })

    return pd.DataFrame(records)


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  Employee Salary Dataset Generator")
    print("=" * 60)

    df = generate_dataset(NUM_RECORDS)

    # Save to CSV
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'employee_salary_dataset.csv')
    df.to_csv(output_path, index=False)

    print(f"\n✅ Generated {len(df)} employee records")
    print(f"📁 Saved to: {output_path}")
    print(f"\n📊 Dataset Summary:")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Domains: {df['domain'].nunique()} ({', '.join(df['domain'].unique())})")
    print(f"\n💰 Salary Statistics (INR):")
    print(f"   Min:    ₹{df['salary_inr'].min():>12,.0f}")
    print(f"   Max:    ₹{df['salary_inr'].max():>12,.0f}")
    print(f"   Mean:   ₹{df['salary_inr'].mean():>12,.0f}")
    print(f"   Median: ₹{df['salary_inr'].median():>12,.0f}")
    print(f"\n📈 Salary by Domain (Mean):")
    for domain, grp in df.groupby('domain')['salary_inr']:
        print(f"   {domain:<15} ₹{grp.mean():>12,.0f}")
    print()
