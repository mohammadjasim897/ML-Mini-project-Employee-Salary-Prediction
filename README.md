# 💰 SalaryIQ — AI Employee Salary Prediction System

A full-stack **Machine Learning** mini project that predicts employee salaries in **₹ Indian Rupees (INR)** across 10 industry domains. Built with Python, Flask, Scikit-learn, and a stunning dark-themed web UI. Includes an intelligent **employee profile summary** generator.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📌 Overview

**SalaryIQ** trains on **5,000+ synthetic employee records** spanning 10 industry domains (IT, Finance, Healthcare, Marketing, HR, Education, Engineering, Legal, Sales, Operations) and uses advanced **ensemble ML models** to predict salaries in ₹ INR based on employee attributes.

### ✨ Features

- 🧠 **ML Models** — Compares Linear Regression, Random Forest & Gradient Boosting; auto-selects the best
- 💰 **INR Predictions** — Salary displayed in Indian Rupee format (₹ Lakhs/Crores)
- 📝 **Employee Summary** — Generates a natural-language profile summary for each prediction
- 🎨 **Premium UI** — Dark glassmorphism theme with animated salary counter & micro-animations
- 📊 **Feature Importance** — Visual breakdown of factors influencing salary
- 📱 **Responsive** — Works on desktop, tablet, and mobile
- 🏢 **10 Domains** — IT, Finance, Healthcare, Marketing, HR, Education, Engineering, Legal, Sales, Operations

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Models | Scikit-learn (Random Forest, Gradient Boosting, Linear Regression) |
| Backend | Python Flask |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Dataset | Synthetic (5,000+ records, 10 domains, INR salaries) |
| Serialization | Pickle |

---

## 🏗️ Architecture

```
User Input → Form Validation → Flask API (/api/predict)
                                    ↓
                            Load Pickle Pipeline
                            (Preprocessor + Model)
                                    ↓
                            Predict Salary (INR)
                                    ↓
                        Generate Employee Summary
                                    ↓
                    Return Prediction + Summary → Render UI
```

---

## 📁 Project Structure

```
ML-Mini-project-Employee-Salary-Prediction/
├── data/
│   ├── generate_dataset.py           # Synthetic dataset generator
│   └── employee_salary_dataset.csv   # Generated dataset (5000 records)
├── model/
│   ├── train_model.py                # ML training pipeline
│   ├── salary_model.pkl              # Trained model (generated)
│   ├── preprocessor.pkl              # Feature preprocessor (generated)
│   └── model_metadata.pkl            # Model info & dropdown options
├── static/
│   ├── css/
│   │   └── style.css                 # Premium dark-themed UI
│   └── js/
│       └── app.js                    # Frontend logic
├── templates/
│   └── index.html                    # Main web page
├── app.py                            # Flask server + API endpoints
├── requirements.txt                  # Python dependencies
├── Procfile                          # Deployment (Gunicorn)
├── render.yaml                       # Render deployment config
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Demo link: http://localhost:8501/




Then open 


## 🧠 How the ML Model Works

### 1. Dataset Generation
- 5,000+ synthetic Indian employee records
- 10 industry domains with domain-specific job titles
- Realistic salary computation based on experience, education, city tier, performance, etc.
- Salary range: ₹2,00,000 – ₹75,00,000+

### 2. Feature Engineering
| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numerical | Employee age (22–60) |
| `gender` | Categorical | Male, Female, Other |
| `education_level` | Categorical | Bachelor's, Master's, PhD |
| `domain` | Categorical | 10 industry domains |
| `job_title` | Categorical | Domain-specific titles |
| `experience_years` | Numerical | Years of work experience |
| `skills_count` | Numerical | Number of technical skills |
| `certifications` | Numerical | Professional certifications |
| `company_size` | Categorical | Startup, SME, MNC, Enterprise |
| `city_tier` | Categorical | Tier 1, Tier 2, Tier 3 |
| `performance_rating` | Numerical | Rating (1.0–5.0) |

### 3. Preprocessing
- **StandardScaler** for numerical features
- **OneHotEncoder** for categorical features
- Wrapped in a `ColumnTransformer` for clean pipeline

### 4. Models Compared
| Model | Strengths |
|-------|-----------|
| Linear Regression | Baseline, interpretable |
| **Random Forest** ⭐ | Handles non-linearity, robust |
| Gradient Boosting | High accuracy, sequential learning |

### 5. Evaluation Metrics
- **R² Score** — Proportion of variance explained
- **MAE** — Mean Absolute Error (in ₹)
- **RMSE** — Root Mean Squared Error (in ₹)

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve main web page |
| `/api/metadata` | GET | Dropdown options & model info |
| `/api/predict` | POST | Predict salary with employee summary |

### Example Request

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Rahul Sharma",
    "age": 28,
    "gender": "Male",
    "education_level": "Master'\''s",
    "domain": "IT",
    "job_title": "Data Scientist",
    "experience_years": 5,
    "skills_count": 8,
    "certifications": 2,
    "company_size": "MNC",
    "city_tier": "Tier 1",
    "performance_rating": 4.2
  }'
```

### Example Response

```json
{
  "predicted_salary": 1245000,
  "salary_formatted": "₹12,45,000",
  "salary_lakhs": "₹12.45 Lakhs",
  "monthly_salary": "₹1,03,750",
  "summary": {
    "text": "Rahul Sharma is a 28-year-old Data Scientist working in the IT domain...",
    "highlights": ["⭐ Top performer", "🎓 Post-graduate degree", "🏢 Working at MNC"]
  },
  "model_used": "Random Forest",
  "model_accuracy": 95.2
}
```

---

## 📊 Employee Summary

Each prediction generates a natural-language summary like:

> *"Rahul Sharma is a 28-year-old Data Scientist working in the IT domain. He is an early-career professional with 5.0 years of experience, holding a Master's degree. Currently employed at an MNC company in a Tier 1 city, Rahul demonstrates a strong performance rating of 4.2/5.0. With 8 technical skills and 2 certifications, his predicted annual salary is ₹12,45,000 (₹12.45 Lakhs) per annum."*

---

## 🎨 UI Features

- 🌑 **Dark glassmorphism** theme with animated gradient background
- 🔢 **Animated salary counter** in Indian numbering format
- 📊 **Feature importance chart** with animated bars
- ⭐ **Employee highlights** as badge tags
- 🎛️ **Interactive sliders** for numerical inputs
- 🔗 **Domain-chained dropdowns** (domain → job titles)

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👥 Authors

- **Mohammad Jasim** — [@mohammadjasim897](https://github.com/mohammadjasim897)

---

*Built as a Mini Project for Machine Learning* 🎓
