# Lead Scoring with Logistic Regression 📊

A machine learning project to predict customer conversion rates using logistic regression, part of the ML Zoomcamp Week -3 curriculum.

## Project Overview

This project implements a lead scoring system to predict whether potential customers will convert based on their interaction patterns, demographics, and engagement metrics. Using logistic regression, we achieve a baseline accuracy of 70% and uncover interesting insights about feature importance.

## Table of Contents

- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Dataset

The project uses the Course Lead Scoring dataset from Alexey Grigorev's repository.

**Source**: https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv

**Size**: 1,462 records with 9 features

**Target Variable**: `converted` (binary: 0 or 1)

## Features

### Numerical Features
- `number_of_courses_viewed` - Number of courses browsed by the lead
- `annual_income` - Lead's annual income
- `interaction_count` - Number of interactions with the platform
- `lead_score` - Pre-calculated lead score

### Categorical Features
- `lead_source` - Origin of the lead (paid_ads, social_media, events, referral)
- `industry` - Lead's industry sector (retail, finance, healthcare, education, etc.)
- `employment_status` - Current employment status (employed, unemployed, self_employed)
- `location` - Geographic location (south_america, australia, europe, etc.)

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab

### Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib jupyter pathlib
```

Or install via requirements.txt:

```bash
pip install -r requirements.txt
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/lead-scoring-logistic-regression.git
cd lead-scoring-logistic-regression
```

## Usage

### Quick Start

1. Open the Jupyter notebook:
```bash
jupyter notebook "Homework Bank Churn (1).ipynb"
```

2. Run all cells sequentially to:
   - Download and prepare the data
   - Perform exploratory data analysis
   - Train the logistic regression model
   - Evaluate feature importance
   - Test different regularization parameters

### Using the Trained Model

```python
# Example usage after training
import pandas as pd

# Create a new lead profile
new_lead = {
    'lead_source': 'paid_ads',
    'industry': 'retail',
    'number_of_courses_viewed': 3,
    'annual_income': 75000,
    'employment_status': 'employed',
    'location': 'europe',
    'interaction_count': 5,
    'lead_score': 0.75
}

# Transform and predict
lead_dict = [new_lead]
X_new = dv.transform(lead_dict)
probability = model.predict_proba(X_new)[0, 1]
prediction = model.predict(X_new)[0]

print(f"Conversion probability: {probability:.2%}")
print(f"Will convert: {'Yes' if prediction == 1 else 'No'}")
```

## 📁 Project Structure

```
week_3/
│
├── Homework Bank Churn.ipynb  # Main notebook with analysis
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── data/                          # Data directory (created on first run)
│   └── course_lead_scoring.csv   # Downloaded dataset
```

## Methodology

### 1. Data Preparation
- Handle missing values (NA for categorical, 0 for numerical)
- Split data: 60% train, 20% validation, 20% test
- Feature encoding using DictVectorizer

### 2. Exploratory Data Analysis
- Correlation analysis of numerical features
- Mutual information scores for categorical features
- Distribution analysis of target variable

### 3. Model Training
- Baseline logistic regression with all features
- Feature importance analysis via ablation study
- Hyperparameter tuning (C values: 0.01, 0.1, 1, 10, 100)

### 4. Evaluation
- Accuracy metric for model performance
- Feature importance ranking
- Impact analysis of individual features

## 🔍 Key Findings

### 1. Feature Correlations
- **Weak correlations** between numerical features (max: 0.027)
- Features are largely independent, no multicollinearity issues

### 2. Mutual Information Scores
```
lead_source          0.026
employment_status    0.013
industry            0.012
location            0.002
```

### 3. Feature Importance 
- **Most Important**: `annual_income` (accuracy drops to 85.3% when removed)
- **Least Important**: `industry` (0.0% difference when removed)
- **Critical Features**: `number_of_courses_viewed` and `interaction_count`

### 4. Regularization Impact
- **Surprising finding**: All C values (0.01 to 100) yielded identical 70% accuracy
- Model is stable and not sensitive to regularization strength

## 📊 Results

| Metric | Value |
|--------|--------|
| **Baseline Accuracy** | 70.0% |
| **Best C Value** | 0.01 - 100 (all equal) |
| **Most Impactful Feature** | annual_income |
| **Least Impactful Feature** | industry |
| **Training Time** | < 1 second |

### Model Performance by Feature Set

| Features Removed | Accuracy | Impact |
|-----------------|----------|---------|
| None (Baseline) | 69.97% | - |
| industry | 69.97% | 0.00% |
| employment_status | 69.62% | -0.34% |
| lead_source | 70.31% | +0.34% |
| location | 70.99% | +1.02% |
| number_of_courses_viewed | 55.63% | -14.33% |
| interaction_count | 55.63% | -14.33% |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Alexey Grigorev** - For the excellent ML Zoomcamp course and dataset
- **ML Zoomcamp Community** - For support and inspiration
- **scikit-learn Documentation** - For comprehensive guides and examples

## 📧 Contact

For questions or feedback, please open an issue in this repository.

---

**Note**: This project is part of the ML Zoomcamp Week -3 homework. The focus is on understanding logistic regression fundamentals rather than achieving state-of-the-art performance.

## 📚 References

- [ML Zoomcamp Course](https://github.com/alexeygrigorev/mlbookcamp-code)
- [Logistic Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Dataset Source](https://github.com/alexeygrigorev/datasets)
