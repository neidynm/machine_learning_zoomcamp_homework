# Week 1: Introduction to Machine Learning 

## Module Overview

This module introduces the fundamental concepts of Machine Learning and sets up the foundation for the entire ML Zoomcamp course. We explore the differences between ML and rule-based systems, understand the CRISP-DM methodology, and get hands-on with essential Python libraries.

## Topics Covered

- **1.1** Introduction to Machine Learning
- **1.2** ML vs Rule-Based Systems
- **1.3** Supervised Machine Learning
- **1.4** CRISP-DM (Cross-Industry Standard Process for Data Mining)
- **1.5** The Modelling Step (Model Selection Process)
- **1.6** Setting up the Environment
- **1.7** Introduction to NumPy
- **1.8** Linear Algebra Refresher
- **1.9** Introduction to Pandas
- **1.10** Summary

## Homework Assignment

The homework focuses on practical data manipulation using NumPy and Pandas with the **Car Fuel Efficiency Dataset**.

### Homework Questions & Solutions:

#### Q1. Pandas Version ✅
Verified Pandas installation and version
```python
import pandas as pd
pd.__version__  # Output: '2.3.1'
```

#### Q2. Records Count ✅
**Question:** How many rows are in the dataset?
```python
number_of_rows = len(car_dataset)
# Answer: 9704 rows
```

#### Q3. Fuel Types ✅
**Question:** How many fuel types are presented in the dataset?
```python
num_fuel_types = car_dataset['fuel_type'].nunique()
# Answer: 2 fuel types
```

#### Q4. Missing Values ✅
**Question:** How many columns in the dataset have missing values?
```python
car_dataset.isnull().sum()
# Columns with missing values:
# - num_cylinders: 482
# - horsepower: 708
# - acceleration: 930
# - num_doors: 502
# Answer: 4 columns have missing values
```

#### Q5. Maximum Fuel Efficiency ✅
**Question:** What's the maximum fuel efficiency of cars from Asia?
```python
max_cars_asia = car_dataset[car_dataset['origin'].isin(['Asia'])]['fuel_efficiency_mpg'].max()
# Answer: 23.759 mpg
```

#### Q6. Median Horsepower ✅
**Question:** Fill missing horsepower values with mode and recalculate median
```python
# Original median
median_horsepower = car_dataset['horsepower'].median()  # 149.0

# Fill with most frequent value
most_frequent = car_dataset['horsepower'].mode()[0]
car_dataset['horsepower'].fillna(most_frequent, inplace=True)

# New median
second_median = car_dataset['horsepower'].median()  # 149.0
# Answer: Median remains 149.0
```

#### Q7. Linear Regression Implementation ✅
**Question:** Implement linear regression using matrix operations
```python
# Filter Asia cars and select features
cars_asia = car_dataset[car_dataset['origin'] == 'Asia']
X = cars_asia[['vehicle_weight', 'model_year']].iloc[:7].to_numpy()

# Matrix operations
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)

# Target values
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

# Calculate weights
w = XTX_inv @ X.T @ y
# Answer: Sum of weights = 0.5188
```

## 🔧 Technical Stack

### Libraries Used:
```python
import pandas as pd        # Version: 2.3.1
import numpy as np         # For numerical operations
from numpy.linalg import inv  # For matrix inversion
```

### Dataset Information:
- **Name:** Car Fuel Efficiency Dataset
- **Source:** [Alexey Grigorev's Datasets Repository](https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv)
- **Size:** 9,704 records
- **Features:** 11 columns including:
  - `engine_displacement`
  - `num_cylinders`
  - `horsepower`
  - `vehicle_weight`
  - `acceleration`
  - `model_year`
  - `origin` (USA, Europe, Asia)
  - `fuel_type` (Gasoline, Diesel)
  - `drivetrain`
  - `num_doors`
  - `fuel_efficiency_mpg` (target variable)

## 💻 Running the Notebook

1. **Clone the repository:**
```bash
git clone https://github.com/neidynm/machine_learning_zoomcamp_homework.git
cd machine_learning_zoomcamp_homework/01-intro
```

2. **Set up environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pandas numpy jupyter
```

3. **Launch Jupyter Notebook:**
```bash
jupyter notebook intro-01.ipynb
```

## 📈 Key Learnings

### Data Exploration Skills:
- **Dataset inspection:** Understanding structure and dimensions
- **Missing value analysis:** Identifying and handling null values
- **Data filtering:** Using boolean indexing with Pandas
- **Statistical analysis:** Computing median, mode, and max values

### NumPy & Linear Algebra:
- **Matrix operations:** Transpose, multiplication, and inversion
- **Array manipulation:** Converting DataFrames to NumPy arrays
- **Linear regression basics:** Implementing the normal equation
  - Formula: `w = (X'X)^(-1) X'y`
  - Understanding the mathematical foundation of ML algorithms

### Pandas Proficiency:
- **Data loading:** Reading CSV files from URLs
- **Column selection:** Using bracket notation and `.iloc`
- **Conditional filtering:** Using `.isin()` for categorical filtering
- **Missing value handling:** Using `.fillna()` with statistical values
- **Aggregation functions:** `.nunique()`, `.median()`, `.mode()`, `.max()`

## Practical Applications

This homework demonstrates fundamental skills essential for ML engineering:
1. **Data Quality Assessment** - Identifying missing values and data types
2. **Statistical Analysis** - Understanding data distribution
3. **Feature Engineering** - Selecting and preparing features for modeling
4. **Matrix Operations** - Foundation for understanding ML algorithms
5. **Linear Regression** - First implementation of an ML algorithm from scratch

## Additional Resources

- [Course GitHub Repository](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/01-intro)
- [Course Videos - Module 1](https://www.youtube.com/playlist?list=PL3MmuxUbc_hL5Jq_B5DPRgRBnBFwPxhZo)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Linear Regression Theory](https://en.wikipedia.org/wiki/Linear_regression)

## My Learning Journey

📝 Follow my ML Zoomcamp journey on Medium: [ML Zoomcamp 2025 Reading List](https://medium.com/@neidy.tunzine/list/ml-zoomcamp-2025-7129a148fd20)

## Completion Status

- [x] Watch all video lectures (1.1 - 1.11)
- [x] Set up Python environment
- [x] Complete homework assignment
- [x] Implement linear regression from scratch
- [x] Submit homework for evaluation
- [x] Document learning progress

## 🚀 Next Steps

Moving forward to **Week 2: Machine Learning for Regression** where we'll:
- Dive deeper into linear regression
- Learn about feature engineering
- Implement regularization techniques
- Work with real-world regression problems

---

**Cohort:** ML Zoomcamp 2025
