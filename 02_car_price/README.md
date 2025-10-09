# Car Fuel Efficiency Prediction with Linear Regression

The resolution of the homework for week 2. A Machine learning project that predicts car fuel efficiency (MPG) using linear regression. This project is part of the ML Zoomcamp Week 2 coursework, demonstrating fundamental regression concepts and best practices in data preprocessing and model evaluation.

## 🎯 Project Overview

This project implements linear regression from scratch to predict how many miles per gallon (MPG) a car can achieve based on its physical characteristics. The implementation explores various preprocessing techniques, regularization methods, and model validation strategies.

**Key Objectives:**
- Predict car fuel efficiency based on vehicle characteristics
- Implement linear regression using the Normal Equation
- Explore data preprocessing techniques and their impact
- Apply regularization to improve model stability
- Validate model performance across different data splits

## 📊 Dataset

The project uses an automotive dataset containing vehicle specifications and their corresponding fuel efficiency ratings.

**Features:**
- `engine_displacement`: Size of the engine
- `horsepower`: Power output of the engine
- `vehicle_weight`: Total weight of the vehicle
- `model_year`: Manufacturing year

**Target Variable:**
- `fuel_efficiency_mpg`: Fuel efficiency in miles per gallon

**Data Source:** [Car Fuel Efficiency Dataset](https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv)

## 🔧 Technical Implementation

### Key Features

1. **Custom Linear Regression Implementation**
   - Built from scratch using NumPy
   - Analytical solution via Normal Equation
   - No dependency on scikit-learn for core algorithm

2. **Data Preprocessing**
   - Missing value imputation (tested zero-fill vs median)
   - Log transformation of target variable
   - 60-20-20 train-validation-test split with shuffling

3. **Regularization**
   - Ridge Regression (L2 regularization)
   - Hyperparameter tuning for optimal λ value
   - Comparison of regularization strengths

4. **Model Validation**
   - Cross-validation with multiple random seeds
   - RMSE evaluation in both log and original space
   - Stability analysis across different data splits

### Mathematical Foundation

**Linear Regression Equation:**
```
y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

**Normal Equation:**
```
w = (XᵀX)⁻¹Xᵀy
```

**Ridge Regression (with regularization):**
```
w = (XᵀX + λI)⁻¹Xᵀy
```

## 📈 Results

### Model Performance

- **Final Test RMSE (log space):** 0.039
- **Final Test RMSE (original space):** 0.607 MPG
- **Model Stability (std dev across 10 seeds):** 0.001

### Key Findings

1. **Missing Value Handling:** Median imputation improved RMSE by ~8% compared to zero-filling
2. **Optimal Regularization:** λ = 0.001 provided the best balance
3. **Model Stability:** Low standard deviation (0.001) across different random seeds confirms robust performance
4. **Practical Accuracy:** Average prediction error of approximately 0.6 MPG

## 🚀 Getting Started

### Prerequisites

```bash
numpy
pandas
matplotlib
seaborn
```

### Installation

```bash
# Clone the repository
git clone https://github.com/neidynm/machine_learning_zoomcamp_homework.git

# Navigate to the project directory
cd machine_learning_zoomcamp_homework/02_car_price

# Install dependencies
pip install numpy pandas matplotlib seaborn
```

### Usage

Open and run the Jupyter notebook:

```bash
jupyter notebook car_fuel_efficiency.ipynb
```

## 📚 Project Structure

```
02_car_price/
│
├── car_fuel_efficiency.ipynb    # Main project notebook
└── README.md                    # Project documentation
```

## 🔍 Key Learnings

1. **Feature Engineering Impact:** Proper handling of missing values can significantly improve model performance
2. **Regularization Benefits:** Even small regularization (λ = 0.001) enhances numerical stability
3. **Validation Importance:** Testing with multiple random seeds reveals true model robustness
4. **Implementation Value:** Building algorithms from scratch deepens understanding of underlying mathematics
5. **Data Shuffling:** Critical for preventing temporal bias in train-test splits

## 📝 Methodology

### 1. Exploratory Data Analysis
- Distribution analysis of fuel efficiency
- Missing value detection (708 missing horsepower values)
- Outlier identification using box plots

### 2. Data Preprocessing
- Missing value imputation comparison
- Log transformation of target variable
- Stratified data splitting with shuffling

### 3. Model Development
- Implementation of linear regression from scratch
- Application of Ridge regularization
- Hyperparameter tuning

### 4. Model Evaluation
- RMSE calculation in log and original space
- Stability testing across multiple random seeds
- Final model testing on holdout set

## 🔗 Related Resources

- 📖 **Medium Article:** [ML Zoomcamp Week 2: Predicting Car Fuel Efficiency with Linear Regression](https://medium.com/@neidy.tunzine/ml-zoomcamp-week-2-predicting-car-fuel-efficiency-with-linear-regression-867e7a8a57b4)
- 🎓 **Course:** ML Zoomcamp by Alexey Grigorev
- 📊 **Dataset:** [Car Fuel Efficiency CSV](https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv)

## 👤 Author

**Neidy Tunzine**
- Medium: [@neidy.tunzine](https://medium.com/@neidy.tunzine)
- GitHub: [@neidynm](https://github.com/neidynm)

## 📄 License

This project is part of the ML Zoomcamp coursework and is available for educational purposes.

## 🙏 Acknowledgments

- ML Zoomcamp course by Alexey Grigorev
- Dataset provided by the ML Zoomcamp community
- IBM documentation on linear regression

---

⭐ If you found this project helpful, please consider giving it a star!

**Note:** This project uses extracts from the internet and summarizes lessons, notes, and code from the ML Zoomcamp course.
