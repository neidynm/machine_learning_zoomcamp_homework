# Week 6: Predicting Fuel Efficiency Using Decision Trees and Ensemble Methods

## Project Overview

This project focuses on predicting car fuel efficiency (MPG - Miles Per Gallon) using machine learning techniques, specifically Decision Trees, Random Forests, and XGBoost. The dataset contains various vehicle characteristics such as engine displacement, horsepower, weight, and other features that influence fuel consumption.

## Dataset

**Source:** [Car Fuel Efficiency Dataset](https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv)

**Features:**
- `engine_displacement`: Engine size
- `num_cylinders`: Number of cylinders
- `horsepower`: Engine power
- `vehicle_weight`: Weight of the vehicle
- `acceleration`: Acceleration capability
- `model_year`: Year of manufacture
- `origin`: Manufacturing region (USA, Europe, Asia)
- `fuel_type`: Type of fuel (Gasoline, Diesel)
- `drivetrain`: Drive configuration (All-wheel, Front-wheel, Rear-wheel)
- `num_doors`: Number of doors

**Target Variable:**
- `fuel_efficiency_mpg`: Fuel efficiency in miles per gallon

## Project Structure

```
week_6/
├── data/
│   └── car_fuel_efficiency.csv
├── Homework.ipynb
└── README.md
```

## Methodology

### 1. Data Preparation

**Missing Value Treatment:**
- Filled missing values with 0 for:
  - `engine_displacement`
  - `num_cylinders`
  - `horsepower`
  - `num_doors`
  - `acceleration`

**Data Splitting:**
- **Training Set:** 60% of data
- **Validation Set:** 20% of data
- **Test Set:** 20% of data
- Random state: 1 for reproducibility

### 2. Feature Engineering

Used `DictVectorizer` with `sparse=False` to convert categorical features to numerical representation, creating one-hot encoded features for:
- Origin (USA, Europe, Asia)
- Fuel type (Gasoline, Diesel)
- Drivetrain type

### 3. Models Implemented

#### Decision Tree (Baseline)
- **Parameters:** `max_depth=1`
- **Purpose:** Identify the most important splitting feature
- **Key Finding:** The root split is based on `vehicle_weight`

#### Random Forest
**Configuration:**
- Number of estimators: 10
- Random state: 1
- n_jobs: -1 (use all CPU cores)

**Initial Results:**
- RMSE: 0.449

**Hyperparameter Tuning:**
Tested combinations of:
- `max_depth`: [10, 15, 20, 25]
- `n_estimators`: range(10, 201, 10)

**Results:**
| Max Depth | Average RMSE |
|-----------|--------------|
| 10        | 0.432        |
| 15        | 0.434        |
| 20        | 0.434        |
| 25        | 0.434        |

**Best Configuration:** `max_depth=10` with lowest average RMSE

#### Feature Importance Analysis

Using Random Forest with `n_estimators=10` and `max_depth=20`:

| Feature                      | Importance |
|------------------------------|------------|
| vehicle_weight               | 0.9592     |
| horsepower                   | 0.0160     |
| acceleration                 | 0.0115     |
| engine_displacement          | 0.0033     |
| model_year                   | 0.0032     |
| num_cylinders                | 0.0023     |
| num_doors                    | 0.0016     |
| origin=USA                   | 0.0006     |
| origin=Asia                  | 0.0005     |
| origin=Europe                | 0.0005     |
| drivetrain=All-wheel drive   | 0.0004     |
| fuel_type=Diesel             | 0.0004     |
| fuel_type=Gasoline           | 0.0003     |
| drivetrain=Front-wheel drive | 0.0003     |

**Key Insight:** Vehicle weight is by far the most important feature (95.92%), followed distantly by horsepower (1.6%) and acceleration (1.15%).

#### XGBoost

**Configuration:**
- `max_depth`: 6
- `min_child_weight`: 1
- `objective`: 'reg:squarederror'
- `num_boost_round`: 100
- `early_stopping_rounds`: 10

**Learning Rate Comparison:**

| Learning Rate (eta) | RMSE   |
|---------------------|--------|
| 0.3                 | 0.436  |
| 0.1                 | 0.425  |

**Best Configuration:** `eta=0.1` achieved the lowest RMSE

## Key Findings

1. **Most Important Feature:** Vehicle weight dominates fuel efficiency predictions, accounting for ~96% of feature importance
2. **Best Model:** Random Forest with `max_depth=10` provided the best balance of performance
3. **Learning Rate Impact:** Lower learning rate (0.1) in XGBoost yielded better results than 0.3
4. **Model Performance:** All ensemble methods (Random Forest, XGBoost) significantly outperformed the single Decision Tree

## Technical Stack

- **Python 3.11.0**
- **Libraries:**
  - pandas: Data manipulation
  - numpy: Numerical operations
  - scikit-learn: Decision Tree, Random Forest, preprocessing
  - xgboost: Gradient boosting
  - matplotlib: Visualization

## Installation

```bash
# Install required packages
pip install pandas numpy scikit-learn xgboost matplotlib

# For XGBoost specifically
pip install --upgrade pip
pip install xgboost --index-url https://pypi.org/simple
```

## Usage

1. **Load the notebook:**
```python
jupyter notebook Homework.ipynb
```

2. **Run all cells sequentially** to:
   - Download and prepare data
   - Train models
   - Evaluate performance
   - Visualize results

## Results Summary

- **Decision Tree (depth=1) RMSE:** Not calculated (visualization only)
- **Random Forest (n=10, depth=default) RMSE:** 0.449
- **Random Forest (optimized) RMSE:** 0.432
- **XGBoost (eta=0.1) RMSE:** 0.425

## Conclusions

1. Ensemble methods substantially improve prediction accuracy over single decision trees
2. Vehicle weight is the overwhelming predictor of fuel efficiency
3. Careful hyperparameter tuning (especially max_depth and learning rate) is crucial for optimal performance
4. XGBoost with proper configuration provides the best results for this regression task

## Future Improvements

- Feature engineering: Create interaction terms (e.g., weight × horsepower)
- Try different ensemble methods (e.g., Gradient Boosting, AdaBoost)
- Implement cross-validation for more robust evaluation
- Explore polynomial features for non-linear relationships
- Consider feature selection to reduce dimensionality

## Author

Neidy - ML Zoomcamp Participant

## Acknowledgments

- Dataset source: Alexey Grigorev
- Course: DataTalks.Club ML Zoomcamp
- Week 6: Decision Trees and Ensemble Learning
