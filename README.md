# 🏠 Real State Price Prediction Project

A machine learning project to accurately predict house prices based on property features. Developed as part of my AI specialization journey.

## 📌 Table of Contents
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Algorithms](#-Algorithms)
- [Results](#-results)
- [Future Improvements](#-future-improvements)
- [License](#-license)

## 🎯 Business Problem
**Objective:** Develop a machine learning regression model to predict real estate sale prices by analyzing key property features like size, location, and condition.

**Impact:**
- Provides data-driven estimates for buyers, sellers, and real estate agents.
- Demonstrates the end-to-end process of a supervised machine learning project.

## 📊 Dataset
**Source:** [Real State Dataset on Kaggle](https://www.kaggle.com/datasets/shree1992/housedata/data)

**Features:**
- `bedrooms`
- `bathrooms`
- `sqft_living`
- `sqft_lot`
- `floors`
- `waterfront`
- `view`
- `condition`
- `grade`
- `sqft_above`
- `sqft_basement`
- `yr_built`
- `yr_renovated`
- `zipcode`
- `lat`
- `long`
- `sqft_living15`
- `sqft_lot15`
- `year`
- `sin_month`
- `cos_month`

## 📊 Preprocessing

### Feature Analysis and Model Selection
In this project the training pipeline was kept consistent across models, with the primary focus directed toward data preprocessing and feature engineering.

Analysis of feature correlations—both individual features and feature interactions (such as latitude and longitude combinations)—revealed a wide number of relationships with the target variable outside a simple direct linear correlation. These interactions indicated that either **Stacking** or **Mixture of experts (MoE)** architectures would provide the optimal balance of computational efficiency and predictive performance.

### Date Feature Engineering
The date field needed treating since it was provided in ISO 8601 format ('YYYYMMDDTHHMMSS'). A few transformation approaches were considered such as:

- **Integer encoding** (`YYYYMMDD`): This method does preserve chronological ordering but would fail to capture distance proportions

- **Separate components** (year, month, day): This method was discarded since it could hinder the model's ability to recognize temporal cycles(eg december is as close to january as to november as opposed to the numbers 12, 1 and 11)

The selected approach was to transform the date into three features:

- **Year** (integer)
- **Month (sine)** 
- **Month (cosine)**

This method preserve the cyclic nature of months while maintaining the proportional relationships between months, the method was intentionally omitted for days of the month. Analysis of the dataset confirmed the theoretical expectation that real estate transaction timing within a month is random and shows no correlation with purchase price.

### Additional Preprocessing Steps
- **Feature Scaling**: [Mention normalization/standardization if applied]
- **Categorical Variables**: [Explain encoding method used]
- **Outlier Treatment**: [Describe if you handled outliers in price or features]

## 🤖 Algorithms
- [Decision Tree](realestate/decision_tree)
- [Random Forest](realstate/random_forest)
- [SVM](realstate/SVM)
- [Neural Network](realstate/neural_network)
- [Linear Regression](realstate/linear_regression)
- [Polinomial Regression](realstate/polynomial_regression)

## 📈 Results
**Performance Comparison:**

|          Model          |    R2    |    MAE    |    MSE    |   RMSE    |
|-------------------------|----------|-----------|-----------|-----------|
|  Random Forest          |  ~0.874  |  ~6,8e04  |  ~1,6e10  |  ~1,2e05  |
|  Neural Network         |  ~0.850  |  ~8,2e04  |  ~2,0e10  |  ~1,4e05  |
|  Polynomial Regression  |  ~0.796  |  ~1,0e05  |  ~2,6e10  |  ~1,6e05  |
|  SVM                    |  ~0.764  |  ~8,5e04  |  ~3,1e10  |  ~1,7e05  |
|  Decision Tree          |  ~0.731  |  ~9,7e04  |  ~3,1e10  |  ~1,7e05  |
|  Linear Regression      |  ~0.700  |  ~1,3e05  |  ~4,6e10  |  ~2,1e05  |

**Key Insight:** The **Random Forest** model demonstrated the best performance, achieving the highest R² score and the lowest error rates.

## 🔧 Future Improvements
**Ensemble or MoE**
During database analysis was noted different types of relationship between features and the target variable which motivated testing algorythms on a few selected features (eg using latitude and longitude on a KNN algorithm to create an aritficial map to detect average price based on location)
