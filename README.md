# House Price Prediction

This project aims to predict house prices using machine learning techniques. The dataset contains various features such as size, location, and amenities.

# Key Features

* **Exploratory Data Analysis (EDA):** The dataset was explored and visualized using Python libraries like Pandas, NumPy, Matplotlib, and Seaborn to understand the distribution of features and relationships between variables.

* **Data Preprocessing:** Preprocessing steps were conducted to handle missing values and prepare the data for modeling. Feature engineering techniques such as log transformations and one-hot encoding were applied to improve model performance.

* **Model Evaluation:** Multiple machine learning models were evaluated, including Linear Regression, Random Forest, and Gradient Boosting. Model performance was assessed using appropriate metrics such as mean squared error and R-squared.

* **Hyperparameter Tuning:** GridSearchCV was used to tune the hyperparameters of the models and optimize their performance. This helped in improving the accuracy and generalization of the models.

* **Model Deployment:** Once the best performing model was selected, it was deployed to make predictions on new data. This involved saving the trained model and building a simple interface for users to input house features and get price predictions.

# Code Snippets

```python
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
data = pd.read_csv("/content/housing.csv")

# Exploratory Data Analysis
data.info()
data.describe()
sns.pairplot(data)

# Data Preprocessing
data.dropna(inplace=True)
data = pd.get_dummies(data)

# Model Training and Evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = data.drop('median_house_value', axis=1)
y = data['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```
# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
param_grid = {...}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Model Deployment
import joblib
joblib.dump(best_model, 'house_price_prediction_model.pkl')

# Conclusion

This project demonstrates the application of machine learning techniques to predict house prices. By leveraging EDA, preprocessing, model evaluation, and hyperparameter tuning, accurate predictions can be made, which can be useful for real estate agents, buyers, and sellers.


