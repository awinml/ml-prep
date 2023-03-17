import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Load the dataset
housing = pd.read_csv("USA_Housing.csv")

# Define the columns for preprocessing
numeric_features = [
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms",
    "Avg. Area Number of Bedrooms",
    "Area Population",
]

# Split the data into training and testing sets
X = housing.drop(["Price"], axis=1)
y = housing["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), numeric_features)]
)

# Define the pipeline
pipe = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
)

# Fit the pipeline to the training data
pipe.fit(X_train, y_train)

# Predict on the testing data
y_pred = pipe.predict(X_test)

# Evaluate the model
score = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print the results
print(f"R-squared score: {score:.3f}")
print(f"Mean squared error: {mse:.3f}")
