import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("Social_Network_Ads.csv")

# Split the data into training and testing sets
X = df.drop(["Purchased", "User ID"], axis=1)
y = df["Purchased"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

# Define preprocessing steps
scaler = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
scale_cols = ["Age", "EstimatedSalary"]
cat_cols = ["Gender"]
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, cat_cols),
        ("scale", scaler, scale_cols),
    ]
)

# Define the pipeline
clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", GaussianNB())])
clf.fit(X_train, y_train)

# Make predictions on the test set and evaluate the model
y_pred = clf.predict(X_test)
acc = round(accuracy_score(y_test, y_pred), 3)
print(f"Accuracy score: {acc*100}%")
print("Classification Report: \n", classification_report(y_test, y_pred))
