# import libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load dataset

df = pd.read_csv(r"C:\Users\hp\Downloads\Amazon Sales Dataset\amazon.csv")

# 2. Convert required columns to numbers
df["discounted_price"] = df["discounted_price"].str.replace("₹","", regex=False).str.replace(",","", regex=False).astype(float)
df["actual_price"] = df["actual_price"].str.replace("₹","", regex=False).str.replace(",","", regex=False).astype(float)
df["discount_percentage"] = df["discount_percentage"].str.replace("%","", regex=False).astype(float)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["rating_count"] = df["rating_count"].str.replace(",","", regex=False)
df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")

# Remove missing values
df = df.dropna(subset=[
    "discounted_price", "actual_price",
    "discount_percentage", "rating", "rating_count"
])

# Features
X = df[[
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating_count"
]]


# REGRESSION - Predict Product Rating


y_reg = df["rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

pred_reg = reg_model.predict(X_test)

print("\n--- Linear Regression ---")
print("MAE :", mean_absolute_error(y_test, pred_reg))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_reg)))
print("R2  :", r2_score(y_test, pred_reg))


# CLASSIFICATION - Highly Rated or Not


# 1 = rating >= 4, 0 = rating < 4
y_class = (df["rating"] >= 4).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# Scaling for Logistic Regression and KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
pred_log = log_model.predict(X_test_scaled)


# Decision Tree
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
pred_tree = tree_model.predict(X_test)


# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)


# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
pred_knn = knn_model.predict(X_test_scaled)



# MODEL COMPARISON


results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN"
    ],
    "Accuracy": [
        accuracy_score(y_test, pred_log),
        accuracy_score(y_test, pred_tree),
        accuracy_score(y_test, pred_rf),
        accuracy_score(y_test, pred_knn)
    ],
    "Precision": [
        precision_score(y_test, pred_log),
        precision_score(y_test, pred_tree),
        precision_score(y_test, pred_rf),
        precision_score(y_test, pred_knn)
    ],
    "Recall": [
        recall_score(y_test, pred_log),
        recall_score(y_test, pred_tree),
        recall_score(y_test, pred_rf),
        recall_score(y_test, pred_knn)
    ],
    "F1 Score": [
        f1_score(y_test, pred_log),
        f1_score(y_test, pred_tree),
        f1_score(y_test, pred_rf),
        f1_score(y_test, pred_knn)
    ]
})

print("\n--- Classification Model Comparison ---")
print(results.to_string(index=False))