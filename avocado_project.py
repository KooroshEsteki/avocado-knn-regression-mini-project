import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_squared_error


# 
# STEP 1 + STEP 2: Reading the Dataset
# 

DATA_PATH = "avocado.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values per column:")
print(df.isna().sum())

# Drop missing values if any exist
df = df.dropna()

print("\nShape after dropping missing values:")
print(df.shape)


# 
# STEP 3: Extract Features
# Exclude region and date from considered features
# Target variable: AveragePrice
# 

target = "AveragePrice"

# Drop columns that should not be used as features
columns_to_drop = [
    target,
    "Date",
    "region"
]

# Some avocado datasets include an index column called Unnamed: 0
if "Unnamed: 0" in df.columns:
    columns_to_drop.append("Unnamed: 0")

X = df.drop(columns=columns_to_drop)
y = df[target]

print("\nSelected feature columns:")
print(X.columns.tolist())

print("\nTarget column:")
print(target)

print("\nFeature matrix shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# 
# STEP 4: Preprocessing
# Scaling numeric features
# Encoding categorical features
# Dealing with NaN values
# 

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# 
# STEP 5: Splitting the Data
# 80% training set
# 10% validation set
# 10% test set
# 

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)

print("\nTraining set:")
print(X_train.shape, y_train.shape)

print("\nValidation set:")
print(X_val.shape, y_val.shape)

print("\nTest set:")
print(X_test.shape, y_test.shape)


# 
# STEP 6: Training KNN Regression
# Try different k values and validate performance
# Regression metric: R-squared
# 

k_values = [1, 3, 5, 7, 9, 11, 13, 15, 21, 25]

knn_results = []

for k in k_values:
    knn_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", KNeighborsRegressor(n_neighbors=k))
        ]
    )

    knn_model.fit(X_train, y_train)

    y_val_pred = knn_model.predict(X_val)

    val_r2 = r2_score(y_val, y_val_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)

    knn_results.append({
        "k": k,
        "validation_r2": val_r2,
        "validation_mse": val_mse
    })

    print(f"k = {k:2d} | Validation R-squared = {val_r2:.4f} | Validation MSE = {val_mse:.4f}")


knn_results_df = pd.DataFrame(knn_results)

print("\nKNN validation results:")
print(knn_results_df)

best_row = knn_results_df.loc[knn_results_df["validation_r2"].idxmax()]
best_k = int(best_row["k"])

print("\nBest k based on validation R-squared:")
print(best_k)


# Final KNN model using best k

final_knn_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", KNeighborsRegressor(n_neighbors=best_k))
    ]
)

final_knn_model.fit(X_train, y_train)

y_test_pred_knn = final_knn_model.predict(X_test)

knn_test_r2 = r2_score(y_test, y_test_pred_knn)
knn_test_mse = mean_squared_error(y_test, y_test_pred_knn)

print("\nFinal KNN Test Results")
print("----------------------")
print("Best k:")
print(best_k)

print("KNN Test R-squared:")
print(round(knn_test_r2, 4))

print("KNN Test MSE:")
print(round(knn_test_mse, 4))


# 
# STEP 7: Challenge Yourself
# Additional regression model: Linear Regression
# 

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

linear_model.fit(X_train, y_train)

y_val_pred_linear = linear_model.predict(X_val)
y_test_pred_linear = linear_model.predict(X_test)

linear_val_r2 = r2_score(y_val, y_val_pred_linear)
linear_test_r2 = r2_score(y_test, y_test_pred_linear)
linear_test_mse = mean_squared_error(y_test, y_test_pred_linear)

print("\nLinear Regression Results")
print("-------------------------")

print("Linear Regression Validation R-squared:")
print(round(linear_val_r2, 4))

print("Linear Regression Test R-squared:")
print(round(linear_test_r2, 4))

print("Linear Regression Test MSE:")
print(round(linear_test_mse, 4))


# 
# Final comparison
# 

comparison_df = pd.DataFrame([
    {
        "model": f"KNN Regressor, k={best_k}",
        "test_r2": knn_test_r2,
        "test_mse": knn_test_mse
    },
    {
        "model": "Linear Regression",
        "test_r2": linear_test_r2,
        "test_mse": linear_test_mse
    }
])

print("\nFinal Model Comparison:")
print(comparison_df.sort_values(by="test_r2", ascending=False))

best_model = comparison_df.loc[comparison_df["test_r2"].idxmax()]

print("\nBest model based on test R-squared:")
print(best_model)

print("\nComment:")
print(
    "The KNN regressor tests several k values and selects the best one using "
    "validation R-squared. Linear Regression is used as an additional regression "
    "model for comparison. The model with the higher test R-squared performs better."
)
