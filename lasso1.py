#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import TransformedTargetRegressor
from category_encoders import GLMMEncoder
import os

def create_stratified_folds(y, n_splits=5):
    bins = pd.qcut(y, q=n_splits, labels=False, duplicates='drop')
    fold_indices = []
    for fold in range(n_splits):
        test_idx = np.where(bins == fold)[0]
        train_idx = np.where(bins != fold)[0]
        fold_indices.append((train_idx, test_idx))
    return fold_indices

def create_pipeline():
    base_regressor = TransformedTargetRegressor(
        regressor=Lasso(random_state=42),
        transformer=StandardScaler()
    )
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', base_regressor)
    ])

def cross_validated_glmm_encoding(df, categorical_columns, target_column, n_splits=5):
    encoded_df = df.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(df):
        train_data, test_data = df.iloc[train_idx], df.iloc[test_idx]
        encoder = GLMMEncoder(cols=categorical_columns)
        encoder.fit(train_data[categorical_columns], train_data[target_column])
        encoded_df.iloc[test_idx, [encoded_df.columns.get_loc(col) for col in categorical_columns]] = encoder.transform(test_data[categorical_columns])
    return encoded_df

def evaluate_cv_method(X, y, cv_type='simple', n_splits=5):
    param_grid = {'regressor__regressor__alpha': np.logspace(-4, 1, 20)}
    scores = []
    if cv_type == 'stratified':
        fold_indices = create_stratified_folds(y, n_splits)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_indices = list(kf.split(X))
    for train_idx, test_idx in fold_indices:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(create_pipeline(), param_grid, cv=inner_cv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        scores.append((rmse, r2))
        print(f"{cv_type} CV - RMSE: {rmse:.2f}, R^2: {r2:.2f}")
    return scores

def preprocess_data(df, target_column):
    df = df.dropna()
    categorical_columns = df.select_dtypes(include=['object']).columns
    encoded_df = cross_validated_glmm_encoding(df, categorical_columns, target_column)
    return encoded_df

def plot_results(simple_cv_scores, stratified_cv_scores):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    cv_results = {'Simple CV': [s[0] for s in simple_cv_scores], 'Stratified CV': [s[0] for s in stratified_cv_scores]}
    sns.boxplot(data=cv_results, ax=axes[0])
    axes[0].set_title('RMSE Distribution')
    axes[0].set_ylabel('RMSE')
    iterations = range(1, len(simple_cv_scores) + 1)
    axes[1].plot(iterations, [s[0] for s in simple_cv_scores], marker='o', label='Simple CV')
    axes[1].plot(iterations, [s[0] for s in stratified_cv_scores], marker='s', label='Stratified CV')
    axes[1].set_title('RMSE Across Folds')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('RMSE')
    axes[1].legend()
    plt.tight_layout()
    plt.show()

def main():
    data_directory = os.path.join(os.getcwd(), 'dataset')
    df = pd.read_csv(os.path.join(data_directory, "ds_salaries.csv"))
    df.columns = df.columns.str.strip()
    target_column = 'salary'
    encoded_df = preprocess_data(df, target_column)
    X, y = encoded_df.drop(columns=[target_column]), encoded_df[target_column]
    print("\nEvaluating Simple CV...")
    simple_cv_scores = evaluate_cv_method(X, y, 'simple')
    print("\nEvaluating Stratified CV...")
    stratified_cv_scores = evaluate_cv_method(X, y, 'stratified')
    plot_results(simple_cv_scores, stratified_cv_scores)
    print(f"Simple CV - Mean RMSE: {np.mean([s[0] for s in simple_cv_scores]):.2f} ± {np.std([s[0] for s in simple_cv_scores]):.2f}")
    print(f"Stratified CV - Mean RMSE: {np.mean([s[0] for s in stratified_cv_scores]):.2f} ± {np.std([s[0] for s in stratified_cv_scores]):.2f}")
    print(f"Improvement: {((np.mean([s[0] for s in simple_cv_scores]) - np.mean([s[0] for s in stratified_cv_scores])) / np.mean([s[0] for s in simple_cv_scores]) * 100):.2f}%")

if __name__ == "__main__":
    main()

