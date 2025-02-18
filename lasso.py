#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import TransformedTargetRegressor
from category_encoders import GLMMEncoder
import os
# %%
def create_stratified_folds(y, n_splits=5):
    """
    Create stratified folds for regression by binning the target variable.
    Returns fold indices that maintain similar target distribution.
    """
    # Create bins for stratification
    bins = pd.qcut(y, q=n_splits, labels=False)
    
    # Initialize folds
    fold_indices = []
    for fold in range(n_splits):
        test_idx = np.where(bins == fold)[0]
        train_idx = np.where(bins != fold)[0]
        fold_indices.append((train_idx, test_idx))
    
    return fold_indices
# %%
def create_pipeline():
    """Create a pipeline with scaling, target transformation, and Lasso regression."""
    # Create the base regressor with transformed target
    base_regressor = TransformedTargetRegressor(
        regressor=Lasso(random_state=42),
        transformer=StandardScaler()
    )
    
    # Create the full pipeline
    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', base_regressor)
    ])
# %%
def cross_validated_glmm_encoding(df, categorical_columns, target_column, n_splits=5):
    """
    Perform GLMM encoding with k-fold cross-validation to prevent data leakage.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    categorical_columns : list
        List of categorical column names
    target_column : str
        Name of the target variable
    n_splits : int
        Number of cross-validation folds
    
    Returns:
    --------
    pd.DataFrame : Encoded dataframe
    """
    # Create a copy of the dataframe to store encoded values
    encoded_df = df.copy()
    
    # Create KFold object
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Perform cross-validated encoding
    for train_idx, test_idx in kf.split(df):
        # Split data
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]
        
        # Fit encoder on training data
        encoder = GLMMEncoder(cols=categorical_columns)
        encoder.fit(train_data[categorical_columns], 
                   train_data[target_column])
        
        # Transform test data
        encoded_values = encoder.transform(test_data[categorical_columns])
        
        # Update encoded dataframe with transformed values
        encoded_df.iloc[test_idx, [encoded_df.columns.get_loc(col) for col in categorical_columns]] = encoded_values
    
    return encoded_df
# %%
def evaluate_cv_method(X, y, cv_type='simple', n_splits=5):
    """
    Evaluate model performance using either simple or stratified cross-validation.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    cv_type : str
        Type of cross-validation ('simple' or 'stratified')
    n_splits : int
        Number of cross-validation folds
    
    Returns:
    --------
    list : Cross-validation scores (RMSE)
    dict : Best parameters for each fold
    """
    param_grid = {
        'regressor__regressor__alpha': np.logspace(-4, 1, 20)
    }
    
    scores = []
    best_params = []
    
    # Create fold indices based on CV type
    if cv_type == 'stratified':
        fold_indices = create_stratified_folds(y, n_splits)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_indices = list(kf.split(X))
    
    # Perform nested cross-validation
    for fold, (train_idx, test_idx) in enumerate(fold_indices):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Inner cross-validation for hyperparameter tuning
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        
        # Grid search with inner cross-validation
        grid_search = GridSearchCV(
            create_pipeline(),
            param_grid,
            cv=inner_cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Fit model
        grid_search.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = grid_search.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(rmse)
        best_params.append(grid_search.best_params_)
        
        print(f"{cv_type} CV - Fold {fold + 1} - RMSE: ${rmse:,.2f}")
        print(f"Best parameters: {grid_search.best_params_}")
    
    return scores, best_params
#%%
def preprocess_data(df, target_column):
    """Preprocess the data with cross-validated GLMM encoding for categorical variables."""
    # Handle missing values
    df = df.dropna()
    
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Apply cross-validated GLMM encoding
    encoded_df = cross_validated_glmm_encoding(
        df, 
        categorical_columns=categorical_columns,
        target_column=target_column
    )
    
    return encoded_df
#%%
def plot_results(simple_cv_scores, stratified_cv_scores):
    """Create visualization comparing CV methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot
    cv_results = {
        'Simple CV': simple_cv_scores,
        'Stratified CV': stratified_cv_scores
    }
    sns.boxplot(data=cv_results, ax=ax1)
    ax1.set_title('Distribution of RMSE Scores')
    ax1.set_ylabel('RMSE ($)')
    
    # Line plot
    iterations = range(1, len(simple_cv_scores) + 1)
    ax2.plot(iterations, simple_cv_scores, marker='o', label='Simple CV', linestyle='-')
    ax2.plot(iterations, stratified_cv_scores, marker='s', label='Stratified CV', linestyle='-')
    ax2.set_title('RMSE Scores Across Folds')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('RMSE ($)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
#%%
def main():
    # Load data
    working_directory = os.getcwd()
    data_directory = os.path.join(working_directory, 'dataset')
    df = pd.read_csv(os.path.join(data_directory, "ds_salaries.csv"))
    print(df.head())
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Preprocess data with cross-validated GLMM encoding
    target_column = 'salary'
    encoded_df = preprocess_data(df, target_column)
    
    # Prepare features and target
    X = encoded_df.drop([target_column], axis=1)
    y = encoded_df[target_column]
    
    # Evaluate both CV methods
    print("\nEvaluating Simple CV...")
    simple_cv_scores, simple_best_params = evaluate_cv_method(X, y, 'simple')
    
    print("\nEvaluating Stratified CV...")
    stratified_cv_scores, stratified_best_params = evaluate_cv_method(X, y, 'stratified')
    
    # Plot results
    plot_results(simple_cv_scores, stratified_cv_scores)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Simple CV - Mean RMSE: ${np.mean(simple_cv_scores):,.2f} (±${np.std(simple_cv_scores):,.2f})")
    print(f"Stratified CV - Mean RMSE: ${np.mean(stratified_cv_scores):,.2f} (±${np.std(stratified_cv_scores):,.2f})")
    
    # Calculate improvement
    improvement = ((np.mean(simple_cv_scores) - np.mean(stratified_cv_scores)) / 
                  np.mean(simple_cv_scores) * 100)
    print(f"\nPercentage improvement with Stratified CV: {improvement:.2f}%")

if __name__ == "__main__":
    main()
#%%
