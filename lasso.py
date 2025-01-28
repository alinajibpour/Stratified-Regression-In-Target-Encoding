#%%
# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from category_encoders import GLMMEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import os
#%%
working_directory = os.getcwd()
data_directory = os.path.join(working_directory, 'dataset')
datasets = {"Ames": "AmesHousing.csv", "IPPS": "Inpatient_Prospective_Payment_System__IPPS__Provider_Summary_for_the_Top_100_Diagnosis-Related_Groups__DRG__-_FY2011.csv", "Salary":"ds_salaries.csv", "Automobile":"clean_automobile_data.csv"}
#%%
# Importing the dataset
df_salary=pd.read_csv(os.path.join(data_directory, datasets["Salary"]))
df_salary.shape
df_salary.head()
df_salary.info()
df_salary.columns = df_salary.columns.str.strip()
#%%
# Handling missing data
columns_with_missing_values = df_salary.columns[df_salary.isnull().any()]
df_salary[columns_with_missing_values].isnull().sum()
AmeHousing= df_salary.dropna(axis=1)
df_salary.shape
df_salary.head()
#%%
#group-size discretization
target_column ='salary'
num_bins = 5
df_salary['salary_disc'] = pd.qcut(df_salary[target_column], num_bins, labels=False)
df_salary['salary_disc'].value_counts()
#%%
encoded_df_salary= df_salary.copy()
# GLMM encoding with 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
categorical_columns = df_salary.select_dtypes(include=['object']).columns
for train_index, test_index in kf.split(df_salary):
    train_data, test_data = df_salary.iloc[train_index], df_salary.iloc[test_index]
    encoder = GLMMEncoder(cols=categorical_columns)
    encoder.fit(train_data, train_data['salary_disc'])
    encoded_df_salary.loc[test_index, categorical_columns] = encoder.transform(test_data)
encoded_df_salary.head()    
#%%    
#Define features (X) and target (y)
X = encoded_df_salary.drop(columns=['salary', 'salary_disc'])  # Drop target columns
y = encoded_df_salary['salary_disc']  # Use stratified target variabl
#Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#%%
# Store results for each fold
fold_results = []

# Define a constant alpha for Lasso
alpha_value = 0.1  # Replace this with the desired constant alpha

for train_index, test_index in skf.split(X, y):
    # Split the data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize Lasso with the constant alpha value
    lasso_model = Lasso(alpha=alpha_value, random_state=42)

    # Fit the model on the training data
    lasso_model.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred = lasso_model.predict(X_test_scaled)

    # Calculate metrics (e.g., RMSE)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    fold_results.append(rmse)

    print(f"Fold RMSE: {rmse}")

# Optionally, print summary statistics for RMSE across folds
print(f"Mean RMSE: {np.mean(fold_results)}")
print(f"Standard Deviation of RMSE: {np.std(fold_results)}")

#%%
# Output overall results
#print(f"Average RMSE across folds: {np.mean(fold_results):.4f}")
#print(f"Standard deviation of RMSE: {np.std(fold_results):.4f}")
#print("\nBest alpha values for each fold:")
#for i, alpha in enumerate(best_alphas):
    #print(f"Fold {i+1}: {alpha}")
#%%
#Output overall results
print(f"Average RMSE across folds: {np.mean(fold_results):.4f}")
#%%
# Nested Cross-Validation with Lasso
def nested_cv_lasso_with_existing_folds(X, y, outer_cv, inner_splits=5):
    # Initialize arrays to store results
    outer_scores = []
    best_params = []
    
    # Outer loop using the existing folds from stratified CV
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        # Split data into outer train and test sets
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale the features
        scaler = StandardScaler()
        X_train_outer_scaled = scaler.fit_transform(X_train_outer)
        X_test_outer_scaled = scaler.transform(X_test_outer)
        
        # Define parameter grid for Lasso
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        }
        
        # Initialize inner cross-validation
        inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)
        
        # Initialize GridSearchCV for inner loop
        grid_search = GridSearchCV(
            estimator=Lasso(random_state=42),
            param_grid=param_grid,
            cv=inner_cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Fit GridSearchCV on outer training data
        grid_search.fit(X_train_outer_scaled, y_train_outer)
        
        # Store best parameters from inner loop
        best_params.append(grid_search.best_params_)
        
        # Get predictions using best model from inner loop
        y_pred = grid_search.predict(X_test_outer_scaled)
        
        # Calculate and store RMSE for outer fold
        rmse = np.sqrt(mean_squared_error(y_test_outer, y_pred))
        outer_scores.append(rmse)
        
        print(f"Outer Fold {outer_fold + 1}:")
        print(f"Best alpha: {grid_search.best_params_['alpha']}")
        print(f"RMSE: {rmse:.4f}\n")
    
    return outer_scores, best_params

# Use the existing stratified folds (skf) from your original code
nested_scores, nested_params = nested_cv_lasso_with_existing_folds(X, y, skf)

# Print overall results
print("Nested Cross-Validation Results:")
print(f"Average RMSE: {np.mean(nested_scores):.4f}")
print(f"Standard deviation of RMSE: {np.std(nested_scores):.4f}")
print("\nBest alpha values for each outer fold:")
for i, params in enumerate(nested_params):
    print(f"Fold {i+1}: {params['alpha']}")

# Compare with original stratified CV results
print("\nComparison with Original Stratified CV:")
print(f"Original CV Average RMSE: {np.mean(fold_results):.4f}")
print(f"Nested CV Average RMSE: {np.mean(nested_scores):.4f}")
#%%
def plot_cv_comparison(fold_results, nested_scores):
    """
    Create comprehensive visualization comparing stratified and nested CV results
    
    Parameters:
    fold_results (list): RMSE results from stratified CV
    nested_scores (list): RMSE results from nested CV
    """
    # Set figure size and create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Bar plot comparison across folds
    folds = range(1, len(fold_results) + 1)
    width = 0.35
    
    ax1.bar([x - width/2 for x in folds], fold_results, width, 
            label='Stratified CV', color='#8884d8', alpha=0.7)
    ax1.bar([x + width/2 for x in folds], nested_scores, width, 
            label='Nested CV', color='#82ca9d', alpha=0.7)
    
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Lasso: RMSE Comparison Across Folds')
    ax1.set_xticks(folds)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 2. Box plot comparison
    ax2.boxplot([fold_results, nested_scores], labels=['Stratified CV', 'Nested CV'])
    ax2.set_ylabel('RMSE')
    ax2.set_title('Lasso: Distribution of RMSE Values')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Line plot showing trends across folds
    ax3.plot(folds, fold_results, 'o-', label='Stratified CV', color='#8884d8')
    ax3.plot(folds, nested_scores, 'o-', label='Nested CV', color='#82ca9d')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('RMSE')
    ax3.set_title('Lasso: RMSE Trends Across Folds')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()
    
    # 4. Mean and std comparison
    means = [np.mean(fold_results), np.mean(nested_scores)]
    stds = [np.std(fold_results), np.std(nested_scores)]
    
    ax4.bar(['Stratified CV', 'Nested CV'], means, yerr=stds, 
            capsize=5, alpha=0.7, color=['#8884d8', '#82ca9d'])
    ax4.set_ylabel('Mean RMSE')
    ax4.set_title('Lasso: Average RMSE with Standard Deviation')
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Add text annotations with statistics
    stats_text = (
        f'Lasso Results:\n\n'
        f'Stratified CV:\n'
        f'Mean: {np.mean(fold_results):.4f}\n'
        f'Std: {np.std(fold_results):.4f}\n\n'
        f'Nested CV:\n'
        f'Mean: {np.mean(nested_scores):.4f}\n'
        f'Std: {np.std(nested_scores):.4f}\n'
    )
    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace')
    
    plt.tight_layout()
    return fig

# Create and display the visualization
fig = plot_cv_comparison(fold_results, nested_scores)
plt.show()  
#%%