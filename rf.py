#%%
# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from category_encoders import GLMMEncoder
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
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
df_automobile=pd.read_csv(os.path.join(data_directory, datasets["Automobile"]))
df_automobile.shape
df_automobile.head()
df_automobile.info()
#%%
# Handling missing data
#columns_with_missing_values = AmesHousing.columns[AmesHousing.isnull().any()]
#AmesHousing[columns_with_missing_values].isnull().sum()
#AmesHousing = AmesHousing.dropna(axis=1)
#AmesHousing.shape
#AmesHousing.head()
#%%
#group-size discretization
target_column = 'price'
num_bins = 5
df_automobile['price_disc'] = pd.qcut(df_automobile[target_column], num_bins, labels=False)
df_automobile['price_disc'].value_counts()
#%%
encoded_df_automobile = df_automobile.copy()
# GLMM encoding with 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
categorical_columns = df_automobile.select_dtypes(include=['object']).columns
for train_index, test_index in kf.split(df_automobile):
    train_data, test_data = df_automobile.iloc[train_index], df_automobile.iloc[test_index]
    encoder = GLMMEncoder(cols=categorical_columns)
    encoder.fit(train_data, train_data['price_disc'])
    encoded_df_automobile.loc[test_index, categorical_columns] = encoder.transform(test_data)
encoded_df_automobile.head()  
#%%
#Define features (X) and target (y)
X = encoded_df_automobile.drop(columns=['price', 'price_disc'])  # Drop target columns
y = encoded_df_automobile['price_disc']  # Use stratified target variabl
#Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#%%
# Store results for each fold
fold_results = []

for train_index, test_index in skf.split(X, y):
    # Split the data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # Predict on the test set
    y_pred = rf.predict(X_test_scaled)
    
    # Calculate metrics (RMSE)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    fold_results.append(rmse)
    print(f"Fold RMSE: {rmse}")

# Output overall results
print(f"Average RMSE across folds: {np.mean(fold_results):.4f}")
#%%
# Nested Cross-Validation with Random Forest
def nested_cv_rf_with_existing_folds(X, y, outer_cv, inner_splits=5):
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
        
        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Initialize inner cross-validation
        inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)
        
        # Initialize GridSearchCV for inner loop
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42),
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
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"RMSE: {rmse:.4f}\n")
    
    return outer_scores, best_params

# Use the existing stratified folds
nested_scores, nested_params = nested_cv_rf_with_existing_folds(X, y, skf)

# Print overall results
print("Nested Cross-Validation Results:")
print(f"Average RMSE: {np.mean(nested_scores):.4f}")
print(f"Standard deviation of RMSE: {np.std(nested_scores):.4f}")
print("\nBest parameters for each outer fold:")
for i, params in enumerate(nested_params):
    print(f"Fold {i+1}: {params}")

# Compare with original stratified CV results
print("\nComparison with Original Stratified CV:")
print(f"Original CV Average RMSE: {np.mean(fold_results):.4f}")
print(f"Nested CV Average RMSE: {np.mean(nested_scores):.4f}")
#%%
def plot_cv_comparison(fold_results, nested_scores):
    """
    Create comprehensive visualization comparing stratified and nested CV results for Random Forest
    using only matplotlib
    
    Parameters:
    fold_results (list): RMSE results from stratified CV
    nested_scores (list): RMSE results from nested CV
    """
    # Create figure and subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Colors for consistency
    strat_color = '#8884d8'
    nested_color = '#82ca9d'
    
    # 1. Bar plot comparison across folds
    folds = range(1, len(fold_results) + 1)
    width = 0.35
    
    ax1.bar([x - width/2 for x in folds], fold_results, width, 
            label='Stratified CV', color=strat_color, alpha=0.7)
    ax1.bar([x + width/2 for x in folds], nested_scores, width, 
            label='Nested CV', color=nested_color, alpha=0.7)
    
    ax1.set_xlabel('Fold Number', fontsize=10)
    ax1.set_ylabel('RMSE Score', fontsize=10)
    ax1.set_title('Random Forest: RMSE Comparison Across Folds', fontsize=12, pad=15)
    ax1.set_xticks(folds)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(fontsize=9)
    
    # 2. Box plot comparison
    box_positions = [1, 2]
    bp = ax2.boxplot([fold_results, nested_scores], 
                     positions=box_positions,
                     patch_artist=True,
                     medianprops=dict(color="black"),
                     flierprops=dict(marker='o', markerfacecolor='gray'))
    
    # Color the boxes
    bp['boxes'][0].set_facecolor(strat_color)
    bp['boxes'][1].set_facecolor(nested_color)
    
    ax2.set_xticks(box_positions)
    ax2.set_xticklabels(['Stratified CV', 'Nested CV'])
    ax2.set_ylabel('RMSE Score', fontsize=10)
    ax2.set_title('Random Forest: Distribution of RMSE Values', fontsize=12, pad=15)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # 3. Line plot showing trends across folds
    ax3.plot(folds, fold_results, 'o-', label='Stratified CV', 
             color=strat_color, linewidth=2, markersize=8)
    ax3.plot(folds, nested_scores, 'o-', label='Nested CV', 
             color=nested_color, linewidth=2, markersize=8)
    
    ax3.set_xlabel('Fold Number', fontsize=10)
    ax3.set_ylabel('RMSE Score', fontsize=10)
    ax3.set_title('Random Forest: RMSE Trends Across Folds', fontsize=12, pad=15)
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend(fontsize=9)
    
    # 4. Mean and std comparison
    means = [np.mean(fold_results), np.mean(nested_scores)]
    stds = [np.std(fold_results), np.std(nested_scores)]
    
    bars = ax4.bar(['Stratified CV', 'Nested CV'], means, yerr=stds, 
                   capsize=5, alpha=0.7, color=[strat_color, nested_color])
    
    ax4.set_ylabel('Mean RMSE Score', fontsize=10)
    ax4.set_title('Random Forest: Average RMSE with Standard Deviation', 
                  fontsize=12, pad=15)
    ax4.grid(True, linestyle='--', alpha=0.3)
    
    # Add text annotations with detailed statistics
    stats_text = (
        f'Random Forest Results\n\n'
        f'Stratified CV:\n'
        f'Mean RMSE: {np.mean(fold_results):.4f}\n'
        f'Std Dev: {np.std(fold_results):.4f}\n'
        f'Min RMSE: {np.min(fold_results):.4f}\n'
        f'Max RMSE: {np.max(fold_results):.4f}\n\n'
        f'Nested CV:\n'
        f'Mean RMSE: {np.mean(nested_scores):.4f}\n'
        f'Std Dev: {np.std(nested_scores):.4f}\n'
        f'Min RMSE: {np.min(nested_scores):.4f}\n'
        f'Max RMSE: {np.max(nested_scores):.4f}'
    )
    # Add the text in a better position
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, family='monospace', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Adjust layout and display
    plt.tight_layout()
    return fig

# To use the visualization:
fig = plot_cv_comparison(fold_results, nested_scores)
plt.show()
#%%
