#%%
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.impute import SimpleImputer  
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV  
from sklearn.pipeline import Pipeline  
from IPython.display import display
from sklearn.linear_model import Lasso  
from category_encoders import GLMMEncoder  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import mean_squared_error  
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer  
from sklearn.base import BaseEstimator, TransformerMixin  
import os  
#%%
#class below addresses data leakage by ensuring no target information contaminates the training process
class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, num_features, cat_features, n_splits=5):
        self.num_features = num_features
        self.cat_features = cat_features
        self.n_splits = n_splits
        self.num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        self.cat_encoders = []  # Stores GLMMEncoders for each fold
        self.fold_masks = []    # Tracks which samples belong to each fold's validation set
        self.X_index = None     # Remembers the index of the training data

    def fit(self, X, y):
        # Store training data index to detect training vs. new data later
        self.X_index = X.index
        
        # Process numerical features
        self.num_pipeline.fit(X[self.num_features])
        
        # Initialize structures for categorical encoding
        X_cat = X[self.cat_features].copy()
        self.encoded_train = np.zeros((X.shape[0], len(self.cat_features)))
        self.cat_encoders = []
        self.fold_masks = []
        
        # Cross-validated GLMM encoding
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X):
            # Train encoder on the fold's training subset
            encoder = GLMMEncoder()
            encoder.fit(X_cat.iloc[train_idx], y.iloc[train_idx])
            self.cat_encoders.append(encoder)
            
            # Store mask for validation indices and encode them
            val_mask = np.zeros(X.shape[0], dtype=bool)
            val_mask[val_idx] = True
            self.fold_masks.append(val_mask)
            self.encoded_train[val_idx] = encoder.transform(X_cat.iloc[val_idx])
        
        return self

    def transform(self, X):
        # Process numerical features
        num_processed = self.num_pipeline.transform(X[self.num_features])
        
        # Process categorical features
        X_cat = X[self.cat_features]
        if X.index.equals(self.X_index):
            # For training data: use precomputed fold-specific encodings
            cat_processed = self.encoded_train
        else:
            # For test/new data: average predictions from all encoders
            encoded_list = [encoder.transform(X_cat) for encoder in self.cat_encoders]
            cat_processed = np.mean(encoded_list, axis=0)
        
        return np.hstack([num_processed, cat_processed])

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
#%%
# Load dataset  
working_dir = os.getcwd()  
data_dir = os.path.join(working_dir, 'dataset')  
df = pd.read_csv(os.path.join(data_dir, 'AmesHousing.csv'))  
#%%
# Define numerical and categorical features
numerical_features= ['Order', 'PID', 'MS SubClass', 'Lot Frontage',
'Lot Area', 'Overall Qual', 'Overall Cond',
'Year Built', 'Year Remod/Add', 'Mas Vnr Area',
'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF',
'Total Bsmt SF', '1st Flr SF', '2nd Flr SF',
'Low Qual Fin SF', 'Gr Liv Area',
'Bsmt Full Bath', 'Bsmt Half Bath',
'Full Bath', 'Half Bath', 'Bedroom AbvGr',
'Kitchen AbvGr', 'TotRms AbvGrd',
'Fireplaces', 'Garage Yr Blt',
'Garage Cars', 'Garage Area',
'Wood Deck SF', 'Open Porch SF', 
'Enclosed Porch', '3Ssn Porch', 
'Screen Porch', 'Pool Area', 
'Misc Val', 'Mo Sold', 'Yr Sold']   # Same as before
categorical_features =['MS Zoning', 'Street', 'Alley', 'Lot Shape', 
'Land Contour', 'Utilities', 'Lot Config', 
'Land Slope', 'Neighborhood', 'Condition 1',
'Condition 2', 'Bldg Type', 'House Style', 
'Roof Style', 'Roof Matl', 'Exterior 1st', 
'Exterior 2nd', 'Mas Vnr Type', 'Exter Qual', 
'Exter Cond', 'Foundation', 'Bsmt Qual', 
'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1',
'BsmtFin Type 2', 'Heating', 'Heating QC',
'Central Air', 'Electrical', 'Kitchen Qual',
'Functional', 'Fireplace Qu', 'Garage Type',
'Garage Finish', 'Garage Qual',
'Garage Cond', 'Paved Drive', 'Pool QC',
'Fence', 'Misc Feature', 'Sale Type',
'Sale Condition']  # Same as before
#%%
target = 'SalePrice'

# Binning target variable for stratified CV  
y_binned = pd.qcut(df[target], q=5, labels=False, duplicates='drop')
print(y_binned.value_counts())
#%%
# Stratified K-Fold  
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  
results = []

# Main processing loop
for train_idx, test_idx in outer_cv.split(df, y_binned):
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
    y_train_binned = pd.qcut(df_train[target], q=5, labels=False, duplicates='drop')
    #print(y_train_binned.value_counts())
    # Create full pipeline
    pipeline = Pipeline([
        ('processor', FeatureProcessor(numerical_features, categorical_features, n_splits=5)),
        ('regressor', TransformedTargetRegressor(
            regressor=Lasso(),
            transformer=StandardScaler()
        ))
    ])
    
    # Parameter grid
    param_grid = {'regressor__regressor__alpha': [0.001, 0.01, 0.1, 1, 10]}
    # Stratified inner CV
    inner_cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_strat = GridSearchCV(
    pipeline,
    param_grid,
    cv=inner_cv_strat.split(df_train, y_train_binned),  # Use stratified splits
    scoring='neg_mean_squared_error'
    ) 
    grid_strat.fit(df_train, df_train[target])
    best_strat = grid_strat.best_estimator_
    preds_strat = best_strat.predict(df_test)
    rmse_strat = np.sqrt(mean_squared_error(df_test[target], preds_strat))
    
    # Unstratified inner CV
    inner_cv_unstrat = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_unstrat = GridSearchCV(pipeline, param_grid, cv=inner_cv_unstrat, scoring='neg_mean_squared_error')
    grid_unstrat.fit(df_train, df_train[target])
    best_unstrat = grid_unstrat.best_estimator_
    preds_unstrat = best_unstrat.predict(df_test)
    rmse_unstrat = np.sqrt(mean_squared_error(df_test[target], preds_unstrat))
    
    results.append({'rmse_strat': rmse_strat, 'rmse_unstrat': rmse_unstrat})
#%%
# Results analysis
results_df = pd.DataFrame(results)
print("Stratified CV Results:")
print(results_df['rmse_strat'].describe())
print("\nUnstratified CV Results:")
print(results_df['rmse_unstrat'].describe())

# Visualization
plt.figure(figsize=(10, 6))
sns.boxplot(data=results_df[['rmse_strat', 'rmse_unstrat']], whis=1.5)
plt.title("RMSE Comparison: Stratified vs Unstratified Inner CV")
plt.ylabel("RMSE")
plt.xticks([0, 1], ['Stratified', 'Unstratified'])
plt.show()

# %%
# Create a comparison table
comparison_table = results_df.agg(['mean', 'std', 'min', 'max']).T
comparison_table.columns = ['Mean RMSE', 'Std RMSE', 'Min RMSE', 'Max RMSE']
comparison_table.index = ['Stratified', 'Unstratified']

# Display the table
print("\nComparison Table: RMSE for Stratified vs. Unstratified Inner CV")
print(comparison_table)
display(comparison_table)
#%%
# %% Check stratification effect on y_train_binned
print("Checking stratification effect...\n")

for fold, (train_idx, val_idx) in enumerate(inner_cv_strat.split(df_train, y_train_binned)):
    print(f"Fold {fold}: Stratified y_train_binned distribution")
    print(pd.Series(y_train_binned.iloc[train_idx]).value_counts(normalize=True))
    print("\n")

for fold, (train_idx, val_idx) in enumerate(inner_cv_unstrat.split(df_train)):
    print(f"Fold {fold}: Unstratified y_train distribution")
    print(df_train.iloc[train_idx][target].describe())
    print("\n")

# %% Compare encoded features from FeatureProcessor
pipeline.fit(df_train, df_train[target])  # Fit the full pipeline
encoded_data_strat = pipeline.named_steps['processor'].transform(df_train)

# If possible, extract encoded categorical features separately
print("First row of processed features (Stratified):", encoded_data_strat[0])

# Fit and transform using unstratified CV (to compare feature transformation)
grid_unstrat.fit(df_train, df_train[target])
encoded_data_unstrat = grid_unstrat.best_estimator_.named_steps['processor'].transform(df_train)

print("First row of processed features (Unstratified):", encoded_data_unstrat[0])

# %% Compare feature differences
print("\nComparing encoded feature differences...")
feature_diffs = np.abs(encoded_data_strat - encoded_data_unstrat)
print("Mean absolute difference across features:", np.mean(feature_diffs))
print("Max absolute difference across features:", np.max(feature_diffs))

# %% Visualization: Checking if the processed features differ significantly
plt.figure(figsize=(10, 5))
plt.hist(feature_diffs.flatten(), bins=50, alpha=0.7, label='Feature Differences')
plt.xlabel("Difference in Encoded Features")
plt.ylabel("Frequency")
plt.title("Histogram of Feature Differences between Stratified and Unstratified Encoders")
plt.legend()
plt.show()
# %%

