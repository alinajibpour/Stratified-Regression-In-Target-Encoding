#%%
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.impute import SimpleImputer  
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV  
from sklearn.pipeline import Pipeline  
from sklearn.linear_model import Lasso  
from category_encoders import GLMMEncoder  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import mean_squared_error  
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer  
from sklearn.base import BaseEstimator, TransformerMixin  
import os  
#%%
# Custom preprocessing transformer with GLMMEncoder using 5-fold CV
class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, num_features, cat_features, n_splits=5):
        self.num_features = num_features
        self.cat_features = cat_features
        self.n_splits = n_splits
        self.num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        self.cat_encoders = []  # Will store GLMMEncoders trained on each fold
    
    def fit(self, X, y):
        # Fit numerical pipeline on training numerical features.
        self.num_pipeline.fit(X[self.num_features])
        
        # Prepare KFold for categorical encoding on the training data only.
        skf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        self.cat_encoders = []
        X_cat = X[self.cat_features].copy()
        encoded_features = np.zeros((X.shape[0], len(self.cat_features)))
        
        # Fit a GLMMEncoder on each fold's training subset and transform the validation fold.
        for train_idx, val_idx in skf.split(X, y):
            encoder = GLMMEncoder()
            encoder.fit(X_cat.iloc[train_idx], y.iloc[train_idx])
            encoded_features[val_idx] = encoder.transform(X_cat.iloc[val_idx])
            self.cat_encoders.append(encoder)
        
        # Note: We no longer fit a global encoder on the entire data.
        # The ensemble of fold-specific encoders (self.cat_encoders) will be used
        # to transform new data by averaging their outputs.
        return self
    
    def transform(self, X):
        # Process numerical features.
        num_processed = self.num_pipeline.transform(X[self.num_features])
        
        # Transform categorical features using each fold-specific encoder.
        X_cat = X[self.cat_features]
        # Get encoded arrays from each stored encoder.
        encoded_list = [encoder.transform(X_cat) for encoder in self.cat_encoders]
        # Average the outputs across folds.
        cat_processed = np.mean(encoded_list, axis=0)
        
        # Combine numerical and categorical features.
        return np.hstack([num_processed, cat_processed])
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
