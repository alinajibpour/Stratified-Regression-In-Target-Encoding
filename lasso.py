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
from sklearn.metrics import mean_squared_error,  mean_absolute_error, r2_score 
from sklearn.compose import TransformedTargetRegressor  
from sklearn.base import BaseEstimator, TransformerMixin  
import os
#%%
class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, num_features, cat_features, n_splits=5, encoder_cv_type='simple', n_bins=5):
        self.num_features = num_features
        self.cat_features = cat_features
        self.n_splits = n_splits
        self.encoder_cv_type = encoder_cv_type  # 'simple' or 'stratified'
        self.n_bins = n_bins  # For stratified encoding
        self.num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        self.cat_encoders = []  
        self.fold_masks = []    
        self.X_index = None     

    def fit(self, X, y):
        self.X_index = X.index
        
        # Process numerical features
        self.num_pipeline.fit(X[self.num_features])
        
        # Prepare categorical features
        X_cat = X[self.cat_features].copy()
        self.encoded_train = np.zeros((X.shape[0], len(self.cat_features)))
        self.cat_encoders = []
        self.fold_masks = []
        
        # Determine CV strategy
        if self.encoder_cv_type == 'stratified':
            y_binned = pd.qcut(y, q=self.n_bins, labels=False, duplicates='drop')
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            splits = kf.split(X, y_binned)
        else:
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            splits = kf.split(X)
        
        # Cross-validated encoding
        for train_idx, val_idx in splits:
            encoder = GLMMEncoder()
            encoder.fit(X_cat.iloc[train_idx], y.iloc[train_idx])
            self.cat_encoders.append(encoder)
            
            val_mask = np.zeros(X.shape[0], dtype=bool)
            val_mask[val_idx] = True
            self.fold_masks.append(val_mask)
            self.encoded_train[val_idx] = encoder.transform(X_cat.iloc[val_idx])
        
        return self

    def transform(self, X):
        num_processed = self.num_pipeline.transform(X[self.num_features])
        X_cat = X[self.cat_features]
        
        if X.index.equals(self.X_index):
            cat_processed = self.encoded_train
        else:
            encoded_list = [encoder.transform(X_cat) for encoder in self.cat_encoders]
            cat_processed = np.median(encoded_list, axis=0)
        
        return np.hstack([num_processed, cat_processed])

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
#%%
# Load data and define features
working_dir = os.getcwd()  
data_dir = os.path.join(working_dir, 'dataset')  
df = pd.read_csv(os.path.join(data_dir, 'vgsales.csv'))   
#df = df_orginal.sample(frac=0.2, random_state=42) 
df.columns = df.columns.str.strip() 
#%%
numerical_features= ['Rank', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']   
categorical_features =['Name', 'Platform', 'Genre', 'Publisher'] 

target = 'Global_Sales'
#%%
# Binning for stratification
y_binned = pd.qcut(df[target], q=5, labels=False, duplicates='drop')
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#%%
results = []
for train_idx, test_idx in outer_cv.split(df, y_binned):
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
    y_train_binned = pd.qcut(df_train[target], q=5, labels=False, duplicates='drop')
    # Create both encoder variants
    pipe_strat_encoder = Pipeline([
        ('processor', FeatureProcessor(
            numerical_features,
            categorical_features,
            n_splits=5,
            encoder_cv_type='stratified',
            n_bins=5
        )),
        ('regressor', TransformedTargetRegressor(
            regressor=Lasso(),
            transformer=StandardScaler()
        ))
    ])
    
    pipe_unstrat_encoder = Pipeline([
        ('processor', FeatureProcessor(
            numerical_features,
            categorical_features,
            n_splits=5,
            encoder_cv_type='simple'
        )),
        ('regressor', TransformedTargetRegressor(
            regressor=Lasso(),
            transformer=StandardScaler()
        ))
    ])
    
    # Shared parameter grid
    param_grid = {'regressor__regressor__alpha': [0.001, 0.01, 0.1, 1, 10]}
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate both encoders
    fold_results = {}
    for name, pipe in [('strat_encoder', pipe_strat_encoder),
                      ('unstrat_encoder', pipe_unstrat_encoder)]:
        # Choose inner CV based on encoder type
        if name == 'strat_encoder':
            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            inner_cv_splits = inner_cv.split(df_train, y_train_binned)
        else:
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
            inner_cv_splits = inner_cv.split(df_train)    
        grid = GridSearchCV(
                pipe,
                param_grid,
                cv=inner_cv_splits,
                scoring='neg_mean_squared_error'
            )
            
        grid.fit(df_train, df_train[target])
        preds = grid.best_estimator_.predict(df_test)
        fold_results[f'rmse_{name}'] = np.sqrt(mean_squared_error(df_test[target], preds))
    
    results.append(fold_results)
#%%
# Analysis and visualization
results_df = pd.DataFrame(results)

print("Stratified Encoder Results:")
print(results_df['rmse_strat_encoder'].describe())
print("\nUnstratified Encoder Results:")
print(results_df['rmse_unstrat_encoder'].describe())

plt.figure(figsize=(10, 6))
sns.boxplot(data=results_df[['rmse_strat_encoder', 'rmse_unstrat_encoder']], whis=1.5)
plt.title("RMSE Comparison: GLMMEncoder CV Strategies")
plt.ylabel("RMSE")
plt.xticks([0, 1], ['Stratified Encoder', 'Unstratified Encoder'])
plt.show()