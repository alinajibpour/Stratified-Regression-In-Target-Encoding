#%%
#Import necessary libraries
import numpy as np  
import pandas as pd  
from joblib import Parallel, delayed  
import random  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.impute import SimpleImputer  
from sklearn.model_selection import (StratifiedKFold, KFold,  
                                   GridSearchCV, train_test_split)  
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import Lasso  
from sklearn.preprocessing import StandardScaler, TargetEncoder 
from sklearn.metrics import mean_squared_error  
from sklearn.compose import TransformedTargetRegressor  
from sklearn.base import BaseEstimator, TransformerMixin  
import os  

#%%
#Set random seed
SEED=42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

#%%
class FeatureProcessor(BaseEstimator, TransformerMixin):
    """
    Custom transformer that handles:
    - Numerical feature imputation and scaling
    - Categorical feature encoding using TargetEncoder
    - Stratified or simple cross-validated encoding
    """
    
    def __init__(self, numerical_features, categorical_features, 
                 encoder_n_splits=5, encoder_cv_type='simple', 
                 stratify_bins=5):
        
        self.numerical_features = numerical_features  
        self.categorical_features = categorical_features  
        self.encoder_n_splits = encoder_n_splits  
        self.encoder_cv_type = encoder_cv_type  
        self.stratify_bins = stratify_bins  
        
        self.num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  
            ('scaler', StandardScaler())  
        ])
        
        self.cat_encoders = []  
        self.X_index = None  
        self.encoded_train = None  

    def fit(self, X, y):
        """Fit the processor to training data"""
        self.X_index = X.index  
        
        # Process numerical features
        self.num_pipeline.fit(X[self.numerical_features])  
        
        # Prepare categorical encoding
        X_cat = X[self.categorical_features].copy()  
        self.encoded_train = np.zeros((X.shape[0], len(self.categorical_features)))  
        
        # Create appropriate CV splits
        if self.encoder_cv_type == 'stratified':
            y_binned = pd.qcut(y, q=self.stratify_bins, labels=False, duplicates='drop')
            kf = StratifiedKFold(n_splits=self.encoder_n_splits, shuffle=True, random_state=SEED)
            splits = kf.split(X, y_binned)
        else:
            kf = KFold(n_splits=self.encoder_n_splits, shuffle=True, random_state=SEED)
            splits = kf.split(X)

        # Fit encoders per fold
        for train_idx, val_idx in splits:
            encoder = TargetEncoder(
                smooth="auto",
                target_type="continuous",
                random_state=SEED
            )
            encoder.fit(X_cat.iloc[train_idx], y.iloc[train_idx])  
            self.cat_encoders.append(encoder)
            self.encoded_train[val_idx] = encoder.transform(X_cat.iloc[val_idx])

        return self

    def transform(self, X):
        """Transform input data using fitted processors"""
        # Process numerical features
        num_processed = self.num_pipeline.transform(X[self.numerical_features])
        
        # Process categorical features
        X_cat = X[self.categorical_features]
        if X.index.equals(self.X_index):  
            cat_processed = self.encoded_train  
        else:
            # For new data, use median of all encoders
            encoded_list = [encoder.transform(X_cat) for encoder in self.cat_encoders]
            cat_processed = np.median(encoded_list, axis=0)  
        
        return np.hstack([num_processed, cat_processed])

#%%
def process_fold(train_idx, test_idx, df, target, numerical_features, categorical_features):
    """
    Process one outer CV fold with:
    - Nested hyperparameter tuning
    - Both encoding strategies
    - Validation set evaluation
    """
    X_train = df.iloc[train_idx].drop(columns=[target])  
    y_train = df.iloc[train_idx][target]  
    X_val = df.iloc[test_idx].drop(columns=[target]) 
    y_val = df.iloc[test_idx][target]  

    results = {}  
    
    # Evaluate both encoding strategies
    for enc_type in ['stratified', 'simple']:
        pipe = Pipeline([
            ('processor', FeatureProcessor(
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                encoder_cv_type=enc_type, 
                stratify_bins=5  # Updated parameter name
            )),
            ('regressor', RandomForestRegressor(
                random_state=SEED,
                n_jobs=1  
            ))
        ])

        # Create inner CV strategy
        if enc_type == 'stratified':
            y_binned = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            inner_splits = inner_cv.split(X_train, y_binned)
        else:
            inner_cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
            inner_splits = inner_cv.split(X_train)

        # Hyperparameter grid remains the same
        param_grid = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [5, 10, 20, None],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__max_features': ['sqrt', 'log2', 0.5, 0.8]
        }
        
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=inner_splits,  
            scoring='neg_mean_squared_error',  
            n_jobs=1, 
            verbose=0
        )
        
        # Train and validate
        grid.fit(X_train, y_train)
        preds = grid.predict(X_val)
        results[f'rmse_{enc_type}'] = np.sqrt(mean_squared_error(y_val, preds))
        
        # Store best parameters
        best_params = grid.best_params_
        results[f'best_n_estimators_{enc_type}'] = best_params['regressor__n_estimators']
        results[f'best_max_depth_{enc_type}'] = best_params['regressor__max_depth']
        results[f'best_min_samples_split_{enc_type}'] = best_params['regressor__min_samples_split']
        results[f'best_min_samples_leaf_{enc_type}'] = best_params['regressor__min_samples_leaf']
        results[f'best_max_features_{enc_type}'] = best_params['regressor__max_features']
    
    return results

#%%
#Main execution block (remainder identical except for parameter name in FeatureProcessor)
if __name__ == "__main__":
    #Load and prepare data
    df = pd.read_csv('dataset/IPPS_sampled.csv')
    df.columns = df.columns.str.strip()
    
    numerical_features = ['Provider Id', 'Provider Zip Code', 'Total Discharges', 'Average Covered Charges', 'Average Medicare Payments']
    categorical_features = ['DRG Definition', 'Provider Name', 'Provider Street Address', 'Provider City', 'Provider State', 'Hospital Referral Region Description']
    target = 'Average Total Payments'

    #Initial train-test split
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=pd.qcut(df[target], q=5, labels=False, duplicates='drop'),
        random_state=SEED
    )

    # Prepare outer CV
    y_binned = pd.qcut(train_df[target], q=5, labels=False, duplicates='drop')
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Parallel processing
    all_results = Parallel(n_jobs=-1, verbose=10, prefer='processes')(
        delayed(process_fold)(
            train_idx, test_idx, train_df, target,
            numerical_features, categorical_features
        )
        for train_idx, test_idx in outer_cv.split(train_df, y_binned)
    )

    # Process results (remainder identical)
    results_df = pd.DataFrame(all_results)
    print("\n=== Cross-Validation Results ===")
    print("Stratified Encoder Performance:")
    print(f"Mean RMSE: {results_df['rmse_stratified'].mean():.2f}")
    print("Best Parameters:")
    print(f"n_estimators: {results_df['best_n_estimators_stratified'].mode()[0]}")
    print(f"max_depth: {results_df['best_max_depth_stratified'].mode()[0]}")
    print(f"min_samples_split: {results_df['best_min_samples_split_stratified'].mode()[0]}")
    print(f"min_samples_leaf: {results_df['best_min_samples_leaf_stratified'].mode()[0]}")
    print(f"max_features: {results_df['best_max_features_stratified'].mode()[0]}")
    
    print("\nSimple Encoder Performance:")
    print(f"Mean RMSE: {results_df['rmse_simple'].mean():.2f}")
    print("Best Parameters:")
    def get_mode_or_default(series, default='Not Available'):
        modes = series.mode()
        return modes[0] if not modes.empty else default
    print(f"n_estimators: {get_mode_or_default(results_df['best_n_estimators_simple'])}")
    print(f"max_depth: {get_mode_or_default(results_df['best_max_depth_simple'])}")
    print(f"min_samples_split: {get_mode_or_default(results_df['best_min_samples_split_simple'])}")
    print(f"min_samples_leaf: {get_mode_or_default(results_df['best_min_samples_leaf_simple'])}")
    print(f"max_features: {get_mode_or_default(results_df['best_max_features_simple'])}")

    # Final test evaluation
    final_results = {}
    for enc_type in ['stratified', 'simple']:
        best_params = {
            'n_estimators': int(results_df[f'best_n_estimators_{enc_type}'].mode()[0]),
            'max_depth': (
                None 
                if pd.isna(results_df[f'best_max_depth_{enc_type}'].mode()[0]) 
                else int(results_df[f'best_max_depth_{enc_type}'].mode()[0])
            ),
            'min_samples_split': int(results_df[f'best_min_samples_split_{enc_type}'].mode()[0]),
            'min_samples_leaf': int(results_df[f'best_min_samples_leaf_{enc_type}'].mode()[0]),
            'max_features': results_df[f'best_max_features_{enc_type}'].mode()[0],
            'random_state': SEED,
            'n_jobs': -1
        }
        
        pipe = Pipeline([
            ('processor', FeatureProcessor(
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                encoder_cv_type=enc_type, 
                stratify_bins=5  # Updated parameter name
            )),
            ('regressor', RandomForestRegressor(**best_params))
        ])
        
        pipe.fit(train_df.drop(columns=[target]), train_df[target])
        preds = pipe.predict(test_df.drop(columns=[target]))
        final_results[f'test_rmse_{enc_type}'] = np.sqrt(mean_squared_error(test_df[target], preds))
    
    print("\n=== Final Test Performance ===")
    print(f"Stratified Encoder Test RMSE: {final_results['test_rmse_stratified']:.2f}")
    print(f"Simple Encoder Test RMSE: {final_results['test_rmse_simple']:.2f}")   
    
    # Save results (remainder identical)
    results_df.to_csv('cv_results.csv', index=False)
    pd.DataFrame(final_results.items(), columns=['metric', 'value']).to_csv('test_results.csv', index=False)
    plot_data = results_df[['rmse_stratified', 'rmse_simple']]
    plot_data.to_csv('plot_data.csv', index=False)    

#%%     
# Visualization function remains identical
def load_and_plot():
    plot_data = pd.read_csv('plot_data.csv')
    test_results = pd.read_csv('test_results.csv')
    
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    sns.boxplot(
        data=plot_data,
        palette=['#4c72b0', '#dd8452'],
        width=0.5,
        linewidth=1.5
    )
    
    sns.swarmplot(
        data=plot_data,
        color='#2d3436',
        size=6,
        alpha=0.7
    )
    
    plt.title("Cross-Validation RMSE Comparison\nStratifiedCV vs SimpleCV for Encoding categorical features", pad=18)
    plt.ylabel("RMSE", labelpad=12)
    plt.xlabel("Encoding Strategy", labelpad=12)
    
    y_min, y_max = ax.get_ylim()
    means = plot_data.mean()
    for i, mean in enumerate(means):
        plt.text(
            i, 
            mean,
            f'Mean: {mean:.2f}',
            ha='center',
            va='top',
            fontsize=12,
            color='#d63031',
            bbox=dict(
                facecolor='white',
                edgecolor='#d63031',
                boxstyle='round,pad=0.2'
            )
        )
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('results_plot.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    load_and_plot()
# %%
