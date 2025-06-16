#%%
# Import necessary libraries
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
from sklearn.linear_model import Lasso  
from sklearn.preprocessing import StandardScaler, TargetEncoder  
from sklearn.metrics import mean_squared_error  
from sklearn.compose import TransformedTargetRegressor  
from sklearn.base import BaseEstimator, TransformerMixin  
import os

#%%
# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

#%%
class FeatureProcessor(BaseEstimator, TransformerMixin):
    """
    Handles feature processing with proper cross-validation isolation
    """
    
    def __init__(self, numerical_features, categorical_features, 
                 encoder_n_splits=5, encoder_cv_type='simple', 
                 stratify_bins=5, min_samples_per_category=10):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.encoder_n_splits = encoder_n_splits  # CV splits for encoding
        self.encoder_cv_type = encoder_cv_type
        self.stratify_bins = stratify_bins
        self.min_samples_per_category= min_samples_per_category  # Bins for stratification
        
        self.num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        self.cat_encoders = []
        self.X_index = None
        self.encoded_train = None
        self.frequent_categories = {} 

    def fit(self, X, y):
        self.X_index = X.index
        
        # Process numerical features
        self.num_pipeline.fit(X[self.numerical_features])
        # ==== ADDED: Handle rare categories ====
        X_cat = X[self.categorical_features].copy()
        for col in self.categorical_features:
            counts = X_cat[col].value_counts()
            frequent_categories = counts[counts >= self.min_samples_per_category].index
            self.frequent_categories[col] = frequent_categories
            # Replace rare categories with 'Other'
            X_cat.loc[~X_cat[col].isin(frequent_categories), col] = 'Other'
        # =======================================
        
        # Prepare categorical encoding
        X_cat = X[self.categorical_features].copy()
        self.encoded_train = np.zeros((X.shape[0], len(self.categorical_features)))
        
        # Create appropriate CV splits
        if self.encoder_cv_type == 'stratified':
            y_binned = pd.qcut(y, q=self.stratify_bins, labels=False, duplicates='drop')
            kf = StratifiedKFold(n_splits=self.encoder_n_splits, 
                                shuffle=True, random_state=SEED)
            splits = kf.split(X, y_binned)
        else:
            kf = KFold(n_splits=self.encoder_n_splits, 
                      shuffle=True, random_state=SEED)
            splits = kf.split(X)

        # Fit encoders per fold
        for train_idx, val_idx in splits:
            encoder = TargetEncoder(
                smooth="auto",
                target_type="continuous",
                random_state=SEED  # Added for reproducibility
            )
            encoder.fit(X_cat.iloc[train_idx], y.iloc[train_idx])
            self.cat_encoders.append(encoder)
            self.encoded_train[val_idx] = encoder.transform(X_cat.iloc[val_idx])

        return self

    def transform(self, X):
        # Process numerical features
        num_processed = self.num_pipeline.transform(X[self.numerical_features])
        # ==== ADDED: Apply same rare category handling ====
        X_cat = X[self.categorical_features].copy()
        for col in self.categorical_features:
            # Replace rare categories with 'Other'
            X_cat.loc[~X_cat[col].isin(self.frequent_categories[col]), col] = 'Other'
        # =================================================
        
        # Process categorical features
        X_cat = X[self.categorical_features]
        if X.index.equals(self.X_index):
            cat_processed = self.encoded_train
        else:
            encoded_list = [encoder.transform(X_cat) for encoder in self.cat_encoders]
            cat_processed = np.median(encoded_list, axis=0)
        
        return np.hstack([num_processed, cat_processed])

#%%
def process_fold(train_idx, test_idx, df, target, numerical_features, categorical_features):
    """Process one outer CV fold with proper nested validation"""
    X_train = df.iloc[train_idx].drop(columns=[target])
    y_train = df.iloc[train_idx][target]
    X_val = df.iloc[test_idx].drop(columns=[target])
    y_val = df.iloc[test_idx][target]

    results = {}
    
    for enc_type in ['stratified', 'simple']:
        pipe = Pipeline([
            ('processor', FeatureProcessor(
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                encoder_n_splits=5,  # Different from outer CV's 5 splits
                encoder_cv_type=enc_type,
                stratify_bins=5
            )),
            ('regressor', TransformedTargetRegressor(
                regressor=Lasso(max_iter=10000, random_state=SEED),
                transformer=StandardScaler()
            ))
        ])

        # Inner CV strategy (5 splits different from encoder's 3)
        if enc_type == 'stratified':
            y_binned = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            inner_splits = inner_cv.split(X_train, y_binned)
        else:
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
            inner_splits = inner_cv.split(X_train)

        param_grid = {'regressor__regressor__alpha': np.logspace(-4, 2, 10)}
        
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=inner_splits,
            scoring='neg_mean_squared_error',
            n_jobs=1,
            verbose=0
        )
        
        grid.fit(X_train, y_train)
        preds = grid.predict(X_val)
        results[f'rmse_{enc_type}'] = np.sqrt(mean_squared_error(y_val, preds))
        results[f'best_alpha_{enc_type}'] = grid.best_params_['regressor__regressor__alpha']
    
    return results

#%%
if __name__ == "__main__":
    df = pd.read_csv('dataset/IPPS.csv')
    df.columns = df.columns.str.strip()
    
    numerical_features = ['Total Discharges', 'Average Covered Charges', 'Average Medicare Payments']
    categorical_features = ['DRG Definition', 'Provider Name', 'Provider Street Address', 'Provider City', 'Provider State', 'Hospital Referral Region Description']
    df['target_transformed'] = np.log1p(df['Average Total Payments'])  # Log-transform target for better distribution
    target = 'target_transformed'

    # Train-test split
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=pd.qcut(df[target], q=5, labels=False, duplicates='drop'),
        random_state=SEED
    )
    
    # Outer CV (5 splits)
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

    # Process results
    results_df = pd.DataFrame(all_results)
    
    # Results reporting (unchanged)
    print("\n=== Cross-Validation Results ===")
    print("Stratified Encoder Performance:")
    print(f"Mean RMSE: {results_df['rmse_stratified'].mean():.2f}")
    print(f"Best Alphas: {results_df['best_alpha_stratified'].tolist()}")
    
    print("\nSimple Encoder Performance:")
    print(f"Mean RMSE: {results_df['rmse_simple'].mean():.2f}")
    print(f"Best Alphas: {results_df['best_alpha_simple'].tolist()}")

    # Final test evaluation
    final_results = {}
    for enc_type in ['stratified', 'simple']:
        best_alpha = results_df[f'best_alpha_{enc_type}'].mode()[0]
        
        pipe = Pipeline([
            ('processor', FeatureProcessor(
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                encoder_n_splits=5,
                encoder_cv_type=enc_type,
                stratify_bins=5,
                min_samples_per_category=1
            )),
            ('regressor', TransformedTargetRegressor(
                regressor=Lasso(alpha=best_alpha, max_iter=10000, random_state=SEED),
                transformer=StandardScaler()
            ))
        ])
        
        pipe.fit(train_df.drop(columns=[target]), train_df[target])
        preds = pipe.predict(test_df.drop(columns=[target]))
        final_results[f'test_rmse_{enc_type}'] = np.sqrt(mean_squared_error(test_df[target], preds))

    print("\n=== Final Test Performance ===")
    print(f"Stratified Encoder Test RMSE: {final_results['test_rmse_stratified']:.2f}")
    print(f"Simple Encoder Test RMSE: {final_results['test_rmse_simple']:.2f}")

    # Save results (unchanged)
    results_df.to_csv('cv_results.csv', index=False)
    pd.DataFrame(final_results.items()).to_csv('test_results.csv', index=False)
    results_df[['rmse_stratified', 'rmse_simple']].to_csv('plot_data.csv', index=False)

#%%
# Visualization function (unchanged)
def load_and_plot():
    plot_data = pd.read_csv('plot_data.csv')
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    sns.boxplot(data=plot_data, palette=['#4c72b0', '#dd8452'], width=0.5, linewidth=1.5)
    sns.swarmplot(data=plot_data, color='#2d3436', size=6, alpha=0.7)
    plt.title("Cross-Validation RMSE Comparison\nStratifiedCV vs SimpleCV for Encoding categorical features", pad=18)
    plt.ylabel("RMSE", labelpad=12)
    plt.xlabel("Encoding Strategy", labelpad=12)
    y_min, y_max = ax.get_ylim()
    for i, mean in enumerate(plot_data.mean()):
        plt.text(i, y_max*0.98, f'Mean: {mean:.2f}', ha='center', va='top',
                 fontsize=12, color='#d63031', bbox=dict(facecolor='white', edgecolor='#d63031'))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('results_plot.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    load_and_plot()
# %%
