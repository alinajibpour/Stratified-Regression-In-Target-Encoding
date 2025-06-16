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
from sklearn.svm import SVR  # Changed from RandomForestRegressor to SVR
from sklearn.preprocessing import TargetEncoder, StandardScaler  
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
                 stratify_bins=5, min_samples_per_category=10):
        
        self.numerical_features = numerical_features  
        self.categorical_features = categorical_features  
        self.encoder_n_splits = encoder_n_splits  
        self.encoder_cv_type = encoder_cv_type  
        self.stratify_bins = stratify_bins  
        self.min_samples_per_category = min_samples_per_category
        
        self.num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  
            ('scaler', StandardScaler())  
        ])
        
        self.cat_encoders = []  
        self.X_index = None  
        self.encoded_train = None
        self.frequent_categories = {}  

    def fit(self, X, y):
        """Fit the processor to training data"""
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
        
        num_processed = self.num_pipeline.transform(X[self.numerical_features])
        
        # ==== ADDED: Apply same rare category handling ====
        X_cat = X[self.categorical_features].copy()
        for col in self.categorical_features:
            # Replace rare categories with 'Other'
            X_cat.loc[~X_cat[col].isin(self.frequent_categories[col]), col] = 'Other'
        # =================================================
        
        X_cat = X[self.categorical_features]
        if X.index.equals(self.X_index):  
            cat_processed = self.encoded_train  
        else:
            
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
    # Consistent naming for data splits
    X_train = df.iloc[train_idx].drop(columns=[target])  
    y_train = df.iloc[train_idx][target]  
    X_val = df.iloc[test_idx].drop(columns=[target]) 
    y_val = df.iloc[test_idx][target]  

    results = {}  
    
    # Evaluate both encoding strategies
    for enc_type in ['stratified', 'simple']:
        # Create pipeline with SVR
        pipe = Pipeline([
            ('processor', FeatureProcessor(
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                encoder_cv_type=enc_type, 
                stratify_bins=5
            )),
            ('regressor', TransformedTargetRegressor(
                regressor=SVR(cache_size=2000),
                transformer=StandardScaler()
            ))
        ])

        # Create inner CV strategy
        if enc_type == 'stratified':
            y_binned = pd.qcut(y_train, q=5, labels=False, duplicates='drop')
            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            inner_splits = inner_cv.split(X_train, y_binned)
        else:
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
            inner_splits = inner_cv.split(X_train)

        # Hyperparameter grid for SVR
        param_grid = {
            'regressor__regressor__C': [0.1, 1, 10, 100],  # Range of C values
            'regressor__regressor__kernel': ['rbf'],
            'regressor__regressor__gamma': ['scale', 'auto', 0.1, 1, 10]  # Range of gamma values
            
        }

        # Configure grid search
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
        results[f'best_C_{enc_type}'] = best_params['regressor__regressor__C']
        results[f'best_kernel_{enc_type}'] = best_params['regressor__regressor__kernel']
        results[f'best_gamma_{enc_type}'] = best_params['regressor__regressor__gamma']
    
    return results

#%%
#Main execution block
if __name__ == "__main__":
    #Load and prepare data
    df = pd.read_csv('dataset/IPPS.csv')
    df.columns = df.columns.str.strip()
    
    #Consistent feature naming throughout
    numerical_features = ['Total Discharges', 'Average Covered Charges', 'Average Medicare Payments']
    
    categorical_features = ['DRG Definition', 'Provider Name', 'Provider Street Address', 'Provider City', 'Provider State', 'Hospital Referral Region Description']
    df['target_transformed'] = np.log1p(df['Average Total Payments'])  # Log-transform target for better distribution
    target = 'target_transformed'

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

    # Parallel processing with consistent parameter names
    all_results = Parallel(n_jobs=-1, verbose=10, prefer='processes')(
        delayed(process_fold)(
            train_idx, test_idx, train_df, target,
            numerical_features, categorical_features  # Correct parameter names
        )
        for train_idx, test_idx in outer_cv.split(train_df, y_binned)
    )

    #Process results
    results_df = pd.DataFrame(all_results)
    # Helper function for parameter extraction
    def get_mode_or_default(series, default='Not Available'):
        modes = series.mode()
        return modes[0] if not modes.empty else default

    # Cross-validation results
    print("\n=== Cross-Validation Results ===")
    print("Stratified Encoder Performance:")
    print(f"Mean RMSE: {results_df['rmse_stratified'].mean():.2f}")
    print("Best Parameters:")
    print(f"C: {get_mode_or_default(results_df['best_C_stratified'])}")
    print(f"kernel: {get_mode_or_default(results_df['best_kernel_stratified'])}")
    print(f"gamma: {get_mode_or_default(results_df['best_gamma_stratified'])}")
    
    
    print("\nSimple Encoder Performance:")
    print(f"Mean RMSE: {results_df['rmse_simple'].mean():.2f}")
    print("Best Parameters:")
    print(f"C: {get_mode_or_default(results_df['best_C_simple'])}")
    print(f"kernel: {get_mode_or_default(results_df['best_kernel_simple'])}")
    print(f"gamma: {get_mode_or_default(results_df['best_gamma_simple'])}")
    
# Final test evaluation
    final_results = {}
    for enc_type in ['stratified', 'simple']:
        # Get best parameters using mode
        best_C = get_mode_or_default(results_df[f'best_C_{enc_type}'])
        best_kernel = get_mode_or_default(results_df[f'best_kernel_{enc_type}'])
        best_gamma = get_mode_or_default(results_df[f'best_gamma_{enc_type}'])
        
        
        
        # Create pipeline with corrected parameters
        pipe = Pipeline([
            ('processor', FeatureProcessor(
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                encoder_cv_type=enc_type, 
                stratify_bins=5,
                min_samples_per_category=1  # Added parameter for rare categories
            )),
            ('regressor', TransformedTargetRegressor(
                regressor=SVR(
                    C=best_C,
                    kernel=best_kernel,
                    gamma=best_gamma,
                    cache_size=2000
                ),
                transformer=StandardScaler()
            ))
        ])
        
        
        # Train and evaluate
        pipe.fit(train_df.drop(columns=[target]), train_df[target])
        preds = pipe.predict(test_df.drop(columns=[target]))
        final_results[f'test_rmse_{enc_type}'] = np.sqrt(mean_squared_error(test_df[target], preds))
    
    # Moved outside the loop
    print("\n=== Final Test Performance ===")
    print(f"Stratified Encoder Test RMSE: {final_results['test_rmse_stratified']:.2f}")
    print(f"Simple Encoder Test RMSE: {final_results['test_rmse_simple']:.2f}")   
    
    # Save results to disk
    results_df.to_csv('cv_results.csv', index=False)
    pd.DataFrame(final_results.items(), columns=['metric', 'value']).to_csv('test_results.csv', index=False)
    
    # Save plot data separately
    plot_data = results_df[['rmse_stratified', 'rmse_simple']]
    plot_data.to_csv('plot_data.csv', index=False)    
#%%     
# Visualize cross-validation results
def load_and_plot():
    # Load pre-computed results
    plot_data = pd.read_csv('plot_data.csv')
    test_results = pd.read_csv('test_results.csv')
    
    # Create larger figure with adjusted proportions
    plt.figure(figsize=(10, 7))  # Increased height from 6 to 7
    ax = plt.gca()
    
    # Create plots
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
    
    # Customize plot with safer margins
    plt.title("Cross-Validation RMSE Comparison\nStratifiedCV vs SimpleCV for Encoding categorical features", pad=18)  # Reduced pad
    plt.ylabel("RMSE", labelpad=12)  # Reduced label padding
    plt.xlabel("Encoding Strategy", labelpad=12)
    
    # Calculate axis limits before annotations
    y_min, y_max = ax.get_ylim()
    
    # Add annotations with dynamic positioning
    means = plot_data.mean()
    for i, mean in enumerate(means):
        plt.text(
            i, 
            mean,  # Position at 98% of y-axis height
            f'Mean: {mean:.2f}',
            ha='center',
            va='top', # Align text to bottom
            fontsize=12,
            color='#d63031',
            bbox=dict(
                facecolor='white',
                edgecolor='#d63031',
                boxstyle='round,pad=0.2'  # Smaller padding
            )
        )
    
    # Manual layout adjustment
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Use constrained layout instead of tight_layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve 5% space at top
    plt.savefig('results_plot.png', bbox_inches='tight')  # Extra margin safety
    plt.close()

if __name__ == "__main__":
    load_and_plot()