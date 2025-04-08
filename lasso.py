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
from category_encoders import GLMMEncoder  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import mean_squared_error  
from sklearn.compose import TransformedTargetRegressor  
from sklearn.base import BaseEstimator, TransformerMixin  
import os  
#%%
# Set global random seed for reproducibility
SEED = 42  
np.random.seed(SEED)  
random.seed(SEED)  
os.environ['PYTHONHASHSEED'] = str(SEED)  
#%%
class FeatureProcessor(BaseEstimator, TransformerMixin):
    """
    Custom transformer that handles:
    - Numerical feature imputation and scaling
    - Categorical feature encoding using GLMM
    - Stratified or simple cross-validated encoding
    """
    
    def __init__(self, numerical_features, categorical_features, n_splits=5, 
                 encoder_cv_type='simple', n_bins=5):
        
        self.numerical_features = numerical_features  
        self.categorical_features = categorical_features  
        self.n_splits = n_splits  
        self.encoder_cv_type = encoder_cv_type  
        self.n_bins = n_bins  
        
        
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
        
        
        self.num_pipeline.fit(X[self.numerical_features])  
        
        
        X_cat = X[self.categorical_features].copy()  
        self.encoded_train = np.zeros((X.shape[0], len(self.categorical_features)))  
        
        
        if self.encoder_cv_type == 'stratified':
            y_binned = pd.qcut(y, q=self.n_bins, labels=False, duplicates='drop')
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=SEED)
            splits = kf.split(X, y_binned)
        else:
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=SEED)
            splits = kf.split(X)

        
        for train_idx, val_idx in splits:
            encoder = GLMMEncoder()  
            encoder.fit(X_cat.iloc[train_idx], y.iloc[train_idx])  
            self.cat_encoders.append(encoder)
            self.encoded_train[val_idx] = encoder.transform(X_cat.iloc[val_idx])

        return self

    def transform(self, X):
        """Transform input data using fitted processors"""
        
        num_processed = self.num_pipeline.transform(X[self.numerical_features])
        
        X_cat = X[self.categorical_features]
        if X.index.equals(self.X_index):  
            cat_processed = self.encoded_train  
        else:
            
            encoded_list = [encoder.transform(X_cat) for encoder in self.cat_encoders]
            cat_processed = np.median(encoded_list, axis=0)  
        
        return np.hstack([num_processed, cat_processed])
#%%
# def validate_feature_processing(processor, X_train, X_test, y_test, numerical_features, categorical_features):
#     """Comprehensive validation of feature processing"""
    
#     print("\n=== Numerical Feature Validation ===")
#     # Check numerical processing
#     num_pipeline = processor.num_pipeline
#     imputer = num_pipeline.named_steps['imputer']
#     scaler = num_pipeline.named_steps['scaler']
    
#     # Imputation validation
#     train_means = X_train[numerical_features].mean().values
#     print("Imputer Statistics vs Training Means:")
#     print(np.c_[imputer.statistics_, train_means])
    
#     # Scaling validation
#     scaled_means = scaler.mean_
#     scaled_stds = scaler.scale_
#     print("\nScaler Parameters vs Training Statistics:")
#     print(f"{'Feature':<15} | {'Scaler Mean':<10} | {'Data Mean':<10} | {'Scaler STD':<10} | {'Data STD':<10}")
#     for i, feat in enumerate(numerical_features):
#         data_mean = X_train[feat].mean()
#         data_std = X_train[feat].std()
#         print(f"{feat:<15} | {scaled_means[i]:<10.2f} | {data_mean:<10.2f} | {scaled_stds[i]:<10.2f} | {data_std:<10.2f}")

#     # Check transformed numerical features
#     X_transformed = processor.transform(X_train)
#     num_transformed = X_transformed[:, :len(numerical_features)]
#     print("\nScaled Numerical Features Statistics:")
#     print(f"Mean: {num_transformed.mean(axis=0).round(2)}")
#     print(f"Std: {num_transformed.std(axis=0).round(2)}")

#     print("\n=== Categorical Feature Validation ===")
#     # Categorical encoding checks
#     print(f"Number of GLMM Encoders: {len(processor.cat_encoders)} (should equal n_splits)")
    
#     # Training data encoding
#     print("\nTraining Data Encoding Sample:")
#     print(processor.encoded_train[:5])
    
#     # Test data encoding
#     X_test_transformed = processor.transform(X_test)
#     cat_transformed_test = X_test_transformed[:, len(numerical_features):]
    
#     # Manual encoding calculation
#     manual_encoded = np.median(
#         [encoder.transform(X_test[categorical_features]) for encoder in processor.cat_encoders],
#         axis=0
#     )
#     print("\nTest Encoding vs Manual Calculation Match:", np.allclose(cat_transformed_test, manual_encoded))

#     print("\n=== Data Leakage Checks ===")
#     # Numerical leakage check
#     test_num_means = X_test[numerical_features].mean().values
#     print("Test Data Means vs Imputer Statistics:")
#     print(np.c_[test_num_means, imputer.statistics_])
    
#     # Categorical leakage check
#     corr_matrix = np.corrcoef(cat_transformed_test.T, y_test)
#     max_corr = np.abs(corr_matrix[-1, :-1]).max()
#     print(f"\nMax Correlation between Test Encodings and Target: {max_corr:.4f}")

#     print("\n=== Dimension Validation ===")
#     expected_features = len(numerical_features) + len(categorical_features)
#     actual_features = X_transformed.shape[1]
#     print(f"Expected Features: {expected_features}, Actual Features: {actual_features}")

#     print("\n=== Missing Values Check ===")
#     print("NaNs in Transformed Data:", np.isnan(X_transformed).sum())

#%%
# def plot_feature_distributions(processor, X_train, numerical_features, categorical_features):
#     """Visual validation of feature distributions"""
#     plt.figure(figsize=(12, 8))
    
#     # Original vs Scaled Numerical Features
#     for i, feat in enumerate(numerical_features):
#         plt.subplot(2, 2, i+1)
#         sns.kdeplot(X_train[feat], label='Original')
#         sns.kdeplot(processor.transform(X_train)[:, i], label='Processed')
#         plt.title(f"{feat} Distribution")
#         plt.legend()
    
#     # Encoded Categorical Features
#     plt.subplot(2, 2, 4)
#     encoded_cats = processor.transform(X_train)[:, len(numerical_features):].flatten()
#     sns.kdeplot(encoded_cats)
#     plt.title("Encoded Categorical Features Distribution")
    
#     plt.tight_layout()
#     plt.show()
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
        # Create pipeline with consistent feature names
        pipe = Pipeline([
            ('processor', FeatureProcessor(
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                encoder_cv_type=enc_type, 
                n_bins=5
            )),
            ('regressor', TransformedTargetRegressor(
                regressor=Lasso(max_iter=10000, random_state=SEED),
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

        # Hyperparameter grid
        param_grid = {
            'regressor__regressor__alpha': np.logspace(-4, 2, 10)  
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
        results[f'best_alpha_{enc_type}'] = grid.best_params_['regressor__regressor__alpha']
    
    return results
#%%
# Main execution block
if __name__ == "__main__":
    # Load and prepare data
    df = pd.read_csv('dataset/autoscout24-germany-dataset_cleaned.csv')
    df.columns = df.columns.str.strip()
    
    # Consistent feature naming throughout
    numerical_features = ['mileage', 'hp', 'year']
    
    categorical_features = ['make', 'model', 'fuel', 'gear', 'offerType']

    target = 'price'

    # Initial train-test split
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=pd.qcut(df[target], q=5, labels=False, duplicates='drop'),
        random_state=SEED
    )
    # ================== VALIDATION CHECKS ==================
    # # Initialize and fit processor
    # validation_processor = FeatureProcessor(
    #     numerical_features=numerical_features,
    #     categorical_features=categorical_features,
    #     encoder_cv_type='stratified'
    # )
    # validation_processor.fit(train_df.drop(columns=[target]), train_df[target])

    # # Run comprehensive validation
    # validate_feature_processing(
    #     processor=validation_processor,
    #     X_train=train_df,
    #     X_test=test_df,
    #     y_test=test_df[target],
    #     numerical_features=numerical_features,
    #     categorical_features=categorical_features
    # )

    # # Visual distribution checks
    # plot_feature_distributions(
    #     processor=validation_processor,
    #     X_train=train_df,
    #     numerical_features=numerical_features,
    #     categorical_features=categorical_features
    # )
    # =======================================================

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

    # Process results
    results_df = pd.DataFrame(all_results)

    # Cross-validation results
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
                encoder_cv_type=enc_type, 
                n_bins=5
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
    plt.title("Cross-Validation RMSE Comparison\nStratified vs Simple Encoding", pad=18)  # Reduced pad
    plt.ylabel("RMSE", labelpad=12)  # Reduced label padding
    plt.xlabel("Encoding Strategy", labelpad=12)
    
    # Calculate axis limits before annotations
    y_min, y_max = ax.get_ylim()
    
    # Add annotations with dynamic positioning
    means = plot_data.mean()
    for i, mean in enumerate(means):
        plt.text(
            i, 
            y_max * 0.98,  # Position at 98% of y-axis height
            f'Mean: {mean:.2f}',
            ha='center',
            va='top',  # Align to top of text
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
    
    
# %%
