#!/usr/bin/env python3
"""
Train and save LightGBM model for GREYMATTER dashboard
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load data and calculate ROI_star target variable"""
    print("📊 Loading data...")
    
    # Try parquet first, fall back to CSV
    if os.path.exists("influencer_modelling_ready.parquet"):
        try:
            df = pd.read_parquet("influencer_modelling_ready.parquet")
            print(f"✅ Loaded {len(df)} records from parquet file")
        except ImportError:
            print("⚠️  Parquet dependencies not available, using CSV...")
            df = pd.read_csv("influencers.csv")
    else:
        df = pd.read_csv("influencers.csv")
        print(f"✅ Loaded {len(df)} records from CSV file")
    
    # Calculate ROI_star (target variable)
    print("🧮 Calculating ROI_star target variable...")
    Q = df["Quality_Audience"]
    E = df["ER"]
    M = df["Turkey"] / 100
    P = df["Est_Post_Price"]
    CM = df["Comment_Rate"]
    IM = 1.0  # Influence multiplier
    
    # ROI calculation: (Q × E × M × IM × (1 + 0.2×CM)) / P
    df["ROI_star"] = (Q * E * M * IM * (1 + 0.2 * CM)) / P
    
    print(f"📈 ROI_star statistics:")
    print(f"   Mean: {df['ROI_star'].mean():.2f}")
    print(f"   Std:  {df['ROI_star'].std():.2f}")
    print(f"   Min:  {df['ROI_star'].min():.2f}")
    print(f"   Max:  {df['ROI_star'].max():.2f}")
    
    return df

def prepare_features_and_target(df):
    """Prepare features and standardized target"""
    print("🔧 Preparing features and target...")
    
    # Features: all columns except ROI_star and text columns
    text_columns = ['Account', 'Category', 'Interests']
    feature_columns = [col for col in df.columns if col not in ['ROI_star'] + text_columns]
    
    X = df[feature_columns].copy()
    
    # Handle any missing values
    X = X.fillna(X.mean())
    
    # Target: standardized ROI_star for better model performance
    roi_mean = df["ROI_star"].mean()
    roi_std = df["ROI_star"].std()
    y = (df["ROI_star"] - roi_mean) / roi_std
    
    print(f"✅ Features: {len(feature_columns)} columns")
    print(f"   Features: {feature_columns}")
    print(f"✅ Target: Standardized ROI_star (mean=0, std=1)")
    
    return X, y, roi_mean, roi_std, feature_columns

def train_lightgbm_model(X, y):
    """Train LightGBM model with proper validation"""
    print("🤖 Training LightGBM model...")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Configure LightGBM model
    reg = lgb.LGBMRegressor(
        objective="regression", 
        metric="rmse",
        learning_rate=0.03, 
        n_estimators=800,
        num_leaves=128, 
        min_data_in_leaf=10,
        subsample=0.8, 
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    # Train model
    reg.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
    )
    
    # Make predictions
    y_train_pred = reg.predict(X_train)
    y_val_pred = reg.predict(X_val)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    print(f"📊 Model Performance:")
    print(f"   Training R²:   {train_r2:.4f} ({train_r2*100:.1f}%)")
    print(f"   Validation R²: {val_r2:.4f} ({val_r2*100:.1f}%)")
    print(f"   Training RMSE: {train_rmse:.4f}")
    print(f"   Validation RMSE: {val_rmse:.4f}")
    
    return reg, train_r2, val_r2, train_rmse, val_rmse

def statistical_validation(X, y, model):
    """Perform statistical significance tests"""
    print("📈 Running statistical validation tests...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"🔄 Cross-Validation Results:")
    print(f"   CV R² Mean: {cv_mean:.4f} ({cv_mean*100:.1f}%)")
    print(f"   CV R² Std:  {cv_std:.4f}")
    print(f"   CV Scores:  {[f'{score:.3f}' for score in cv_scores]}")
    
    # Test if CV scores are significantly better than 0
    t_stat, p_value = stats.ttest_1samp(cv_scores, 0)
    print(f"📊 Statistical Significance Test:")
    print(f"   H0: Model performance = 0 (no predictive power)")
    print(f"   H1: Model performance > 0 (has predictive power)")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print(f"   ✅ SIGNIFICANT: Model has statistically significant predictive power (p < 0.05)")
    else:
        print(f"   ❌ NOT SIGNIFICANT: Cannot reject null hypothesis (p >= 0.05)")
    
    return cv_mean, cv_std, p_value

def save_model_and_metadata(model, roi_mean, roi_std, feature_columns, metrics):
    """Save trained model and all necessary metadata"""
    print("💾 Saving model and metadata...")
    
    # Save the trained model
    joblib.dump(model, 'roi_lightgbm_model.pkl')
    
    # Save preprocessing parameters
    scaler_info = {
        'roi_mean': roi_mean,
        'roi_std': roi_std,
        'feature_columns': feature_columns
    }
    joblib.dump(scaler_info, 'roi_model_scaler.pkl')
    
    # Save model performance metrics
    model_info = {
        'model_type': 'LightGBM Regressor',
        'train_r2': metrics['train_r2'],
        'val_r2': metrics['val_r2'],
        'cv_mean': metrics['cv_mean'],
        'cv_std': metrics['cv_std'],
        'statistical_significance_p': metrics['p_value'],
        'is_statistically_significant': metrics['p_value'] < 0.05,
        'feature_count': len(feature_columns),
        'training_date': pd.Timestamp.now().isoformat()
    }
    joblib.dump(model_info, 'roi_model_info.pkl')
    
    print("✅ Model artifacts saved:")
    print("   - roi_lightgbm_model.pkl (trained model)")
    print("   - roi_model_scaler.pkl (preprocessing parameters)")
    print("   - roi_model_info.pkl (performance metrics)")

def main():
    print("🚀 GREYMATTER LightGBM Model Training")
    print("=" * 50)
    
    try:
        # Check if lightgbm is available
        import lightgbm as lgb
        print("✅ LightGBM is available")
    except ImportError:
        print("❌ LightGBM not installed. Install with: pip install lightgbm")
        return
    
    # Load and prepare data
    df = load_and_prepare_data()
    X, y, roi_mean, roi_std, feature_columns = prepare_features_and_target(df)
    
    # Train model
    model, train_r2, val_r2, train_rmse, val_rmse = train_lightgbm_model(X, y)
    
    # Statistical validation
    cv_mean, cv_std, p_value = statistical_validation(X, y, model)
    
    # Save everything
    metrics = {
        'train_r2': train_r2,
        'val_r2': val_r2,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'p_value': p_value
    }
    
    save_model_and_metadata(model, roi_mean, roi_std, feature_columns, metrics)
    
    print("\n🎉 Model training completed successfully!")
    print(f"📊 Final Performance Summary:")
    print(f"   Validation R²: {val_r2:.1%}")
    print(f"   Cross-Val R²:  {cv_mean:.1%} ± {cv_std:.1%}")
    print(f"   Statistical Significance: {'✅ YES' if p_value < 0.05 else '❌ NO'} (p={p_value:.6f})")
    
    if val_r2 > 0.90:
        print("🏆 Excellent model performance (>90% validation accuracy)!")
    elif val_r2 > 0.80:
        print("👍 Good model performance (>80% validation accuracy)")
    else:
        print("⚠️  Model performance could be improved")

if __name__ == "__main__":
    main()