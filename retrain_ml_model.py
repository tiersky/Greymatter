#!/usr/bin/env python3
"""
Retrain LightGBM model with enhanced post-campaign metrics
This script automatically detects available features and trains accordingly
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

def detect_enhanced_features(df):
    """Detect which features are available in the dataset"""
    print("🔍 Analyzing available features...")
    
    # Core features (always available)
    core_features = ['Followers', 'Quality_Audience', 'ER', 'Est_Post_Price', 
                    'Avg_Reels_View', 'Turkey', 'Comment_Rate', 'Score']
    
    # Enhanced post-campaign features
    enhanced_features = ['Sponsored_Reels_View', 'Campaign_Reach', 'Performance_Ratio',
                        'Cost_Per_Reach', 'Link_Clicks', 'Cost_Per_Click', 'Actual_Campaign_Cost']
    
    available_core = [f for f in core_features if f in df.columns]
    available_enhanced = [f for f in enhanced_features if f in df.columns]
    
    print(f"✅ Core features available: {len(available_core)}/{len(core_features)}")
    print(f"✅ Enhanced features available: {len(available_enhanced)}/{len(enhanced_features)}")
    
    total_features = available_core + available_enhanced
    print(f"📊 Total features for training: {len(total_features)}")
    print(f"   Features: {total_features}")
    
    return total_features

def calculate_target_roi(df):
    """Calculate target ROI with enhanced formula if post-campaign data is available"""
    print("🧮 Calculating target ROI variable...")
    
    if 'Performance_Ratio' in df.columns and 'Cost_Per_Reach' in df.columns:
        print("   Using enhanced ROI calculation with post-campaign metrics")
        
        # Enhanced ROI calculation for rows with post-campaign data
        def enhanced_roi(row):
            try:
                performance_score = float(row.get('Performance_Ratio', 1.0))
                reach_efficiency = 1 / max(float(row.get('Cost_Per_Reach', 0.01)), 0.001)
                engagement_quality = float(row['ER']) * 10
                audience_quality = float(row['Quality_Audience']) / 100000
                
                click_efficiency = 0
                if row.get('Cost_Per_Click', 0) > 0:
                    click_efficiency = 100 / max(float(row.get('Cost_Per_Click', 100)), 1)
                
                market_factor = float(row['Turkey']) / 100
                
                advanced_roi = (
                    performance_score * 0.25 +
                    reach_efficiency * 0.20 +
                    engagement_quality * 0.20 +
                    audience_quality * 0.15 +
                    click_efficiency * 0.10 +
                    market_factor * 0.10
                ) * 1000
                
                return advanced_roi
            except:
                # Fallback to ML-guided formula
                return calculate_ml_guided_roi(row)
        
        df["ROI_star"] = df.apply(enhanced_roi, axis=1)
    else:
        print("   Using ML-guided ROI calculation (core features only)")
        df["ROI_star"] = df.apply(calculate_ml_guided_roi, axis=1)
    
    print(f"📈 ROI statistics:")
    print(f"   Mean: {df['ROI_star'].mean():.2f}")
    print(f"   Std:  {df['ROI_star'].std():.2f}")
    print(f"   Min:  {df['ROI_star'].min():.2f}")
    print(f"   Max:  {df['ROI_star'].max():.2f}")
    
    return df

def calculate_ml_guided_roi(row):
    """ML-guided ROI calculation based on feature importance"""
    try:
        # Based on LightGBM feature importance analysis
        engagement_score = float(row['ER']) * 0.19
        reach_score = float(row['Avg_Reels_View']) / 1000000 * 0.171
        audience_size_score = float(row['Followers']) / 1000000 * 0.157
        quality_score = float(row['Quality_Audience']) / 100000 * 0.12
        market_score = float(row['Turkey']) / 100 * 0.10
        influence_score = float(row['Score']) / 100 * 0.08
        comment_engagement = float(row['Comment_Rate']) * 0.06
        cost_efficiency = 100000 / max(float(row['Est_Post_Price']), 1) * 0.05
        
        ml_roi = (engagement_score + reach_score + audience_size_score + quality_score +
                 market_score + influence_score + comment_engagement + cost_efficiency) * 1000
        
        return ml_roi
    except:
        return 0

def train_enhanced_model(X, y, feature_columns):
    """Train LightGBM model with appropriate hyperparameters"""
    print("🤖 Training enhanced LightGBM model...")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Enhanced hyperparameters for more features
    n_features = len(feature_columns)
    
    if n_features > 10:  # Enhanced dataset
        reg = lgb.LGBMRegressor(
            objective="regression", 
            metric="rmse",
            learning_rate=0.02,  # Lower for more complex model
            n_estimators=1000,
            num_leaves=min(256, 2**min(n_features//2, 8)),  # Adaptive based on features
            min_data_in_leaf=8,
            subsample=0.8, 
            colsample_bytree=0.7,  # Lower to prevent overfitting with more features
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            random_state=42,
            verbose=-1
        )
    else:  # Core dataset
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
    
    print(f"📊 Enhanced Model Performance:")
    print(f"   Training R²:   {train_r2:.4f} ({train_r2*100:.1f}%)")
    print(f"   Validation R²: {val_r2:.4f} ({val_r2*100:.1f}%)")
    print(f"   Training RMSE: {train_rmse:.4f}")
    print(f"   Validation RMSE: {val_rmse:.4f}")
    print(f"   Features used: {n_features}")
    
    return reg, train_r2, val_r2, train_rmse, val_rmse

def main():
    print("🚀 GREYMATTER Enhanced LightGBM Model Training")
    print("=" * 60)
    
    # Load data
    try:
        df = pd.read_csv("influencers.csv")
        print(f"✅ Loaded {len(df)} records from CSV")
    except FileNotFoundError:
        print("❌ influencers.csv not found")
        return
    
    # Detect available features
    feature_columns = detect_enhanced_features(df)
    
    if len(feature_columns) < 5:
        print("❌ Insufficient features for training")
        return
    
    # Calculate target ROI
    df = calculate_target_roi(df)
    
    # Prepare features
    X = df[feature_columns].copy()
    X = X.fillna(X.mean())  # Handle any missing values
    
    # Target: standardized ROI
    roi_mean = df["ROI_star"].mean()
    roi_std = df["ROI_star"].std()
    y = (df["ROI_star"] - roi_mean) / roi_std
    
    print(f"✅ Prepared {len(feature_columns)} features for training")
    
    # Train model
    model, train_r2, val_r2, train_rmse, val_rmse = train_enhanced_model(X, y, feature_columns)
    
    # Statistical validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"🔄 Cross-Validation Results:")
    print(f"   CV R² Mean: {cv_mean:.4f} ({cv_mean*100:.1f}%)")
    print(f"   CV R² Std:  {cv_std:.4f}")
    
    # Statistical significance
    t_stat, p_value = stats.ttest_1samp(cv_scores, 0)
    print(f"📊 Statistical Significance:")
    print(f"   p-value: {p_value:.6f}")
    print(f"   Significant: {'✅ YES' if p_value < 0.05 else '❌ NO'}")
    
    # Save enhanced model
    joblib.dump(model, 'roi_lightgbm_model.pkl')
    
    scaler_info = {
        'roi_mean': roi_mean,
        'roi_std': roi_std,
        'feature_columns': feature_columns
    }
    joblib.dump(scaler_info, 'roi_model_scaler.pkl')
    
    model_info = {
        'model_type': 'Enhanced LightGBM Regressor',
        'train_r2': train_r2,
        'val_r2': val_r2,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'statistical_significance_p': p_value,
        'is_statistically_significant': p_value < 0.05,
        'feature_count': len(feature_columns),
        'enhanced_features': len([f for f in feature_columns if f.startswith(('Sponsored_', 'Campaign_', 'Performance_', 'Cost_', 'Link_'))]),
        'training_date': pd.Timestamp.now().isoformat()
    }
    joblib.dump(model_info, 'roi_model_info.pkl')
    
    print("\n🎉 Enhanced model training completed!")
    print(f"📊 Final Performance:")
    print(f"   Validation R²: {val_r2:.1%}")
    print(f"   Enhanced Features: {model_info['enhanced_features']}")
    print(f"   Statistical Significance: {'✅ YES' if p_value < 0.05 else '❌ NO'}")
    
    if val_r2 > 0.95:
        print("🏆 Excellent model performance (>95% validation accuracy)!")
    elif val_r2 > 0.85:
        print("👍 Good model performance (>85% validation accuracy)")
    
    print("\n🔄 Restart dashboard backend to load the enhanced model")

if __name__ == "__main__":
    main()