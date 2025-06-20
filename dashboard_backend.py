#!/usr/bin/env python3
"""
GREYMATTER Dashboard Backend
Handles data processing and CSV updates for the web dashboard
"""

import json
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from datetime import datetime
import logging
import joblib

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File paths
CSV_FILE = "influencers.csv"
PROCESSED_FILE = "influencer_modelling_ready.parquet"

class InfluencerDataManager:
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        self.ml_model = None
        self.ml_scaler = None
        self.ml_info = None
        self.load_data()
        self.load_ml_model()
    
    def load_data(self):
        """Load influencer data from CSV and parquet files"""
        try:
            if os.path.exists(CSV_FILE):
                self.raw_data = pd.read_csv(CSV_FILE)
                logger.info(f"Loaded {len(self.raw_data)} records from CSV")
            else:
                logger.warning(f"CSV file {CSV_FILE} not found")
                
            # Try to load parquet file, but don't fail if dependencies are missing
            if os.path.exists(PROCESSED_FILE):
                try:
                    self.processed_data = pd.read_parquet(PROCESSED_FILE)
                    logger.info(f"Loaded processed data with shape {self.processed_data.shape}")
                except ImportError as e:
                    logger.warning(f"Parquet support not available: {e}")
                    logger.info("Dashboard will work with CSV data only")
                    self.processed_data = None
            else:
                logger.warning(f"Processed file {PROCESSED_FILE} not found")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Don't raise if we have CSV data
            if self.raw_data is None:
                raise
    
    def load_ml_model(self):
        """Load trained LightGBM model and preprocessing parameters"""
        try:
            if os.path.exists('roi_lightgbm_model.pkl'):
                self.ml_model = joblib.load('roi_lightgbm_model.pkl')
                self.ml_scaler = joblib.load('roi_model_scaler.pkl')
                self.ml_info = joblib.load('roi_model_info.pkl')
                logger.info(f"✅ ML Model loaded successfully")
                logger.info(f"   Model type: {self.ml_info['model_type']}")
                logger.info(f"   Validation R²: {self.ml_info['val_r2']:.1%}")
                logger.info(f"   Statistical significance: {'YES' if self.ml_info['is_statistically_significant'] else 'NO'}")
            else:
                logger.warning("⚠️  ML model files not found. Using mathematical formula fallback.")
                logger.info("   Run 'python3 train_ml_model.py' to train the model")
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            logger.info("Using mathematical formula fallback")
    
    def calculate_roi(self, row):
        """Calculate ROI score using ML model or mathematical formula fallback"""
        try:
            # Try ML model first if available
            if self.ml_model is not None and self.ml_scaler is not None:
                return self.predict_roi_ml(row)
            else:
                # Fallback to mathematical formula
                return self.calculate_roi_formula(row)
        except Exception as e:
            logger.error(f"Error calculating ROI: {e}")
            return 0
    
    def predict_roi_ml(self, row):
        """Use trained LightGBM model to predict ROI"""
        try:
            # Prepare feature vector
            feature_columns = self.ml_scaler['feature_columns']
            features = []
            
            for col in feature_columns:
                if col in row.index:
                    features.append(float(row[col]))
                else:
                    # Handle missing features with default values
                    features.append(0.0)
            
            # Make prediction (model outputs standardized values)
            features_array = np.array(features).reshape(1, -1)
            standardized_pred = self.ml_model.predict(features_array)[0]
            
            # Convert back to original scale
            roi_mean = self.ml_scaler['roi_mean']
            roi_std = self.ml_scaler['roi_std']
            roi_pred = (standardized_pred * roi_std) + roi_mean
            
            return round(roi_pred, 2)
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            # Fallback to formula
            return self.calculate_roi_formula(row)
    
    def calculate_roi_formula(self, row):
        """Calculate ROI using ML-guided formula based on LightGBM feature importance"""
        try:
            # Based on LightGBM feature importance analysis:
            # ER: 19.0%, Avg_Reels_View: 17.1%, Followers: 15.7%, Quality_Audience: ~12%
            
            # Core engagement metrics (highest importance)
            engagement_score = float(row['ER']) * 0.19  # 19% weight
            reach_score = float(row['Avg_Reels_View']) / 1000000 * 0.171  # 17.1% weight, normalized
            audience_size_score = float(row['Followers']) / 1000000 * 0.157  # 15.7% weight, normalized
            
            # Quality and targeting metrics
            quality_score = float(row['Quality_Audience']) / 100000 * 0.12  # ~12% weight, normalized
            market_score = float(row['Turkey']) / 100 * 0.10  # Market targeting
            influence_score = float(row['Score']) / 100 * 0.08  # Overall influence score
            comment_engagement = float(row['Comment_Rate']) * 0.06  # Comment engagement
            
            # Cost efficiency (inverse relationship)
            cost_efficiency = 100000 / max(float(row['Est_Post_Price']), 1) * 0.05  # 5% weight
            
            # ML-guided ROI calculation
            ml_roi = (
                engagement_score + 
                reach_score + 
                audience_size_score + 
                quality_score + 
                market_score + 
                influence_score + 
                comment_engagement + 
                cost_efficiency
            ) * 1000  # Scale for readability
            
            return round(ml_roi, 2)
            
        except Exception as e:
            logger.error(f"Error in ML-guided formula calculation: {e}")
            # Fallback to simple calculation
            try:
                Q = row['Quality_Audience']
                E = row['ER'] 
                P = row['Est_Post_Price']
                roi = (Q * E) / P
                return round(roi, 2)
            except:
                return 0
    
    def calculate_advanced_roi(self, row):
        """Calculate advanced ROI using post-campaign metrics and ML insights"""
        try:
            # If we have post-campaign data, use it for more accurate ROI
            if 'Performance_Ratio' in row and 'Cost_Per_Reach' in row:
                # Advanced ROI calculation using real campaign performance
                performance_score = float(row.get('Performance_Ratio', 1.0))  # actual vs predicted performance
                reach_efficiency = 1 / max(float(row.get('Cost_Per_Reach', 0.01)), 0.001)  # higher reach per cost = better
                engagement_quality = float(row['ER']) * 10  # Scale engagement rate
                audience_quality = float(row['Quality_Audience']) / 100000  # Normalize quality
                
                # Click efficiency (if available)
                click_efficiency = 0
                if row.get('Cost_Per_Click', 0) > 0:
                    click_efficiency = 100 / max(float(row.get('Cost_Per_Click', 100)), 1)
                
                # Market penetration factor
                market_factor = float(row['Turkey']) / 100
                
                # Advanced ROI incorporating actual campaign performance
                advanced_roi = (
                    performance_score * 0.25 +      # 25% weight on performance vs baseline
                    reach_efficiency * 0.20 +       # 20% weight on reach efficiency  
                    engagement_quality * 0.20 +     # 20% weight on engagement quality
                    audience_quality * 0.15 +       # 15% weight on audience quality
                    click_efficiency * 0.10 +       # 10% weight on click efficiency
                    market_factor * 0.10             # 10% weight on market targeting
                ) * 1000  # Scale for readability
                
                return round(advanced_roi, 2)
            else:
                # Fallback to ML-guided formula for existing data
                return self.calculate_roi_formula(row)
                
        except Exception as e:
            logger.error(f"Error in advanced ROI calculation: {e}")
            return self.calculate_roi_formula(row)
    
    def get_follower_tier(self, followers):
        """Determine follower tier based on follower count"""
        followers = int(followers)
        if followers < 10000:
            return 'Nano'
        elif followers < 100000:
            return 'Micro'
        elif followers < 500000:
            return 'Mid'
        elif followers < 2000000:
            return 'Macro'
        elif followers < 10000000:
            return 'Star'
        else:
            return 'Superstar'
    
    def get_dashboard_data(self, limit=None, category_filter='', tier_filter=''):
        if self.raw_data is None or len(self.raw_data) == 0:
            logger.error("No raw data available")
            return {"error": "No data available"}
        
        logger.info(f"Processing {len(self.raw_data)} records from CSV")
        
        # Calculate ROI for all records
        data_with_roi = self.raw_data.copy()
        data_with_roi['ROI_star'] = data_with_roi.apply(self.calculate_roi, axis=1)
        
        # Apply category filter if provided
        if category_filter:
            # Case-insensitive partial match in Category column
            mask = data_with_roi['Category'].str.contains(category_filter, case=False, na=False)
            data_with_roi = data_with_roi[mask]
            logger.info(f"Applied category filter '{category_filter}': {len(data_with_roi)} records remain")
        
        # Apply tier filter if provided
        if tier_filter:
            # Calculate tiers for all records
            data_with_roi['Tier'] = data_with_roi['Followers'].apply(self.get_follower_tier)
            # Filter by selected tier
            tier_mask = data_with_roi['Tier'] == tier_filter
            data_with_roi = data_with_roi[tier_mask]
            logger.info(f"Applied tier filter '{tier_filter}': {len(data_with_roi)} records remain")
        
        # Sort by ROI
        data_with_roi = data_with_roi.sort_values('ROI_star', ascending=False)
        
        # Apply limit if specified
        if limit:
            data_with_roi = data_with_roi.head(limit)
            logger.info(f"Limited results to top {limit} influencers")
        
        # Prepare dashboard data
        dashboard_data = []
        for _, row in data_with_roi.iterrows():
            try:
                # Calculate tier for this influencer
                tier = self.get_follower_tier(row['Followers'])
                
                dashboard_data.append({
                    'account': str(row['Account']),
                    'roi': round(float(row['ROI_star']), 0),
                    'followers': int(row['Followers']),
                    'engagement': round(float(row['ER']), 2),
                    'postPrice': int(row['Est_Post_Price']),
                    'turkey': int(row['Turkey']),
                    'qualityAudience': int(row['Quality_Audience']),
                    'commentRate': round(float(row['Comment_Rate']), 3),
                    'avgReelsView': int(row['Avg_Reels_View']),
                    'score': int(row['Score']),
                    'category': str(row['Category']) if pd.notna(row['Category']) else '',
                    'interests': str(row['Interests']) if pd.notna(row['Interests']) else '',
                    'tier': tier
                })
            except Exception as e:
                logger.error(f"Error processing row: {e}")
                continue
        
        logger.info(f"Returning {len(dashboard_data)} processed records")
        return dashboard_data
    
    def add_campaign_data(self, campaign_data):
        """Add new campaign data to CSV"""
        try:
            # Validate required fields
            required_fields = ['account', 'followers', 'qualityAudience', 'avgReelsView', 
                             'turkey', 'score', 'category', 'interests', 'commentRate',
                             'sponsoredReelsView', 'campaignEngagementRate', 'campaignReach', 
                             'actualCost']
            
            for field in required_fields:
                if field not in campaign_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Calculate derived metrics from post-campaign data
            sponsored_views = int(campaign_data['sponsoredReelsView'])
            avg_views = int(campaign_data['avgReelsView'])
            performance_ratio = sponsored_views / max(avg_views, 1)
            
            campaign_reach = int(campaign_data['campaignReach'])
            actual_cost = float(campaign_data['actualCost'])
            cost_per_reach = actual_cost / max(campaign_reach, 1)
            
            link_clicks = int(campaign_data.get('linkClicks', 0))
            cost_per_click = (actual_cost / max(link_clicks, 1)) if link_clicks > 0 else 0
            
            # Use campaign engagement rate as the main ER (real performance)
            campaign_er = float(campaign_data['campaignEngagementRate'])
            
            # Create new row with enhanced ML-focused features
            new_row = {
                'Account': campaign_data['account'],
                'Followers': int(campaign_data['followers']),
                'Quality_Audience': int(campaign_data['qualityAudience']),
                'ER': campaign_er,  # Use actual campaign engagement rate
                'Est_Post_Price': actual_cost,  # Use actual cost for more accurate future predictions
                'Avg_Reels_View': avg_views,  # Historical baseline
                'Category': campaign_data['category'],
                'Interests': campaign_data['interests'],
                'Turkey': int(campaign_data['turkey']),
                'Comment_Rate': float(campaign_data['commentRate']),
                'Score': int(campaign_data['score']),
                # New ML features for better predictions
                'Sponsored_Reels_View': sponsored_views,
                'Campaign_Reach': campaign_reach,
                'Performance_Ratio': round(performance_ratio, 3),
                'Cost_Per_Reach': round(cost_per_reach, 4),
                'Link_Clicks': link_clicks,
                'Cost_Per_Click': round(cost_per_click, 2) if cost_per_click > 0 else 0,
                'Actual_Campaign_Cost': actual_cost
            }
            
            # Calculate advanced ROI using post-campaign metrics
            roi = self.calculate_advanced_roi(new_row)
            
            # Add to dataframe
            if self.raw_data is None:
                self.raw_data = pd.DataFrame([new_row])
            else:
                self.raw_data = pd.concat([self.raw_data, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save to CSV
            self.raw_data.to_csv(CSV_FILE, index=False)
            logger.info(f"Added new campaign data for {campaign_data['account']} with ROI {roi}")
            
            return {"success": True, "roi": roi, "message": "Campaign data added successfully"}
            
        except Exception as e:
            logger.error(f"Error adding campaign data: {e}")
            return {"success": False, "error": str(e)}
    
    def get_statistics(self):
        """Get summary statistics"""
        if self.raw_data is None:
            return {}
        
        data_with_roi = self.raw_data.copy()
        data_with_roi['ROI_star'] = data_with_roi.apply(self.calculate_roi, axis=1)
        
        return {
            'total_influencers': len(data_with_roi),
            'avg_roi': round(data_with_roi['ROI_star'].mean(), 1),
            'max_roi': round(data_with_roi['ROI_star'].max(), 0),
            'avg_engagement': round(data_with_roi['ER'].mean(), 1),
            'avg_followers': int(data_with_roi['Followers'].mean()),
            'total_reach': int(data_with_roi['Followers'].sum())
        }

# Initialize data manager
data_manager = InfluencerDataManager()

@app.route('/')
def index():
    """Serve the dashboard HTML"""
    return send_from_directory('.', 'dashboard.html')

@app.route('/greymatter_materials/<path:filename>')
def serve_materials(filename):
    """Serve greymatter materials (fonts, logos)"""
    return send_from_directory('greymatter_materials', filename)

@app.route('/api/influencers')
def get_influencers():
    """Get all influencer data"""
    try:
        logger.info("Getting influencer data for dashboard...")
        limit = request.args.get('limit', type=int)
        category_filter = request.args.get('category', '')
        tier_filter = request.args.get('tier', '')
        data = data_manager.get_dashboard_data(limit=limit, category_filter=category_filter, tier_filter=tier_filter)
        
        if isinstance(data, dict) and 'error' in data:
            return jsonify(data), 500
            
        logger.info(f"Returning {len(data)} influencer records")
        if len(data) > 0:
            logger.info(f"Sample record: {data[0]['account']}")
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error in get_influencers: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/statistics')
def get_statistics():
    """Get dashboard statistics"""
    try:
        stats = data_manager.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error in get_statistics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/top-influencers/<int:limit>')
def get_top_influencers(limit):
    """Get top N influencers by ROI"""
    try:
        data = data_manager.get_dashboard_data()
        if isinstance(data, dict) and 'error' in data:
            return jsonify(data), 500
        
        top_data = data[:limit]
        return jsonify(top_data)
    except Exception as e:
        logger.error(f"Error in get_top_influencers: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/add-campaign', methods=['POST'])
def add_campaign():
    """Add new campaign data"""
    try:
        campaign_data = request.get_json()
        
        if not campaign_data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        result = data_manager.add_campaign_data(campaign_data)
        
        if result['success']:
            # Reload data to get updated dataset
            data_manager.load_data()
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error in add_campaign: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/roi-distribution')
def get_roi_distribution():
    """Get ROI distribution for charts"""
    try:
        data = data_manager.get_dashboard_data()
        if isinstance(data, dict) and 'error' in data:
            return jsonify(data), 500
        
        # Create ROI buckets
        buckets = {
            '0-500': 0,
            '500-1000': 0,
            '1000-2000': 0,
            '2000-5000': 0,
            '5000+': 0
        }
        
        for item in data:
            roi = item['roi']
            if roi < 500:
                buckets['0-500'] += 1
            elif roi < 1000:
                buckets['500-1000'] += 1
            elif roi < 2000:
                buckets['1000-2000'] += 1
            elif roi < 5000:
                buckets['2000-5000'] += 1
            else:
                buckets['5000+'] += 1
        
        return jsonify(buckets)
        
    except Exception as e:
        logger.error(f"Error in get_roi_distribution: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/backup-data')
def backup_data():
    """Create a backup of current data"""
    try:
        if data_manager.raw_data is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"backup_influencers_{timestamp}.csv"
            data_manager.raw_data.to_csv(backup_file, index=False)
            return jsonify({"success": True, "backup_file": backup_file})
        else:
            return jsonify({"success": False, "error": "No data to backup"}), 400
            
    except Exception as e:
        logger.error(f"Error in backup_data: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/influencer-analysis/<account>')
def get_influencer_analysis(account):
    """Get comprehensive influencer analysis with decision tree breakdown"""
    try:
        logger.info(f"Getting detailed analysis for account: {account}")
        if data_manager.raw_data is None:
            return jsonify({"error": "No data available"}), 500
        
        # Find the influencer
        influencer = data_manager.raw_data[data_manager.raw_data['Account'] == account]
        if influencer.empty:
            logger.error(f"Influencer not found: {account}")
            return jsonify({"error": "Influencer not found"}), 404
        
        row = influencer.iloc[0]
        
        # Extract all influencer data
        influencer_data = {
            'account': str(row['Account']),
            'followers': int(row['Followers']),
            'quality_audience': int(row['Quality_Audience']),
            'engagement_rate': float(row['ER']),
            'post_price': int(row['Est_Post_Price']),
            'avg_reels_view': int(row['Avg_Reels_View']),
            'turkey_market': int(row['Turkey']),
            'comment_rate': float(row['Comment_Rate']),
            'score': int(row['Score']),
            'category': str(row['Category']) if pd.notna(row['Category']) else '',
            'interests': str(row['Interests']) if pd.notna(row['Interests']) else ''
        }
        
        # Calculate ROI using the same method as the main table (ML-guided)
        final_roi = data_manager.calculate_roi(row)
        
        # For detailed breakdown, show ML-guided calculation steps
        engagement_score = float(row['ER']) * 0.19  # 19% weight
        reach_score = float(row['Avg_Reels_View']) / 1000000 * 0.171  # 17.1% weight
        audience_size_score = float(row['Followers']) / 1000000 * 0.157  # 15.7% weight
        quality_score = float(row['Quality_Audience']) / 100000 * 0.12  # 12% weight
        market_score = float(row['Turkey']) / 100 * 0.10  # 10% weight
        influence_score = float(row['Score']) / 100 * 0.08  # 8% weight
        comment_engagement = float(row['Comment_Rate']) * 0.06  # 6% weight
        cost_efficiency = 100000 / max(float(row['Est_Post_Price']), 1) * 0.05  # 5% weight
        
        # Calculate dataset averages for comparison (numeric columns only)
        numeric_columns = data_manager.raw_data.select_dtypes(include=[np.number]).columns
        avg_data = data_manager.raw_data[numeric_columns].mean()
        avg_roi = data_manager.raw_data.apply(data_manager.calculate_roi, axis=1).mean()
        
        # ML Feature importance analysis
        ml_components = {
            'engagement_score': engagement_score * 1000,
            'reach_score': reach_score * 1000,
            'audience_size_score': audience_size_score * 1000,
            'quality_score': quality_score * 1000,
            'market_score': market_score * 1000,
            'influence_score': influence_score * 1000,
            'comment_engagement': comment_engagement * 1000,
            'cost_efficiency': cost_efficiency * 1000
        }
        
        # ROI Decision Tree Analysis
        decision_tree = {
            'final_roi': round(final_roi, 0),
            'vs_average_roi': round(((final_roi / avg_roi) - 1) * 100, 1),
            'roi_category': 'Excellent' if final_roi > avg_roi * 2 else 'Good' if final_roi > avg_roi * 1.5 else 'Average' if final_roi > avg_roi * 0.8 else 'Below Average',
            'ml_components': ml_components
        }
        
        
        # Add comparison data for highlighting below-average contributors
        dataset_averages = {
            'engagement_rate': float(avg_data['ER']),
            'avg_reels_view': float(avg_data['Avg_Reels_View']),
            'followers': float(avg_data['Followers']),
            'quality_audience': float(avg_data['Quality_Audience']),
            'turkey_market': float(avg_data['Turkey']),
            'score': float(avg_data['Score']),
            'comment_rate': float(avg_data['Comment_Rate']),
            'post_price': float(avg_data['Est_Post_Price'])
        }
        
        return jsonify({
            'success': True,
            'influencer_data': influencer_data,
            'decision_tree': decision_tree,
            'ml_components': ml_components,
            'dataset_averages': dataset_averages,
            'roi_formula': "ML ROI★ = (ES×0.19 + RS×0.171 + AS×0.157 + QS×0.12 + MS×0.10 + IS×0.08 + CE×0.06 + CF×0.05) × 1000"
        })
        
    except Exception as e:
        logger.error(f"Error in get_influencer_analysis: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/debug/accounts')
def get_debug_accounts():
    """Debug endpoint to see available accounts"""
    try:
        if data_manager.raw_data is None:
            return jsonify({"error": "No data available"})
        
        accounts = data_manager.raw_data['Account'].head(10).tolist()
        return jsonify({"sample_accounts": accounts})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/ml-info')
def get_ml_info():
    """Get machine learning model information and performance"""
    try:
        if data_manager.ml_info is not None:
            ml_status = {
                'ml_enabled': True,
                'model_type': data_manager.ml_info['model_type'],
                'validation_accuracy': round(data_manager.ml_info['val_r2'] * 100, 1),
                'training_accuracy': round(data_manager.ml_info['train_r2'] * 100, 1),
                'cross_validation_mean': round(data_manager.ml_info['cv_mean'] * 100, 1),
                'cross_validation_std': round(data_manager.ml_info['cv_std'] * 100, 1),
                'statistical_significance_p': data_manager.ml_info['statistical_significance_p'],
                'is_statistically_significant': bool(data_manager.ml_info['is_statistically_significant']),
                'feature_count': data_manager.ml_info['feature_count'],
                'training_date': data_manager.ml_info['training_date']
            }
        else:
            ml_status = {
                'ml_enabled': False,
                'fallback_method': 'Mathematical Formula',
                'message': 'Machine learning model not available. Using mathematical ROI calculation.'
            }
        
        return jsonify(ml_status)
    except Exception as e:
        logger.error(f"Error in get_ml_info: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/categories')
def get_categories():
    """Get list of available categories for filtering"""
    try:
        if data_manager.raw_data is None:
            return jsonify({"error": "No data available"}), 500
        
        # Extract individual category keywords from the Category column
        categories_set = set()
        
        for category_string in data_manager.raw_data['Category'].dropna():
            # Split by comma and clean up each category
            individual_categories = [cat.strip() for cat in str(category_string).split(',')]
            categories_set.update(individual_categories)
        
        # Sort categories and get most common ones
        categories_list = sorted(list(categories_set))
        
        # Count occurrences to get the most popular categories
        category_counts = {}
        for category_string in data_manager.raw_data['Category'].dropna():
            for cat in categories_list:
                if cat.lower() in str(category_string).lower():
                    category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Sort by popularity and take top categories
        popular_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return jsonify({
            'categories': [cat for cat, count in popular_categories],
            'category_counts': dict(popular_categories),
            'total_categories': len(categories_list)
        })
        
    except Exception as e:
        logger.error(f"Error in get_categories: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/tiers')
def get_tiers():
    """Get available follower tiers with counts"""
    try:
        if data_manager.raw_data is None:
            return jsonify({"error": "No data available"}), 500
        
        # Calculate tiers for all influencers
        tiers = ['Nano', 'Micro', 'Mid', 'Macro', 'Star', 'Superstar']
        tier_counts = {}
        
        for tier in tiers:
            tier_counts[tier] = 0
        
        for _, row in data_manager.raw_data.iterrows():
            tier = data_manager.get_follower_tier(row['Followers'])
            tier_counts[tier] += 1
        
        # Create tier definitions
        tier_definitions = {
            'Nano': '<10K followers',
            'Micro': '10K-100K followers',
            'Mid': '100K-500K followers', 
            'Macro': '500K-2M followers',
            'Star': '2M-10M followers',
            'Superstar': '10M+ followers'
        }
        
        return jsonify({
            'tiers': tiers,
            'tier_counts': tier_counts,
            'tier_definitions': tier_definitions,
            'total_influencers': len(data_manager.raw_data)
        })
        
    except Exception as e:
        logger.error(f"Error in get_tiers: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/model-feature-importance')
def get_model_feature_importance():
    """Get LightGBM model feature importance for overall model analysis"""
    try:
        if data_manager.ml_model is None or data_manager.ml_scaler is None:
            return jsonify({
                "error": "ML model not available", 
                "fallback_method": "Mathematical Formula"
            }), 404
        
        # Get feature importance from LightGBM model
        feature_columns = data_manager.ml_scaler['feature_columns']
        feature_importance = data_manager.ml_model.feature_importances_
        
        # Create feature importance analysis
        importance_data = []
        for i, feature in enumerate(feature_columns):
            importance_data.append({
                'feature': feature,
                'importance': float(feature_importance[i]),
                'importance_normalized': float(feature_importance[i] / feature_importance.sum() * 100)
            })
        
        # Sort by importance
        importance_data.sort(key=lambda x: x['importance'], reverse=True)
        
        # Calculate sample dataset statistics for context
        sample_stats = {}
        if data_manager.raw_data is not None:
            for feature in feature_columns:
                if feature in data_manager.raw_data.columns:
                    sample_stats[feature] = {
                        'mean': float(data_manager.raw_data[feature].mean()),
                        'std': float(data_manager.raw_data[feature].std()),
                        'min': float(data_manager.raw_data[feature].min()),
                        'max': float(data_manager.raw_data[feature].max())
                    }
        
        # Generate model performance summary
        model_summary = {
            'total_features': len(feature_columns),
            'top_3_features': [item['feature'] for item in importance_data[:3]],
            'model_type': 'LightGBM Regressor',
            'feature_importance_sum': float(feature_importance.sum()),
            'validation_accuracy': data_manager.ml_info['val_r2'] if data_manager.ml_info else 0.0
        }
        
        return jsonify({
            'success': True,
            'feature_importance': importance_data,
            'sample_statistics': sample_stats,
            'model_summary': model_summary,
            'explanation': 'Feature importance shows how much each feature contributes to the LightGBM model predictions across all influencers.'
        })
        
    except Exception as e:
        logger.error(f"Error in get_model_feature_importance: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    import os
    
    # Get port from environment variable (for web deployment) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    print("Starting GREYMATTER Dashboard Backend...")
    print(f"Dashboard available at: http://localhost:{port}")
    print("API endpoints:")
    print("  GET /api/influencers - Get all influencer data")
    print("  GET /api/statistics - Get summary statistics")
    print("  GET /api/top-influencers/<limit> - Get top N influencers")
    print("  POST /api/add-campaign - Add new campaign data")
    print("  GET /api/roi-distribution - Get ROI distribution")
    print("  GET /api/backup-data - Create data backup")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)