# GREYMATTER Influencer ROI Dashboard

![GREYMATTER Logo](./greymatter_materials/greymatter_png.png)

## 🚀 Overview

An advanced AI-powered influencer analytics dashboard that uses machine learning to predict ROI and analyze influencer performance. Built with LightGBM ML models and featuring real-time analytics, tier-based filtering, and comprehensive campaign management.

## ✨ Features

### 🤖 Machine Learning & AI
- **LightGBM Model**: 95.8% validation accuracy for ROI prediction
- **Feature Importance Analysis**: ML-guided ROI calculation with weighted factors
- **Real-time ML Insights**: Dynamic model performance indicators
- **SHAP Analysis**: Model explainability and feature contributions

### 📊 Analytics Dashboard
- **Interactive Charts**: Top influencers, engagement heatmaps, ROI distribution
- **Tier-based Filtering**: Nano, Micro, Mid, Macro, Star, Superstar categories
- **Dynamic Performance Highlighting**: Red/green indicators for above/below average metrics
- **Real-time Statistics**: Live KPI tiles with performance metrics

### 🎯 Campaign Management
- **Campaign Input System**: Pre/post-campaign data collection
- **ROI Tracking**: ML-enhanced return on investment calculations
- **Performance Metrics**: Engagement rates, reach analysis, cost efficiency
- **Data Export**: CSV backup and data management

### 🎨 User Experience
- **Responsive Design**: Mobile and desktop optimized
- **Modern UI**: Clean, professional interface with GREYMATTER branding
- **Tab-based Navigation**: Dashboard, Influencers, Campaign Input
- **Advanced Filtering**: Multi-criteria search and filtering options

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Backend**: Python Flask with CORS support
- **ML Framework**: LightGBM, Scikit-learn, SHAP
- **Data Processing**: Pandas, NumPy
- **Visualization**: Chart.js, Plotly.js
- **Data Storage**: CSV, Parquet files

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/GREYMATTER.git
cd GREYMATTER
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install flask flask-cors pandas numpy lightgbm scikit-learn joblib
```

4. **Run the dashboard**
```bash
python dashboard_backend.py
```

5. **Open in browser**
Navigate to `http://localhost:5000`

## 📁 Project Structure

```
GREYMATTER/
├── dashboard.html              # Main frontend dashboard
├── dashboard_backend.py        # Flask API backend
├── train_ml_model.py          # ML model training
├── retrain_ml_model.py        # Enhanced model retraining
├── influencers.csv            # Main dataset
├── roi_lightgbm_model.pkl     # Trained ML model
├── roi_model_scaler.pkl       # Model preprocessing
├── roi_model_info.pkl         # Model metadata
├── greymatter_materials/      # Branding assets
└── README.md                  # Project documentation
```

## 🤖 Machine Learning Details

### Model Performance
- **Algorithm**: LightGBM Regressor
- **Validation R²**: 95.8%
- **Statistical Significance**: p < 0.05
- **Features**: 8 core influencer metrics

### Feature Importance (ML-Guided ROI)
1. **Engagement Rate (19.0%)** - Highest predictive power
2. **Average Reels Views (17.1%)** - Content reach indicator
3. **Follower Count (15.7%)** - Audience size factor
4. **Quality Audience (12.0%)** - Audience quality score
5. **Turkey Market (10.0%)** - Geographic targeting
6. **Score Rating (8.0%)** - Overall influence score
7. **Comment Rate (6.0%)** - Engagement depth
8. **Cost Efficiency (5.0%)** - Price optimization

### ROI Formula
```
ROI★ = (ES×0.19 + RS×0.171 + AS×0.157 + QS×0.12 + MS×0.10 + IS×0.08 + CE×0.06 + CF×0.05) × 1000
```

## 🎯 Influencer Tiers

| Tier | Followers | Description |
|------|-----------|-------------|
| Nano | <10K | Micro-influencers with niche audiences |
| Micro | 10K-100K | Small but engaged communities |
| Mid | 100K-500K | Growing influence and reach |
| Macro | 500K-2M | Significant market presence |
| Star | 2M-10M | Major influencers with broad appeal |
| Superstar | 10M+ | Celebrity-level influence |

## 🌐 Web Deployment

### Option 1: Heroku
1. Create `requirements.txt`
2. Add `Procfile`
3. Deploy to Heroku

### Option 2: Vercel/Netlify
1. Static frontend deployment
2. Separate API deployment

### Option 3: DigitalOcean/AWS
1. Full-stack deployment
2. Database integration

## 📊 API Endpoints

- `GET /` - Main dashboard
- `GET /api/influencers` - Get influencer data
- `GET /api/statistics` - Dashboard statistics
- `GET /api/top-influencers/<limit>` - Top performers
- `POST /api/add-campaign` - Add campaign data
- `GET /api/influencer-analysis/<account>` - Detailed analysis
- `GET /api/model-feature-importance` - ML model insights

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**GREYMATTER** - Advanced Influencer Analytics
- Website: [Your Website]
- Email: [Your Email]
- LinkedIn: [Your LinkedIn]

---

*Built with ❤️ by the GREYMATTER team*