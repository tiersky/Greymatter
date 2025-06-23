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

