# GREYMATTER Dashboard

A sleek, interactive web dashboard for influencer ROI analytics and campaign management.

## 🚀 Quick Start

1. **Install Dependencies** (if not already installed):
   ```bash
   pip install flask flask-cors pandas numpy
   ```

2. **Launch Dashboard**:
   ```bash
   python run_dashboard.py
   ```

3. **Open Browser**:
   - Navigate to: http://localhost:5000
   - Dashboard loads automatically

## 📊 Features

### Dashboard Tab
- **Real-time Statistics**: Total influencers, average ROI, top performer metrics
- **ROI Distribution Chart**: Visual breakdown of ROI score ranges
- **Top 10 Influencers**: Bar chart of highest performing accounts
- **Engagement Heatmap**: Interactive scatter plot showing followers vs engagement vs ROI

### Influencers Tab
- **Sortable Table**: Click column headers to sort by any metric
- **Search Functionality**: Filter influencers by account name
- **Real-time Data**: Reflects latest additions from input tab

### Input Tab
- **Campaign Data Entry**: Add new influencer campaign results
- **Automatic ROI Calculation**: Real-time ROI computation
- **CSV Integration**: Data automatically saved to influencers.csv
- **Machine Learning Enhancement**: New data improves algorithm precision

## 🎯 Key Metrics

- **ROI_star Score**: (Quality_Audience × ER × Turkey_Market × (1 + 0.2×Comment_Rate)) / Post_Price
- **Engagement Rate**: Average engagement percentage
- **Quality Audience**: Verified/real followers count
- **Turkey Market**: Turkish audience percentage

## 🔧 Technical Details

### Backend (dashboard_backend.py)
- **Flask API**: RESTful endpoints for data operations
- **Pandas Integration**: CSV/Parquet data processing
- **Real-time Updates**: Immediate data reflection
- **Error Handling**: Robust validation and logging

### Frontend (dashboard.html)
- **Chart.js**: Interactive charts and visualizations
- **Plotly**: Advanced heatmap functionality
- **Responsive Design**: Mobile-friendly interface
- **Real-time Updates**: Dynamic data loading

## 📁 File Structure

```
GREYMATTER/
├── dashboard.html              # Main dashboard interface
├── dashboard_backend.py        # Flask API backend
├── run_dashboard.py           # Easy launcher script
├── influencers.csv            # Raw influencer data
├── influencer_modelling_ready.parquet  # Processed data
├── greymatter_inf_analysis.ipynb       # ML analysis notebook
└── gerymatter_analysis.py              # Data processing script
```

## 🔌 API Endpoints

- `GET /api/influencers` - Get all influencer data
- `GET /api/statistics` - Get summary statistics
- `GET /api/top-influencers/<limit>` - Get top N influencers
- `POST /api/add-campaign` - Add new campaign data
- `GET /api/roi-distribution` - Get ROI distribution
- `GET /api/backup-data` - Create data backup

## 🎨 Design Features

- **Minimal & Sleek**: Clean, professional interface
- **Gradient Backgrounds**: Modern visual appeal
- **Responsive Tables**: Mobile-optimized data display
- **Interactive Charts**: Hover effects and animations
- **Color-coded Data**: Visual hierarchy and clarity

## 💡 Usage Tips

1. **Sorting**: Click any column header in the Influencers tab to sort
2. **Search**: Use the search box to filter by account name
3. **Data Entry**: Fill all fields in Input tab for accurate ROI calculation
4. **Backup**: API automatically creates timestamped backups
5. **Mobile**: Dashboard works on tablets and phones

## 🔄 Data Flow

1. **Input**: New campaign data via Input tab
2. **Processing**: Backend calculates ROI and validates data
3. **Storage**: Data saved to influencers.csv
4. **Analysis**: ML algorithm processes new data points
5. **Visualization**: Dashboard updates with latest insights

## 🛠 Troubleshooting

- **Port 5000 in use**: Change port in dashboard_backend.py
- **Missing data**: Ensure influencers.csv exists in project directory
- **Import errors**: Install missing packages via pip
- **Browser issues**: Try Chrome/Firefox, clear cache

## 🔮 Future Enhancements

- Real-time campaign tracking
- Predictive ROI modeling
- Instagram API integration
- Advanced filtering options
- Export functionality
- Team collaboration features