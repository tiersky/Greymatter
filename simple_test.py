#!/usr/bin/env python3
"""
Simple test to debug the dashboard issues
"""

import pandas as pd
import json

def test_data_loading():
    print("Testing data loading...")
    
    # Load CSV
    df = pd.read_csv("influencers.csv")
    print(f"✅ Loaded {len(df)} records")
    print(f"✅ Columns: {list(df.columns)}")
    print(f"✅ Sample accounts: {df['Account'].head(5).tolist()}")
    
    # Test ROI calculation
    def calculate_roi(row):
        Q = float(row['Quality_Audience'])
        E = float(row['ER']) 
        M = float(row['Turkey']) / 100
        CM = float(row['Comment_Rate'])
        P = float(row['Est_Post_Price'])
        IM = 1.0
        
        roi = (Q * E * M * IM * (1 + 0.2 * CM)) / P
        return round(roi, 2)
    
    # Test with first row
    first_row = df.iloc[0]
    roi = calculate_roi(first_row)
    print(f"✅ Sample ROI calculation: {first_row['Account']} = {roi}")
    
    # Calculate all ROIs and sort
    df['ROI_star'] = df.apply(calculate_roi, axis=1)
    df_sorted = df.sort_values('ROI_star', ascending=False)
    
    print("\n🏆 Top 10 influencers:")
    for i, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['Account']:20} ROI: {row['ROI_star']:8.0f}")
    
    # Test JSON serialization
    top_5 = []
    for _, row in df_sorted.head(5).iterrows():
        top_5.append({
            'account': str(row['Account']),
            'roi': round(float(row['ROI_star']), 0),
            'followers': int(row['Followers']),
            'engagement': round(float(row['ER']), 2),
            'postPrice': int(row['Est_Post_Price']),
            'turkey': int(row['Turkey'])
        })
    
    print(f"\n✅ JSON serialization test:")
    print(json.dumps(top_5[0], indent=2))

if __name__ == "__main__":
    test_data_loading()