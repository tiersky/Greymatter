#!/usr/bin/env python3
"""
Debug the influencer analysis function
"""

import pandas as pd
import numpy as np

def test_analysis():
    # Load data
    df = pd.read_csv('influencers.csv')
    
    # Find simayrie
    influencer = df[df['Account'] == 'simayrie']
    if influencer.empty:
        print("Influencer not found")
        return
    
    row = influencer.iloc[0]
    print(f"Testing analysis for: {row['Account']}")
    
    try:
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
        print("✅ Influencer data extraction successful")
        
        # Calculate ROI components step by step
        Q = float(row['Quality_Audience'])
        E = float(row['ER']) 
        M = float(row['Turkey']) / 100
        CM = float(row['Comment_Rate'])
        P = float(row['Est_Post_Price'])
        IM = 1.0  # Influence multiplier
        
        print(f"Q={Q}, E={E}, M={M}, CM={CM}, P={P}, IM={IM}")
        
        # ROI Calculation: (Q × E × M × IM × (1 + 0.2×CM)) / P
        comment_boost = 1 + (0.2 * CM)
        numerator = Q * E * M * IM * comment_boost
        final_roi = numerator / P
        print(f"Final ROI: {final_roi}")
        
        # Calculate dataset averages for comparison
        avg_data = df.mean()
        def calculate_roi(r):
            try:
                q = float(r['Quality_Audience'])
                e = float(r['ER']) 
                m = float(r['Turkey']) / 100
                cm = float(r['Comment_Rate'])
                p = float(r['Est_Post_Price'])
                im = 1.0
                roi = (q * e * m * im * (1 + 0.2 * cm)) / p
                return roi
            except:
                return 0
        
        avg_roi = df.apply(calculate_roi, axis=1).mean()
        print(f"Average ROI: {avg_roi}")
        
        # Test string formatting that might be causing issues
        print("Testing string formatting...")
        
        # Quality Audience impact
        qa_contribution = (Q - avg_data['Quality_Audience']) / avg_data['Quality_Audience'] * 100 if avg_data['Quality_Audience'] > 0 else 0
        print(f"QA contribution: {qa_contribution}")
        
        test_contribution = {
            'factor': 'Quality Audience',
            'value': int(Q),
            'contribution_pct': round(qa_contribution, 1),
            'is_positive': qa_contribution > 0,
            'impact_level': 'High' if abs(qa_contribution) > 50 else 'Medium' if abs(qa_contribution) > 20 else 'Low',
            'description': f"{'Above' if qa_contribution > 0 else 'Below'} average by {abs(qa_contribution):.1f}%"
        }
        print("✅ QA contribution successful")
        
        # Engagement Rate impact
        er_contribution = (E - avg_data['ER']) / avg_data['ER'] * 100 if avg_data['ER'] > 0 else 0
        test_er = {
            'factor': 'Engagement Rate',
            'value': f"{E:.2f}%",
            'contribution_pct': round(er_contribution, 1),
            'is_positive': er_contribution > 0,
            'impact_level': 'High' if abs(er_contribution) > 50 else 'Medium' if abs(er_contribution) > 20 else 'Low',
            'description': f"{'Above' if er_contribution > 0 else 'Below'} average by {abs(er_contribution):.1f}%"
        }
        print("✅ ER contribution successful")
        
        # Turkey Market impact
        tm_contribution = (row['Turkey'] - avg_data['Turkey']) / avg_data['Turkey'] * 100 if avg_data['Turkey'] > 0 else 0
        test_tm = {
            'factor': 'Turkey Market Share',
            'value': f"{int(row['Turkey'])}%",
            'contribution_pct': round(tm_contribution, 1),
            'is_positive': tm_contribution > 0,
            'impact_level': 'High' if abs(tm_contribution) > 20 else 'Medium' if abs(tm_contribution) > 10 else 'Low',
            'description': f"{'Above' if tm_contribution > 0 else 'Below'} average by {abs(tm_contribution):.1f}%"
        }
        print("✅ TM contribution successful")
        
        # Comment Rate impact
        cr_contribution = (CM - avg_data['Comment_Rate']) / avg_data['Comment_Rate'] * 100 if avg_data['Comment_Rate'] > 0 else 0
        test_cr = {
            'factor': 'Comment Rate',
            'value': f"{CM:.3f}",
            'contribution_pct': round(cr_contribution, 1),
            'is_positive': cr_contribution > 0,
            'impact_level': 'High' if abs(cr_contribution) > 100 else 'Medium' if abs(cr_contribution) > 50 else 'Low',
            'description': f"{'Above' if cr_contribution > 0 else 'Below'} average by {abs(cr_contribution):.1f}%"
        }
        print("✅ CR contribution successful")
        
        # Post Price impact (inverted - lower price is better)
        pp_contribution = -(P - avg_data['Est_Post_Price']) / avg_data['Est_Post_Price'] * 100 if avg_data['Est_Post_Price'] > 0 else 0
        test_pp = {
            'factor': 'Post Price (Cost Efficiency)',
            'value': f"${int(P):,}",
            'contribution_pct': round(pp_contribution, 1),
            'is_positive': pp_contribution > 0,
            'impact_level': 'High' if abs(pp_contribution) > 50 else 'Medium' if abs(pp_contribution) > 20 else 'Low',
            'description': f"{'More cost-effective' if pp_contribution > 0 else 'More expensive'} than average by {abs(pp_contribution):.1f}%"
        }
        print("✅ PP contribution successful")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_analysis()