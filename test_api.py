#!/usr/bin/env python3
"""
Test API endpoints to debug connection issues
"""

import requests
import json

def test_api():
    base_url = "http://localhost:5000"
    
    print("Testing GREYMATTER Dashboard API...")
    print("=" * 50)
    
    # Test basic connection
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"✅ Dashboard HTML: {response.status_code}")
    except Exception as e:
        print(f"❌ Dashboard HTML failed: {e}")
        return
    
    # Test influencers API
    try:
        response = requests.get(f"{base_url}/api/influencers?limit=3", timeout=10)
        print(f"✅ Influencers API: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Records returned: {len(data)}")
            if len(data) > 0:
                print(f"   Sample account: {data[0].get('account', 'N/A')}")
                print(f"   Sample ROI: {data[0].get('roi', 'N/A')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Influencers API failed: {e}")
    
    # Test statistics API
    try:
        response = requests.get(f"{base_url}/api/statistics", timeout=5)
        print(f"✅ Statistics API: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Total influencers: {data.get('total_influencers', 'N/A')}")
    except Exception as e:
        print(f"❌ Statistics API failed: {e}")
    
    # Test SHAP API
    try:
        response = requests.get(f"{base_url}/api/shap-analysis/yasemoz88", timeout=5)
        print(f"✅ SHAP API: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Account: {data.get('account', 'N/A')}")
            print(f"   Actual ROI: {data.get('actual_roi', 'N/A')}")
    except Exception as e:
        print(f"❌ SHAP API failed: {e}")

if __name__ == "__main__":
    test_api()