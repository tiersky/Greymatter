#!/usr/bin/env python3
"""
Open GREYMATTER Dashboard in Browser
"""

import webbrowser
import time
import requests

def check_server():
    """Check if dashboard server is running"""
    try:
        response = requests.get("http://localhost:5000", timeout=2)
        return response.status_code == 200
    except:
        return False

def main():
    print("🔍 Checking if dashboard is running...")
    
    if check_server():
        print("✅ Dashboard is running!")
        print("🌐 Opening in browser...")
        webbrowser.open("http://localhost:5000")
    else:
        print("❌ Dashboard is not running")
        print("🚀 Please start the dashboard first with:")
        print("   python run_dashboard.py")

if __name__ == "__main__":
    main()