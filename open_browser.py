#!/usr/bin/env python3
"""
Open GREYMATTER Dashboard in browser
"""

import webbrowser
import time
import requests

def wait_for_server():
    """Wait for server to be ready"""
    print("🔍 Waiting for dashboard server...")
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://localhost:5000", timeout=2)
            if response.status_code == 200:
                print("✅ Dashboard server is ready!")
                return True
        except:
            pass
        time.sleep(1)
        print(f"   Waiting... ({i+1}/30)")
    
    print("❌ Dashboard server not responding")
    return False

def main():
    print("🚀 Opening GREYMATTER Dashboard...")
    
    if wait_for_server():
        print("🌐 Opening browser at http://localhost:5000")
        webbrowser.open("http://localhost:5000")
        print("📝 Check browser console (F12) for any API errors")
        print("🔧 If you see dummy data, the frontend may not be connecting to the API")
    else:
        print("❌ Cannot open dashboard - server not running")
        print("💡 Start the server first with: python3 dashboard_backend.py")

if __name__ == "__main__":
    main()