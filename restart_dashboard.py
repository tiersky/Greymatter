#!/usr/bin/env python3
"""
Restart GREYMATTER Dashboard
Quick script to restart the dashboard server
"""

import subprocess
import sys
import time
import signal
import os

def main():
    print("🔄 Restarting GREYMATTER Dashboard...")
    
    # Find any existing dashboard processes and kill them
    try:
        result = subprocess.run(['pgrep', '-f', 'dashboard_backend.py'], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"🛑 Stopping existing process {pid}")
                    os.kill(int(pid), signal.SIGTERM)
                    time.sleep(1)
    except:
        pass
    
    print("🚀 Starting dashboard...")
    print("📱 Dashboard will be available at: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "dashboard_backend.py"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped!")

if __name__ == "__main__":
    main()