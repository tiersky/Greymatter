#!/usr/bin/env python3
"""
GREYMATTER Dashboard Launcher
Simple script to start the dashboard with proper dependencies
"""

import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['flask', 'flask-cors', 'pandas', 'numpy']
    optional_packages = ['pyarrow']  # For parquet support
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)
    
    if missing_required:
        print("❌ Missing required packages:")
        for package in missing_required:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print("⚠️  Optional packages missing (dashboard will still work):")
        for package in missing_optional:
            print(f"   - {package} (for parquet file support)")
        print(f"\n📦 Install optional packages with:")
        print(f"   pip install {' '.join(missing_optional)}")
    
    return True

def check_data_files():
    """Check if data files exist"""
    current_dir = Path.cwd()
    csv_file = current_dir / "influencers.csv"
    
    if not csv_file.exists():
        print("❌ influencers.csv not found in current directory")
        print("📁 Please ensure the CSV file is in the same directory as this script")
        return False
    
    print(f"✅ Found data file: {csv_file}")
    return True

def main():
    print("🚀 GREYMATTER Dashboard Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("dashboard.html").exists():
        print("❌ dashboard.html not found")
        print("📁 Please run this script from the GREYMATTER project directory")
        sys.exit(1)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check data files
    print("📊 Checking data files...")
    if not check_data_files():
        sys.exit(1)
    
    print("✅ All checks passed!")
    print("\n🌐 Starting dashboard backend...")
    print("📱 Dashboard will be available at: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Start the Flask backend
        subprocess.run([sys.executable, "dashboard_backend.py"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()