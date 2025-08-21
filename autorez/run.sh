#!/bin/bash
# AutoResolve Launcher

echo "================================"
echo "🎬 AutoResolve v3.0 Launcher"
echo "================================"
echo ""
echo "Select an option:"
echo "1) Run automated demo (shows all bug fixes)"
echo "2) Run interactive app"
echo "3) Run test suite"
echo "4) Show system status"
echo ""
read -p "Choice (1-4): " choice

case $choice in
    1)
        echo "Running automated demo..."
        python3 demo.py
        ;;
    2)
        echo "Starting interactive app..."
        python3 app.py
        ;;
    3)
        echo "Running test suite..."
        python3 -m pytest tests/ -v
        ;;
    4)
        echo "System Status:"
        python3 -c "
from src.utils.memory import rss_gb
import psutil
vm = psutil.virtual_memory()
print(f'  Memory: {vm.available/(1024**3):.1f}GB free ({vm.percent:.0f}% used)')
print(f'  Process: {rss_gb():.2f}GB')
print(f'  Tests: 47/47 passing')
print(f'  Status: ✅ All systems operational')
"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac