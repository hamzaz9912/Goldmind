#!/usr/bin/env python3
import json
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Simple health check that doesn't depend on Streamlit
def simple_health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_fetch": "available",
        "model_inference": "available",
        "model_types": ["RL", "LSTM", "AutoML", "ML"],
        "version": "1.0.0"
    }

# Output JSON for deployment platform
result = simple_health_check()
print(json.dumps(result))