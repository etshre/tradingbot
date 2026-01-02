import sys
import os

# This shows the path to the Python you are currently using
print(f"Current Python: {sys.executable}")

# This lists all folders where Python looks for modules
print(f"Search Paths: {sys.path}")

try:
    import yfinance as yf
    print("Success: yfinance is found!")
except ImportError:
    print("Error: yfinance is still not found in the paths above.")
