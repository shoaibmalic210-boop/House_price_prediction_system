"""
House Price Prediction - Evaluation Entry Point
Run this script from the project root to evaluate the model.
"""

import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from evaluate import evaluate

if __name__ == "__main__":
    evaluate()
