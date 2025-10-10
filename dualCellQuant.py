#!/usr/bin/env python3
"""
DualCellQuant - Backward compatibility wrapper.

This file maintains backward compatibility with the original dualCellQuant.py.
All functionality has been moved to the dualcellquant package.

To use:
    python dualCellQuant.py  # Launches Gradio UI
    
Or import as module:
    from dualCellQuant import *
"""

# Re-export all functions from the package for backward compatibility
from dualcellquant import *

# Import build_ui from the refactored location
from dualcellquant.ui import build_ui

# Main entry point
if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch()
