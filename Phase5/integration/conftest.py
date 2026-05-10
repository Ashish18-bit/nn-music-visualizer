import sys
import os

# Add all phases to path
for phase in ["phase1", "phase2", "phase3", "phase4"]:
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", phase
    ))