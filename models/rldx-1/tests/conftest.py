# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
import os
import sys

# Make the model workspace importable (mirrors the flat imports the
# container uses: the model folder is the working directory).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
