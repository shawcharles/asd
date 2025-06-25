import sys
import pathlib

# Ensure the project 'src' directory is on the Python path so that top-level
# packages such as `trimmed_match` and `adaptive_supergeos` can be imported
# when running the tests via `pytest` without requiring an editable install.
SRC_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
