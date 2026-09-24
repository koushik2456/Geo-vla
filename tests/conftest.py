"""Force a hermetic environment: synthetic data, no LLM, no checkpoints."""
import os
import sys
import tempfile

os.environ["GEO_VLA_OFFLINE"] = "1"
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["MODEL_CHECKPOINT_DIR"] = tempfile.mkdtemp(prefix="geo-vla-ckpt-")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
