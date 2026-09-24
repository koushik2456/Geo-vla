"""Force a hermetic environment: synthetic data, no LLM, no checkpoints, temp storage."""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="geo-vla-test-")
os.environ["GEO_VLA_OFFLINE"] = "1"
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["LLM_PROVIDER"] = "none"
os.environ["MODEL_CHECKPOINT_DIR"] = os.path.join(_tmp, "checkpoints")
os.environ["MODEL_REGISTRY_DIR"] = os.path.join(_tmp, "registry")
os.environ["GEO_VLA_DATA_DIR"] = os.path.join(_tmp, "data")
os.environ["MONITOR_INTERVAL_SEC"] = "0"
os.environ["ADMIN_PASSWORD"] = ""

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
