import os
from pathlib import Path
from dotenv import dotenv_values, load_dotenv

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Explicitly load local .env overriding any conflicting environment variables
ENV_PATH = PROJECT_ROOT / ".env"
env_vars = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}

# Priority: explicitly specified in project .env, then os.environ
GEMINI_API_KEY = env_vars.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = env_vars.get("GEMINI_MODEL", "gemini-flash-latest")

# Storage Paths
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "fact_layer.sqlite3"
PAGES_DIR = DATA_DIR / "pages"
UPLOADS_DIR = DATA_DIR / "uploads"
PDFS_DIR = DATA_DIR / "pdfs"           # raw PDFs served for citation viewer
STARTER_DATASETS_DIR = PROJECT_ROOT / "starter-datasets"

# Create required directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
PAGES_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PDFS_DIR.mkdir(parents=True, exist_ok=True)

# Application Settings
API_HOST = os.environ.get("FACT_LAYER_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("FACT_LAYER_PORT", "8000"))
NEXT_PORT = int(os.environ.get("NEXT_PORT", "3000"))

# Reconciliation threshold
RECONCILIATION_SIMILARITY_THRESHOLD = 0.72
VALUE_TOLERANCE_PERCENT = 0.01  # 1% tolerance for floating point rounding/representation
