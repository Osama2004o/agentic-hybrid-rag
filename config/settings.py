import os
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

LLM_MODEL = os.getenv("LLM_MODEL", "openrouter/free")
GENERATION_TEMPERATURE = 0.5
MAX_OUTPUT_TOKENS = 8192
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL
EMBEDDING_DEVICE = "cuda"

CHROMA_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "chroma_db"
)
CHROMA_COLLECTION_NAME = "code_knowledge_base"

TOP_K_RESULTS = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

MAX_MEMORY_TURNS = 20
CODE_EXECUTION_TIMEOUT = 10

# ── Redis / Memurai Configuration ──
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Cache TTLs (seconds)
CACHE_TTL_RETRIEVAL = 60 * 60 * 24       # 24 hours for retrieval cache
CACHE_TTL_RESPONSE = 60 * 60 * 12        # 12 hours for generator response cache
CACHE_TTL_MEMORY = 60 * 60 * 24 * 7      # 7 days for generator session memory

# Semantic Cache Configuration (Memurai vectorset)
SEMANTIC_CACHE_ENABLED = True
SEMANTIC_CACHE_THRESHOLD = 0.80
SEMANTIC_CACHE_KEY = "semantic_cache:prompts"
SEMANTIC_ANSWER_PREFIX = "semantic_cache:answer:"

# Generator defaults
MAX_REFINEMENT_LOOPS = 4
