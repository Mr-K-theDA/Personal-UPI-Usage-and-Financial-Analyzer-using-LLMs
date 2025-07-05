# config.py
import os
import psutil
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # System optimization for your PC
    MAX_MEMORY_USAGE = int(os.getenv('MAX_MEMORY_USAGE', 4096))  # 4GB
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 10))
    
    # Check available memory (fixed division)
    AVAILABLE_MEMORY = psutil.virtual_memory().available / (1024**3)  # GB as float
    
    # PDF Processing - optimized for your system
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB (reduced for your RAM)
    ALLOWED_EXTENSIONS = ['.pdf']
    PROCESSING_CHUNK_SIZE = 1000  # Process in small chunks
    
    # Categories
    EXPENSE_CATEGORIES = [
        'Food & Dining', 'Transportation', 'Shopping', 'Entertainment',
        'Bills & Utilities', 'Healthcare', 'Education', 'Travel',
        'Groceries', 'Fuel', 'Investment', 'Transfer', 'Others'
    ]
    
    # LLM Settings - optimized for free tiers
    MAX_TOKENS = 500  # Reduced for efficiency
    TEMPERATURE = 0.7
    
    # Use lightweight model
    DEFAULT_MODEL = "gemini-1.5-flash-latest"  # Free and efficient
    OPENAI_MODEL = "gpt-3.5-turbo"
    HUGGINGFACE_MODEL = "mistralai/Mistral-Nemo-Instruct-2407"
    HUGGINGFACE_FALLBACK_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
    DEFAULT_PROVIDER = "huggingface"
    
    @classmethod
    def get_memory_usage(cls):
        """Monitor memory usage"""
        return psutil.virtual_memory().percent
    
    @classmethod
    def should_gc(cls):
        """Check if garbage collection needed"""
        return cls.get_memory_usage() > 80

# Add main block for direct execution
if __name__ == "__main__":
    print("=== Configuration Settings ===")
    print(f"GEMINI_API_KEY loaded: {bool(Config.GEMINI_API_KEY)}")
    print(f"Available Memory: {Config.AVAILABLE_MEMORY:.2f} GB")
    print(f"Current Memory Usage: {Config.get_memory_usage()}%")
    print(f"Should GC: {Config.should_gc()}")
    print(f"Default Model: {Config.DEFAULT_MODEL}")
    print(f"Default Provider: {Config.DEFAULT_PROVIDER}")
    print("Categories:", Config.EXPENSE_CATEGORIES)
