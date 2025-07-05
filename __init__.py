# __init__.py
"""
Personal UPI Usage and Financial Analyzer
==========================================

A comprehensive tool for analyzing UPI transaction statements from multiple apps
(Paytm, GPay, PhonePe, etc.) and generating actionable insights using LLMs.

Author: Your Name
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes for easy access
from .pdf_processor import MemoryOptimizedUPIProcessor
from .data_cleaner import MemoryOptimizedDataCleaner
from .llm_analyzer import LightweightLLMAnalyzer
from .recommendation_engine import FinancialRecommendationEngine
from .utils import (
    FileUtils, 
    MemoryUtils, 
    DataUtils, 
    DateUtils, 
    SecurityUtils,
    ValidationUtils,
    log_function_performance
)

# Package metadata
__all__ = [
    'MemoryOptimizedUPIProcessor',
    'MemoryOptimizedDataCleaner', 
    'LightweightLLMAnalyzer',
    'FinancialRecommendationEngine',
    'FileUtils',
    'MemoryUtils',
    'DataUtils',
    'DateUtils',
    'SecurityUtils',
    'ValidationUtils',
    'log_function_performance'
]

# Package info
PACKAGE_INFO = {
    'name': 'UPI Financial Analyzer',
    'version': __version__,
    'description': 'AI-powered UPI transaction analysis and financial insights',
    'features': [
        'Multi-format PDF processing',
        'Intelligent transaction categorization',
        'LLM-powered insights',
        'Memory-optimized for 8GB RAM systems',
        'Personalized financial recommendations'
    ],
    'supported_apps': ['Paytm', 'Google Pay', 'PhonePe', 'Generic UPI'],
    'requirements': {
        'python': '>=3.8',
        'memory': '8GB RAM recommended',
        'storage': '1GB free space'
    }
}

def get_package_info():
    """Return package information"""
    return PACKAGE_INFO

def check_system_compatibility():
    """Check if system meets requirements"""
    import psutil
    import sys
    
    compatibility = {
        'python_version': sys.version_info >= (3, 8),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'memory_sufficient': psutil.virtual_memory().total >= 6 * (1024**3),  # 6GB minimum
        'cpu_cores': psutil.cpu_count(),
        'platform': sys.platform
    }
    
    return compatibility

# Initialize logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info(f"UPI Financial Analyzer v{__version__} initialized")