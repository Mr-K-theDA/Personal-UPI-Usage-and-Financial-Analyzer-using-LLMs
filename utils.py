# utils.py
import os
import pandas as pd
import numpy as np
import logging
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import base64
from pathlib import Path
import gc
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileUtils:
    """File handling utilities optimized for your system"""
    
    @staticmethod
    def ensure_directory_exists(directory_path: str) -> None:
        """Create directory if it doesn't exist"""
        Path(directory_path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """Get file size in MB"""
        return os.path.getsize(file_path) / (1024 * 1024)
    
    @staticmethod
    def validate_pdf_file(file_path: str, max_size_mb: int = 5) -> Tuple[bool, str]:
        """Validate PDF file for processing"""
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        if not file_path.lower().endswith('.pdf'):
            return False, "File is not a PDF"
        
        file_size = FileUtils.get_file_size_mb(file_path)
        if file_size > max_size_mb:
            return False, f"File size ({file_size:.1f}MB) exceeds limit ({max_size_mb}MB)"
        
        return True, "Valid PDF file"
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, file_path: str, format: str = 'csv') -> bool:
        """Save DataFrame to file"""
        try:
            FileUtils.ensure_directory_exists(os.path.dirname(file_path))
            
            if format.lower() == 'csv':
                df.to_csv(file_path, index=False)
            elif format.lower() == 'json':
                df.to_json(file_path, orient='records', date_format='iso')
            elif format.lower() == 'excel':
                df.to_excel(file_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"DataFrame saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save DataFrame: {e}")
            return False
    
    @staticmethod
    def load_dataframe(file_path: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from file"""
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                return pd.read_json(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
                
        except Exception as e:
            logger.error(f"Failed to load DataFrame: {e}")
            return None

class MemoryUtils:
    """Memory management utilities for your 8GB system"""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percentage': memory.percent
        }
    
    @staticmethod
    def should_optimize_memory() -> bool:
        """Check if memory optimization is needed"""
        return psutil.virtual_memory().percent > 80
    
    @staticmethod
    def force_garbage_collection():
        """Force garbage collection"""
        gc.collect()
        logger.info("Garbage collection completed")
    
    @staticmethod
    def log_memory_usage(operation: str = ""):
        """Log current memory usage"""
        memory_info = MemoryUtils.get_memory_usage()
        logger.info(f"Memory usage {operation}: {memory_info['percentage']:.1f}% "
                   f"({memory_info['used_gb']:.1f}GB/{memory_info['total_gb']:.1f}GB)")

class DataUtils:
    """Data processing utilities"""
    
    @staticmethod
    def safe_convert_to_float(value: Any) -> float:
        """Safely convert value to float"""
        try:
            if isinstance(value, str):
                # Remove currency symbols and commas
                cleaned = value.replace('₹', '').replace(',', '').replace(' ', '')
                return float(cleaned)
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    @staticmethod
    def extract_numbers_from_string(text: str) -> List[float]:
        """Extract all numbers from a string"""
        import re
        numbers = re.findall(r'\d+\.?\d*', text)
        return [float(num) for num in numbers]
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return str(text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep basic punctuation
        import re
        text = re.sub(r'[^\w\s\-\.\,\:]', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float) -> float:
        """Calculate percentage change between two values"""
        if old_value == 0:
            return 0.0
        return ((new_value - old_value) / old_value) * 100
    
    @staticmethod
    def group_small_categories(df: pd.DataFrame, column: str, 
                              min_percentage: float = 5) -> pd.DataFrame:
        """Group small categories into 'Others'"""
        total_count = len(df)
        value_counts = df[column].value_counts()
        
        # Calculate percentages
        percentages = (value_counts / total_count) * 100
        
        # Find categories below threshold
        small_categories = percentages[percentages < min_percentage].index
        
        # Replace small categories with 'Others'
        df_copy = df.copy()
        df_copy[column] = df_copy[column].replace(small_categories, 'Others')
        
        return df_copy

class DateUtils:
    """Date and time utilities"""
    
    @staticmethod
    def parse_flexible_date(date_str: str) -> Optional[datetime]:
        """Parse date string with multiple possible formats"""
        formats = [
            '%d/%m/%Y',
            '%d-%m-%Y',
            '%Y-%m-%d',
            '%d %b %Y',
            '%d %B %Y',
            '%m/%d/%Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse date: {date_str}")
        return None
    
    @staticmethod
    def get_date_range_description(start_date: datetime, end_date: datetime) -> str:
        """Get human-readable date range description"""
        delta = end_date - start_date
        
        if delta.days <= 7:
            return "This week"
        elif delta.days <= 31:
            return "This month"
        elif delta.days <= 93:
            return "Last 3 months"
        elif delta.days <= 186:
            return "Last 6 months"
        else:
            return f"{delta.days} days"
    
    @staticmethod
    def get_month_year_key(date: datetime) -> str:
        """Get month-year key for grouping"""
        return f"{date.year}-{date.month:02d}"

class SecurityUtils:
    """Security utilities for handling sensitive financial data"""
    
    @staticmethod
    def hash_sensitive_data(data: str) -> str:
        """Hash sensitive data for privacy"""
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    @staticmethod
    def mask_account_number(account_num: str) -> str:
        """Mask account number for display"""
        if len(account_num) <= 4:
            return account_num
        return '*' * (len(account_num) - 4) + account_num[-4:]
    
    @staticmethod
    def validate_data_privacy(df: pd.DataFrame) -> Dict[str, bool]:
        """Check if DataFrame contains potentially sensitive information"""
        sensitive_checks = {
            'has_account_numbers': False,
            'has_phone_numbers': False,
            'has_email_addresses': False,
            'has_personal_names': False
        }
        
        # Convert DataFrame to string for checking
        df_str = df.to_string().lower()
        
        # Check for account numbers (sequences of 10+ digits)
        import re
        if re.search(r'\d{10,}', df_str):
            sensitive_checks['has_account_numbers'] = True
        
        # Check for phone numbers
        if re.search(r'[+]?[\d\s\-$$$$]{10,}', df_str):
            sensitive_checks['has_phone_numbers'] = True
        
        # Check for email addresses
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', df_str):
            sensitive_checks['has_email_addresses'] = True
        
        return sensitive_checks

class VisualizationUtils:
    """Utilities for creating visualizations optimized for your system"""
    
    @staticmethod
    def get_color_palette(n_colors: int) -> List[str]:
        """Get a color palette for visualizations"""
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        if n_colors <= len(colors):
            return colors[:n_colors]
        
        # Generate additional colors if needed
        import matplotlib.cm as cm
        import numpy as np
        
        additional_colors = cm.Set3(np.linspace(0, 1, n_colors - len(colors)))
        additional_hex = ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) 
                         for r, g, b, _ in additional_colors]
        
        return colors + additional_hex
    
    @staticmethod
    def optimize_plot_for_memory():
        """Set matplotlib backend for memory efficiency"""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend

class ValidationUtils:
    """Data validation utilities"""
    
    @staticmethod
    def validate_transaction_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate transaction DataFrame"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        required_columns = ['date', 'amount', 'description', 'transaction_type']
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing columns: {missing_columns}")
        
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("DataFrame is empty")
            return validation_results
        
        # Check for null values
        null_counts = df[required_columns].isnull().sum()
        high_null_columns = null_counts[null_counts > len(df) * 0.1].index.tolist()
        if high_null_columns:
            validation_results['warnings'].append(f"High null values in: {high_null_columns}")
        
        # Check amount values
        if 'amount' in df.columns:
            negative_amounts = (df['amount'] < 0).sum()
            zero_amounts = (df['amount'] == 0).sum()
            
            if negative_amounts > 0:
                validation_results['warnings'].append(f"{negative_amounts} negative amounts found")
            
            if zero_amounts > 0:
                validation_results['warnings'].append(f"{zero_amounts} zero amounts found")
        
        # Statistics
        validation_results['statistics'] = {
            'total_rows': len(df),
            'date_range': {
                'start': df['date'].min() if 'date' in df.columns else None,
                'end': df['date'].max() if 'date' in df.columns else None
            },
            'total_amount': df['amount'].sum() if 'amount' in df.columns else 0
        }
        
        return validation_results

class ConfigUtils:
    """Configuration utilities"""
    
    @staticmethod
    def load_config_from_env() -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {
            'api_keys': {
                'openai': os.getenv('OPENAI_API_KEY'),
                'gemini': os.getenv('GEMINI_API_KEY'),
                'huggingface': os.getenv('HUGGINGFACE_API_TOKEN')
            },
            'memory_limit': int(os.getenv('MAX_MEMORY_USAGE', 4096)),
            'batch_size': int(os.getenv('BATCH_SIZE', 10)),
            'debug_mode': os.getenv('DEBUG_MODE', 'False').lower() == 'true'
        }
        
        return config
    
    @staticmethod
    def check_api_availability(config: Dict[str, Any]) -> Dict[str, bool]:
        """Check which APIs are available"""
        availability = {}
        
        for service, api_key in config['api_keys'].items():
            availability[service] = bool(api_key and api_key.strip())
        
        return availability

# Convenience functions for common operations
def log_function_performance(func):
    """Decorator to log function performance"""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        MemoryUtils.log_memory_usage(f"before {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"{func.__name__} completed in {duration:.2f} seconds")
            MemoryUtils.log_memory_usage(f"after {func.__name__}")
            
            return result
            
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            raise
    
    return wrapper

def create_sample_data() -> pd.DataFrame:
    """Create sample transaction data for testing"""
    sample_data = {
        'date': pd.date_range(start='2024-01-01', end='2024-03-31', freq='D'),
        'amount': np.random.uniform(100, 10000, size=len(pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')))
    }