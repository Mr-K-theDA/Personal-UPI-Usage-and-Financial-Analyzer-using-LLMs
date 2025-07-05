# data_cleaner.py
import pandas as pd
import numpy as np
import re
import gc
from typing import Dict, List
import logging
from config import Config

logger = logging.getLogger(__name__)

class MemoryOptimizedDataCleaner:
    def __init__(self):
        self.config = Config()
        self.categories = self.config.EXPENSE_CATEGORIES
        
        # Optimized keyword mapping for your system
        self.category_keywords = {
            'Food & Dining': ['zomato', 'swiggy', 'restaurant', 'food', 'dining', 'pizza', 'dominos', 'kfc'],
            'Transportation': ['uber', 'ola', 'cab', 'metro', 'bus', 'auto', 'fuel', 'petrol', 'rapido'],
            'Shopping': ['amazon', 'flipkart', 'myntra', 'shopping', 'mall', 'ajio', 'nykaa'],
            'Entertainment': ['netflix', 'prime', 'hotstar', 'spotify', 'bookmyshow', 'gaming'],
            'Bills & Utilities': ['electricity', 'airtel', 'jio', 'vi', 'broadband', 'recharge'],
            'Groceries': ['bigbasket', 'grofers', 'dmart', 'reliance', 'grocery', 'vegetables'],
            'Transfer': ['transfer', 'sent to', 'received from', 'wallet', 'bank'],
            'Others': ['atm', 'cash', 'miscellaneous']
        }
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and optimize DataFrame for your system"""
        logger.info("Starting data cleaning...")
        
        # Memory check
        initial_memory = self.config.get_memory_usage()
        logger.info(f"Memory before cleaning: {initial_memory}%")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Basic cleaning
        df_clean = self._remove_duplicates(df_clean)
        df_clean = self._clean_amounts(df_clean)
        df_clean = self._clean_descriptions(df_clean)
        df_clean = self._categorize_transactions(df_clean)
        df_clean = self._add_derived_features(df_clean)
        
        # Memory optimization
        df_clean = self._optimize_datatypes(df_clean)
        
        # Cleanup
        gc.collect()
        
        final_memory = self.config.get_memory_usage()
        logger.info(f"Memory after cleaning: {final_memory}%")
        logger.info(f"Cleaned {len(df_clean)} transactions")
        
        return df_clean
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate transactions"""
        before_count = len(df)
        
        # Remove exact duplicates
        df = df.drop_duplicates()
        
        # Remove near-duplicates (same amount, date, description)
        df = df.drop_duplicates(subset=['date', 'amount', 'description'], keep='first')
        
        after_count = len(df)
        logger.info(f"Removed {before_count - after_count} duplicates")
        
        return df
    
    def _clean_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate amounts"""
        # Remove negative amounts (keep absolute values)
        df['amount'] = df['amount'].abs()
        
        # Remove zero amounts
        df = df[df['amount'] > 0]
        
        # Remove unrealistic amounts (> 1 lakh)
        df = df[df['amount'] <= 100000]
        
        return df
    
    def _clean_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean transaction descriptions"""
        # Convert to lowercase for consistency
        df['description_clean'] = df['description'].str.lower()
        
        # Remove extra spaces and special characters
        df['description_clean'] = df['description_clean'].str.replace(r'[^\w\s]', ' ', regex=True)
        df['description_clean'] = df['description_clean'].str.replace(r'\s+', ' ', regex=True)
        df['description_clean'] = df['description_clean'].str.strip()
        
        # Extract merchant/recipient names
        df['merchant'] = df['description_clean'].apply(self._extract_merchant)
        
        return df
    
    def _extract_merchant(self, description: str) -> str:
        """Extract merchant name from description"""
        if not isinstance(description, str):
            return 'Unknown'

        # More robust patterns for merchant extraction
        patterns = [
            r'(?:to|from|paid to|sent to|received from)\s+([a-z\s&]+?)(?:\s+on|\s+for|\s+via|$)'
        ]

        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                merchant = match.group(1).strip()
                # Avoid overly generic matches
                if merchant not in ['self', 'bank', 'account']:
                    return merchant.title()

        # Fallback for simpler descriptions
        words = description.split()
        if len(words) > 1 and words[0] in ['to', 'from']:
            return words[1].title()
        
        # If no specific pattern is matched, return the first few words as a guess
        return ' '.join(words[:2]).title()
    
    def _categorize_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize transactions using keywords"""
        df['category'] = 'Others'  # Default category
        
        # Process in batches to manage memory
        batch_size = 1000
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            for category, keywords in self.category_keywords.items():
                mask = batch_df['description_clean'].str.contains(
                    '|'.join(keywords), case=False, na=False
                )
                df.loc[batch_df.index[mask], 'category'] = category
            
            # Memory check
            if i % 5000 == 0:
                gc.collect()
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for analysis"""
        # Time-based features
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        df['day_of_month'] = pd.to_datetime(df['datetime']).dt.day
        df['is_weekend'] = pd.to_datetime(df['datetime']).dt.dayofweek >= 5
        
        # Amount-based features
        df['amount_range'] = pd.cut(df['amount'], 
                                   bins=[0, 100, 500, 1000, 5000, float('inf')],
                                   labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Transaction frequency by merchant
        merchant_counts = df['merchant'].value_counts()
        df['merchant_frequency'] = df['merchant'].map(merchant_counts)
        
        return df
    
    def _optimize_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize datatypes to reduce memory usage"""
        # Convert object columns to category where appropriate
        categorical_columns = ['app_source', 'transaction_type', 'category', 'amount_range']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Optimize numeric columns
        df['amount'] = pd.to_numeric(df['amount'], downcast='float')
        df['merchant_frequency'] = pd.to_numeric(df['merchant_frequency'], downcast='integer')
        
        return df
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics"""
        summary = {
            'total_transactions': len(df),
            'total_amount': df['amount'].sum(),
            'avg_transaction': df['amount'].mean(),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'categories': df['category'].value_counts().to_dict(),
            'top_merchants': df['merchant'].value_counts().head(5).to_dict(),
            'transaction_types': df['transaction_type'].value_counts().to_dict()
        }
        
        return summary
