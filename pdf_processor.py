# pdf_processor.py
import sys
sys.path.append('upi-financial-analyzer')
import fitz  # PyMuPDF - lightweight
import pandas as pd
import re
import gc
from datetime import datetime
from typing import List, Dict, Optional
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryOptimizedUPIProcessor:
    def __init__(self):
        self.config = Config()
        self.app_patterns = {
            'paytm': {
                'date_pattern': r'(\d{2}/\d{2}/\d{4})',
                'time_pattern': r'(\d{2}:\d{2}:\d{2})',
                'amount_pattern': r'₹\s*([\d,]+\.?\d*)',
                'date_format': '%d/%m/%Y'
            },
            'gpay': {
                'date_pattern': r'(\d{1,2}\s+\w+\s+\d{4})',
                'time_pattern': r'(\d{1,2}:\d{2}\s+(?:AM|PM))',
                'amount_pattern': r'₹\s*([\d,]+\.?\d*)',
                'date_format': '%d %b %Y'
            },
            'phonepe': {
                'date_pattern': r'(\d{2}-\d{2}-\d{4})',
                'time_pattern': r'(\d{2}:\d{2})',
                'amount_pattern': r'₹\s*([\d,]+\.?\d*)',
                'date_format': '%d-%m-%Y'
            }
        }
    
    def check_memory_and_gc(self):
        """Check memory usage and garbage collect if needed"""
        if self.config.should_gc():
            gc.collect()
            logger.info("Memory cleaned up")
    
    def detect_app_type(self, text_sample: str) -> str:
        """Detect UPI app type from small text sample"""
        # Only check first 1000 characters for efficiency
        text_lower = text_sample[:1000].lower()
        
        if 'paytm' in text_lower:
            return 'paytm'
        elif 'google pay' in text_lower or 'gpay' in text_lower:
            return 'gpay'
        elif 'phonepe' in text_lower or 'phone pe' in text_lower:
            return 'phonepe'
        else:
            return 'generic'
    
    def extract_text_memory_efficient(self, pdf_path: str) -> str:
        """Memory-efficient text extraction"""
        try:
            doc = fitz.open(pdf_path)
            text_chunks = []
            
            # Process pages one by one to save memory
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                text_chunks.append(page_text)
                
                # Check memory usage periodically
                if page_num % 5 == 0:
                    self.check_memory_and_gc()
            
            doc.close()
            full_text = "\n".join(text_chunks)
            
            # Clean up chunks from memory
            del text_chunks
            gc.collect()
            
            return full_text
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""
    
    def parse_transactions_chunked(self, text: str, app_type: str) -> List[Dict]:
        """Parse transactions in chunks to manage memory"""
        lines = text.split('\n')
        transactions = []
        
        # Process in chunks
        chunk_size = self.config.PROCESSING_CHUNK_SIZE
        
        for i in range(0, len(lines), chunk_size):
            chunk = lines[i:i + chunk_size]
            chunk_text = '\n'.join(chunk)
            
            # Parse chunk based on app type
            if app_type == 'paytm':
                chunk_transactions = self._parse_paytm_chunk(chunk_text)
            elif app_type == 'gpay':
                chunk_transactions = self._parse_gpay_chunk(chunk_text)
            elif app_type == 'phonepe':
                chunk_transactions = self._parse_phonepe_chunk(chunk_text)
            else:
                chunk_transactions = self._parse_generic_chunk(chunk_text)
            
            transactions.extend(chunk_transactions)
            
            # Memory check after each chunk
            self.check_memory_and_gc()
        
        return transactions
    
    def _parse_paytm_chunk(self, text: str) -> List[Dict]:
        """Parse Paytm transactions from text chunk"""
        transactions = []
        
        # Simplified pattern for better performance
        pattern = r'(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2}:\d{2})\s+(.*?)\s+(₹\s*[\d,]+\.?\d*)'
        matches = re.findall(pattern, text, re.MULTILINE)
        
        for match in matches[:50]:  # Limit per chunk for memory
            try:
                date_str, time_str, description, amount_str = match
                
                # Clean and convert amount
                amount_clean = re.sub(r'[₹,\s]', '', amount_str)
                amount = float(amount_clean)
                
                transaction = {
                    'date': datetime.strptime(date_str, '%d/%m/%Y').date(),
                    'time': datetime.strptime(time_str, '%H:%M:%S').time(),
                    'description': description.strip()[:100],  # Limit description length
                    'amount': abs(amount),
                    'transaction_type': 'debit' if 'paid' in description.lower() else 'credit',
                    'app_source': 'paytm'
                }
                transactions.append(transaction)
                
            except Exception as e:
                continue  # Skip problematic transactions
        
        return transactions
    
    def _parse_gpay_chunk(self, text: str) -> List[Dict]:
        """Parse Google Pay transactions from text chunk with improved pattern matching"""
        transactions = []
        # This pattern is designed to be more flexible for GPay statements
        pattern = re.compile(
            r"(Paid to|Received from)\s*(.*?)\s*"
            r"(\d{1,2}\s\w{3}\s\d{4},\s\d{1,2}:\d{2}\s[AP]M)\s*"
            r"₹\s*([\d,]+\.\d{2})",
            re.IGNORECASE
        )
        matches = pattern.finditer(text)

        for match in matches:
            try:
                trans_type_str, merchant, datetime_str, amount_str = match.groups()
                
                trans_type = 'debit' if 'Paid' in trans_type_str else 'credit'
                amount = float(amount_str.replace(',', ''))
                
                # GPay format: '1 Jan 2023, 9:00 PM'
                trans_datetime = datetime.strptime(datetime_str, '%d %b %Y, %I:%M %p')

                transaction = {
                    'date': trans_datetime.date(),
                    'time': trans_datetime.time(),
                    'description': f"{trans_type_str} {merchant}".strip(),
                    'amount': amount,
                    'transaction_type': trans_type,
                    'app_source': 'gpay'
                }
                transactions.append(transaction)
            except Exception as e:
                logger.warning(f"Skipping a GPay transaction due to parsing error: {e}")
                continue
        
        return transactions
    
    def _parse_phonepe_chunk(self, text: str) -> List[Dict]:
        """Parse PhonePe transactions from text chunk"""
        transactions = []
        
        pattern = r'(\d{2}-\d{2}-\d{4})\s+(\d{2}:\d{2})\s+(.*?)\s+(₹\s*[\d,]+\.?\d*)'
        matches = re.findall(pattern, text, re.MULTILINE)
        
        for match in matches[:50]:  # Limit per chunk
            try:
                date_str, time_str, description, amount_str = match
                
                amount_clean = re.sub(r'[₹,\s]', '', amount_str)
                amount = float(amount_clean)
                
                transaction = {
                    'date': datetime.strptime(date_str, '%d-%m-%Y').date(),
                    'time': datetime.strptime(time_str, '%H:%M').time(),
                    'description': description.strip()[:100],
                    'amount': amount,
                    'transaction_type': 'debit' if 'sent' in description.lower() else 'credit',
                    'app_source': 'phonepe'
                }
                transactions.append(transaction)
                
            except:
                continue
        
        return transactions
    
    def _parse_generic_chunk(self, text: str) -> List[Dict]:
        """Generic parsing for unknown formats"""
        transactions = []
        lines = text.split('\n')
        
        for line in lines:
            if '₹' in line:
                try:
                    amount_match = re.search(r'₹\s*([\d,]+\.?\d*)', line)
                    if amount_match:
                        amount = float(amount_match.group(1).replace(',', ''))
                        
                        transaction = {
                            'date': datetime.now().date(),
                            'time': datetime.now().time(),
                            'description': line.strip()[:100],
                            'amount': amount,
                            'transaction_type': 'debit',
                            'app_source': 'generic'
                        }
                        transactions.append(transaction)
                        
                        if len(transactions) >= 30:
                            break
                except:
                    continue
        
        return transactions
    
    def process_pdf(self, pdf_path: str) -> pd.DataFrame:
        """Main processing method optimized for your system"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Check initial memory
        initial_memory = self.config.get_memory_usage()
        logger.info(f"Initial memory usage: {initial_memory}%")
        
        # Extract text efficiently
        text = self.extract_text_memory_efficient(pdf_path)
        if not text:
            raise Exception("Failed to extract text from PDF")
        
        # Detect app type from sample
        app_type = self.detect_app_type(text)
        logger.info(f"Detected app type: {app_type}")
        
        # Parse transactions in chunks
        transactions = self.parse_transactions_chunked(text, app_type)
        
        if not transactions:
            raise Exception("No transactions found in PDF")
        
        # Create DataFrame efficiently
        df = pd.DataFrame(transactions)
        
        # Add essential columns only
        df['datetime'] = pd.to_datetime(df['date'].astype(str))
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        
        # Final cleanup
        del text, transactions
        gc.collect()
        
        final_memory = self.config.get_memory_usage()
        logger.info(f"Final memory usage: {final_memory}%")
        logger.info(f"Extracted {len(df)} transactions")
        
        return df
