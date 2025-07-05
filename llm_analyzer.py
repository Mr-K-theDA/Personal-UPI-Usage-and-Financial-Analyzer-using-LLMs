# llm_analyzer.py
import google.generativeai as genai
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import InferenceClient
import pandas as pd
import json
from typing import Dict, List
import logging
from config import Config
import time

logger = logging.getLogger(__name__)

class LightweightLLMAnalyzer:
    def __init__(self, llm_provider: str = Config.DEFAULT_PROVIDER):
        self.config = Config()
        self.llm_provider = llm_provider.lower()
        self.model = None
        self.tokenizer = None # For local Hugging Face models
        self.client = None # For OpenAI
        self.llm_available = self._initialize_llm()

    def _initialize_llm(self) -> bool:
        """Initialize the selected LLM provider."""
        try:
            if self.llm_provider in ['gemini', 'hugging face', 'openai']:
                if self.config.GEMINI_API_KEY:
                    genai.configure(api_key=self.config.GEMINI_API_KEY)
                    self.model = genai.GenerativeModel(self.config.DEFAULT_MODEL)
                    self.llm_provider = 'gemini'  # Force provider to gemini
                    return True
                else:
                    logger.warning("Gemini API key not found.")
                    return False
            
            elif self.llm_provider == 'hugging face':
                if self.config.HUGGINGFACE_API_TOKEN:
                    try:
                        logger.info(f"Attempting to load gated model '{self.config.HUGGINGFACE_MODEL}' locally.")
                        self.tokenizer = AutoTokenizer.from_pretrained(self.config.HUGGINGFACE_MODEL, token=self.config.HUGGINGFACE_API_TOKEN)
                        self.model = AutoModelForCausalLM.from_pretrained(self.config.HUGGINGFACE_MODEL, token=self.config.HUGGINGFACE_API_TOKEN)
                        logger.info("Gated Hugging Face model loaded successfully.")
                        return True
                    except Exception as e:
                        if "gated repo" in str(e):
                            logger.warning("Gated model access denied. Falling back to public model.")
                            logger.info(f"Loading public fallback model '{self.config.HUGGINGFACE_FALLBACK_MODEL}'.")
                            self.tokenizer = AutoTokenizer.from_pretrained(self.config.HUGGINGFACE_FALLBACK_MODEL, token=self.config.HUGGINGFACE_API_TOKEN)
                            self.model = AutoModelForCausalLM.from_pretrained(self.config.HUGGINGFACE_FALLBACK_MODEL, token=self.config.HUGGINGFACE_API_TOKEN)
                            logger.info("Public Hugging Face model loaded successfully.")
                            return True
                        elif "PyPreTokenizerTypeWrapper" in str(e):
                            logger.error("Hugging Face tokenizer failed to initialize. Disabling Hugging Face provider.")
                            return False
                        else:
                            raise e
                else:
                    logger.warning("Hugging Face API token not found for local model loading.")
                    return False
            
            else:
                logger.error(f"Unsupported LLM provider: {self.llm_provider}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider {self.llm_provider}: {e}")
            return False

    def analyze_spending_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze spending patterns with LLM assistance"""
        summary_data = self._prepare_summary_data(df)
        
        if self.llm_available:
            if self.llm_provider == 'gemini':
                insights = self._generate_gemini_insights(summary_data)
            elif self.llm_provider == 'openai':
                insights = self._generate_openai_insights(summary_data)
            elif self.llm_provider == 'hugging face':
                insights = self._generate_huggingface_insights(summary_data)
            else:
                insights = self._generate_rule_based_insights(summary_data)
        else:
            insights = self._generate_rule_based_insights(summary_data)
        
        return insights

    def _prepare_summary_data(self, df: pd.DataFrame) -> Dict:
        """Prepare summary data for analysis"""
        summary = {
            'total_spent': float(df[df['transaction_type'] == 'debit']['amount'].sum()),
            'total_received': float(df[df['transaction_type'] == 'credit']['amount'].sum()),
            'transaction_count': len(df),
            'avg_transaction': float(df['amount'].mean()),
            'category_spending': df[df['transaction_type'] == 'debit'].groupby('category')['amount'].sum().to_dict(),
            'monthly_spending': df[df['transaction_type'] == 'debit'].groupby(['year', 'month'])['amount'].sum().to_dict(),
            'top_merchants': df['merchant'].value_counts().head(5).to_dict(),
            'weekend_vs_weekday': {
                'weekend': float(df[df['is_weekend'] == True]['amount'].sum()),
                'weekday': float(df[df['is_weekend'] == False]['amount'].sum())
            }
        }
        return summary

    def _create_prompt(self, summary_data: Dict) -> str:
        """Create a standardized prompt for financial analysis."""
        return f"""
        Analyze this financial data and provide insights in a structured JSON format:

        Financial Summary:
        - Total Spent: ₹{summary_data['total_spent']:.2f}
        - Total Received: ₹{summary_data['total_received']:.2f}
        - Average Transaction: ₹{summary_data['avg_transaction']:.2f}
        - Transaction Count: {summary_data['transaction_count']}

        Category-wise Spending:
        {json.dumps(summary_data['category_spending'], indent=2)}

        Top Merchants:
        {json.dumps(summary_data['top_merchants'], indent=2)}

        Please provide your analysis in the following JSON format:
        {{
            "key_insights": ["insight 1", "insight 2", "insight 3"],
            "improvement_areas": ["area 1", "area 2"],
            "positive_habits": ["habit 1", "habit 2"],
            "budget_recommendations": ["recommendation 1", "recommendation 2"],
            "additional_notes": "Any additional notes or explanations"
        }}

        Keep each insight concise and actionable. Your response must be a valid JSON object.
        """

    def _parse_llm_response(self, response_text: str, summary_data: Dict) -> Dict:
        """Parse the LLM response and extract structured data."""
        insights = {
            'key_insights': [], 'improvement_areas': [], 'positive_habits': [],
            'budget_recommendations': [], 'additional_notes': "",
            'summary_data': summary_data, 'insights_type': f'{self.llm_provider}_generated'
        }
        try:
            # Clean up response text before parsing
            # Find the start of the JSON object
            json_start = response_text.find('{')
            if json_start == -1:
                raise json.JSONDecodeError("No JSON object found in the response.", response_text, 0)
            
            # Find the end of the JSON object
            json_end = response_text.rfind('}') + 1
            if json_end <= json_start:
                 raise json.JSONDecodeError("Could not find end of JSON object.", response_text, 0)

            json_string = response_text[json_start:json_end]
            response_json = json.loads(json_string)
            
            insights.update({k: response_json.get(k, []) for k in insights if isinstance(insights[k], list)})
            insights['additional_notes'] = response_json.get('additional_notes', "")
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"LLM response is not in a valid JSON format: {e}. Using fallback.")
            insights['llm_analysis'] = response_text
        return insights

    def _generate_gemini_insights(self, summary_data: Dict) -> Dict:
        """Generate insights using Gemini LLM."""
        try:
            prompt = self._create_prompt(summary_data)
            response = self.model.generate_content(prompt)
            time.sleep(1)  # Rate limiting
            return self._parse_llm_response(response.text, summary_data)
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return self._generate_rule_based_insights(summary_data)

    def _generate_openai_insights(self, summary_data: Dict) -> Dict:
        """Generate insights using OpenAI LLM."""
        try:
            prompt = self._create_prompt(summary_data)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE
            )
            time.sleep(1) # Rate limiting
            return self._parse_llm_response(response.choices[0].message.content, summary_data)
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return self._generate_rule_based_insights(summary_data)

    def _generate_huggingface_insights(self, summary_data: Dict) -> Dict:
        """Generate insights using a local Hugging Face LLM."""
        try:
            prompt = self._create_prompt(summary_data)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Generate output from the model
            outputs = self.model.generate(**inputs, max_new_tokens=self.config.MAX_TOKENS)
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # The generated text often includes the prompt, so we remove it from the response.
            if response_text.strip().startswith(prompt.strip()):
                response_text = response_text.strip()[len(prompt.strip()):]

            return self._parse_llm_response(response_text, summary_data)
        except Exception as e:
            logger.error(f"Hugging Face local analysis failed: {e}")
            return self._generate_rule_based_insights(summary_data)

    def _generate_rule_based_insights(self, summary_data: Dict) -> Dict:
        """Generate insights using rule-based analysis"""
        insights = {
            'key_insights': [], 'improvement_areas': [], 'positive_habits': [],
            'budget_recommendations': [], 'summary_data': summary_data,
            'insights_type': 'rule_based'
        }
        if summary_data['total_spent'] > summary_data['total_received']:
            insights['key_insights'].append("Your spending exceeds income - consider budgeting")
        if summary_data['category_spending']:
            top_category = max(summary_data['category_spending'], key=summary_data['category_spending'].get)
            insights['key_insights'].append(f"Highest spending: {top_category} (₹{summary_data['category_spending'][top_category]:.2f})")
        if summary_data['avg_transaction'] > 1000:
            insights['key_insights'].append("High average transaction amount - mostly large purchases")
        else:
            insights['key_insights'].append("Moderate spending pattern with small frequent transactions")
        if summary_data['category_spending'].get('Food & Dining', 0) > summary_data['total_spent'] * 0.3:
            insights['improvement_areas'].append("Food spending is high - consider home cooking")
        if summary_data['category_spending'].get('Entertainment', 0) > summary_data['total_spent'] * 0.2:
            insights['improvement_areas'].append("Entertainment expenses are significant")
        if summary_data['category_spending'].get('Investment', 0) > 0:
            insights['positive_habits'].append("Good job on investing!")
        monthly_avg = summary_data['total_spent'] / max(len(summary_data['monthly_spending']), 1)
        insights['budget_recommendations'].append(f"Monthly spending average: ₹{monthly_avg:.2f}")
        insights['budget_recommendations'].append("Try to save 20% of income")
        return insights

    def generate_recommendations(self, insights: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        if insights['insights_type'].endswith('_generated') and not insights.get('llm_analysis'):
            if insights.get('key_insights'):
                recommendations.append("🔍 Key Insights:")
                recommendations.extend([f"- {item}" for item in insights['key_insights']])
            if insights.get('improvement_areas'):
                recommendations.append("\n💡 Improvement Areas:")
                recommendations.extend([f"- {item}" for item in insights['improvement_areas']])
            if insights.get('positive_habits'):
                recommendations.append("\n👍 Positive Habits:")
                recommendations.extend([f"- {item}" for item in insights['positive_habits']])
            if insights.get('budget_recommendations'):
                recommendations.append("\n💰 Budget Recommendations:")
                recommendations.extend([f"- {item}" for item in insights['budget_recommendations']])
            if insights.get('additional_notes'):
                recommendations.append(f"\n📝 Additional Notes: {insights['additional_notes']}")
        elif insights.get('llm_analysis'):
            recommendations.append("Based on AI analysis of your spending patterns:")
            recommendations.append(insights['llm_analysis'][:500] + "...")
        else:
            recommendations.extend([
                "💡 Track your daily expenses to identify spending patterns",
                "🎯 Set monthly budgets for each category",
                "💰 Consider automating savings (even small amounts help)",
                "📱 Use UPI transaction limits to control spending",
                "📊 Review your spending weekly to stay on track"
            ])
            recommendations.extend(insights['improvement_areas'])
        return recommendations
