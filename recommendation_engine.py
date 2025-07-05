import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FinancialRecommendationEngine:
    def __init__(self):
        self.recommendation_rules = {
            'high_spending_category': {
                'threshold': 0.25,
                'message': "Your spending on {} is {:.1%} of your total expenses. Consider setting a budget for this category.",
                'impact': 'High'
            },
            'low_savings_rate': {
                'threshold': 0.15,
                'message': "Your savings rate is only {:.1%}. Aim for at least 20% by reducing non-essential spending.",
                'impact': 'High'
            },
            'frequent_small_transactions': {
                'threshold_count': 20,
                'threshold_amount': 200,
                'message': "You have {} small transactions under ₹{}. These can add up; try tracking them for a week.",
                'impact': 'Medium'
            },
            'potential_subscriptions': {
                'frequency': 3,
                'message': "You have recurring payments to {}. Ensure you are actively using this subscription.",
                'impact': 'Medium'
            }
        }

    def generate_recommendations(self, df: pd.DataFrame) -> List[Dict]:
        """Generate a list of personalized financial recommendations."""
        recommendations = []
        
        if df.empty:
            return recommendations

        # Calculate core metrics
        debits = df[df['transaction_type'] == 'debit']
        credits = df[df['transaction_type'] == 'credit']
        total_expense = debits['amount'].sum()
        total_income = credits['amount'].sum()

        # 1. High Spending Category Analysis
        if total_expense > 0:
            category_spending = debits.groupby('category')['amount'].sum()
            for category, amount in category_spending.items():
                if category != 'Transfer' and (amount / total_expense) > self.recommendation_rules['high_spending_category']['threshold']:
                    recommendations.append({
                        'category': 'Budgeting',
                        'recommendation': self.recommendation_rules['high_spending_category']['message'].format(category, amount / total_expense),
                        'impact': self.recommendation_rules['high_spending_category']['impact']
                    })

        # 2. Savings Rate Analysis
        if total_income > 0:
            savings_rate = (total_income - total_expense) / total_income
            if savings_rate < self.recommendation_rules['low_savings_rate']['threshold']:
                recommendations.append({
                    'category': 'Savings',
                    'recommendation': self.recommendation_rules['low_savings_rate']['message'].format(savings_rate),
                    'impact': self.recommendation_rules['low_savings_rate']['impact']
                })

        # 3. Frequent Small Transactions
        small_transactions = debits[debits['amount'] < self.recommendation_rules['frequent_small_transactions']['threshold_amount']]
        if len(small_transactions) > self.recommendation_rules['frequent_small_transactions']['threshold_count']:
            recommendations.append({
                'category': 'Spending Habits',
                'recommendation': self.recommendation_rules['frequent_small_transactions']['message'].format(
                    len(small_transactions), self.recommendation_rules['frequent_small_transactions']['threshold_amount']
                ),
                'impact': self.recommendation_rules['frequent_small_transactions']['impact']
            })
            
        # 4. Potential Subscriptions
        recurring_payments = debits.groupby('merchant').filter(lambda x: len(x) >= self.recommendation_rules['potential_subscriptions']['frequency'])
        potential_subscriptions = recurring_payments['merchant'].unique()
        for merchant in potential_subscriptions:
            if merchant != 'Unknown':
                recommendations.append({
                    'category': 'Subscriptions',
                    'recommendation': self.recommendation_rules['potential_subscriptions']['message'].format(merchant),
                    'impact': self.recommendation_rules['potential_subscriptions']['impact']
                })

        return recommendations
