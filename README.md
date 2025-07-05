# Personal UPI Usage and Financial Analyzer using LLMs

## 🔍 Overview

An AI-powered FinTech application that extracts, processes, and analyzes UPI transaction statements (from Paytm, GPay, PhonePe, etc.) and delivers personalized financial insights using Large Language Models (LLMs). The system provides users with actionable advice, spending pattern summaries, and budgeting tips, all from a user-friendly interface.

---

## 💡 Features

- 📄 PDF Statement Parsing from multiple UPI apps  
- 🧹 Data Cleaning and Structuring with Pandas  
- 📊 Spending Pattern & Trend Analysis  
- 🧠 LLM-Powered Recommendations (OpenAI / Hugging Face)  
- 🔄 Langflow Integration for modular AI workflows  
- 🌐 Gradio / Streamlit Web App Interface  
- 🚀 Deployed on Hugging Face Spaces  

---

## 📁 Project Structure

upi-financial-analyzer/
│
├── data/ # Sample UPI PDFs
├── src/
│ ├── parser.py # PDF parsing logic
│ ├── data_cleaner.py # Data normalization and transformation
│ ├── analyzer.py # Spending pattern analysis
│ ├── recommender.py # LLM-based advice generation
│ └── app.py # Streamlit/Gradio interface
│
├── prompts/ # LLM prompt templates
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── .huggingface/ # Deployment config (for HF Spaces)

yaml
Copy
Edit

---

## 🚀 How It Works

1. **Upload UPI Statement PDFs** from Paytm, PhonePe, or GPay  
2. **Extract and Clean Data** using `pdfplumber` and `pandas`  
3. **Analyze Financial Patterns** — detect wasteful spending, top merchants, trends  
4. **Generate Recommendations** with LLMs using Langflow chain  
5. **View Insights in Real-Time** on the Streamlit/Gradio dashboard  

---

## 🛠 Tech Stack

- **Python 3.10+**
- **pdfplumber / PyMuPDF** – PDF parsing
- **Pandas** – Data wrangling
- **OpenAI / Hugging Face Transformers** – LLMs
- **Langflow** – LLM flow orchestration
- **Gradio / Streamlit** – UI for insight generation
- **Hugging Face Spaces** – Deployment platform

---

## 🔧 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/upi-financial-analyzer.git
cd upi-financial-analyzer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run src/app.py
# or
python src/app.py  # for Gradio
🎯 Use Cases
Personal Budgeting Assistant

Spending Habit Analyzer

Unified Financial Tracker for UPI

Expense Optimization and Alerts

📈 Evaluation Metrics
✅ Accuracy of Data Extraction

🧠 Relevance of LLM Recommendations

📊 Completeness of Structured Data

⏱️ Speed of Insight Generation

⭐ User Feedback on Financial Suggestions

🧠 Skills Learned
PDF Data Extraction & Normalization

Langflow-based LLM Chain Design

Prompt Engineering

NLP & Financial Data Analysis

Web App Deployment on Hugging Face Spaces

📄 License
This project is open-source and available under the MIT License.

🙋‍♂️ Author
Karthick Gurusamy
For questions or collaboration: LinkedIn

yaml
Copy
Edit

---

Let me know if you'd like a version customized for GitHub Pages or with sample screenshots/logos included.
