# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
from datetime import datetime, timedelta
import json

# Import our custom modules
from src.pdf_processor import MemoryOptimizedUPIProcessor
from src.data_cleaner import MemoryOptimizedDataCleaner
from src.llm_analyzer import LightweightLLMAnalyzer
from src.recommendation_engine import FinancialRecommendationEngine
from src.utils import FileUtils, MemoryUtils, ValidationUtils
from config import Config

# Page configuration
st.set_page_config(
    page_title="UPI Financial Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .recommendation-item {
        background-color: #262730;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border-left: 3px solid #28a745;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

class UPIAnalyzerApp:
    def __init__(self):
        self.config = Config()
        self.processor = MemoryOptimizedUPIProcessor()
        self.cleaner = MemoryOptimizedDataCleaner()
        # Defer analyzer initialization
        self.analyzer = None
        self.recommender = FinancialRecommendationEngine()
        
        # Initialize session state
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None

    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">💰 UPI Financial Analyzer</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                AI-powered analysis of your UPI transactions • Get personalized financial insights
            </p>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render sidebar with system info and settings"""
        st.sidebar.header("📊 System Status")
        
        # Memory usage
        memory_info = MemoryUtils.get_memory_usage()
        st.sidebar.metric(
            "Memory Usage", 
            f"{memory_info['percentage']:.1f}%",
            f"{memory_info['used_gb']:.1f}GB / {memory_info['total_gb']:.1f}GB"
        )
        
        # System compatibility
        st.sidebar.header("⚙️ Settings")
        
        # File upload settings
        max_file_size = st.sidebar.slider("Max PDF Size (MB)", 1, 10, 5)
        
        # Analysis settings
        use_llm = st.sidebar.checkbox("Use AI Analysis", value=True, 
                                     help="Enable LLM-powered insights (requires API key)")
        
        # LLM provider selection
        llm_provider = st.sidebar.selectbox(
            "Choose AI Provider",
            options=['Gemini', 'OpenAI', 'Hugging Face'],
            index=2, # Set Hugging Face as default
            help="Select the LLM provider for analysis"
        )
        
        # Display API status
        st.sidebar.header("🔑 API Status")
        api_status = {
            "Gemini": bool(self.config.GEMINI_API_KEY),
            "OpenAI": bool(self.config.OPENAI_API_KEY),
            "Hugging Face": bool(self.config.HUGGINGFACE_API_TOKEN)
        }
        
        for api, status in api_status.items():
            emoji = "✅" if status else "❌"
            st.sidebar.write(f"{emoji} {api}")
        
        return max_file_size, use_llm, llm_provider

    def process_uploaded_file(self, uploaded_file, max_file_size):
        """Process uploaded PDF file"""
        try:
            # Validate file
            if uploaded_file.size > max_file_size * 2048 * 2048:
                st.error(f"File size exceeds {max_file_size}MB limit")
                return None
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Process PDF
            with st.spinner("🔄 Extracting transaction data from PDF..."):
                # Show progress
                progress_bar = st.progress(0)
                progress_bar.progress(25)
                
                # Extract transactions
                df_raw = self.processor.process_pdf(tmp_file_path)
                progress_bar.progress(50)
                
                # Clean data
                df_clean = self.cleaner.clean_dataframe(df_raw)
                progress_bar.progress(75)
                
                # Validate data
                validation_results = ValidationUtils.validate_transaction_data(df_clean)
                progress_bar.progress(100)
                
                # Cleanup temp file
                os.unlink(tmp_file_path)
                
                if not validation_results['is_valid']:
                    st.error("Data validation failed:")
                    for error in validation_results['errors']:
                        st.error(f"• {error}")
                    return None
                
                # Show warnings if any
                if validation_results['warnings']:
                    for warning in validation_results['warnings']:
                        st.warning(f"⚠️ {warning}")
                
                st.success(f"✅ Successfully processed {len(df_clean)} transactions!")
                return df_clean
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            return None

    def display_overview_metrics(self, df):
        """Display overview metrics"""
        st.header("📈 Transaction Overview")
        
        # Calculate metrics
        total_transactions = len(df)
        total_spent = df[df['transaction_type'] == 'debit']['amount'].sum()
        total_received = df[df['transaction_type'] == 'credit']['amount'].sum()
        avg_transaction = df['amount'].mean()
        date_range = f"{df['date'].min()} to {df['date'].max()}"
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", f"{total_transactions:,}")
        
        with col2:
            st.metric("Total Spent", f"₹{total_spent:,.2f}")
        
        with col3:
            st.metric("Total Received", f"₹{total_received:,.2f}")
        
        with col4:
            net_flow = total_received - total_spent
            st.metric("Net Flow", f"₹{net_flow:,.2f}", 
                     delta=f"₹{abs(net_flow):,.2f}")
        
        # Additional info
        st.info(f"📅 Date Range: {date_range} • 💳 Average Transaction: ₹{avg_transaction:.2f}")

    def create_visualizations(self, df):
        """Create interactive and customizable visualizations"""
        st.header("📊 Interactive Visual Analysis")

        # Add visualization options
        st.sidebar.header("🎨 Visualization Options")

        # Chart type options
        chart_type = st.sidebar.selectbox(
            "Chart Type",
            ["Default", "Dark Theme", "Light Theme", "High Contrast"],
            index=0
        )

        # Color scheme options
        color_scheme = st.sidebar.selectbox(
            "Color Scheme",
            ["Plotly", "Sequential", "Diverging", "Cyclical", "Qualitative"],
            index=0
        )

        # Add export options
        export_format = st.sidebar.selectbox(
            "Export Format",
            ["PNG", "JPEG", "SVG", "PDF"],
            index=0
        )

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Category Analysis",
            "📅 Time Trends",
            "📈 Transaction Patterns",
            "🏪 Merchant Analysis"
        ])

        # Apply theme based on selection
        if chart_type == "Dark Theme":
            plotly_theme = "plotly_dark"
        elif chart_type == "Light Theme":
            plotly_theme = "plotly_white"
        elif chart_type == "High Contrast":
            plotly_theme = "ggplot2"
        else:
            plotly_theme = None

        # Apply color scheme
        if color_scheme == "Sequential":
            color_seq = px.colors.sequential.Plasma
        elif color_scheme == "Diverging":
            color_seq = px.colors.diverging.Tealrose
        elif color_scheme == "Cyclical":
            color_seq = px.colors.cyclical.IceFire
        elif color_scheme == "Qualitative":
            color_seq = px.colors.qualitative.Plotly
        else:
            color_seq = None

        # Pass theme and color scheme to visualization methods
        with tab1:
            self.render_category_analysis(df, plotly_theme, color_seq)

        with tab2:
            self.render_time_trends(df, plotly_theme, color_seq)

        with tab3:
            self.render_transaction_patterns(df, plotly_theme, color_seq)

        with tab4:
            self.render_merchant_analysis(df, plotly_theme, color_seq)

        # Add export button
        st.sidebar.markdown("---")
        if st.sidebar.button("Export All Visualizations"):
            self.export_visualizations(df, export_format)

    def render_category_analysis(self, df, theme=None, color_seq=None):
        """Render category-wise analysis with customization options"""
        # Category-wise spending (pie chart)
        category_spending = df[df['transaction_type'] == 'debit'].groupby('category')['amount'].sum().reset_index()

        col1, col2 = st.columns(2)

        with col1:
            fig_pie = px.pie(
                category_spending,
                values='amount',
                names='category',
                title="Spending by Category",
                color_discrete_sequence=color_seq if color_seq else None
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')

            if theme:
                fig_pie.update_layout(template=theme)

            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Category bar chart
            fig_bar = px.bar(
                category_spending.sort_values('amount', ascending=True),
                x='amount',
                y='category',
                orientation='h',
                title="Category-wise Spending Amount",
                color_discrete_sequence=color_seq if color_seq else None
            )

            if theme:
                fig_bar.update_layout(template=theme)

            st.plotly_chart(fig_bar, use_container_width=True)

        # Add interactive controls
        st.sidebar.subheader("Category Analysis Options")

        # Add filter for categories
        all_categories = category_spending['category'].unique()
        selected_categories = st.sidebar.multiselect(
            "Filter Categories",
            options=all_categories,
            default=all_categories
        )

        # Filter data based on selection
        filtered_df = df[df['category'].isin(selected_categories)]

        # Category statistics table
        st.subheader("Category Statistics")
        category_stats = filtered_df[filtered_df['transaction_type'] == 'debit'].groupby('category').agg({
            'amount': ['sum', 'mean', 'count'],
            'merchant': 'nunique'
        }).round(2)

        category_stats.columns = ['Total Amount', 'Average Amount', 'Transaction Count', 'Unique Merchants']
        category_stats = category_stats.sort_values('Total Amount', ascending=False)
        st.dataframe(category_stats)

    def render_time_trends(self, df, theme=None, color_seq=None):
        """Render time-based analysis with customization options"""
        # Add interactive controls
        st.sidebar.subheader("Time Trends Options")

        # Date range selector
        min_date = df['date'].min()
        max_date = df['date'].max()

        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )

        # Convert date column to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        # Filter data based on date range
        filtered_df = df[(df['date'] >= pd.Timestamp(date_range[0])) &
                         (df['date'] <= pd.Timestamp(date_range[1]))]

        # Daily spending trend
        daily_spending = filtered_df[filtered_df['transaction_type'] == 'debit'].groupby('date')['amount'].sum().reset_index()

        fig_daily = px.line(
            daily_spending,
            x='date',
            y='amount',
            title="Daily Spending Trend",
            color_discrete_sequence=color_seq if color_seq else None
        )
        fig_daily.update_layout(xaxis_title="Date", yaxis_title="Amount (₹)")

        if theme:
            fig_daily.update_layout(template=theme)

        st.plotly_chart(fig_daily, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            # Monthly spending
            filtered_df['month_year'] = filtered_df['datetime'].dt.to_period('M')
            monthly_spending = filtered_df[filtered_df['transaction_type'] == 'debit'].groupby('month_year')['amount'].sum()

            fig_monthly = px.bar(
                x=monthly_spending.index.astype(str),
                y=monthly_spending.values,
                title="Monthly Spending",
                color_discrete_sequence=color_seq if color_seq else None
            )
            fig_monthly.update_layout(xaxis_title="Month", yaxis_title="Amount (₹)")

            if theme:
                fig_monthly.update_layout(template=theme)

            st.plotly_chart(fig_monthly, use_container_width=True)

        with col2:
            # Day of week analysis
            filtered_df['day_name'] = filtered_df['datetime'].dt.day_name()
            day_spending = filtered_df[filtered_df['transaction_type'] == 'debit'].groupby('day_name')['amount'].sum()

            # Reorder days
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_spending = day_spending.reindex(days_order)

            fig_dow = px.bar(
                x=day_spending.index,
                y=day_spending.values,
                title="Spending by Day of Week",
                color_discrete_sequence=color_seq if color_seq else None
            )

            if theme:
                fig_dow.update_layout(template=theme)

            st.plotly_chart(fig_dow, use_container_width=True)

    def render_transaction_patterns(self, df, theme=None, color_seq=None):
        """Render transaction pattern analysis with customization options"""
        # Add interactive controls
        st.sidebar.subheader("Transaction Patterns Options")

        # Amount range selector
        min_amount = df['amount'].min()
        max_amount = df['amount'].max()

        amount_range = st.sidebar.slider(
            "Select Amount Range",
            min_value=float(min_amount),
            max_value=float(max_amount),
            value=(float(min_amount), float(max_amount))
        )

        # Filter data based on amount range
        filtered_df = df[(df['amount'] >= amount_range[0]) &
                         (df['amount'] <= amount_range[1])]

        col1, col2 = st.columns(2)

        with col1:
            # Transaction amount distribution
            fig_hist = px.histogram(
                filtered_df,
                x='amount',
                nbins=30,
                title="Transaction Amount Distribution",
                color_discrete_sequence=color_seq if color_seq else None
            )
            fig_hist.update_layout(xaxis_title="Amount (₹)", yaxis_title="Frequency")

            if theme:
                fig_hist.update_layout(template=theme)

            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            # Transaction type distribution
            type_counts = filtered_df['transaction_type'].value_counts()
            fig_type = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Transaction Type Distribution",
                color_discrete_sequence=color_seq if color_seq else None
            )

            if theme:
                fig_type.update_layout(template=theme)

            st.plotly_chart(fig_type, use_container_width=True)

        # Hour-wise transaction pattern
        if 'hour' in filtered_df.columns:
            hourly_pattern = filtered_df.groupby('hour')['amount'].agg(['sum', 'count']).reset_index()

            fig_hourly = make_subplots(specs=[[{"secondary_y": True}]])

            fig_hourly.add_trace(
                go.Bar(
                    x=hourly_pattern['hour'],
                    y=hourly_pattern['sum'],
                    name='Total Amount',
                    marker_color=color_seq[0] if color_seq else None
                ),
                secondary_y=False
            )

            fig_hourly.add_trace(
                go.Scatter(
                    x=hourly_pattern['hour'],
                    y=hourly_pattern['count'],
                    mode='lines+markers',
                    name='Transaction Count',
                    line_color=color_seq[1] if color_seq and len(color_seq) > 1 else None
                ),
                secondary_y=True
            )

            fig_hourly.update_layout(
                title="Hourly Transaction Pattern",
                xaxis_title="Hour of Day",
                yaxis_title="Amount (₹)",
                legend_title="Metrics"
            )
            fig_hourly.update_yaxes(title_text="Transaction Count", secondary_y=True)

            if theme:
                fig_hourly.update_layout(template=theme)

            st.plotly_chart(fig_hourly, use_container_width=True)

    def render_merchant_analysis(self, df, theme=None, color_seq=None):
        """Render merchant-specific analysis with customization options"""
        # Add interactive controls
        st.sidebar.subheader("Merchant Analysis Options")

        # Merchant filter
        all_merchants = df['merchant'].unique()
        selected_merchants = st.sidebar.multiselect(
            "Filter Merchants",
            options=all_merchants,
            default=all_merchants
        )

        # Filter data based on selection
        filtered_df = df[df['merchant'].isin(selected_merchants)]

        st.header("🏪 Merchant Analysis")

        # Top merchants by transaction count
        top_merchants = filtered_df['merchant'].value_counts().head(10).reset_index()
        top_merchants.columns = ['Merchant', 'Transaction Count']

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top 10 Merchants by Transactions")
            fig_merchants = px.bar(
                top_merchants,
                x='Transaction Count',
                y='Merchant',
                orientation='h',
                color_discrete_sequence=color_seq if color_seq else None
            )

            if theme:
                fig_merchants.update_layout(template=theme)

            st.plotly_chart(fig_merchants, use_container_width=True)

        with col2:
            st.subheader("Merchant Statistics")
            merchant_stats = filtered_df.groupby('merchant').agg({
                'amount': ['sum', 'mean', 'count'],
                'category': pd.Series.mode
            }).round(2)

            merchant_stats.columns = ['Total Amount', 'Average Amount', 'Transaction Count', 'Primary Category']
            merchant_stats = merchant_stats.sort_values('Total Amount', ascending=False).head(10)
            st.dataframe(merchant_stats)

        # Add merchant spending trend visualization
        st.subheader("Merchant Spending Trends")

        # Get top 5 merchants by spending
        top_merchants_spending = filtered_df.groupby('merchant')['amount'].sum().nlargest(5).index

        # Filter data for top merchants
        top_merchants_data = filtered_df[filtered_df['merchant'].isin(top_merchants_spending)]

        # Create line chart for spending trends
        merchant_trends = top_merchants_data.groupby(['merchant', 'date'])['amount'].sum().reset_index()

        fig_trends = px.line(
            merchant_trends,
            x='date',
            y='amount',
            color='merchant',
            title="Top Merchants Spending Trends",
            color_discrete_sequence=color_seq if color_seq else None
        )

        if theme:
            fig_trends.update_layout(template=theme)

        fig_trends.update_layout(
            xaxis_title="Date",
            yaxis_title="Amount (₹)",
            legend_title="Merchant"
        )

        st.plotly_chart(fig_trends, use_container_width=True)

    def export_visualizations(self, df, format='PNG'):
        """Export all visualizations as image files in a zip archive"""
        import tempfile
        import zipfile
        from io import BytesIO
        
        with st.spinner(f"🔄 Exporting visualizations as {format} files..."):
            try:
                # Create temporary directory
                with tempfile.TemporaryDirectory() as tmp_dir:
                    zip_buffer = BytesIO()
                    filenames = []
                    
                    # Recreate all visualizations with current settings
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        # Category Analysis
                        fig_pie, fig_bar = self._create_category_visualizations(df)
                        filenames.extend([
                            self._save_figure(fig_pie, tmp_dir, 'category_pie', format),
                            self._save_figure(fig_bar, tmp_dir, 'category_bar', format)
                        ])
                        
                        # Time Trends
                        fig_daily, fig_monthly, fig_dow = self._create_time_visualizations(df)
                        filenames.extend([
                            self._save_figure(fig_daily, tmp_dir, 'daily_trend', format),
                            self._save_figure(fig_monthly, tmp_dir, 'monthly_spending', format),
                            self._save_figure(fig_dow, tmp_dir, 'day_of_week', format)
                        ])
                        
                        # Transaction Patterns
                        fig_hist, fig_type, fig_hourly = self._create_transaction_visualizations(df)
                        filenames.extend([
                            self._save_figure(fig_hist, tmp_dir, 'amount_distribution', format),
                            self._save_figure(fig_type, tmp_dir, 'transaction_types', format),
                            self._save_figure(fig_hourly, tmp_dir, 'hourly_pattern', format) if fig_hourly else None
                        ])
                        
                        # Merchant Analysis
                        fig_merchants, fig_trends = self._create_merchant_visualizations(df)
                        filenames.extend([
                            self._save_figure(fig_merchants, tmp_dir, 'top_merchants', format),
                            self._save_figure(fig_trends, tmp_dir, 'merchant_trends', format)
                        ])
                        
                        # Add files to zip
                        for filename in filter(None, filenames):
                            zip_file.write(filename, arcname=filename.split('/')[-1])
                    
                    # Offer zip file for download
                    st.success("✅ Visualizations exported successfully!")
                    st.download_button(
                        label="📥 Download All Visualizations",
                        data=zip_buffer.getvalue(),
                        file_name=f"upi_visualizations_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                        mime="application/zip"
                    )
                    
            except Exception as e:
                st.error(f"Failed to export visualizations: {str(e)}")

    def _save_figure(self, fig, directory, base_name, format):
        """Helper to save a figure to a file and return the path"""
        if fig is None:
            return None
        filename = f"{directory}/{base_name}.{format.lower()}"
        fig.write_image(filename, format=format.lower())
        return filename

    def _create_category_visualizations(self, df):
        """Recreate category analysis visualizations"""
        category_spending = df[df['transaction_type'] == 'debit'].groupby('category')['amount'].sum().reset_index()
        
        fig_pie = px.pie(
            category_spending,
            values='amount',
            names='category',
            title="Spending by Category"
        )
        
        fig_bar = px.bar(
            category_spending.sort_values('amount', ascending=True),
            x='amount',
            y='category',
            orientation='h',
            title="Category-wise Spending Amount"
        )
        
        return fig_pie, fig_bar

    def _create_time_visualizations(self, df):
        """Recreate time trend visualizations"""
        # Daily spending
        daily_spending = df[df['transaction_type'] == 'debit'].groupby('date')['amount'].sum().reset_index()
        fig_daily = px.line(daily_spending, x='date', y='amount', title="Daily Spending Trend")
        
        # Monthly spending
        df['month_year'] = df['datetime'].dt.to_period('M')
        monthly_spending = df[df['transaction_type'] == 'debit'].groupby('month_year')['amount'].sum()
        fig_monthly = px.bar(
            x=monthly_spending.index.astype(str),
            y=monthly_spending.values,
            title="Monthly Spending"
        )
        
        # Day of week
        df['day_name'] = df['datetime'].dt.day_name()
        day_spending = df[df['transaction_type'] == 'debit'].groupby('day_name')['amount'].sum()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_spending = day_spending.reindex(days_order)
        fig_dow = px.bar(x=day_spending.index, y=day_spending.values, title="Spending by Day of Week")
        
        return fig_daily, fig_monthly, fig_dow

    def _create_transaction_visualizations(self, df):
        """Recreate transaction pattern visualizations"""
        # Amount distribution
        fig_hist = px.histogram(df, x='amount', nbins=30, title="Transaction Amount Distribution")
        
        # Transaction types
        type_counts = df['transaction_type'].value_counts()
        fig_type = px.pie(values=type_counts.values, names=type_counts.index, title="Transaction Type Distribution")
        
        # Hourly pattern if available
        fig_hourly = None
        if 'hour' in df.columns:
            hourly_pattern = df.groupby('hour')['amount'].agg(['sum', 'count']).reset_index()
            fig_hourly = make_subplots(specs=[[{"secondary_y": True}]])
            fig_hourly.add_trace(
                go.Bar(x=hourly_pattern['hour'], y=hourly_pattern['sum'], name='Total Amount'),
                secondary_y=False
            )
            fig_hourly.add_trace(
                go.Scatter(x=hourly_pattern['hour'], y=hourly_pattern['count'], mode='lines+markers', name='Transaction Count'),
                secondary_y=True
            )
            fig_hourly.update_layout(
                title="Hourly Transaction Pattern",
                xaxis_title="Hour of Day",
                yaxis_title="Amount (₹)",
                legend_title="Metrics"
            )
            fig_hourly.update_yaxes(title_text="Transaction Count", secondary_y=True)
        
        return fig_hist, fig_type, fig_hourly

    def _create_merchant_visualizations(self, df):
        """Recreate merchant analysis visualizations"""
        # Top merchants
        top_merchants = df['merchant'].value_counts().head(10).reset_index()
        top_merchants.columns = ['Merchant', 'Transaction Count']
        fig_merchants = px.bar(
            top_merchants,
            x='Transaction Count',
            y='Merchant',
            orientation='h',
            title="Top 10 Merchants by Transactions"
        )
        
        # Merchant trends
        top_merchants_spending = df.groupby('merchant')['amount'].sum().nlargest(5).index
        top_merchants_data = df[df['merchant'].isin(top_merchants_spending)]
        merchant_trends = top_merchants_data.groupby(['merchant', 'date'])['amount'].sum().reset_index()
        fig_trends = px.line(
            merchant_trends,
            x='date',
            y='amount',
            color='merchant',
            title="Top Merchants Spending Trends"
        )
        
        return fig_merchants, fig_trends

    def generate_ai_insights(self, df, llm_provider):
        """Generate AI-powered insights"""
        with st.spinner(f"🧠 Generating AI insights using {llm_provider}..."):
            try:
                # Initialize analyzer with selected provider
                self.analyzer = LightweightLLMAnalyzer(llm_provider=llm_provider)
                
                insights = self.analyzer.analyze_spending_patterns(df)
                st.session_state.analysis_results = insights
                
                st.header("🤖 AI-Powered Insights")
                
                recommendations = self.analyzer.generate_recommendations(insights)
                
                for rec in recommendations:
                    st.markdown(f"""
                    <div class="recommendation-item">
                        {rec}
                    </div>
                    """, unsafe_allow_html=True)
                
                return True
            except Exception as e:
                st.error(f"Failed to generate AI insights: {str(e)}")
                return False

    def generate_recommendations(self, df):
        """Generate financial recommendations"""
        with st.spinner("💡 Generating recommendations..."):
            try:
                recommendations = self.recommender.generate_recommendations(df)
                
                st.header("📌 Personalized Recommendations")
                
                for rec in recommendations:
                    st.markdown(f"""
                    <div class="recommendation-item">
                        <strong>{rec['category']}:</strong> {rec['recommendation']}
                        <br><small>Impact: {rec['impact']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                return True
            except Exception as e:
                st.error(f"Failed to generate recommendations: {str(e)}")
                return False

    def run(self):
        """Main application runner"""
        self.render_header()
        
        # Get settings from sidebar
        max_file_size, use_llm, llm_provider = self.render_sidebar()
        
        # File upload section
        st.header("📤 Upload Your UPI Statement")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            # Process the uploaded file
            df = self.process_uploaded_file(uploaded_file, max_file_size)
            
            if df is not None:
                # Store in session state
                st.session_state.processed_data = df
                
                # Display overview metrics
                self.display_overview_metrics(df)
                
                # Create visualizations
                self.create_visualizations(df)
                
                # Generate AI insights if enabled
                if use_llm:
                    self.generate_ai_insights(df, llm_provider)
                
                # Generate recommendations
                self.generate_recommendations(df)
                
                # Add download button for processed data
                st.download_button(
                    label="📥 Download Processed Data",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name='processed_upi_transactions.csv',
                    mime='text/csv'
                )

# Run the application
if __name__ == "__main__":
    app = UPIAnalyzerApp()
    app.run()
