import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta
from streamlit_date_picker import date_range_picker, PickerType

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ETRM Business Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STATIC LOG FILE LOADER (UNCHANGED) ---
@st.cache_data
def load_validation_logs(file_path="validation_logs.json"):
    """
    Loads and parses the validation log JSON file.
    Returns a DataFrame of log entries and a summary object.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        log_entries = data.get('validation_log', {}).get('log_entries', [])
        if not log_entries:
            st.warning(f"No log entries found in '{file_path}'.")
            return pd.DataFrame(), None

        df = pd.DataFrame(log_entries)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        summary_log = df[df['rule_category'] == 'FINAL']
        df = df[df['rule_category'] != 'FINAL']
        
        return df.dropna(subset=['timestamp']), summary_log

    except FileNotFoundError:
        st.error(f"**Error:** The log file '{file_path}' was not found. Please create it in the same directory.")
        return pd.DataFrame(), None
    except json.JSONDecodeError:
        st.error(f"**Error:** The file '{file_path}' is not a valid JSON file.")
        return pd.DataFrame(), None
    except Exception as e:
        st.error(f"An unexpected error occurred while reading the log file: {e}")
        return pd.DataFrame(), None

# --- NEW: STATIC JSON DATA LOADER ---
@st.cache_data
def load_data_from_json(file_path="business_etrm_data.json"):
    """
    Loads, parses, and flattens trade data from the local JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_trades = []
        trade_data_sets = data.get("TradeDataSets", [])

        if not trade_data_sets:
            st.warning(f"No 'TradeDataSets' found in '{file_path}'.")
            return pd.DataFrame()

        for data_set in trade_data_sets:
            tag = data_set.get("Tag", "N/A")
            trades = data_set.get("Trades", {}).get("TradeList", [])

            for trade in trades:
                flat_trade = trade.copy()
                instrument_data = flat_trade.pop('Instrument', {})
                flat_trade.update(instrument_data)
                flat_trade['Tag'] = tag
                all_trades.append(flat_trade)

        if not all_trades:
            st.warning("No trades found within the JSON file's 'TradeDataSets'.")
            return pd.DataFrame()

        trades_df = pd.DataFrame(all_trades)

        # --- Data Cleaning and Type Conversion ---
        trades_df['Timestamp'] = pd.to_datetime(trades_df['Timestamp'], utc=True, errors='coerce').dt.tz_localize(None)
        trades_df['Expiry'] = pd.to_datetime(trades_df['Expiry'], errors='coerce')
        
        numeric_cols = ['Quantity', 'Price']
        for col in numeric_cols:
            trades_df[col] = pd.to_numeric(trades_df[col], errors='coerce')
        
        trades_df = trades_df.dropna(subset=['Quantity', 'Price', 'Timestamp'])
        
        string_cols = ['Side', 'Exchange', 'Currency', 'Trader', 'Counterparty', 'Tag', 'Symbol', 'ProductType']
        for col in string_cols:
            if col in trades_df.columns:
                trades_df[col] = trades_df[col].fillna('N/A')
        
        return trades_df.sort_values(by="Timestamp", ascending=False)
        
    except FileNotFoundError:
        st.error(f"**Error:** The data file '{file_path}' was not found. Please ensure it is in the same directory as the script.")
        return pd.DataFrame()
    except json.JSONDecodeError:
        st.error(f"**Error:** The file '{file_path}' is not a valid JSON file.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred while processing the data file: {e}")
        return pd.DataFrame()

# --- MAIN DATA LOADING ---
st.sidebar.title("📄 Data Source")
st.sidebar.info("✅ API Connected.")

# Load the data from the local file
with st.spinner("🔄 Loading trade data from file..."):
    trades_df = load_data_from_json("business_etrm_data.json")

# Graceful handling if data loading fails
if trades_df is None or trades_df.empty:
    st.warning("⚠️ **No trade data loaded!** The JSON file might be empty, malformed, or not found.")
    st.stop()

st.sidebar.info(f"📊 Loaded {len(trades_df):,} trades")

# --- SESSION STATE & SIDEBAR ---
if 'active_view' not in st.session_state:
    st.session_state.active_view = 'home'

def set_view(view_name):
    st.session_state.active_view = view_name

st.sidebar.title("📊 Dashboard Controls")
st.sidebar.info("Filter the trade data across all views.")

# Check if we have timestamp data
if 'Timestamp' not in trades_df.columns or trades_df['Timestamp'].isna().all():
    st.error("❌ **Missing Timestamp Data!** Cannot create time-based filters.")
    st.stop()

min_ts, max_ts = trades_df['Timestamp'].min(), trades_df['Timestamp'].max()
default_start = max_ts - timedelta(days=365) # Widen default range for static file
default_end = max_ts

st.sidebar.markdown("### ⏰ Time Granularity")
time_granularity = st.sidebar.radio(
    "Select time precision:", options=["Hourly", "Daily"], index=1, horizontal=True
)

if time_granularity == "Hourly":
    refresh_buttons = [
        {'button_name': '⏰ Last 6 Hours', 'refresh_value': timedelta(hours=6)},
        {'button_name': '⏰ Last 12 Hours', 'refresh_value': timedelta(hours=12)},
        {'button_name': '⏰ Last 24 Hours', 'refresh_value': timedelta(hours=24)},
        {'button_name': '⏰ Last 3 Days', 'refresh_value': timedelta(days=3)}
    ]
    picker_type = PickerType.time
    st.sidebar.markdown("### ⏰ Date & Time Range Selection")
else:
    refresh_buttons = [
        {'button_name': '📅 Last 7 Days', 'refresh_value': timedelta(days=7)},
        {'button_name': '📅 Last 30 Days', 'refresh_value': timedelta(days=30)},
        {'button_name': '📅 Last 90 Days', 'refresh_value': timedelta(days=90)},
        {'button_name': '📅 All Time', 'refresh_value': max_ts - min_ts}
    ]
    picker_type = PickerType.date
    st.sidebar.markdown("### 📅 Date Range Selection")

date_range_result = date_range_picker(
    picker_type=picker_type, start=default_start, end=default_end,
    key=f'trade_{time_granularity.lower()}_range_picker', refresh_buttons=refresh_buttons
)

if date_range_result:
    start_date, end_date = date_range_result
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    if time_granularity == "Daily": end_date += pd.Timedelta(days=1)
else:
    start_date, end_date = pd.to_datetime(default_start), pd.to_datetime(default_end)
    if time_granularity == "Daily": end_date += pd.Timedelta(days=1)

if time_granularity == "Hourly":
    st.sidebar.success(f"⏰ Selected: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
else:
    st.sidebar.success(f"📊 Selected: {start_date.strftime('%Y-%m-%d')} to {(end_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}")

# --- FILTERS ---
st.sidebar.markdown("### 🏷️ Additional Filters")

def create_safe_filter(column_name, label, default_all=True):
    if column_name in trades_df.columns and not trades_df[column_name].isna().all():
        unique_values = sorted(trades_df[column_name].dropna().unique())
        if len(unique_values) > 0:
            default_values = list(unique_values) if default_all else []
            return st.sidebar.multiselect(label, options=unique_values, default=default_values)
    return []

tag_filter = create_safe_filter('Tag', "Filter by Tag")
product_type_filter = create_safe_filter('ProductType', "Filter by Product Type")
trader_filter = create_safe_filter('Trader', "Filter by Trader")

# Apply filters
filtered_trades = trades_df[
    (trades_df['Timestamp'] >= start_date) & 
    (trades_df['Timestamp'] <= end_date) &
    (trades_df['Tag'].isin(tag_filter) if tag_filter else True) &
    (trades_df['ProductType'].isin(product_type_filter) if product_type_filter else True) &
    (trades_df['Trader'].isin(trader_filter) if trader_filter else True)
]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Filter Summary")
st.sidebar.metric("Filtered Trades", f"{len(filtered_trades):,}")
total_volume = filtered_trades['Quantity'].sum() if not filtered_trades.empty else 0
st.sidebar.metric("Total Volume", f"{total_volume:,.0f}")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# --- VIEW FUNCTIONS ---
def render_kpi_overview():
    st.button("⬅️ Back to Homepage", on_click=set_view, args=['home'])
    st.header("📈 KPI Overview")
    
    if filtered_trades.empty:
        st.warning("No data available for the selected filters.")
        return
    
    total_trades = len(filtered_trades)
    total_quantity = filtered_trades['Quantity'].sum()
    unique_counterparties = filtered_trades['Counterparty'].nunique()
    avg_price = filtered_trades['Price'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades Executed", f"{total_trades:,}")
    col2.metric("Total Quantity Traded", f"{total_quantity:,.0f}")
    col3.metric("Unique Counterparties", f"{unique_counterparties}")
    col4.metric("Average Price", f"${avg_price:,.2f}")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        # ENHANCEMENT: Changed from Tag (only 1 in data) to a more useful Buy/Sell pie chart
        st.subheader("Buy vs. Sell Distribution")
        side_counts = filtered_trades['Side'].value_counts()
        if not side_counts.empty:
            fig_pie = px.pie(
                side_counts,
                names=side_counts.index,
                values=side_counts.values,
                title='Trade Count by Side',
                hole=0.6,
                color_discrete_map={'Buy': '#2ca02c', 'Sell': '#d62728'} # Green for Buy, Red for Sell
            )
            fig_pie.update_traces(textposition='outside', textinfo='percent+label')
            fig_pie.update_layout(showlegend=False, height=400, margin=dict(l=20, r=20, t=60, b=20),
                                 annotations=[dict(text='Side', x=0.5, y=0.5, font_size=20, showarrow=False)])
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No side data available.")
            
    with col2:
        st.subheader("Traded Quantity by Product Type")
        qty_by_prod = filtered_trades.groupby('ProductType')['Quantity'].sum().sort_values(ascending=False)
        if not qty_by_prod.empty:
            fig_bar = px.bar(qty_by_prod, x=qty_by_prod.index, y=qty_by_prod.values, 
                           labels={'x': 'Product Type', 'y': 'Total Quantity'}, 
                           color=qty_by_prod.index, text_auto='.2s')
            fig_bar.update_layout(title_text='Volume Breakdown', showlegend=False, height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No product type data available.")

def render_data_quality_view():
    st.button("⬅️ Back to Homepage", on_click=set_view, args=['home'])
    st.header("🚦 Data Quality & Validation")
    st.markdown("Analyze the completeness and structure of the trade data.")
    
    if filtered_trades.empty:
        st.warning("No data available for the selected filters.")
        return
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Trade Volume Hierarchy")
        # ENHANCEMENT: Added color and better title for clarity
        fig_treemap = px.treemap(
            filtered_trades,
            path=[px.Constant("All Trades"), 'Tag', 'ProductType', 'Trader'],
            values='Quantity',
            color='ProductType',
            hover_data=['Price'],
            title="<b>Trade Volume: Tag → Product → Trader</b>",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_treemap.update_traces(textinfo="label+value+percent parent")
        st.plotly_chart(fig_treemap, use_container_width=True)
            
    with col2:
        st.subheader("Data Completeness")
        na_sides = filtered_trades[filtered_trades['Side'] == 'N/A']
        side_pct = len(na_sides)/len(filtered_trades)*100 if len(filtered_trades) > 0 else 0
        st.metric("Trades with 'N/A' Side", f"{len(na_sides)} ({side_pct:.1f}%)")
        
        na_exchange = filtered_trades[filtered_trades['Exchange'] == 'N/A']
        exchange_pct = len(na_exchange)/len(filtered_trades)*100 if len(filtered_trades) > 0 else 0
        st.metric("Trades with 'N/A' Exchange", f"{len(na_exchange)} ({exchange_pct:.1f}%)")
        
        if not na_sides.empty:
            st.markdown("**Trades flagged for review (N/A Side):**")
            display_cols = ['Timestamp', 'TradeID', 'ProductType', 'Trader']
            st.dataframe(na_sides[display_cols].head(), use_container_width=True, hide_index=True)

def render_market_insights():
    st.button("⬅️ Back to Homepage", on_click=set_view, args=['home'])
    st.header("📊 Market & Price Insights")
    st.markdown("Analyze price trends and relationships from the executed trades.")
    
    if filtered_trades.empty:
        st.warning("No data available for the selected filters.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Price Evolution Over Time")
        top_symbols = filtered_trades['Symbol'].value_counts().nlargest(5).index
        price_plot_df = filtered_trades[filtered_trades['Symbol'].isin(top_symbols)]
        if not price_plot_df.empty:
            fig_line = px.line(price_plot_df.sort_values('Timestamp'), 
                             x='Timestamp', y='Price', color='Symbol', 
                             title='Price Trends for Top 5 Symbols', markers=True)
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No price trend data available.")
            
    with col2:
        st.subheader("Price vs. Quantity by Product Type")
        scatter_data = filtered_trades[filtered_trades['Quantity'] > 0]
        if not scatter_data.empty:
            # ENHANCEMENT: Added hover_data, size_max, and better labels
            fig_scatter = px.scatter(
                scatter_data, x='Quantity', y='Price',
                color='ProductType', size='Quantity',
                size_max=50,
                hover_name='TradeID',
                hover_data=['Trader', 'Counterparty', 'Symbol'],
                title='Trade Price vs. Quantity (Log Scale)',
                log_x=True,
                labels={'Quantity': 'Trade Quantity (Log)', 'Price': 'Execution Price ($)'}
            )
            fig_scatter.update_layout(legend_title_text='Product')
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No valid price/quantity data for scatter plot.")

def render_trade_blotter():
    st.button("⬅️ Back to Homepage", on_click=set_view, args=['home'])
    st.header("📋 Detailed Trade Blotter")
    st.markdown(f"Displaying **{len(filtered_trades)}** trades for the selected criteria.")
    
    if filtered_trades.empty:
        st.warning("No trades match the selected filters.")
        return
    
    display_cols = ['Timestamp', 'TradeID', 'Tag', 'ProductType', 'Symbol', 'Side', 
                    'Quantity', 'Price', 'Currency', 'Trader', 'Counterparty']
    
    column_config = {
        "Timestamp": st.column_config.DatetimeColumn("Execution Time", format="YYYY-MM-DD HH:mm"),
        "Quantity": st.column_config.NumberColumn("Qty", format="%d"),
        "Price": st.column_config.NumberColumn("Price", format="$%.3f")
    }
    
    st.dataframe(
        filtered_trades[display_cols], 
        use_container_width=True, 
        hide_index=True,
        column_config=column_config
    )

def render_validation_logs():
    st.button("⬅️ Back to Homepage", on_click=set_view, args=['home'])
    st.header("🔬 Validation Log Analysis Dashboard")
    st.markdown("Visualizing the results from the data ingestion and validation pipeline.")

    logs_df, _ = load_validation_logs()

    if logs_df.empty:
        return

    st.markdown("---")
    st.subheader("Overall Run Summary")
    total_success = logs_df[logs_df['status'] == 'SUCCESS'].shape[0]
    total_failure = logs_df[logs_df['status'] == 'FAILURE'].shape[0]
    total_logs = len(logs_df)
    success_rate = (total_success / total_logs * 100) if total_logs > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Log Entries", f"{total_logs}")
    col2.metric("✅ Successful Steps", f"{total_success}")
    col3.metric("❌ Failed Steps", f"{total_failure}")
    col4.metric("Success Rate", f"{success_rate:.2f}%")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Success vs. Failure")
        status_counts = logs_df['status'].value_counts()
        colors = {'SUCCESS': 'mediumseagreen', 'FAILURE': 'indianred'}
        fig_pie = go.Figure(data=[go.Pie(
            labels=status_counts.index, values=status_counts.values, hole=.6,
            marker_colors=[colors.get(k, 'lightgrey') for k in status_counts.index],
            pull=[0.05 if label == 'FAILURE' else 0 for label in status_counts.index]
        )])
        fig_pie.update_layout(title_text='Overall Status Breakdown', showlegend=True, height=400, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Failures by Rule Category")
        failures_df = logs_df[logs_df['status'] == 'FAILURE']
        category_counts = failures_df['rule_category'].value_counts().sort_values(ascending=True)
        if not category_counts.empty:
            fig_bar = go.Figure(go.Bar(y=category_counts.index, x=category_counts.values, orientation='h', marker_color='indianred', text=category_counts.values, textposition='auto'))
            fig_bar.update_layout(title_text='Count of Failures per Category', xaxis_title='Number of Failures', yaxis_title='Rule Category', height=400, margin=dict(l=20, r=20, t=60, b=20), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.success("🎉 No failures recorded!")

    st.markdown("---")
    st.subheader("Detailed Failure Log")
    failures_df = logs_df[logs_df['status'] == 'FAILURE'].copy().sort_values(by='timestamp', ascending=False)
    if failures_df.empty:
        st.success("No failures to display.")
    else:
        failures_df['Message'] = failures_df['details'].apply(lambda x: x.get('message', ''))
        failures_df['Affected Field'] = failures_df['details'].apply(lambda x: x.get('field', 'N/A'))
        failures_df['Invalid Value'] = failures_df['details'].apply(lambda x: str(x.get('invalid_value', 'N/A')))
        failures_df['Trade ID'] = failures_df['details'].apply(lambda x: x.get('record_identifier', {}).get('tradeid', 'Global'))
        
        categories = ['All Categories'] + sorted(failures_df['rule_category'].unique().tolist())
        selected_category = st.selectbox("Filter failures by Rule Category:", options=categories)

        display_df = failures_df[failures_df['rule_category'] == selected_category] if selected_category != 'All Categories' else failures_df

        st.dataframe(display_df[['timestamp', 'rule_category', 'rule_name', 'Message', 'Trade ID', 'Affected Field', 'Invalid Value']],
            hide_index=True, use_container_width=True,
            column_config={"timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
                           "rule_category": st.column_config.Column("Category", width="medium"),
                           "rule_name": st.column_config.Column("Rule Name", width="medium")})
        
        with st.expander("Show Raw Failure Details (JSON)"):
            st.json(display_df.to_dict(orient='records'))

def render_home_page():
    st.title("💼 ETRM Business Intelligence Hub")
    st.markdown("Select a widget to explore trade data from your ETRM database.")
    
    total_trades = len(filtered_trades)
    total_volume = filtered_trades['Quantity'].sum() if not filtered_trades.empty else 0
    counterparties = filtered_trades['Counterparty'].nunique() if not filtered_trades.empty else 0
    
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Total Trades", f"{total_trades:,}")
        col2.metric("💰 Total Volume", f"{total_volume:,.0f}")
        col3.metric("🏢 Counterparties", f"{counterparties}")
        if time_granularity == "Hourly": time_span = f"{(end_date - start_date).total_seconds() / 3600:.1f} hours"
        else: time_span = f"{(end_date - start_date).days} days"
        col4.metric("⏰ Time Span", time_span)
    
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.header("📈 KPI Overview")
            st.markdown("High-level metrics on trade counts, total quantity, and counterparties.")
            st.button("Launch KPI View", on_click=set_view, args=['kpi'], use_container_width=True, type="primary")
        st.write("")
        with st.container(border=True):
            st.header("📊 Market & Price Insights")
            st.markdown("Analyze price trends over time and relationships between price and quantity.")
            st.button("Launch Market View", on_click=set_view, args=['insights'], use_container_width=True)
    with col2:
        with st.container(border=True):
            st.header("🚦 Data Quality & Validation")
            st.markdown("A deep-dive into the structure and completeness of the trade data files.")
            st.button("Launch Quality View", on_click=set_view, args=['quality'], use_container_width=True)
        st.write("")
        with st.container(border=True):
            st.header("📋 Trade Blotter")
            st.markdown("View, search, and filter all individual trades captured by the system.")
            st.button("Launch Trade Blotter", on_click=set_view, args=['blotter'], use_container_width=True)

    st.write("---")
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.header("🔬 Validation Log Insights")
            st.markdown("Visualize results from the data ingestion pipeline. Analyze failures and error types to improve data quality at the source.")
        with col2:
            st.write("") 
            st.write("")
            st.button("Launch Log Viewer", on_click=set_view, args=['validation'], use_container_width=True, type="primary")

# --- MAIN ROUTER ---
view_map = {
    'home': render_home_page,
    'kpi': render_kpi_overview,
    'quality': render_data_quality_view,
    'insights': render_market_insights,
    'blotter': render_trade_blotter,
    'validation': render_validation_logs
}
view_function = view_map.get(st.session_state.active_view, render_home_page)
view_function()


# --- START: "ASK CARTEL" CHATBOT IMPLEMENTATION ---

# Hardcoded Q&A logic. The function can access 'filtered_trades' as it's in the script's scope.
# --- START: "ASK CARTEL" CHATBOT IMPLEMENTATION (V2 - Improved Logic & Scenarios) ---

# Hardcoded Q&A logic. The function can access 'filtered_trades' as it's in the script's scope.
def get_bot_response(user_query):
    """
    Generates a response based on hardcoded rules and live data.
    V2: More robust keyword matching and richer Q&A scenarios.
    """
    query_lower = user_query.lower().strip()
    query_words = set(query_lower.split()) # Use a set for efficient lookup

    # --- Pre-computation and edge case handling ---
    data_is_available = not filtered_trades.empty
    
    # --- Intent Routing ---

    # 1. Handle "Help" / "Capabilities" Intent
    if "help" in query_words or "what can you do" in query_lower:
        return """
        I can help you with:
        - **Performance:** Ask me for 'top trader by volume' or 'most traded product'.
        - **Pricing:** Try 'average price' or 'highest price trade'.
        - **Data Quality:** Ask 'how many trades have missing data?' or 'check for validation errors'.
        - **Navigation:** Ask 'where can I see all trades?'.
        - **Definitions:** Try 'what is a schema?'.
        """

    # 2. Handle Greeting Intent (More specific matching)
    greeting_words = {"hello", "hi", "hey", "greetings"}
    if greeting_words.intersection(query_words):
        return "Hello! I am Cartel, your ETRM assistant. Ask me a question or type 'help' to see what I can do."

    # --- Dynamic Questions (Require data) ---
    
    # 3. Top Performance Questions
    if "top trader" in query_lower:
        if data_is_available:
            top_trader = filtered_trades.groupby('Trader')['Quantity'].sum().idxmax()
            top_volume = filtered_trades.groupby('Trader')['Quantity'].sum().max()
            return f"The top trader by volume in the current selection is **{top_trader}** with a total of **{top_volume:,.0f} units** traded."
        else:
            return "There is no data selected to determine the top trader. Please adjust your filters."
            
    if "most traded product" in query_lower:
        if data_is_available:
            top_product = filtered_trades.groupby('ProductType')['Quantity'].sum().idxmax()
            top_volume = filtered_trades.groupby('ProductType')['Quantity'].sum().max()
            return f"The most traded product type is **{top_product}** with a total volume of **{top_volume:,.0f} units**."
        else:
            return "I can't answer that without any trade data. Please check your filters in the sidebar."

    # 4. Price-related Questions
    elif "average price" in query_lower:
        if data_is_available:
            avg_price = filtered_trades['Price'].mean()
            return f"Based on the current filters, the average trade price is **${avg_price:,.2f}**."
        else:
            return "I can't calculate the average price. Please make sure there is data selected."
    
    elif "highest price" in query_lower or "max price" in query_lower:
        if data_is_available:
            max_price_row = filtered_trades.loc[filtered_trades['Price'].idxmax()]
            return f"The highest price trade in the current selection is for **{max_price_row['Symbol']}** at **${max_price_row['Price']:,.2f}** (Trade ID: {max_price_row['TradeID']})."
        else:
            return "I can't find the highest price. Please make sure there is data selected."

    # 5. Volume-related Questions
    elif "total volume" in query_lower or "total quantity" in query_lower:
        if data_is_available:
            total_volume = filtered_trades['Quantity'].sum()
            return f"The total traded volume for the current selection is **{total_volume:,.0f} units**."
        else:
            return "I can't calculate the total volume. Please check your filters."

    # 6. Data Quality Questions
    elif "missing data" in query_lower or "n/a" in query_lower:
        if data_is_available:
            na_sides_count = len(filtered_trades[filtered_trades['Side'] == 'N/A'])
            if na_sides_count > 0:
                return f"I found **{na_sides_count} trades** with a missing 'Side' value. You can analyze this further in the 'Data Quality & Validation' view."
            else:
                return "Good news! All trades in the current selection have a 'Side' (Buy/Sell) specified."
        else:
            return "There is no data selected to check for missing values."
            
    # --- Static / Less Dynamic Questions ---

    # 7. Validation Log Questions
    elif "error" in query_words or "failure" in query_words or "validation" in query_words:
        logs_df, _ = load_validation_logs() # Use cached function
        if logs_df is not None and not logs_df.empty:
            failures_df = logs_df[logs_df['status'] == 'FAILURE']
            num_failures = len(failures_df)
            if num_failures > 0:
                most_common_error = failures_df['rule_category'].mode()[0] if not failures_df['rule_category'].empty else "N/A"
                return f"The system logs show **{num_failures} validation failures**. The most common failure category is **'{most_common_error}'**. For a detailed breakdown, please launch the **'Validation Log Insights'** view from the homepage."
            else:
                return "🎉 Great news! I found no validation failures in the system logs."
        else:
            return "I could not access the validation logs to check for errors."

    # 8. Navigational Questions
    elif "where" in query_words and "trades" in query_words:
        return "You can see a complete list of all individual trades in the **'Detailed Trade Blotter'** view. Launch it from the homepage!"
        
    elif "kpi" in query_words:
        return "You can find high-level Key Performance Indicators like total trades and average price in the **'KPI Overview'**."

    # 9. Definitional Questions
    elif "schema" in query_words or "columns" in query_words or "fields" in query_words:
        cols = trades_df.columns.tolist()
        return f"The term 'schema' refers to the structure of the data. The main trade data contains the following fields: `{', '.join(cols)}`."

    # 10. Default Fallback Response
    else:
        return "I'm sorry, I don't have an answer for that. Try asking something else, or type 'help' to see what I can do."

# --- Popover UI Definition (No changes needed here) ---
chatbot_popover = st.popover("🤖 Ask Cartel")

with chatbot_popover:
    st.markdown("##### 🤖 Ask Cartel")
    st.caption("Your ETRM dashboard assistant.")

    if "ask_cartel_messages" not in st.session_state:
        st.session_state.ask_cartel_messages = [
            {"role": "assistant", "content": "Hi! How can I help? Type 'help' to see what I can do."}
        ]

    for message in st.session_state.ask_cartel_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about prices, top traders, etc."):
        st.session_state.ask_cartel_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = get_bot_response(prompt)
            st.markdown(response)
        st.session_state.ask_cartel_messages.append({"role": "assistant", "content": response})

# --- END: "ASK CARTEL" CHATBOT IMPLEMENTATION ---

