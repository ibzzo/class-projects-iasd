import json
import streamlit as st
from kafka import KafkaConsumer
import pandas as pd
from datetime import datetime
import threading
import queue
import time
import os

# Queue for thread-safe communication
alert_queue = queue.Queue(maxsize=100)
consumer_started = False

def consume_fraud_alerts():
    """Background thread to consume fraud alerts from Kafka"""
    while True:
        try:
            consumer = KafkaConsumer(
                'fraud-alerts',
                bootstrap_servers=[os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')],
                auto_offset_reset='latest',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                consumer_timeout_ms=1000
            )
            
            print("Connected to Kafka successfully!")
            
            for message in consumer:
                alert = message.value
                alert['received_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Add to queue (remove oldest if full)
                if alert_queue.full():
                    alert_queue.get()
                alert_queue.put(alert)
                
        except Exception as e:
            print(f"Error connecting to Kafka: {e}")
            time.sleep(5)  # Wait 5 seconds before retrying

# Start consumer thread only once
if not consumer_started:
    consumer_thread = threading.Thread(target=consume_fraud_alerts, daemon=True)
    consumer_thread.start()
    consumer_started = True

# Streamlit UI
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("🚨 Real-Time Fraud Detection Dashboard")

# Create placeholders
metrics_placeholder = st.empty()
alerts_placeholder = st.empty()

# Main loop
while True:
    # Get all alerts from queue
    alerts = []
    while not alert_queue.empty():
        try:
            alerts.append(alert_queue.get_nowait())
        except queue.Empty:
            break
    
    # Display metrics
    with metrics_placeholder.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Alerts", len(alerts))
        
        with col2:
            if alerts:
                fraud_types = pd.DataFrame(alerts)['fraud_type'].value_counts()
                st.metric("Most Common Type", fraud_types.index[0] if len(fraud_types) > 0 else "N/A")
        
        with col3:
            if alerts:
                total_amount = sum(alert.get('amount', 0) for alert in alerts)
                st.metric("Total Suspicious Amount", f"${total_amount:,.2f}")
    
    # Display recent alerts
    with alerts_placeholder.container():
        st.subheader("Recent Fraud Alerts")
        
        if alerts:
            # Convert to DataFrame for better display
            df = pd.DataFrame(alerts)
            
            # Select and reorder columns
            columns = ['received_at', 'user_id', 'transaction_id', 'amount', 
                      'currency', 'location', 'method', 'fraud_type']
            df = df[columns]
            
            # Display as table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                    "received_at": "Time",
                    "user_id": "User ID",
                    "transaction_id": "Transaction ID",
                    "fraud_type": "Fraud Type"
                }
            )
            
            # Fraud type distribution
            st.subheader("Fraud Type Distribution")
            fraud_type_counts = df['fraud_type'].value_counts()
            st.bar_chart(fraud_type_counts)
            
        else:
            st.info("No fraud alerts yet. Waiting for suspicious transactions...")
    
    # Refresh every 2 seconds
    time.sleep(2)
    st.rerun()