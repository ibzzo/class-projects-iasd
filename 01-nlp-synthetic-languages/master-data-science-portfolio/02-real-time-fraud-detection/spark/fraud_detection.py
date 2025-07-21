from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import os

def create_spark_session():
    """Create Spark session with Kafka support"""
    return SparkSession.builder \
        .appName("FraudDetection") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .getOrCreate()

def get_schema():
    """Define the schema for transaction data as per wiki"""
    return StructType([
        StructField("user_id", StringType(), True),
        StructField("transaction_id", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("currency", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("location", StringType(), True),
        StructField("method", StringType(), True)
    ])

def detect_fraud(df):
    """Apply the 3 fraud detection rules as specified in the wiki"""
    
    # Convert timestamp string to timestamp type
    df = df.withColumn("timestamp", to_timestamp(col("timestamp")))
    
    # Rule 1: Flag high-value transactions (> 1000)
    high_value_df = df.filter(col("amount") > 1000) \
        .withColumn("fraud_type", lit("high_value_transaction"))
    
    # Rule 2: Detect more than 3 transactions from same user in < 1 minute
    rapid_transactions_df = df \
        .withWatermark("timestamp", "5 minutes") \
        .groupBy(
            window(col("timestamp"), "1 minute"),
            col("user_id")
        ) \
        .agg(
            count("*").alias("transaction_count"),
            collect_list("transaction_id").alias("transaction_ids"),
            sum("amount").alias("total_amount"),
            first("currency").alias("currency"),
            first("location").alias("location"),
            first("method").alias("method")
        ) \
        .filter(col("transaction_count") > 3) \
        .select(
            col("user_id"),
            col("transaction_ids")[0].alias("transaction_id"),
            col("total_amount").alias("amount"),
            col("currency"),
            col("window.start").alias("timestamp"),
            col("location"),
            col("method")
        ) \
        .withColumn("fraud_type", lit("rapid_transactions"))
    
    # Rule 3: Detect transactions in multiple countries within 5 minutes
    # (Using location as proxy for country)
    location_changes_df = df \
        .withWatermark("timestamp", "5 minutes") \
        .groupBy(
            window(col("timestamp"), "5 minutes"),
            col("user_id")
        ) \
        .agg(
            approx_count_distinct("location").alias("location_count"),
            collect_list("location").alias("locations"),
            first("transaction_id").alias("transaction_id"),
            sum("amount").alias("amount"),
            first("currency").alias("currency"),
            first("method").alias("method")
        ) \
        .filter(col("location_count") > 2) \
        .select(
            col("user_id"),
            col("transaction_id"),
            col("amount"),
            col("currency"),
            col("window.start").alias("timestamp"),
            col("locations")[0].alias("location"),
            col("method")
        ) \
        .withColumn("fraud_type", lit("multiple_locations"))
    
    # Union all fraud alerts
    return high_value_df.union(rapid_transactions_df).union(location_changes_df)

def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    # Read from Kafka
    df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')) \
        .option("subscribe", "transactions") \
        .option("startingOffsets", "latest") \
        .load()
    
    # Parse JSON data
    schema = get_schema()
    transactions_df = df.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")
    
    # Detect fraud
    fraud_alerts_df = detect_fraud(transactions_df)
    
    # Output 1: Console
    console_query = fraud_alerts_df.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .trigger(processingTime="10 seconds") \
        .start()
    
    # Output 2: Parquet files
    parquet_query = fraud_alerts_df.writeStream \
        .outputMode("append") \
        .format("parquet") \
        .option("path", "fraud_alerts_parquet") \
        .option("checkpointLocation", "checkpoint_parquet") \
        .trigger(processingTime="30 seconds") \
        .start()
    
    # Output 3: Kafka topic
    kafka_query = fraud_alerts_df.select(
        to_json(struct("*")).alias("value")
    ).writeStream \
        .outputMode("append") \
        .format("kafka") \
        .option("kafka.bootstrap.servers", os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')) \
        .option("topic", "fraud-alerts") \
        .option("checkpointLocation", "checkpoint_kafka") \
        .trigger(processingTime="10 seconds") \
        .start()
    
    # Wait for all streams
    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    main()