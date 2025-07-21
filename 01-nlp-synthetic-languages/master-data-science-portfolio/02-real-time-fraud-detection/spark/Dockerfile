FROM bitnami/spark:3.5

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/spark-apps

COPY fraud_detection.py .

RUN chown -R 1001:1001 /opt/spark-apps

USER 1001

ENV PYTHONUNBUFFERED=1
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:29092

CMD ["/opt/bitnami/spark/bin/spark-submit", \
     "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0", \
     "--master", "spark://spark-master:7077", \
     "--deploy-mode", "client", \
     "/opt/spark-apps/fraud_detection.py"]