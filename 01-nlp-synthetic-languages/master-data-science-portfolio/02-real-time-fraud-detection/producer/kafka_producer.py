import json
import random
import time
import os
from datetime import datetime, timedelta
from kafka import KafkaProducer
from faker import Faker

fake = Faker()

class TransactionProducer:
    def __init__(self, bootstrap_servers=None):
        if bootstrap_servers is None:
            bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.producer = KafkaProducer(
            bootstrap_servers=[bootstrap_servers],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.transaction_count = 0
        
    def generate_transaction(self):
        """Generate a single transaction following the exact format from the wiki"""
        self.transaction_count += 1
        return {
            "user_id": f"u{random.randint(1000, 9999)}",
            "transaction_id": f"t-{self.transaction_count:07}",
            "amount": round(random.uniform(5.0, 5000.0), 2),
            "currency": random.choice(["EUR", "USD", "GBP"]),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "location": fake.city(),
            "method": random.choice(["credit_card", "debit_card", "paypal", "crypto"])
        }
    
    def run(self, transactions_per_second=50):
        """Run the producer, simulating 10-100 transactions per second"""
        print(f"Starting transaction producer at {transactions_per_second} TPS...")
        
        try:
            while True:
                # Vary the rate between 10-100 TPS
                current_tps = random.randint(10, min(100, transactions_per_second * 2))
                
                for _ in range(current_tps):
                    transaction = self.generate_transaction()
                    self.producer.send('transactions', value=transaction)
                    
                    if self.transaction_count % 100 == 0:
                        print(f"Sent {self.transaction_count} transactions...")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("Shutting down producer...")
        finally:
            self.producer.flush()
            self.producer.close()

if __name__ == "__main__":
    producer = TransactionProducer()
    producer.run()