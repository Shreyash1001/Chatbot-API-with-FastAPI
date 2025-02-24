import redis
import os
from collections import defaultdict, deque

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

try:
    # Attempt to connect to Redis
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()  # Check if Redis server is available
    chat_memory = None  # If Redis is connected, no fallback needed
    print("Connected to Redis!")
except Exception as e:
    # If Redis fails to connect, use in-memory chat memory
    redis_client = None
    chat_memory = defaultdict(lambda: deque(maxlen=10))  # Fallback to in-memory
    print("Redis connection failed, using in-memory storage.")

# Testing Redis Client
if redis_client:
    # Try setting a key-value pair to test Redis functionality
    redis_client.set("test_key", "test_value")
    print(f"Redis test key: {redis_client.get('test_key')}")

# Test chat memory (fallback) functionality
if chat_memory is not None:
    # Add a test message to chat memory
    user_id = "test_user"
    chat_memory[user_id].append("Test message 1")
    chat_memory[user_id].append("Test message 2")
    print(f"Chat memory for {user_id}: {list(chat_memory[user_id])}")
