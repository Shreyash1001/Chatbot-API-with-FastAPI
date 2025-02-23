from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key from the environment
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Check if the API key was successfully retrieved
if not SERPAPI_KEY:
    print("API Key not found. Please check the .env file.")
else:
    print("API Key successfully retrieved.")
