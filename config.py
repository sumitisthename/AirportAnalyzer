import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("GROQ API Key:", GROQ_API_KEY)  # Ensure it's loaded correctly

# Base URL for scraping
BASE_URL = "https://www.airlinequality.com/airline-reviews/qatar-airways"
