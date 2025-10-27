import time
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import os
from requests import ReadTimeout
from google.genai import types
import numpy as np
from scipy.spatial.distance import cosine

# Load variables from environment
load_dotenv()
api_key=os.getenv("GEMINI_API_KEY")

# Set API timeout
timeout_sec = 10
client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_sec * 1000))

# Gemini API response and rate limit with retry protection
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=30))
def get_gemini_response(prompt: str):
    """
    Getting answer from Gemini API with Rate Limit defence and timeouts
    """
    time.sleep(0.5) # delay between requests
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )
        return response.text
    except ReadTimeout:
        print(f"Gemini API request exceeds {timeout_sec} time limit.")
        raise
    except Exception as e:
        if "429" in str(e):
            print("Rate limit exceeded. Too much requests. Wait a few minutes and try again.")
            time.sleep(5)
        else:
            print(f"Error: {e}")
            raise

# Get text embeddings
def get_embeddings(text: str):
    """
    Gets an embedded vector representation of text from Gemini API

    :param text: The string of text that you want to recieve embedded for
    :return: List of numbers representing the embedded vector
    """
    try:
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=[text])
        return np.array(response.embeddings[0].values)
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return None


if __name__ == "__main__":
    prompt = "What medieval armour considered the best"
    try:
        result = get_gemini_response(prompt)
        print("\nGemini response:\n", result)
    except Exception as e:
        print("\nCould not get response from Gemini API after all attempts:", e)

#   Embedding similarity test
    print("\n---Embedding test---")
    text1 = "Medieval knights wore full metal armour for defense"
    text2 = "In Middle age, knights wore plate armour for protection"

    embed1 = get_embeddings(text1)
    embed2 = get_embeddings(text2)

    if embed1 is not None and embed2 is not None:
        similarity = 1 - cosine(embed1, embed2)
        print(f"Text similarity: {similarity:.4f}")

