import os
import faiss
import numpy as np
import openai
import groq
from selenium import webdriver
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from config import GROQ_API_KEY, BASE_URL
import os
from dotenv import load_dotenv
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Load the .env file
load_dotenv()

# Retrieve the API key
groq_api_key = os.getenv("GROQ_API_KEY")

print("GROQ API Key:", groq_api_key)  # Ensure it's loaded correctly


# Function to scrape reviews
def scrape_reviews(base_url, num_pages=5, file_path="reviews.txt"):
    """Scrapes customer reviews from the given URL."""
    if os.path.exists(file_path):
        os.remove(file_path)
    
    driver = webdriver.Chrome()  # Ensure ChromeDriver is installed
    for page_num in range(1, num_pages + 1):
        url = f"{base_url}/page/{page_num}" if page_num > 1 else base_url
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        reviews = soup.find_all('div', class_='body')

        with open(file_path, 'a', encoding='utf-8') as file:
            for i, review in enumerate(reviews):
                text = review.text.strip()
                file.write(f"Review {i + 1 + (page_num - 1) * len(reviews)}:\n{text}\n{'-' * 80}\n")
    
    driver.quit()

# Function to load scraped text
def load_text(file_path="reviews.txt"):
    """Loads customer reviews from a file."""
    with open(file_path, encoding='utf-8') as f:
        return f.read()

# Function to split text into chunks
def split_text(data):
    """Splits the content into individual reviews."""
    return [review.strip() for review in data.split("Review")[1:]]

# Function to initialize sentence embeddings
def initialize_embeddings(docs):
    """Generates embeddings using SentenceTransformer."""
    embedding_model = SentenceTransformer("avsolatorio/GIST-Embedding-v0")
    embeddings_list = embedding_model.encode(docs, convert_to_tensor=True)
    return embedding_model, embeddings_list

# Function to convert user query to embeddings
def convert_query_to_embedding(question, embedding_model):
    """Converts a question into an embedding."""
    search_vec = embedding_model.encode(question, convert_to_tensor=True)
    svec = np.array(search_vec).reshape(1, -1)
    return np.ascontiguousarray(svec, dtype="float32")

# Function to retrieve relevant reviews
def retrieve_reviews(question, docs, embedding_model, embeddings_list):
    """Retrieves the most relevant reviews using FAISS."""
    svec = convert_query_to_embedding(question, embedding_model)
    dim = embeddings_list.shape[1]
    
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_list)
    
    distances, indices = index.search(svec, k=5)
    return [docs[i] for i in indices[0]]

# Function to process customer feedback using LLM
def analyze_feedback(question, documents, model="llama3-8b-8192", temperature=0.7, max_tokens=1000):
    """Uses LLM to analyze customer feedback."""
    client = groq.Client(api_key="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")
    batch_size = 3
    results = []

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        input_text = f"{question}\n\nDocuments:\n" + "\n".join(batch_docs)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Analyze the customer feedback in a structured format.\n\n"
                        "**Customer Feedback Analysis – Qatar Airways**\n\n"
                        "**Summary:**\n- Provide a high-level overview.\n\n"
                        "**Key Insights:**\n- Identify positive and negative experiences.\n\n"
                        "**Specific Examples:**\n- Mention customer names and issues.\n\n"
                        "**Actionable Recommendations:**\n- Provide practical solutions.\n"
                    )
                },
                {"role": "user", "content": input_text}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        results.append(response.choices[0].message.content.strip())

    return "\n\n".join(results)


# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

def compute_bleu(reference, generated):
    """
    Computes BLEU score between reference and generated response.

    Args:
        reference (str): Ground truth response.
        generated (str): AI-generated response.

    Returns:
        float: BLEU score (0 to 1, higher is better).
    """
    reference_tokens = [nltk.word_tokenize(reference.lower())]
    generated_tokens = nltk.word_tokenize(generated.lower())
    
    return sentence_bleu(reference_tokens, generated_tokens)


# Main function to run the backend logic
if __name__ == "__main__":
    # Example usage
    scrape_reviews(BASE_URL, num_pages=5)
    reviews_text = load_text()
    chunks = split_text(reviews_text)
    
    embedding_model, embeddings_list = initialize_embeddings(chunks)
    
    question = "What did Michael Schade say about Qatar Airways?"
    relevant_reviews = retrieve_reviews(question, chunks, embedding_model, embeddings_list)
    
    response = analyze_feedback(question, relevant_reviews)
    print(response)  # Print the analysis result

    with open("reference.txt", "r") as f:
        reference = f.read()

    bleu_score = compute_bleu(reference, response)
    print(f"BLEU Score: {bleu_score:.3f}")  # Print the BLEU score