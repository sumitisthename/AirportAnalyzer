import streamlit as st
from backend import scrape_reviews, load_text, split_text, initialize_embeddings, retrieve_reviews, analyze_feedback
from config import BASE_URL
from backend import compute_bleu, compute_rouge, compute_bertscore, gpt4_evaluate

# Streamlit App Title
st.set_page_config(page_title="Qatar Airways Feedback Analysis", layout="wide")
st.title("✈️ Qatar Airways Customer Feedback Analysis")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Settings")
    num_pages = st.slider("Number of Pages to Scrape", 1, 10, 5)
    st.write("Click below to scrape reviews:")
    if st.button("Scrape Reviews"):
        with st.spinner("Scraping customer reviews..."):
            scrape_reviews(BASE_URL, num_pages)
        st.success("Reviews scraped successfully!")

# Load and Process Reviews
if st.button("Load Reviews"):
    reviews_text = load_text()
    chunks = split_text(reviews_text)
    
    with st.spinner("Generating embeddings..."):
        embedding_model, embeddings_list = initialize_embeddings(chunks)

    st.session_state["chunks"] = chunks
    st.session_state["embedding_model"] = embedding_model
    st.session_state["embeddings_list"] = embeddings_list

    st.success("Reviews loaded and processed!")

# User Input Section
st.subheader("🔍 Search Customer Feedback")
question = st.text_input("Enter your query (e.g., 'What did Michael Schade say about Qatar Airways?')")

if st.button("Analyze Feedback") and "chunks" in st.session_state:
    with st.spinner("Retrieving relevant reviews..."):
        relevant_reviews = retrieve_reviews(
            question,
            st.session_state["chunks"],
            st.session_state["embedding_model"],
            st.session_state["embeddings_list"]
        )

    with st.spinner("Analyzing feedback using AI..."):
        response = analyze_feedback(question, relevant_reviews)

    st.subheader("📌 Customer Feedback Analysis")
    st.markdown(response)

# Evaluation Metrics Section
st.title("📝 Response Evaluation Metrics")

# User Inputs
reference_text = st.text_area("Enter the Reference (Ground Truth) Response")
generated_text = response

if st.button("Evaluate Response"):
    if reference_text and generated_text:
        # Compute Metrics
        bleu_score = compute_bleu(reference_text, generated_text)
        #rouge_scores = compute_rouge(reference_text, generated_text)
        #bert_score = compute_bertscore(reference_text, generated_text)
        #gpt_feedback = gpt4_evaluate(reference_text, generated_text)

        # Display Results
        st.subheader("📊 Evaluation Results")
        st.metric("BLEU Score", f"{bleu_score:.3f}")
        #st.write("**ROUGE Scores:**")
        #st.json(rouge_scores)
        #st.metric("BERTScore", f"{bert_score:.3f}")

        #st.subheader("💡 GPT-4 Feedback")
        #st.write(gpt_feedback)

    else:
        st.error("Please enter both Reference and Generated responses.")


# Footer
st.markdown("---")
st.caption("Developed by AI-Powered Insights 🚀")
