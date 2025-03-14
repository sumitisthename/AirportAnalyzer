import nltk
from nltk.translate.bleu_score import sentence_bleu

# Ensure NLTK resources are downloaded
#nltk.download('punkt', quiet=True)

def compute_bleu(reference, generated):
    """
    Computes BLEU score between reference and generated response.

    Args:
        reference (str): Ground truth response.
        generated (str): AI-generated response.

    Returns:
        float: BLEU score (0 to 1, higher is better).
    """
    reference_tokens = [nltk.word_tokenize(reference.lower())]  # List of reference sentences
    generated_tokens = nltk.word_tokenize(generated.lower())  # Generated response tokens
    
    return sentence_bleu(reference_tokens, generated_tokens)

# Example Sentences
reference_sentence = "The quick brown fox jumps over the lazy dog."
generated_sentence = "A fast brown fox leaps over a sleepy dog."

# Compute BLEU Score
bleu_score = compute_bleu(reference_sentence, generated_sentence)
print(f"BLEU Score: {bleu_score:.3f}")
