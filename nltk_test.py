import nltk
from nltk.translate.bleu_score import sentence_bleu

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
print(nltk.data.path)


nltk.data.find("tokenizers/punkt")
print("Punkt tokenizer is available!")




''''
import nltk
import os

# Manually specify NLTK download path if necessary
nltk.data.path.append(os.path.expanduser("~") + "/nltk_data")

# Force download of `punkt` before using it
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=False)  # Set quiet=True if you don’t want download messages

# Now use nltk.word_tokenize()
from nltk.tokenize import word_tokenize

text = "Hello, how are you?"
tokens = word_tokenize(text)
print(tokens)


'''