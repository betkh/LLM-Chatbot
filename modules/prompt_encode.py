# Encode text into embeddings
from sentence_transformers import SentenceTransformer
from nomic import embed
import numpy as np


def nomic_encode(text):
    """
    Encodes the input text into a vector using Nomic embeddings.
    """
    try:
        # Use the nomic.embed module to generate embeddings
        embeddings = embed.text(texts=[text], model='nomic-embed-text')
        # Return the first (and only) embedding
        return np.array(embeddings['embeddings'][0])
    except Exception as e:
        print(f"Error encoding text: {e}")
        return None


# input = "suggest flights to go from chicago to NY"

# encoded_ = nomic_encode(input)

# print(encoded_)


def encode_sen_transformer(text):
    # Lightweight and efficient
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text)
    return embedding


input_text = "suggest flights to go from Chicago to NY"
encoded_text = encode_sen_transformer(input_text)
print(encoded_text)
