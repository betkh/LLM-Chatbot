from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


userQuery = ["Suggest gift items for Mother’s Day",]

embeddings = model.encode(userQuery)
print("shape: ", embeddings.shape)
print(embeddings[0])  # Print the first embedding vector
