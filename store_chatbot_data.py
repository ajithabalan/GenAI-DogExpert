import os
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as PineconeStore
from breedDescription import BREED_DESCRIPTIONS
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys securely
API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=API_KEY)

# Define Pinecone index name
index_name = "dog-chatbot"

# Use a free open-source embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Check if index exists, create if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Matching MiniLM embedding output size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to index
index = pc.Index(index_name)

# Convert breed descriptions into vector embeddings and store them
vectors = []
for breed, description in BREED_DESCRIPTIONS.items():
    vector = embedding_model.encode(description).tolist()  # Generate vector embedding
    vectors.append((breed, vector, {"description": description}))

index.upsert(vectors)

print("✅ Chatbot Data stored successfully in Pinecone!")
