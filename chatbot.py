import os
import json
import re
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint  
from pydantic import BaseModel, Field

# ✅ Load API keys securely
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACEHUB_API_KEY = os.getenv("HUGGINGFACEHUB_API_KEY")

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "dog-chatbot"

# ✅ Check if index exists, else create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  
        metric="cosine"
    )

# ✅ Connect to Pinecone
index = pc.Index(index_name)

# ✅ Load Sentence Transformer Model for embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ Initialize Hugging Face LLM with task type
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    task="text-generation",  
    temperature=0.7,          # ✅ Pass explicitly
    max_new_tokens=200,       # ✅ Pass explicitly
    huggingfacehub_api_token=HUGGINGFACEHUB_API_KEY
)
# ✅ Dog-related Keywords
DOG_KEYWORDS = {"dog", "puppy", "breed", "canine", "retriever", "shepherd", "terrier", "hound", "bulldog", "mastiff", "poodle"}

def is_dog_related(query):
    """Check if the query is about dogs using keywords and LLM classification."""
    
    query_lower = query.lower()
    if any(word in query_lower for word in DOG_KEYWORDS):
        return True

    # 🧠 Improved LLM Prompt for classification
    classification_prompt = f"""
    SYSTEM: You are an AI classifier. Your task is to determine whether a question is about dogs.
    Respond only with "yes" or "no".

    USER: "{query}"
    """

    response = llm.invoke(classification_prompt).strip().lower()
    return response == "yes"

def chat_with_dog_ai(user_query):
    """General chatbot interface that answers dog-related questions dynamically."""

    if not is_dog_related(user_query):
        return "⚠️ Sorry, I can only answer dog related questions."

    # 🔍 Generate embedding for the query
    query_embedding = embedding_model.encode(user_query).tolist()

    # 🔍 Retrieve relevant information from Pinecone
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    print(f"📌 Raw Pinecone results: {results}")
    if not results["matches"]:
        return "Sorry, I couldn't find any relevant information."

    # 📜 Extract breed names and descriptions
    breed_info = [f"**{match['id']}**: {match['metadata']['description']}" for match in results["matches"]]

    # 🔹 Prepare enhanced AI prompt
   
    prompt = f"""
    SYSTEM:You are an expert in dog breeds. Provide clear and well-structured answers without explaining your reasoning.
    CONTEXT: 
    {''.join(breed_info)}
    USER QUESTION: "{user_query}"

    ANSWER :
    """

    raw_response = llm.invoke(prompt).strip()
    print(f"\n📌 Raw AI Response: {raw_response}")
    return raw_response

# ✅ Test the chatbot
if __name__ == "__main__":
    while True:
        user_input = input("\n🐶 Ask me anything about dog breeds (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        response = chat_with_dog_ai(user_input)
        print(f"\n🤖 AI Answer: {response}")
