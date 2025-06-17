# import statements

import os
import discord
import google.generativeai as genai
from dotenv import load_dotenv

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load .env values
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
discord_token = os.getenv("DISCORD_TOKEN")

# Check keys
if not api_key or not discord_token:
    print("❌ Missing keys in .env")
    exit()

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Bot intents
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# user_id, role, message, embedding
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_DIM = 384  # Dimensions for MiniLM
VECTOR_STORE_PATH = "vector_store.pkl"
vector_store = {}

def load_vector_store():
    global vector_store
    if os.path.exists(VECTOR_STORE_PATH):
        with open(VECTOR_STORE_PATH, "rb") as f:
            vector_store = pickle.load(f)

def save_vector_store():
    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(vector_store, f)

load_vector_store()

def store_message(user_id, role, message):
    global vector_store
    embedding = embedding_model.encode([message])[0]
    if user_id not in vector_store:
        vector_store[user_id] = []
    vector_store[user_id].append((embedding, message, role))
    save_vector_store()

def summarize_conversation(user_id):
    if user_id not in vector_store:
        return "No history to summarize."
    history = vector_store[user_id]
    prompt = "Summarize the following conversation in a nicely formatted paragraph. Make sure that it is readable with headers. Under each Header I want bullet points of the main points:\n"
    for _, msg, role in history:
        prompt += f"{role}: {msg}\n"
    response = model.generate_content(prompt)
    return response.text.strip()

def query_data(sql_query):
    prompt = (
        "show me an SQL query for the following message and assume that whatever tables/rows/column names you decide to use are correct. "
        "DO NOT GIVE ME ANY EXPLANATIONS, I ONLY WANT TO SEE THE SQL QUERIES. I DO NOT WANT ANY OPTIONS. JUST CHOOSE AN OPTION AND SEND IT TO ME:\n"
        f"{sql_query}"
    )
    response = model.generate_content(prompt)
    return response.text.strip()

def search_user_conversation(user_id, query, top_k=5):
    if user_id not in vector_store or not vector_store[user_id]:
        return "❌ No history to search."

    user_data = vector_store[user_id]
    embeddings = np.array([entry[0] for entry in user_data])
    messages = [(entry[2], entry[1]) for entry in user_data]  # (role, message)
        
    index = faiss.IndexFlatL2(VECTOR_DIM)
    index.add(embeddings)

    query_vector = embedding_model.encode([query])[0].reshape(1, -1)
    D, I = index.search(query_vector, top_k)

    relevant_msgs = [messages[i] for i in I[0] if i < len(messages)]

    prompt = f"Search for the following topic in our conversation and bulletpoint anything relevant to '{query}':\n\n"
    for role, msg in relevant_msgs:
        prompt += f"{role}: {msg}\n"

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error during search: {e}"

def clear_user_history(user_id):
    if user_id in vector_store:
        del vector_store[user_id]
        save_vector_store()

def clear_all_memory():
    global vector_store
    vector_store = {}  # Clear in-memory store

    # Remove saved file if it exists
    if os.path.exists("vector_store.pkl"):
        os.remove("vector_store.pkl")
        print("🧠 All memory cleared and file deleted.")
    else:
        print("🧠 In-memory store cleared. No file found.")

@client.event
async def on_ready():
    print(f"✅ Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    user_id = message.author.id
    user_message = message.content.strip()

    # Handle special commands
    if user_message.lower() == "exit":
        clear_user_history(user_id)
        await message.channel.send("🧠 Memory cleared!")
        return

    if user_message.lower() == "summary":
        summary = summarize_conversation(user_id)
        await message.channel.send(f"📋 Summary:\n{summary}")
        return
    
    if user_message.lower().startswith("query: "):
        sql_query = user_message[7:].strip()
        query = query_data(sql_query)
        await message.channel.send(f"{query}")
        return

    if user_message.lower().startswith("search: "):
        search_query = user_message[8:].strip()
        search_result = search_user_conversation(user_id, search_query)
        await message.channel.send(f"🔎 Search:\n{search_result}")
        return
    
    if user_message.lower() == "dump":
        history = vector_store.get(user_id, [])
        if not history:
            await message.channel.send("🗃️ No messages found.")
        else:
            dump = "\n".join([f"{role}: {msg}" for _, msg, role in history])
            await message.channel.send(f"🧠 Your conversation history:\n{dump[:1900]}")  # Discord limit
        return
    
    if user_message.lower() == "clear all":
        clear_all_memory()

    store_message(user_id, "User", user_message)

    history = vector_store.get(user_id, [])
    recent_history = history[-10:]
    full_prompt = "Do not give me super long responses or bullet points unless asked to do so.\n"
    for _, msg, role in recent_history:
        full_prompt += f"{role}: {msg}\n"
    #full_prompt += "Bot:"

    try:
        response = model.generate_content(full_prompt)
        bot_reply = response.text.strip()
    except Exception as e:
        await message.channel.send(f"❌ Error: {e}")
        return

    #await message.channel.send(bot_reply)
    store_message(user_id, "Bot", bot_reply)

client.run(discord_token)