from sentence_transformers import SentenceTransformer
from llama_index.core import (
    Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank

import logging
import os
from dotenv import load_dotenv
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("local_rag")
logger.info("Imports loaded successfully.")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

Settings.text_splitter = SentenceSplitter(chunk_size=400, chunk_overlap=10)
Settings.llm = Ollama(
    model="qwen3:4b-instruct",
    context_window=8192,
    request_timeout=300.0,
    system_prompt=
    """
    You are a helpful assistant that can search through documents to answer questions. 
    Answer only from the supplied context. 
    If it does not explicitly answer the question, say that the 'Az információ nem szerepel az adatbázisban'. 
    Do not infer policies from related statements. 
    Answer only the question asked.
    """,
)

env_path = Path(__file__).resolve().parent / ".env"
loaded = load_dotenv(env_path, override=True)

vector_store = PGVectorStore.from_params(
    host=os.getenv("PG_HOST"),
    port=int(os.getenv("PG_PORT")),
    database=os.getenv("PG_DATABASE"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    table_name=os.getenv("PG_TABLE_NAME"),
    embed_dim=int(os.getenv("EMBED_DIM")),
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
documents = SimpleDirectoryReader("./data").load_data()


index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    show_progress=True,
)

"""
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True,
)
"""

query_engine = index.as_query_engine(
    similarity_top_k=6,
    node_postprocessors=[
        SentenceTransformerRerank(
            model="Qwen/Qwen3-Reranker-0.6B",
            top_n=3,
        )
    ],
)

query = ""
print("Szia!")
while(True):
    query = input("Kérdezz bátran: ")
    if(query == "exit"):
        print("Viszlát!")
        break
    response = query_engine.query(query)
    print(response)
    