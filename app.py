from sentence_transformers import SentenceTransformer
from llama_index.core import (
    Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore
import logging
import os
from dotenv import load_dotenv
#from llama_index.core.evaluation import FaithfulnessEvaluator
#import nest_asyncio

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
text_splitter = SentenceSplitter(chunk_size=400, chunk_overlap=10)
Settings.text_splitter = text_splitter
Settings.llm = Ollama(
    model="qwen3:4b-instruct",
    context_window=8192,
    request_timeout=300.0,
    system_prompt="""You are a helpful assistant that can search through documents to answer questions. Abstain when not given enough context, and when the question is not understandable. """,
)

vector_store = PGVectorStore.from_params(
    host=os.getenv("PG_HOST", "localhost"),
    port=int(os.getenv("PG_PORT", "5432")),
    database=os.environ["PG_DATABASE"],
    user=os.environ["PG_USER"],
    password=os.environ["PG_PASSWORD"],
    table_name=os.getenv("PG_TABLE_NAME", "policy_chunks"),
    embed_dim=int(os.getenv("EMBED_DIM", "384")),
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
documents = SimpleDirectoryReader("./data").load_data()

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    transformations=[text_splitter],
)

"""
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    transformations=[text_splitter],
)
"""

#evaluator = FaithfulnessEvaluator(llm=Settings.llm)

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L6-v2", top_n=3
)

query_engine = index.as_query_engine(similarity_top_k=3, node_postprocessors=[reranker])
x2
query = ""
while(True):
    query = input("Szia! Kérdezz bátran: ")
    if(query == "exit"):
        print("Viszlát!")
        break
    response = query_engine.query(query)
    print(response)
    #eval_result = evaluator.evaluate_response(response=response)
    #print(str(eval_result.passing))
    #nest_asyncio.apply()
    