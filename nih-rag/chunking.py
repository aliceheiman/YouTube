# STEP 1: NIH API
import requests
from bs4 import BeautifulSoup

NIH_API = "https://ods.od.nih.gov/api/"

response = requests.get(NIH_API)
soup = BeautifulSoup(response.text, "html.parser")
links = soup.find_all("a", string="HTML")
web_paths = [NIH_API + link["href"] for link in links if "español" not in link["href"]]

# STEP 2: RETRIEVE, CHUNK, AND INDEX WEB PAGES
from langchain_community.document_loaders import WebBaseLoader

print("Fetching content...")
loader = WebBaseLoader(
    web_paths=web_paths,
)
docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# SAVE TO DISK
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from chromadb.config import Settings

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma.db",
    client_settings=Settings(
        anonymized_telemetry=False,
        is_persistent=True,
    ),
)
