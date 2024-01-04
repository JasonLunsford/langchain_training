from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma    # our Vector Store
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n",  # this separates chunks by finding nearest separator, done SECOND
    chunk_size=200,  # the amount of text chunked, done FIRST
    chunk_overlap=0  # the amount of overlap allowed between chunks
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter
)

# Running this line of code causes the embeddings to be calculated, which costs a
# small amount of money
# sets up database and adds documents
db = Chroma.from_documents(
    docs,
    embedding=embeddings,  # notice the naming mismatch, due to differences between libraries
    persist_directory="emb"
)

# results = db.similarity_search_with_score(
#     "What is an interesting fact about the English language?"
#     k=1  # determines how many results you get back
# )

results = db.similarity_search(
    "What is an interesting fact about the English language?"
)

for result in results:
    print("\n")
    print(result.page_content)

# Use this loop with "similarity_search_with_score" as it uses this data structure
# for result in results:
#     print("\n")
#     print(result[1])
#     print(result[0].page_content)
