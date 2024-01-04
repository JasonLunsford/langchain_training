from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundent_filter_retriever import RedundentFilterRetriever
from dotenv import load_dotenv

# activate debugging when needed
# import langchain
# langchain.debug = True

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()

# Create instance of database, but don't populate it with stuff
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings  # different syntax bc Chroma is a bit disorganized
)

# this is a piece of "glue code" that allows RetrievalQA (which knows nothing about specific vector stores)
# and Chroma (a specific vector store) to work together, because "retriever" (once created) has
# a method called "get_relevant_documents" inside of it
retriever = RedundentFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"  # most basic chain type. take some context from vector store and "stuff" it into the prompt
)

result = chain.run("What is an interesting fact about the English language?")

print(result)