from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

# verbose=true is optional, shows more info during execution
chat = ChatOpenAI(verbose=True)

memory = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    llm=chat
    # chat_memory=FileChatMessageHistory("messages.json"), <-- works better with ConversationBufferMemory 
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    verbose=True  # <-- totally optional
)

while True:
    content = input(">> ")

    if content.lower() == 'exit':
        print('Chat terminated by user.')
        break

    result = chain({"content": content})

    print(result["text"])
