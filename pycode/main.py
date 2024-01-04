from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse

load_dotenv()

# Fetch arguments, assign defaults to them, add them to "args" object
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

llm = OpenAI()

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

test_prompt = PromptTemplate(
    template="Write a unit test for the following {language} code:\m{code}",
    input_variables=["language", "code"]
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

# use SequentialChain to link multiple chains together
chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"], # incoming variables to the Sequential chain
    output_variables=["test", "code"] # exit variables we care about from last chain
)

result = chain({
    "language": args.language,
    "task": args.task
})

print(">>>>>>>>>> Generated Code:")
print(result["code"])

print(">>>>>>>>>> Generated Test:")
print(result["test"])