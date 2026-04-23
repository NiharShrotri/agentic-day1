from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-nano")

resp1 = llm.invoke("We are building an AI system for processing medical insurance claims.")

resp2 = llm.invoke("What are the main risks in this system?")

print(resp1.content)
print(resp2.content)

#Why the second question may fail or behave inconsistently without conversation history?
#Answer: The second question may fail or behave inconsistently without conversation history because the model does not have a conversation history.
#The model is not able to understand the context of the first question and the second question is not related to the first question.
# LLMs are stateless → each call is independent unless context is explicitly passed.