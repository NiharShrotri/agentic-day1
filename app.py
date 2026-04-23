from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# Initialize model
llm = ChatOpenAI(model="gpt-4.1-nano")


# ---------------- CONTEXT BREAK (Naive) ---------------- #
print("\n--- Without Context (Naive Calls) ---")

resp1 = llm.invoke("We are building an AI system for processing medical insurance claims.")
resp2 = llm.invoke("What are the main risks in this system?")

print(resp1.content)
print(resp2.content)

# Why this fails:
# LLMs are stateless — each call is independent unless context is explicitly passed.
# The second question does not have access to the first message.


# ---------------- CONTEXT FIX (Messages API) ---------------- #
print("\n--- With Context (Messages API) ---")

messages = [
    SystemMessage(content="You are a senior AI architect reviewing production systems."),
    HumanMessage(content="We are building an AI system for processing medical insurance claims."),
    HumanMessage(content="What are the main risks in this system?")
]

resp3 = llm.invoke(messages)

print(resp3.content)


# ---------------- REFLECTION ---------------- #
"""
Reflection:

1. Why did string-based invocation fail?
String-based invocation failed because LLMs are stateless. Each call is independent and does not retain previous context, so the second question lacks necessary background.

2. Why does message-based invocation work?
Message-based invocation works because it explicitly passes the conversation history, allowing the model to understand context and provide coherent responses.

3. What would break in a production AI system if we ignore message history?
Without message history, systems would produce inconsistent, incorrect, or irrelevant responses. This would break user experience, reduce trust, and lead to failures in real-world applications like customer support or decision systems.
"""