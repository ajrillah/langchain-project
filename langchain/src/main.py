from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.messages import (
    HumanMessage,
    SystemMessage,
)

# Load API Key
from dotenv import load_dotenv

try:
    load_dotenv()
    print("[Checkpoint] " + "Load API Key Succesfull")
except:
    print("[Checkpoint] " + "Load API Key Failed")


# Initiating Model Open Source
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",
)

chat_model = ChatHuggingFace(llm=llm)

# Create Message
messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content="jelaskan tentang digitalisasi maksimal 1 paragraf"),
]

# Invoke/Send Message to Model
ai_msg = None

try:
    ai_msg = chat_model.invoke(messages)
    print("[Checkpoint] " + "Invoke Success")
except Exception as e:
    print("[Checkpoint] " + "Invoke Failed")
    print(e)

# Filter Thingking
content = ai_msg.content
if "<think>" in content:
    content = content.split("</think>")[-1].strip()

# Result Printing
print("[Result] " + content)
