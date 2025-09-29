import os
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Create an assistant (only once, you can reuse its ID later)
assistant = client.beta.assistants.create(
    name="FOC Website Q&A Bot",
    instructions="You are a helpful assistant that answers questions about the Faculty of Computing, University of Sri Jayewardenepura.",
    model="gpt-4o-mini",   # fast + cost efficient
)

print("Assistant created:", assistant.id)

# Step 2: Create a thread (a conversation context)
thread = client.beta.threads.create()

# Step 3: Add a user message
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="What are the available courses?"
)

# Step 4: Run the assistant on the thread
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# Step 5: Print responses
messages = client.beta.threads.messages.list(thread_id=thread.id)
for msg in messages.data:
    if msg.role == "assistant":
        print("Assistant:", msg.content[0].text.value)
