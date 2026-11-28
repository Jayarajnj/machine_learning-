#Package Imstallation and Instance Creation

!pip install -q fastapi uvicorn
!pip install -q -U transformers accelerate bitsandbytes fastmcp


#Model(LLM) Implementation


import os, torch, json, asyncio
from fastmcp import FastMCP, Client
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


HF_TOKEN = "hf_token" # Optional: make it available to Hugging Face libraries os.environ["HF_TOKEN"] = HF_TOKEN

model_id = "google/gemma-2-2b-it"

bnb_config = BitsAndBytesConfig( load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16 )

tokenizer = AutoTokenizer.from_pretrained( model_id, token=HF_TOKEN )

model = AutoModelForCausalLM.from_pretrained( model_id, token=HF_TOKEN, quantization_config=bnb_config, device_map="auto" )

print("Model successfully loaded!")
print(f"Model: {model_id}")



# Cell 1: generation helper
def chat_with_gemma(history, max_new_tokens=512, temperature=0.7, top_p=0.9):
    inputs = tokenizer.apply_chat_template(
        history,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# Cell 2: interactive chat loop (type "exit" to stop, "reset" to clear history)
history = []

print("Chat started. Type 'exit' to quit, 'reset' to clear history.\n")
while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ("exit", "quit"):
        print("Chat ended.")
        break
    if user_input.lower() == "reset":
        history = []
        print("History cleared.\n")
        continue

    history.append({"role": "user", "content": user_input})
    reply = chat_with_gemma(history)
    history.append({"role": "assistant", "content": reply})

    print(f"Gemma: {reply}\n")

import threading
from typing import List, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Gemma 2 Chat API")

# The model lives on one GPU, so only let one request generate at a time
gpu_lock = threading.Lock()


# ---- Request / response schemas (FastAPI validates these automatically) ----
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: int = Field(256, ge=1, le=1024)
    temperature: float = Field(0.7, ge=0.1, le=2.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)


class ChatResponse(BaseModel):
    reply: str


# ---- Endpoints ----
@app.get("/health")
def health():
    return {"status": "ok", "model": model_id}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages or req.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from the user.")

    history = [m.model_dump() for m in req.messages]

    with gpu_lock:
        reply = chat_with_gemma(
            history,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
    return ChatResponse(reply=reply)

import uvicorn

config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
server = uvicorn.Server(config)

thread = threading.Thread(target=server.run, daemon=True)
thread.start()



######################## Single request/chat.


import requests

BASE = "http://127.0.0.1:8000"

print(requests.get(f"{BASE}/health").json())

payload = {
    "messages": [{"role": "user", "content": "Explain FastAPI in two sentences."}],
    "max_new_tokens": 150,
}
r = requests.post(f"{BASE}/chat", json=payload)
print(r.status_code)
print(r.json()["reply"])






#######################   multi turn chat


import requests

BASE = "http://127.0.0.1:8000"

# ===== ADJUST THIS NUMBER =====
MAX_TURNS = 10   # number of past exchanges (your message + Gemma's reply) to remember
# ==============================


def api_chat(messages, max_new_tokens=256, temperature=0.7):
    r = requests.post(
        f"{BASE}/chat",
        json={
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["reply"]


history = []
print(f"Chat started (remembers last {MAX_TURNS} exchanges). Type 'exit' to quit, 'reset' to clear.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ("exit", "quit"):
        print("Chat ended.")
        break
    if user_input.lower() == "reset":
        history = []
        print("History cleared.\n")
        continue

    history.append({"role": "user", "content": user_input})

    # Keep only the last MAX_TURNS exchanges, plus the new user message
    # Each exchange = 2 messages, so we keep (2 * MAX_TURNS) + 1 messages
    history = history[-(2 * MAX_TURNS + 1):]

    try:
        reply = api_chat(history)
    except requests.exceptions.RequestException as e:
        history.pop()  # remove the unanswered user message
        print(f"API error: {e}\n")
        continue

    history.append({"role": "assistant", "content": reply})
    print(f"Gemma: {reply}\n")
