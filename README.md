# 🥐 Maison Noisette: Agent-Powered Business Assistant

**EECE 503P/798S Agentic Systems · C3 Introduction to Agents**

Camille is the virtual concierge of *Maison Noisette*, a fictional French-Lebanese pâtisserie in Beirut. For every message, the agent:

- routes it with an LLM classifier,
- answers from the business profile, using a separate persona and instructions,
- calls tools to log leads and feedback,
- reflects on its draft before replying.

It runs in a Gradio chat UI and is powered by the **OpenAI** API.

## 🔗 Links

- 🎥 **Demo video:** [Watch on YouTube](https://youtu.be/GTi9oyQgRJ4?si=0UjIICWgV2aLFbBp)
- 🤗 **Live demo (Hugging Face Spaces):** [sambkttt/maison-noisette](https://huggingface.co/spaces/sambkttt/maison-noisette)
- 
## How it works

```
user message + recent history
        │
   ROUTER (LLM, JSON output)  →  general | lead | feedback
        │
   route note + allowed tools
        │
   AGENT LOOP: model(system_prompt + history + message, tools)
        ├── tool call → run tool (print + logs/*.jsonl) → result back to model → loop
        └── text → draft
        │
   REFLECTION: check draft vs instructions → pass / revise (both logged)
        │
   reply → Gradio ChatInterface (+ live agent-trace panel)
```

| Route | Handling | Tools available |
|---|---|---|
| `general` | Answer from the business info; log the question if it isn't covered | `record_feedback` |
| `lead` | Collect name, email and request details, then record the lead | `record_customer_interest`, `record_feedback` |
| `feedback` | Thank the customer and log feedback, or log an unanswerable question | `record_feedback` |

## Project structure

```
business_bot/
├── me/
│   ├── about_business.pdf      # full business profile (services, prices, team, policies)
│   └── business_summary.txt    # short summary
├── business_agent.ipynb        # main notebook: persona, instructions, tools, router, agent loop, reflection, Gradio
├── app.py                      # same agent code as a Gradio app (Hugging Face Spaces)
├── requirements.txt
├── .env.example                # copy to .env and add OPENAI_API_KEY (.env is git-ignored)
└── logs/                       # created at runtime: leads.jsonl, feedback.jsonl, reflection.jsonl
```

## Run locally (Linux)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyterlab ipykernel
cp .env.example .env            # then put your OpenAI key in .env
jupyter lab business_agent.ipynb   # Kernel → Restart & Run All
# or run just the app:
python app.py                   # http://127.0.0.1:7860
```

## Requirements checklist

| Requirement | Where |
|---|---|
| Business identity (name, mission, services, team, UVP) | `me/about_business.pdf`, `me/business_summary.txt` |
| Persona and instructions as separate, labeled blocks | Notebook §2 (🎭 Persona / 📋 Instructions) |
| Router (LLM) sends each message to its own handling | Notebook §5, with a labeled test demo |
| `record_customer_interest(email, name, message)` and `record_feedback(question)` | Notebook §3: print + `logs/*.jsonl`, with MCP-style schemas |
| System prompt: in character, uses both sources, logs unknowns, encourages leads | Notebook §4 |
| Agent loop: message + history + tools, executes tool calls | Notebook §6 |
| Gradio ChatInterface | Notebook §9 / `app.py` |
| ⭐ Reflection step with logged draft and revised reply | Notebook §7, `logs/reflection.jsonl` |
| ⭐ Hugging Face Spaces | `app.py`; link above |

## Deploy to Hugging Face Spaces

1. Create a new Space with the **Gradio** SDK.
2. Upload `app.py`, `requirements.txt` and the `me/` folder.
3. Go to Space **Settings → Variables and secrets** and add a secret named `OPENAI_API_KEY`.
