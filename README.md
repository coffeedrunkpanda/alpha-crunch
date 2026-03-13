# AlphaCrunch: Automated Investment Analyst Agent 🏦🤖

AlphaCrunch is a LangGraph-powered conversational AI agent using a QLoRA fine-tuned Mistral-7B-Instruct-v0.2 Finance LLM, trained on virattt/financial-qa-10K SEC Q&A data. It powers RAG (ChromaDB) analysis of S&P 500 10-K filings from the jlohding/sp500-edgar-10k with multi-turn memory and Gradio UI.

## 📁 Repository Structure
```
├── .env.sample              # Env vars (Modal token, paths)
├── .gitignore
├── LICENSE
├── pyproject.toml           # UV deps
├── uv.lock
├── README.md                # This file
├── requirements.txt         # Legacy pip
├── data/                    # Chroma_db, fiqa, etc
├── notebooks/               # EDA, experiments
├── outputs/                 # LoRA adapter, checkpoints
│   └── finance-llm-adapter/
├── scripts/                 # ingest_data.py, test_agent.py
├── src/alpha_crunch/        # Core package (see sub-structure)
├── .venv/
└── wandb/                   # W&B runs (LoRA eval)
```

## Core Features

### Finance LLM (QLoRA Fine-Tuning)

Fine-tuned on 70% RAG-simulated (context + question → answer) and 30% pure knowledge format; 80/10/10 train/val/test split.

| Key Params | Value                    | Reason                                   |
| ---------- | ------------------------ | ---------------------------------------- |
| LoRA rank  | r=16, alpha=32           | Capacity/VRAM balance (fits RTX 5070 Ti) |
| Quant      | 4-bit NF4 + double quant | 16GB VRAM                                |
| Targets    | q,k,v,o,gate,up,down     | Near full fine-tune quality              |
| Template   | Mistral [INST] chat      | Instruction-tuned base                   |

Adapter (~30MB) saved in ./outputs/finance-llm-adapter; eval with BERTScore, LLM-as-Judge, logged to W&B.

- [WandB LoRA fine-tunning dashboard](https://wandb.ai/coffeedrunk/finance-llm/workspace?nw=nwusercoffeedrunk)
- [WandB Evals dashboard](https://wandb.ai/coffeedrunk/finance-llm-evals/workspace?nw=nwusercoffeedrunk)


### RAG
- Data pipeline: Top 50 S&P by weight (Mar 2026 Slickcharts), CIK-filtered from jlohding/sp500-edgar-10k, 47 unique names post-deduplication.
- Deterministic NER with regex on immutable COMPANY_REGISTRY tuple (500+ S&P names), longest-first matching, and alias resolution (e.g., "Google" → "ALPHABET"). 
- Hybrid RAG: ChromaDB filters by exact company metadata before semantic search with all-mpnet-base-v2 embeddings; @lru_cache singleton optimizations.

### Agent
- LangGraph flow: intent_node → conditional (rag/analyst/help) → Mistral Finance LLM; MemorySaver for sessions.
- Multi-turn memory via LangGraph add_messages reducer and MemorySaver checkpointer with thread_id sessions. 
- Intent routing: "rag", "analyst", "help" categories; state hygiene clears retrieved_context to prevent leakage. 

### UI
- Gradio 6.0 UI: Streaming, custom Anta/Courier fonts, mesh gradient CSS, glassmorphism.

## Architecture

```bash
Gradio UI → LangGraph (AgentState: messages, intent, context, answer)
           ↓ intent_node (Finance LLM)
      ┌──────┼──────┐
  RAG    Analyst   Help
(Chroma) (Reason) (Info)
         ↓ Answer

```


```
src/alpha_crunch/
  agent/
    config.py     # Immutable registry, paths, aliases
    vector_store.py # ChromaDB singleton
    rag_node.py   # LangGraph RAG execution
  state.py       # Pydantic AgentState
```
User input flows: Gradio ChatInterface → LangGraph graph (intent_node → conditional → llm_node) → Mistral 7B on Modal. [gradio](https://www.gradio.app/4.44.1/docs/gradio/chatinterface)

## Quickstart

1. Clone and install with uv:
   ```bash
   git clone git@github.com:coffeedrunkpanda/alpha-crunch.git
   cd alphacrunch
   uv sync
   ```

Copy .env.sample → .env (add Modal API key).

2. Ingest top 50 S&P 500 filings (CIK-filtered, name-standardized):
   ```bash
   uv run scripts/rag/ingest_data.py
   ```

3. Launch Gradio UI:
   ```bash
    uv run python src/alpha_crunch/app.py
   ```

   Features custom CSS (loomy mesh gradient, Anta/Courier Prime fonts, glassmorphism). [gradio](https://www.gradio.app/4.44.1/docs/gradio/chatinterface)

## Example Usage

Query: "What are Apple's main supply chain risks?"  
Answer: Apple's supply chain risks include disruptions in manufacturing or logistics, sole-sourcing reliance on certain vendors for critical components, and foreign currency exchange rate fluctuations. These risks could materially affect the Company’s financial condition and operating results.

### Sample questions:

 - what is asset allocation?

- what is the sec fillings 10k? why does it matter? 

- what are the most important concepts in investment? 

- What are Apple's main supply chain risks?

- Describe tesla's business.

## Limitations & Next

- Single company/query only—no multi-compare yet.
- Add more metadata filtering to improve the accuracy of the agent. 
- No information on stock prices. Next: yfinance integration.

## Tech Stack

- LangGraph for stateful agents with custom nodes and routing. 
- ChromaDB local vector store; uv package management; python-dotenv.
- QLoRA-tuned Mistral-7B-Instruct (finance-specialized on SEC Q&A); Gradio ChatInterface. 
