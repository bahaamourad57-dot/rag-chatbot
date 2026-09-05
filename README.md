# RAG Chatbot: Documentation Q&A

A small retrieval-augmented generation (RAG) chatbot that answers
questions using only a local set of documents as its knowledge base —
and says "I don't know" rather than hallucinating when the answer
isn't in there. Runs entirely on free tiers (no paid API required).

## Why RAG, and why this design

An LLM on its own can't answer questions about documents it's never
seen (your internal notes, a private knowledge base, etc.), and asking
it to just "know" your content invites hallucination. RAG fixes this
by retrieving the relevant text first, then asking the model to answer
using only that retrieved context.

**Retrieval and generation are deliberately decoupled** into separate
modules (`retrieve.py` vs `chat.py`). Retrieval uses TF-IDF + cosine
similarity rather than a neural embedding model — it needs no API
calls, no GPU, and is fully deterministic, which makes it something
you can actually evaluate and debug offline. For this size of document
set it retrieves well (see Results below); swapping in a real
embedding model later would only mean changing `ingest.py` and
`retrieve.py`, not the generation logic.

## How it works

```
Question → Retriever (TF-IDF cosine similarity, top-3 chunks)
         → if nothing scores above threshold: return "I don't know"
         → else: build a prompt with the retrieved chunks as context
         → Llama 3.3 70B (via Groq's free API) generates an answer,
           citing which doc it came from
```

## Results

Retrieval evaluated against a labeled set of 6 test questions
(`data/eval_questions.json`), checking whether the correct source
document appears in the top-3 retrieved chunks:

**6/6 (100%) retrieval accuracy**

An off-topic query (e.g. asking for a cake recipe) correctly returns
no results rather than forcing a match — this is what lets the chatbot
say "I don't know" instead of hallucinating an answer from irrelevant
context.

## Project structure

```
rag-chatbot/
├── data/
│   ├── docs/                # knowledge base (add your own .md files here)
│   └── eval_questions.json  # labeled test set for retrieval evaluation
├── src/
│   ├── ingest.py            # load, chunk, and index documents (TF-IDF)
│   ├── retrieve.py          # cosine-similarity retrieval over the index
│   ├── chat.py              # retrieval + Claude generation, CLI chat loop
│   └── eval_retrieval.py    # retrieval accuracy against labeled test set
├── index/                    # generated TF-IDF index (from ingest.py)
├── requirements.txt
├── .env.example
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

# Add a free Groq API key (no credit card required) — get one at
# https://console.groq.com/keys
cp .env.example .env
# then edit .env and paste your real key in place of "your-key-here"

cd src
python ingest.py            # builds the retrieval index from data/docs/
python eval_retrieval.py    # checks retrieval accuracy (no API key needed)
python chat.py              # starts the interactive chatbot (needs API key)
```

## Using your own documents

Drop `.md` files into `data/docs/`, then re-run `python ingest.py` to
rebuild the index. No other code changes needed.

## Limitations

- TF-IDF retrieval is keyword-based, not semantic — it can miss a
  relevant chunk that uses very different wording than the query, even
  if the meaning matches. A production system with a larger, more
  varied document set would likely benefit from a real embedding model.
- The "I don't know" threshold was tuned by hand against this small
  document set; it would need re-checking against a larger corpus.
- No conversation memory — each question is answered independently,
  with no awareness of prior turns in the chat.

## Stack

Python, scikit-learn (TF-IDF, cosine similarity), Groq API (Llama 3.3 70B)
