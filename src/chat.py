"""
Combine retrieval with generation: given a user question, retrieve
relevant chunks, then ask an LLM to answer using ONLY that retrieved
context, citing which document it came from.

Uses Groq's free API (OpenAI-compatible) running Llama 3.3 70B, so
this project runs with zero API cost. Requires the GROQ_API_KEY
environment variable to be set (see README for how to set this via a
.env file, and how to get a free key at console.groq.com).
"""

import os
import sys
from pathlib import Path

from retrieve import Retriever

ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = """You are a documentation assistant. Answer the user's \
question using ONLY the context provided below. If the context does not \
contain enough information to answer, say so plainly instead of guessing \
or using outside knowledge.

When you answer, mention which source document(s) you used, e.g. \
"(source: precision_recall.md)".

Context:
{context}
"""


def load_env_file(path: Path = ROOT / ".env"):
    """Minimal .env loader (avoids requiring python-dotenv as a dependency)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def format_context(chunks):
    parts = []
    for c in chunks:
        parts.append(f"[from {c['source']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def get_client():
    try:
        from openai import OpenAI
    except ImportError:
        print(
            "The 'openai' package isn't installed.\n"
            "Run: pip install openai",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "GROQ_API_KEY is not set. Add it to a .env file in the "
            "project root (see README) or export it in your shell.\n"
            "Get a free key at https://console.groq.com/keys",
            file=sys.stderr,
        )
        sys.exit(1)

    # Groq exposes an OpenAI-compatible API, so the openai SDK works
    # unchanged aside from pointing base_url at Groq's endpoint.
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def answer_question(question: str, retriever: Retriever, client, model: str = "openai/gpt-oss-120b"):
    chunks = retriever.retrieve(question)

    if not chunks:
        return (
            "I don't have information about that in my knowledge base, "
            "so I can't answer confidently.",
            [],
        )

    context = format_context(chunks)
    system = SYSTEM_PROMPT.format(context=context)

    response = client.chat.completions.create(
        model=model,
        max_tokens=500,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
    )

    answer_text = response.choices[0].message.content
    sources = sorted(set(c["source"] for c in chunks))
    return answer_text, sources


def main():
    load_env_file()
    retriever = Retriever()
    client = get_client()

    print("RAG chatbot ready. Ask a question (or type 'quit' to exit).\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        answer, sources = answer_question(question, retriever, client)
        print(f"\nBot: {answer}")
        if sources:
            print(f"     (retrieved from: {', '.join(sources)})")
        print()


if __name__ == "__main__":
    main()
