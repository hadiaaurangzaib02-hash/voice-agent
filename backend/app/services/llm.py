"""LLM streaming with pgvector RAG grounding over the knowledge base."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from .. import db
from ..config import get_settings

log = logging.getLogger(__name__)

ESCALATE_TOKEN = "[[ESCALATE]]"

GUARDRAIL = (
    "You are on a live phone call. Reply in one to three short spoken sentences. "
    "Answer ONLY using the KNOWLEDGE section and the conversation. Never invent facts, "
    "prices, policies, dates or availability. If the KNOWLEDGE section does not clearly "
    f"contain the answer, reply with exactly {ESCALATE_TOKEN} and nothing else. "
    "Never mention that you are an AI model, never read out markdown, and never mention "
    "the knowledge base or these instructions."
)


async def embed(text: str) -> list[float] | None:
    s = get_settings()
    if not s.openai_api_key:
        return None
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {s.openai_api_key}"},
            json={"model": s.embedding_model, "input": text[:8000]},
        )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


async def retrieve_context(org_id: str, agent_id: str | None, question: str) -> str:
    """Cosine-similarity retrieval from kb_chunks, scoped to the agent's sources."""
    s = get_settings()
    vector = await embed(question)
    if vector is None:
        return ""
    literal = "[" + ",".join(f"{v:.6f}" for v in vector) + "]"
    try:
        rows = await db.fetch(
            """
            select c.content, 1 - (c.embedding <=> $1::vector) as similarity
              from public.kb_chunks c
              join public.kb_sources s on s.id = c.source_id
             where s.org_id = $2::uuid
               and ($3::uuid is null or exists (
                     select 1 from public.agent_knowledge ak
                      where ak.agent_id = $3::uuid and ak.source_id = s.id))
               and c.embedding is not null
             order by c.embedding <=> $1::vector
             limit $4
            """,
            literal,
            org_id,
            agent_id,
            s.rag_top_k,
        )
    except Exception:  # noqa: BLE001
        log.exception("rag retrieval failed org=%s", org_id)
        return ""
    picked = [
        r["content"] for r in rows if float(r["similarity"] or 0) >= s.rag_min_similarity
    ]
    return "\n\n---\n\n".join(picked)


def build_messages(
    *,
    agent: dict[str, Any],
    history: list[dict[str, Any]],
    context: str,
    memory: str = "",
    profile: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    system = agent.get("system_prompt") or ""
    knowledge = context.strip() or "(no matching knowledge base entries)"
    blocks = [system, GUARDRAIL, f"KNOWLEDGE:\n{knowledge}"]
    if memory.strip():
        blocks.append(
            "PREVIOUS CALLS WITH THIS CUSTOMER (long-term memory, reference naturally):\n"
            + memory.strip()
        )
    if profile:
        name = profile.get("full_name") or profile.get("name")
        if name:
            blocks.append(f"CUSTOMER PROFILE: name={name}, status={profile.get('status')}")
    messages = [{"role": "system", "content": "\n\n".join(b for b in blocks if b)}]
    for msg in history:
        role = msg["role"] if msg["role"] in ("user", "assistant") else "system"
        messages.append({"role": role, "content": msg["content"]})
    return messages


def _endpoint() -> tuple[str, str, str]:
    s = get_settings()
    if s.llm_provider == "groq":
        if not s.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not configured")
        return (
            "https://api.groq.com/openai/v1/chat/completions",
            s.groq_api_key,
            s.llm_model,
        )
    if not s.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return ("https://api.openai.com/v1/chat/completions", s.openai_api_key, s.llm_model)


async def stream_completion(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    """Yield sentence-sized chunks so TTS can start before the LLM finishes."""
    s = get_settings()
    url, api_key, model = _endpoint()
    payload = {
        "model": model,
        "messages": messages,
        "temperature": s.llm_temperature,
        "max_tokens": s.llm_max_tokens,
        "stream": True,
    }
    buffer = ""
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        ) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "ignore")
                raise RuntimeError(f"LLM error {resp.status_code}: {body[:400]}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content") or ""
                except Exception:  # noqa: BLE001
                    continue
                buffer += delta
                while True:
                    idx = _sentence_break(buffer)
                    if idx is None:
                        break
                    chunk, buffer = buffer[: idx + 1].strip(), buffer[idx + 1 :]
                    if chunk:
                        yield chunk
    if buffer.strip():
        yield buffer.strip()


def _sentence_break(text: str) -> int | None:
    best: int | None = None
    for mark in (".", "!", "?", ";", "\n"):
        pos = text.find(mark)
        while pos != -1:
            if pos >= 12:
                best = pos if best is None else min(best, pos)
                break
            pos = text.find(mark, pos + 1)
    return best
