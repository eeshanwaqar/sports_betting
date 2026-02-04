# ADR 003: RAG Architecture

## Status
Accepted

## Context
Need natural language interface for predictions.

## Decision
Use LangChain + ChromaDB + Ollama for:
- LangChain: Industry standard for RAG
- ChromaDB: Simple, local-first vector DB
- Ollama: Free local LLM inference

## Consequences
- Can switch to OpenAI for production
- Need to manage embeddings
