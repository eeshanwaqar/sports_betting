# ADR 002: Feature Store

## Status
Accepted

## Context
Need consistent features between training and serving.

## Decision
Use Feast with Redis for:
- Open source
- Supports offline/online stores
- Industry adoption

## Consequences
- Additional infrastructure (Redis)
- Learning curve for Feast
