# Lab 20: Multi-Agent Research System

Xây dựng hệ thống nghiên cứu gồm **Supervisor + Researcher + Analyst + Writer + Critic** và benchmark với single-agent baseline.

## Architecture

```text
User Query
   |
   v
Supervisor / Router
   |------> Researcher Agent  -> sources + research_notes
   |------> Analyst Agent     -> analysis_notes
   |------> Writer Agent      -> final_answer
   |------> Critic Agent      -> critique (optional)
   |
   v
Trace + Benchmark Report
```

## Cấu trúc repo

```
src/multi_agent_research_lab/
├── agents/              # Supervisor, Researcher, Analyst, Writer, Critic
├── core/                # Config, state, schemas, errors
├── graph/               # LangGraph workflow
├── services/            # LLM client (OpenAI), Search client (Tavily)
├── evaluation/          # Benchmark runner, report generator
├── observability/       # Tracing (LangSmith)
└── cli.py               # CLI entrypoint
```

## Quickstart

### 1. Tạo môi trường

```bash
pip install -e ".[dev]"
cp .env.example .env
```

### 2. Cấu hình API keys

```bash
OPENAI_API_KEY=...     # Required
TAVILY_API_KEY=...     # Optional (mock available)
LANGSMITH_API_KEY=...  # Optional (tracing)
```

### 3. Chạy

```bash
# Baseline
make run-baseline --query "Research GraphRAG state-of-the-art"

# Multi-agent
make run-multi --query "Research GraphRAG state-of-the-art"

# Benchmark comparison
make run-benchmark --query "Research GraphRAG state-of-the-art"

# Tests
make test
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `baseline` | Single-agent baseline |
| `multi-agent` | Multi-agent workflow |
| `benchmark` | Compare baseline vs multi-agent |
| `compare` | Compare across multiple queries |

## Features

- **LLM Client**: OpenAI with retry, cost tracking, mock fallback
- **Search**: Tavily API with mock fallback
- **Agents**: Supervisor (routing), Researcher, Analyst, Writer, Critic
- **Workflow**: LangGraph with conditional edges
- **Tracing**: Local + LangSmith integration
- **Benchmark**: Latency, cost, quality, citation coverage metrics

## Metrics

| Metric | Baseline | Multi-Agent |
|--------|----------|-------------|
| Latency | Lower | Higher |
| Cost | Lower | Higher |
| Quality | Baseline | Better citations |
| Debugability | Low | High (full state) |

## References

- [LangGraph concepts](https://langchain-ai.github.io/langgraph/concepts/)
- [LangSmith tracing](https://docs.smith.langchain.com/)
- [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
