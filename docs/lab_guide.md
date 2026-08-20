# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer, Critic.

## Architecture

```
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

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Quick Start

```bash
# Cài đặt
pip install -e ".[dev]"

# Chạy baseline
make run-baseline

# Chạy multi-agent
make run-multi

# Benchmark comparison
make run-benchmark

# Tests
make test
```

## File Structure

| Module | Description |
|--------|-------------|
| `services/llm_client.py` | OpenAI LLM client với retry, cost tracking |
| `services/search_client.py` | Tavily search với mock fallback |
| `agents/supervisor.py` | Routing policy |
| `agents/researcher.py` | Search + notes |
| `agents/analyst.py` | Analysis + credibility |
| `agents/writer.py` | Final answer + citations |
| `agents/critic.py` | Fact-check + review |
| `graph/workflow.py` | LangGraph workflow |
| `observability/tracing.py` | LangSmith tracing |
| `evaluation/benchmark.py` | Benchmark runner |

## Benchmark Metrics

| Metric | Cách đo |
|--------|---------|
| Latency | wall-clock time |
| Cost | token usage từ API response |
| Quality | heuristic score (0-10) |
| Citation coverage | cited sources / total sources |
| Failure rate | failed runs / total runs |

## Running with Offline Corpus

```bash
# Chạy benchmark với offline corpus
python scripts/run_corpus_benchmark.py
```

Corpus tại: `ai_agent_offline_research_corpus_v2/`

## Troubleshooting

### macOS: lỗi SSL certificate

```bash
# Cách 1: Cài certificate
/Applications/Python\ 3.12/Install\ Certificates.command

# Cách 2: Set env variable
export SSL_CERT_FILE=$(python -m certifi)
```

### Tavily not installed

```bash
pip install tavily-python
```

### LangSmith tracing

```bash
# Thêm vào .env
LANGSMITH_API_KEY=xxx
```

---

## Exit Ticket

### 1. Case nào nên dùng multi-agent? Vì sao?

**Nên dùng multi-agent khi:**

| Lý do | Giải thích |
|-------|-------------|
| **Chất lượng câu trả lời quan trọng** | Phân chia task cho nhiều agents chuyên biệt (researcher tìm kiếm, analyst phân tích, writer viết, critic review) cho ra kết quả tốt hơn single agent làm tất cả. |
| **Cần citations/nguồn đáng tin cậy** | Multi-agent có workflow rõ ràng để thu thập và đánh giá nguồn, đảm bảo trích dẫn chính xác. |
| **Cần debug được** | State được chia sẻ qua lại giữa các agents, dễ trace xem agent nào làm gì, lỗi ở đâu. |
| **Task phức tạp, nhiều bước** | Khi cần research → analyze → write → critique, multi-agent tách bạch rõ ràng, dễ maintain. |
| **Production cần mở rộng** | Thêm agent mới (validator, fact-checker) không cần sửa agent cũ. |
| **Kiểm soát rủi ro** | Critic agent giúp catch hallucinations và weak claims trước khi trả lời. |

**Tóm lại:** Multi-agent phù hợp khi quality, debuggability, và reliability quan trọng hơn speed và cost.

---

### 2. Case nào không nên dùng multi-agent? Vì sao?

**Không nên dùng multi-agent khi:**

| Lý do | Giải thích |
|-------|-------------|
| **Query đơn giản, chỉ cần factual answer** | "Thời tiết hôm nay thế nào?" → single agent nhanh hơn, multi-agent chỉ thêm overhead không cần thiết. |
| **Tốc độ là ưu tiên #1** | Multi-agent gọi nhiều LLM calls hơn → latency cao hơn 3-5 lần. Mỗi agent call mất thêm 5-15s. |
| **Chi phí phải thấp** | Mỗi agent call tốn thêm tokens và API costs. Simple query không justify được chi phí gấp 4-5 lần. |
| **Prototype nhanh** | Khi chỉ cần test ý tưởng, single agent với system prompt đủ dùng. Multi-agent cần setup nhiều hơn. |
| **Infrastructure hạn chế** | Multi-agent cần quản lý state phức tạp hơn, nhiều điểm failure hơn, khó debug hơn. |
| **Task deterministic quan trọng** | Pipeline dài tăng cơ hội output không deterministic do nhiều LLM calls. |
| **Context window hạn chế** | Multi-agent state passing tốn thêm tokens cho context. |

**Tóm lại:** Single-agent phù hợp khi speed, cost, và simplicity quan trọng hơn quality và debuggability.

---

### Summary

| Criteria | Baseline | Multi-Agent |
|----------|----------|-------------|
| Speed | ⭐⭐⭐⭐⭐ Fast | ⭐⭐ Slower |
| Cost | ⭐⭐⭐⭐⭐ Low | ⭐⭐ Higher |
| Quality | ⭐⭐⭐ Average | ⭐⭐⭐⭐⭐ High |
| Debugability | ⭐⭐ Limited | ⭐⭐⭐⭐⭐ Full |
| Extensibility | ⭐⭐ Limited | ⭐⭐⭐⭐⭐ High |

**Recommendation:**
- **Dùng Baseline**: simple queries, prototyping, cost-sensitive, latency-critical
- **Dùng Multi-Agent**: production, quality-critical, complex research, long-term maintainability
