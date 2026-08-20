# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

TODO(student): thay baseline placeholder bằng một call LLM thật.

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

TODO(student): implement routing policy.

Gợi ý câu hỏi thiết kế:

- Khi nào gọi Researcher?
- Khi nào gọi Analyst?
- Khi nào gọi Writer?
- Khi nào stop?
- Nếu agent fail thì retry hay fallback?

## Milestone 3: Worker agents

File gợi ý:

- `src/multi_agent_research_lab/agents/researcher.py`
- `src/multi_agent_research_lab/agents/analyst.py`
- `src/multi_agent_research_lab/agents/writer.py`

TODO(student): implement từng worker.

## Milestone 4: Trace và benchmark

File gợi ý:

- `src/multi_agent_research_lab/observability/tracing.py`
- `src/multi_agent_research_lab/evaluation/benchmark.py`
- `src/multi_agent_research_lab/evaluation/report.py`

Benchmark tối thiểu:

| Metric | Cách đo gợi ý |
|---|---|
| Latency | wall-clock time |
| Cost | token usage hoặc provider usage |
| Quality | rubric 0-10 do peer review |
| Citation coverage | số claims có source / tổng claims chính |
| Failure rate | số query fail / tổng query |

## Troubleshooting

### macOS: lỗi SSL certificate khi gọi API qua HTTPS (Tavily, OpenAI, ...)

Triệu chứng: khi implement `SearchClient` (hoặc bất kỳ HTTPS call nào) trên macOS, bạn có thể gặp lỗi kiểu:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Nguyên nhân: Python cài từ python.org trên macOS **không dùng** certificate store của hệ điều hành, nên không tìm thấy CA bundle hợp lệ. Đây là lỗi môi trường, **không phải** do API key sai.

Cách khắc phục (chọn 1 trong 3):

1. **Chạy script cài certificate đi kèm Python** (nhanh nhất):

   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

   (thay `3.12` bằng version Python của bạn)

2. **Dùng `certifi` trong code** — thêm `certifi` vào dependencies, rồi tạo SSL context khi gọi HTTPS:

   ```python
   import certifi
   import ssl
   from urllib.request import urlopen

   ssl_context = ssl.create_default_context(cafile=certifi.where())
   urlopen(request, timeout=timeout, context=ssl_context)
   ```

3. **Set biến môi trường** trỏ tới CA bundle của certifi (không cần đổi code):

   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

## Running Benchmark

Chạy benchmark để so sánh baseline vs multi-agent:

```bash
# Cài đặt dependencies
pip install -e ".[dev,llm]"

# Chạy baseline
make run-baseline

# Chạy multi-agent
make run-multi
```

Để chạy benchmark tự động với nhiều queries, tạo file `scripts/run_benchmark.py`:

```python
from multi_agent_research_lab.evaluation.benchmark import run_comparative_benchmark
from multi_agent_research_lab.evaluation.report import render_comparison_report
from multi_agent_research_lab.cli import baseline, multi_agent

queries = [
    "Research GraphRAG state-of-the-art",
    "Compare single-agent and multi-agent workflows",
    "Summarize production guardrails for LLM agents"
]

results = run_comparative_benchmark(queries, baseline, multi_agent)
report = render_comparison_report(
    [m for _, m in results["baseline"]],
    [m for _, m in results["multi_agent"]]
)
print(report)
```

## Exit ticket

### 1. Case nào nên dùng multi-agent? Vì sao?

**Nên dùng multi-agent khi:**

- **Chất lượng câu trả lời quan trọng** — Phân chia task cho nhiều agents chuyên biệt (researcher tìm kiếm, analyst phân tích, writer viết) cho ra kết quả tốt hơn single agent làm tất cả.

- **Cần citations/nguồn đáng tin cậy** — Multi-agent có workflow rõ ràng để thu thập và đánh giá nguồn, đảm bảo trích dẫn chính xác.

- **Cần debug được** — State được chia sẻ qua lại giữa các agents, dễ trace xem agent nào làm gì, lỗi ở đâu.

- **Task phức tạp, nhiều bước** — Khi cần research → analyze → write, multi-agent tách bạch rõ ràng, dễ maintain.

- **Production system cần mở rộng** — Thêm agent mới (critic, validator) không cần sửa agent cũ.

### 2. Case nào không nên dùng multi-agent? Vì sao?

**Không nên dùng multi-agent khi:**

- **Query đơn giản, chỉ cần factual answer** — "Thời tiết hôm nay thế nào?" → single agent nhanh hơn, multi-agent chỉ thêm overhead.

- **Tốc độ là ưu tiên #1** — Multi-agent gọi nhiều LLM calls hơn → latency cao hơn 3-5 lần.

- **Chi phí phải thấp** — Mỗi agent call tốn thêm tokens và API costs. Simple query không justify được chi phí.

- **Prototype nhanh** — Khi chỉ cần test ý tưởng, single agent với system prompt đủ dùng.

- **Infrastructure hạn chế** — Multi-agent cần quản lý state phức tạp hơn, nhiều điểm failure hơn.

- **Task có đầu ra cần nhất quán** — Pipeline dài tăng cơ hội output không deterministic.

**Tóm lại:** Multi-agent là trade-off giữa quality/debuggability và speed/cost. Chọn dựa trên requirements cụ thể của bài toán.
