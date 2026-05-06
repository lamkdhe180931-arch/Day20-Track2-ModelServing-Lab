# Reflection — Lab 20 (Personal Report)

> **Đây là báo cáo cá nhân.** Mỗi học viên chạy lab trên laptop của mình, với spec của mình. Số liệu của bạn không so sánh được với bạn cùng lớp — chỉ so sánh **before vs after trên chính máy bạn**. Grade rubric tính theo độ rõ ràng của setup + tuning của bạn, không phải tốc độ tuyệt đối.

---

**Họ Tên:** Kiều Đức Lâm
**Cohort:** A20
**Ngày submit:** 2026-05-06

---

## 1. Hardware spec (từ `00-setup/detect-hardware.py`)

- **OS:** macOS (Apple Silicon)
- **CPU:** Apple M2
- **Cores:** 8 physical / 8 logical
- **CPU extensions:** NEON (ARM64)
- **RAM:** 8.0 GB
- **Accelerator:** Apple Metal (Apple Silicon unified memory)
- **llama.cpp backend đã chọn:** Metal (`-DGGML_METAL=on`)
- **Recommended model tier:** Qwen2.5-1.5B-Instruct (Q4_K_M)

**Setup story:** Setup chạy suôn sẻ trên macOS Apple Silicon. `macos-setup.sh` tự detect arm64 và build llama-cpp-python với Metal backend. Điểm duy nhất cần xử lý thêm là `python -m llama_cpp.server` thiếu dependencies (`uvicorn`, `starlette`, `fastapi`) — cài bổ sung bằng pip. Sau đó dùng binary `llama-server` từ Homebrew để có `/metrics` Prometheus endpoint.

---

## 2. Track 01 — Quickstart numbers (từ `benchmarks/01-quickstart-results.md`)

Settings: `n_threads=8`, `n_ctx=2048`, `n_batch=512`, `n_gpu_layers=99`

| Model | Load (ms) | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | E2E P50/P95/P99 (ms) | Decode rate (tok/s) |
|---|--:|--:|--:|--:|--:|
| qwen2.5-1.5b-instruct-q4_k_m.gguf | 12076 | 56 / 202 | 16.6 / 17.0 | 1118 / 1169 / 1196 | 60.3 |
| qwen2.5-1.5b-instruct-q2_k.gguf   | 1164  | 54 / 119 | 15.5 / 16.6 | 1058 / 1097 / 1100 | 64.6 |

**Một quan sát:** Q2_K nhanh hơn Q4_K_M ~7% về decode rate (64.6 vs 60.3 tok/s) và load nhanh hơn 10× (1164 ms vs 12076 ms). Tuy nhiên chênh lệch latency E2E rất nhỏ (~60 ms ở P50). Trên 8 GB RAM, Q4_K_M là lựa chọn tốt hơn — chất lượng output rõ ràng hơn trong khi latency gần như tương đương. Q2_K chỉ nên dùng khi RAM thực sự tight (< 4 GB).

---

## 3. Track 02 — llama-server load test

Server khởi động với: `llama-server --model qwen2.5-1.5b-instruct-q4_k_m.gguf --host 0.0.0.0 --port 8080 --threads 8 --n-gpu-layers 99 --parallel 4 --ctx-size 2048 --metrics`

| Concurrency | Total RPS | TTFB P50 (ms) | E2E P95 (ms) | E2E P99 (ms) | Failures |
|--:|--:|--:|--:|--:|--:|
| 10 | 1.34 | 5600 | 7900 | 9000 | 0 |
| 50 | — | — | — | — | — |

*(load-50 chưa hoàn thành — cần chạy thêm `make load-50`)*

**KV-cache observation:** Từ `/metrics` sau smoke test: `llamacpp:tokens_predicted_total=21`, `llamacpp:prompt_tokens_total=31`. Ở concurrency 10 với `--parallel 4`, server phải queue requests vì chỉ có 4 slots — điều này giải thích P95 latency ~7900 ms cao hơn đáng kể so với single-request latency ~1100 ms. Peak `llamacpp:kv_cache_usage_ratio` ở load-10 ước tính ~0.80–0.90 (4 slots × 2048 ctx trên 8 GB RAM unified memory).

---

## 4. Track 03 — Milestone integration

- **N16 (Cloud/IaC):** stub — localhost only, không có K8s cluster
- **N17 (Data pipeline):** stub — không có Airflow DAG
- **N18 (Lakehouse):** stub — không có Delta Lake / Iceberg
- **N19 (Vector + Feature Store):** stub — dùng in-memory TOY_DOCS với keyword overlap scoring (không có Qdrant/Feast thật)

**Nơi tốn nhiều ms nhất** trong pipeline (đo bằng `time.perf_counter`):

- embed/retrieve: ~0.1 ms (toy keyword matching, không có real embedder)
- llama-server: 756 ms – 6236 ms (chiếm >99.9% total time)
- total: 756 ms – 6236 ms tùy query

**Reflection:** Bottleneck tuyệt đối nằm ở llama-server — retrieval toy chỉ mất < 1 ms. Điều này khớp kỳ vọng: với real vector index (Qdrant), embed + retrieve sẽ tốn thêm 20–100 ms, nhưng LLM decode vẫn chiếm 95%+ của total latency. Để cải thiện TTFT trong production RAG, cần tập trung vào prompt prefix caching (system prompt không đổi → cache hit) hơn là optimize retrieval layer.

---

## 5. Bonus — The single change that mattered most

**Change:** Bật Metal GPU offload với `--n-gpu-layers 99` (toàn bộ model layers lên Apple Silicon GPU / unified memory)

**Before vs after** (ước tính dựa trên llama.cpp benchmark patterns trên M2 8GB):

```
before: CPU-only inference, n_gpu_layers=0 → ~15–20 tok/s (estimated)
after:  Metal full offload, n_gpu_layers=99 → 60.3 tok/s (Q4_K_M measured)
speedup: ~3–4×
```

**Tại sao nó work:** Apple Silicon dùng unified memory architecture — CPU và GPU chia sẻ cùng một bộ nhớ vật lý, không cần copy data qua PCIe bus như NVIDIA. Khi chạy inference trên CPU, mỗi matrix multiplication trong attention phải load weights từ RAM qua CPU memory controller. Khi offload lên Metal GPU, cùng weights đó được access bởi GPU compute units với bandwidth cao hơn vì GPU có nhiều execution units để pipeline memory access song song. Qwen2.5-1.5B ở Q4_K_M chiếm ~1.0 GB — fit hoàn toàn trong 8 GB unified memory, không có eviction hay swapping. Đây là lý do Metal offload trên Apple Silicon cho speedup mạnh nhất ở dải model nhỏ: model fit VRAM, GPU compute thắng rõ ràng so với CPU sequential execution.

---

## 6. Điều ngạc nhiên nhất

Load time của Q4_K_M (12076 ms) gấp 10× Q2_K (1164 ms) dù file size chỉ gấp ~1.6×. Lý do là Q4_K_M dùng k-quant mixed precision cần dequantize phức tạp hơn trong quá trình load vào Metal buffer — không phải linear với file size.

---

## 7. Self-graded checklist

- [x] `hardware.json` đã commit
- [x] `models/active.json` đã commit (primary + compare GGUF)
- [x] `benchmarks/01-quickstart-results.md` đã commit
- [ ] `benchmarks/02-server-results.md` (hoặc CSV từ `record-metrics.py`) đã commit
- [ ] `benchmarks/bonus-*.md` đã commit (ít nhất 1 sweep)
- [ ] Ít nhất 6 screenshots trong `submission/screenshots/` (cần thêm: `02-quickstart-bench.png`, `05-locust-50.png`)
- [ ] `make verify` exit 0 (chạy ngay trước khi push)
- [ ] Repo trên GitHub ở chế độ **public**
- [ ] Đã paste public repo URL vào VinUni LMS

---

**Quan trọng:** repo phải **public** đến khi điểm được công bố. Nếu private, grader không xem được → 0 điểm.
