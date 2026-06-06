# EXACT 2026 — Hệ thống AI giải bài toán Logic & Vật lý

Hệ thống Agentic AI xây dựng cho cuộc thi **EXACT 2026** (lần thứ 2), sử dụng phương pháp **Neurosymbolic Hybrid** — kết hợp LLM mã nguồn mở với các bộ giải ký hiệu (Z3, SymPy) để trả lời chính xác các câu hỏi giáo dục.

## Mục tiêu

- Trả lời câu hỏi suy luận logic (quy chế đào tạo) — **Type 1**
- Giải bài toán vật lý STEM (mạch điện, tụ điện, lực Coulomb...) — **Type 2**
- Cung cấp lời giải thích chi tiết cho từng câu trả lời

## Ràng buộc cuộc thi

- Chỉ được dùng LLM mã nguồn mở, tối đa **8 tỷ tham số**
- Cấm sử dụng GPT, Claude, Gemini (model đóng)
- API endpoint phải trả kết quả trong tối đa **60 giây**
- Output phải có cấu trúc JSON (answer + explanation)

## Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| LLM | Qwen2.5-7B-Instruct (Fine-tuned Neurosymbolic LoRA) |
| Quantization | BitsAndBytes 4-bit NF4 + bfloat16 |
| Logic Solver | Z3 Theorem Prover (Microsoft) |
| Math Solver | SymPy |
| Pipeline | LangGraph (Python Sandbox) |
| RAG | LlamaIndex + Qdrant + BM25 Hybrid Search |
| Embedding | BAAI/bge-m3 |
| Reranker | BAAI/bge-reranker-base |
| API | FastAPI + Uvicorn |
| Wrapper | CustomChatWrapper (thay thế LangChain ChatHuggingFace) |

## Cấu trúc thư mục

```
EXACT_2026/
├── config/
│   ├── setting.yaml          # Cấu hình chính (model, timeout, RAG...)
│   └── logging.yaml          # Cấu hình logging
│
├── data/
│   ├── EXACT2026_dataset_2026-05-15/ # Dữ liệu gốc từ BTC
│   ├── sft_dataset/                  # Dữ liệu đã xử lý để fine-tuning
│   └── *.pdf                         # Tài liệu quy chế, slide đào tạo
│
├── src/
│   ├── api/                  # FastAPI endpoint (POST /solve)
│   ├── agent/                # LangGraph pipeline
│   │   ├── nodes/            # Các node xử lý
│   │   ├── state.py          # Định nghĩa trạng thái pipeline
│   │   ├── graph.py          # Luồng xử lý chính
│   │   └── schema.py         # Pydantic schema cho request/response
│   ├── llm/                  # Quản lý LLM (factory, provider)
│   ├── prompt/               # Prompt templates (Z3, SymPy, Explanation...)
│   ├── retrieval/            # RAG engine (vector_db, hybrid search)
│   ├── sandbox/              # Chạy code Z3/SymPy an toàn
│   └── utils/                # Logger, output parser
│
├── scripts/                  # Script tiện ích
├── logs/                     # File log tự động tạo
├── storage/                  # Vector DB (Qdrant) tự động tạo
├── requirements.txt
├── solution.md               # Mô tả giải pháp (nộp bài)
└── README.md
```

## Luồng xử lý (Pipeline)

```
Câu hỏi JSON --> Phân loại (có premises? logic : physics)
                      |
          +-----------+-----------+
          |                       |
       LOGIC                   PHYSICS
          |                       |
      Z3 Solver              RAG (tìm công thức)
      (retry x1)                  |
          |                  SymPy Solver
     Thành công?             (retry x1)
      /       \                   |
    Có       Không          Thành công?
     |          |             /       \
     |     LLM Direct       Có       Không
     |      (fallback)       |          |
     |          |            |     LLM Direct
     +-----+---+            +-----+----+
           |                      |
     Answer Selector        Answer Selector
           |                      |
           +----------+-----------+
                      |
               JSON Response
          {answer, explanation}
```

**Điểm nổi bật (Tối ưu hóa hệ thống):**
- **Neurosymbolic AI**: Ép buộc LLM không được lười biếng mà phải giải thích step-by-step (`print(Step 1...)`) cùng lúc với việc sinh code Z3/SymPy.
- **Tự động sửa lỗi (Self-Correction)**: Nếu Sandbox chạy code lỗi, sẽ tự động ném lỗi đó ngược lại cho LLM để tự sửa (tối đa 1 lần).
- **Trích xuất an toàn**: Fallback lấy log từ Sandbox hoặc tự động trích xuất comment Python làm lời giải thích.
- **Tối ưu VRAM (OOM Protection)**: Hệ thống giữ trạng thái `torch.cuda.empty_cache()` liên tục, tự động dọn dẹp bộ nhớ sau mỗi lần xử lý để dành VRAM cho model Qwen2.5-7B.
- **Bypass LangChain Bug**: Sử dụng `CustomChatWrapper` tự viết thay vì `ChatHuggingFace` để không bị crash khi load LoRA Adapter ở định dạng 4-bit.

---

## Hướng dẫn cài đặt

### Bước 1: Tạo môi trường Conda

```bash
conda create -n exact-env python=3.11 -y
conda activate exact-env
```

### Bước 2: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

**Lưu ý:**
- Thư viện `bitsandbytes` chỉ hỗ trợ GPU NVIDIA (CUDA). Trên Colab thì đã có sẵn.
- `torch` cần phiên bản tương thích với CUDA. Xem: https://pytorch.org/get-started/locally/

### Bước 3: Cấu hình model (tùy chọn)

Mặc định hệ thống sẽ dùng model Qwen2.5-7B đã được tinh chỉnh (fine-tune) trực tiếp của bạn. Bạn có thể sửa repo Hugging Face chứa LoRA Adapter của bạn trong file `config/setting.yaml`:

```yaml
llm:
  model_id: "platypus123/Qwen-Z3-Merged-BTAM17026"   
  load_in_4bit: true                                     # 4-bit quantization
```

**Không cần tải model thủ công** — hệ thống tự động tải và cache.

---

## Ví dụ thực tế (Đã test trên Colab)

Dưới đây là 3 ví dụ thực tế đã được test độc lập trên Google Colab sử dụng hàm `run_pipeline()`.

### 1. Bài toán Vật lý: Định luật 2 Newton
```python
import time
from src.agent.graph import run_pipeline

print("\n--- TEST PHYSICS: NEWTONS SECOND LAW ---")
start = time.time()

result = run_pipeline(
    question="A wooden block with a mass of 5 kg is placed on a horizontal surface. A horizontal force F = 25 N is applied to pull the block. The coefficient of kinetic friction between the block and the surface is mu = 0.2. Given that the acceleration due to gravity g = 10 m/s^2, calculate the magnitude of the friction force and the resulting acceleration of the block."
)

print(f"Time: {time.time() - start:.1f}s")
print(f"Answer: {result['answer']}")
print(f"Solver Used: {result['solver_used']}")
print(f"Explanation:\n{result['explanation']}")
```

**Kết quả Output:**
```text
Time: 47.4s
Answer: f_k = 10.0000000000000, a = 3.00000000000000 m/s²
Solver Used: sympy
Explanation:
**Chi tiết thực thi (SymPy):**
Step 1: m = 5 kg, F = 25 N, μ = 0.2, g = 10 m/s²
Step 2: Normal force N = m × g = 5 × 10 = g*m
Step 3: Frictional force f_k = μ × N = 0.2 × g*m = 10.0000000000000
Step 4: Net force along x-axis: F_net = F - f_k = 25 - 10.0000000000000 = 15.0000000000000
Step 5: Acceleration a = F_net / m = 15.0000000000000 / 5 = 3.00000000000000
```

### 2. Bài toán Vật lý: Chuyển động ném xiên (Nâng cao)
```python
import time
from src.agent.graph import run_pipeline

print("\n--- TEST PHYSICS 10: PROJECTILE FULL ANALYSIS ---")
start = time.time()

result = run_pipeline(
    question="A projectile is launched from the ground with an initial velocity of 40 m/s at an angle of 45 degrees above the horizontal. Neglecting air resistance and taking g = 9.8 m/s^2, determine: (1) the maximum height reached, (2) the total time of flight, and (3) the horizontal range."
)

print(f"Time: {time.time() - start:.1f}s")
print(f"Answer: {result['answer']}")
print(f"Solver Used: {result['solver_used']}")
print(f"Explanation:\n{result['explanation']}")
```

**Kết quả Output:**
```text
Time: 34.6s
Answer: 40.82 m; 5.77 s; 163.27 m
Solver Used: sympy
Explanation:
**Chi tiết thực thi (SymPy):**
Step 1: The maximum height reached is h_max = 40.82 m.
Step 2: The total time of flight is t_total = 5.77 s.
Step 3: The horizontal range is R = 163.27 m.
```

### 3. Bài toán Logic: Thay thế điểm GPA
```python
import time
from src.agent.graph import run_pipeline

print("\n--- TEST LOGIC: GRADE REPLACEMENT ---")
start = time.time()

result = run_pipeline(
    question="Will Student D's new grade replace their old grade in the GPA calculation?",
    premises=[
        "According to the academic regulations, when a student retakes a course, the new grade replaces the original grade only if the new grade is higher.",
        "Student D originally received a D in Chemistry.",
        "Student D retook Chemistry this semester and received an F."
    ]
)

print(f"Time: {time.time() - start:.1f}s")
print(f"Answer: {result['answer']}")
print(f"Solver Used: {result['solver_used']}")
print(f"Explanation:\n{result['explanation']}")
```

**Kết quả Output:**
```text
Time: 37.6s
Answer: No
Solver Used: z3
Explanation:
**Lập luận (LLM):**
The new grade (F) is not higher than the old grade (D). Therefore, according to regulation 1, the new grade will NOT replace the old grade.

Formal Logic (FOL):
  P1: ∀x∀y((RetakeCourse(x, y) ∧ NewGradeHigherThanOld(y)) → ReplaceGrade(x))
  P2: RetakeCourse(D, Chem)
  P3: OriginalGrade(Chem, D)

**Chi tiết thực thi (Z3 Theorem Prover):**
Step 1: Setup variables.
Step 2: Apply logic.
Step 3: Result -> Replaced = No
```

---

## Hướng dẫn chạy API Server (Dùng cho nộp bài thi)

Để khởi động máy chủ API theo đúng tiêu chuẩn của cuộc thi (FastAPI + Uvicorn), bạn hãy thực hiện các bước sau:

**Bước 1: Chạy Server**
```bash
# Đảm bảo bạn đang ở thư mục EXACT_2026 và đã kích hoạt môi trường
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
*Lần đầu tiên khởi động, Server sẽ mất vài phút để load các Model (LLM, Embedding) vào VRAM.*

**Bước 2: Gửi Request (Test API)**
Mở một Terminal khác hoặc dùng Python để bắn Request:
```python
import requests

response = requests.post("http://localhost:8000/solve", json={
    "question": "Is Socrates mortal?",
    "premises": [
        "All men are mortal.",
        "Socrates is a man."
    ]
})
print(response.json())
```

## API Reference

### POST `/solve`
**Request:**
```json
{
    "question": "Nội dung câu hỏi Vật lý hoặc Logic...",
    "premises": ["Giả thiết 1", "Giả thiết 2"], 
    "type": "logic" 
}
```
*(Lưu ý: `premises` và `type` là tùy chọn, hệ thống có thể tự động nhận diện).*

**Response:**
```json
{
    "answer": "3.0 m/s²",
    "explanation": "**Chi tiết thực thi (SymPy):**\nStep 1: Normal force...\nStep 2: Frictional force...",
    "fol": "import sympy as sp\n...",
    "cot": [],
    "premises": [],
    "confidence": 1.0,
    "task_type": "physics",
    "solver_used": "sympy"
}
```

### GET `/health`
Kiểm tra trạng thái server:
```bash
curl http://localhost:8000/health
```

---

## Cấu hình

Toàn bộ cấu hình nằm trong file `config/setting.yaml`:

| Tham số | Mô tả | Giá trị mặc định |
|---|---|---|
| `llm.model_id` | Model ID trên HuggingFace | `platypus123/Qwen-Z3-Merged-BTAM17026` |
| `llm.temperature` | Nhiệt độ sinh text (0 = deterministic) | `0.0` |
| `llm.max_new_tokens` | Số token tối đa LLM sinh ra | `1024` |
| `llm.load_in_4bit` | Bật 4-bit quantization | `true` |
| `pipeline.global_timeout` | Timeout tổng cho pipeline (giây) | `120` |
| `pipeline.solver_timeout` | Timeout cho mỗi lần chạy solver | `10` |
| `pipeline.max_retry` | Số lần retry khi solver lỗi | `1` |
| `retrieval.top_k` | Số candidates ban đầu cho RAG | `10` |
| `retrieval.rerank_top_n` | Số kết quả sau rerank | `3` |

---

## Hướng dẫn chạy trên Google Colab (Chi tiết)

Hệ thống được thiết kế để có thể chạy mượt mà trên môi trường Google Colab miễn phí. Bạn hãy làm theo các bước sau:

**Bước 1: Bật GPU trên Colab**
- Mở Google Colab, tạo một Notebook mới.
- Vào menu **Runtime (Thời gian chạy)** > **Change runtime type (Thay đổi loại thời gian chạy)**.
- Chọn Hardware accelerator là **T4 GPU** (Bắt buộc để load model 7B).

**Bước 2: Tải mã nguồn và cài đặt**
Copy đoạn code sau vào một ô (cell) và chạy:
```python
# Clone mã nguồn từ nhánh main
!git clone https://github.com/Platypus27-coder/EXACT_2026.git
%cd /content/EXACT_2026

# Cài đặt các thư viện lõi
!pip install -r requirements.txt
```

**Bước 3: Chạy test trực tiếp**
Tạo một ô code mới và chạy thử hàm `run_pipeline`:
```python
import time
from src.agent.graph import run_pipeline

start = time.time()
result = run_pipeline(
    question="A wooden block with a mass of 5 kg is placed on a horizontal surface. A horizontal force F = 25 N is applied to pull the block. The coefficient of kinetic friction between the block and the surface is mu = 0.2. Given that the acceleration due to gravity g = 10 m/s^2, calculate the magnitude of the friction force and the resulting acceleration of the block."
)

print(f"Time: {time.time() - start:.1f}s")
print(f"Answer: {result['answer']}")
print(f"Explanation:\n{result['explanation']}")
```

*Lưu ý: Ở lần chạy câu hỏi đầu tiên, hệ thống sẽ mất khoảng 3 - 5 phút để tự động tải các mô hình (Qwen, BGE-M3, Reranker) từ HuggingFace về máy. Các câu hỏi sau sẽ chạy rất nhanh (khoảng 30-40 giây).*

---

## Ghi chú

- Model tự động tải từ HuggingFace lần đầu và được cache lại
- Kiến thức vật lý tự động nạp vào Vector DB lần đầu chạy (không cần chạy script riêng)
- Log chi tiết được ghi vào `logs/exact_2026.log`
- Vector DB (Qdrant) lưu tại `storage/qdrant_storage/`
- Dữ liệu fine-tuning ở `data/sft_dataset/` đã sẵn sàng dùng
- File `solution.md` mô tả giải pháp (nộp bài thi)
