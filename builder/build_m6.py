"""
builder/build_m6.py
Generates M6_S01.ipynb to M6_S03.ipynb for Module 6 (LLM Assisted).
"""

from utils import create_notebook, save_notebook

def build_m6_s01():
    objectives = [
        "Hiểu sự khác biệt giữa việc 'lệ thuộc vào AI một cách mù quáng' và 'sử dụng LLM như một người cộng sự nghiên cứu (Research Co-pilot)'.",
        "Làm chủ khung đặt lệnh có cấu trúc RTCC (Role - Task - Context - Constraint) chuyên dụng cho Khoa học Dữ liệu và AI.",
        "Thiết lập các kịch bản Prompting phương pháp: Giải thích mã nguồn phức tạp, Brainstorm ý tưởng đặc trưng mới, và Phản biện giải pháp (Code Review).",
        "Duy trì năng lực tư duy độc lập (Human Agency) và nguyên tắc xác minh thực nghiệm (Empirical Verification)."
    ]
    prerequisites = [
        "Đã hoàn thành các Module từ M0 đến M5.",
        "Hiểu cách thức hoạt động cơ bản của các mô hình ngôn ngữ lớn (LLMs)."
    ]
    why_content = """
Trong thời đại bùng nổ của Trí tuệ Nhân tạo Tạo sinh, nhiều sinh viên năm nhất mắc phải một căn bệnh nguy hiểm: **Sao chép mã nguồn mù quáng từ ChatGPT/Claude/Gemini mà không hề hiểu bản chất**. Khi gặp một đề thi Olympic AI mới lạ với những ràng buộc khắt khe, hoặc khi LLM sinh ra một đoạn code bị lỗi ngầm tinh vi (Ảo giác - Hallucination), những sinh viên này hoàn toàn tê liệt vì đã đánh mất hoàn toàn cơ bắp tư duy của chính mình!

Module M6 **không dạy bạn cách viết văn vẻ để nịnh AI**. Chúng ta tiếp cận LLM dưới góc độ của một kỹ sư đẳng cấp: Sử dụng LLM như một trợ lý gia tốc, một người thầy phản biện khắt khe và một công cụ mở rộng tư duy. Người chiến thắng trong kỷ nguyên AI không phải là người cấm dùng AI, cũng không phải người phụ thuộc vào AI, mà là người **biết cách điều khiển và kiểm chứng AI bằng tư duy sắc bén của một chuyên gia**!
"""
    concept_content = r"""
### 1. Khung cấu trúc Prompt RTCC chuyên dụng cho AI Engineering
Để nhận được câu trả lời chất lượng cao từ LLM thay vì những đoạn văn chung chung vô thưởng vô phạt, bạn bắt buộc phải tuân theo 4 thành phần:
* **R (Role - Vai trò):** Định vị chuyên gia cho mô hình (ví dụ: *"Bạn là Huấn luyện viên Olympic AI chuyên về PyTorch tối ưu hóa"*).
* **T (Task - Nhiệm vụ cụ thể):** Nêu rõ hành động cần làm (ví dụ: *"Hãy phân tích đoạn code huấn luyện sau và chỉ ra 3 điểm nghẽn hiệu năng"*).
* **C (Context - Bối cảnh & Ràng buộc kỹ thuật):** Cung cấp thông tin môi trường (ví dụ: *"Tập dữ liệu bảng 100,000 dòng, tài nguyên 1 GPU T4 16GB, metric đánh giá là Macro-F1"*).
* **C (Constraint - Ràng buộc đầu ra):** Giới hạn định dạng trả lời (ví dụ: *"Không viết lại toàn bộ code, chỉ giải thích nguyên lý toán học và đưa ra gợi ý gợi mở dưới 200 từ"*).

### 2. Các mẫu Prompt trợ lực học tập đỉnh cao
1. **Prompt Giải thích mã nguồn từng dòng (Line-by-Line Socratic Explanation):**
   - *"Giải thích cho tôi luồng gradient của đoạn code PyTorch này như thể tôi là sinh viên năm nhất, tập trung vào tại sao phải dùng detach() ở dòng 12."*
2. **Prompt Brainstorm Kỹ nghệ Đặc trưng (Feature Brainstorming):**
   - *"Với bài toán dự đoán gian lận thẻ tín dụng có các cột [Thời gian, Số tiền, Mã cửa hàng], hãy đề xuất 5 ý tưởng đặc trưng tương tác phi hiển nhiên."*
3. **Prompt Phản biện Kiến trúc (Adversarial Code Review):**
   - *"Hãy đóng vai một giám khảo khó tính của kỳ thi Olympic AI, soi xét đoạn code chia fold sau đây và chỉ ra nguy cơ rò rỉ dữ liệu tiềm ẩn."*
"""
    visual_intuition_content = """
```
Mô hình Tương tác với LLM có Phương pháp:
   [Kỹ sư AI] ──(Tư duy logic & Hiểu bài toán)──┐
       ▲                                        ▼
       │                                 [Khung RTCC Prompt]
       │                                        │
       │                                        ▼
[Kiểm chứng Thực nghiệm] ◄──(Phản biện)─── [Trợ lý LLM]
(Chạy thử nghiệm trên Benchmark)
```
"""
    code_cells = [
        (
            """# 1. Framework tao Prompt tu dong theo chuan RTCC trong Python
class RTPromptBuilder:
    def __init__(self, role):
        self.role = role
        self.context = ""
        self.task = ""
        self.constraints = []
        
    def set_context(self, context_str):
        self.context = context_str
        return self
        
    def set_task(self, task_str):
        self.task = task_str
        return self
        
    def add_constraint(self, constraint_str):
        self.constraints.append(constraint_str)
        return self
        
    def build(self):
        prompt = f"### ROLE:\\n{self.role}\\n\\n"
        if self.context:
            prompt += f"### CONTEXT & ENVIRONMENT:\\n{self.context}\\n\\n"
        prompt += f"### TASK:\\n{self.task}\\n\\n"
        if self.constraints:
            prompt += "### CONSTRAINTS & OUTPUT RULES:\\n"
            for c in self.constraints:
                prompt += f"- {c}\\n"
        return prompt

# Tao mot Prompt mau yeu cau LLM dong vai Reviewer phan bien code
builder = RTPromptBuilder(role="Senior Olympic AI Judge & Performance Optimization Specialist")
builder.set_context("Bai toan phan loai anh y te tren PyTorch, GPU T4 16GB, tap du lieu nho 2,000 anh.")
builder.set_task("Review doan code Data Augmentation ben duoi va chi ra 2 rui ro gay suy giam do chinh xac hoac lam cham toc do nạp batch.")
builder.add_constraint("Khong duoc viet san dap an thay the hoan chinh.")
builder.add_constraint("Giai thich bang truc giac toan hoc va co che hoat dong cua GPU.")
builder.add_constraint("Dua ra 2 cau hoi goi mo de toi tu suy nghi giai phap.")

generated_prompt = builder.build()
print("===== PROMPT CHUAN KHOA HOC SINH RA BOI FRAMEWORK =====")
print(generated_prompt)
""",
            """Khối code xây dựng một lớp Python `RTPromptBuilder`:
- Tự động chuẩn hóa việc đặt câu hỏi cho LLM theo đúng chuẩn kỹ sư chuyên nghiệp (Role - Context - Task - Constraints).
- Ngăn chặn việc nhận về những câu trả lời lười biếng hoặc copy-paste vô bổ bằng các ràng buộc mang tính định hướng tư duy (Socratic questioning)."""
        ),
        (
            """# 2. Mo phong qua trinh Kiem chung Thuc nghiem (Empirical Verification Sandbox)
# Nguyen tac: Bat ky loi khuyen nao cua LLM deu phai duoc do luong bang so lieu thuc te tren Benchmark!

def verify_llm_suggestion(baseline_score, improved_score, metric_name="Macro-F1"):
    diff = improved_score - baseline_score
    print(f"===== KIEM CHUNG LOI KHUYEN CUA LLM TREN THUC NGHIEM =====")
    print(f"Diem Baseline noi bo : {baseline_score:.4f}")
    print(f"Diem sau khi ap dung : {improved_score:.4f}")
    
    if diff > 0.005:
        print(f"-> XAC NHAN THU TRUC: Y tuong cai thien ro ret (+{diff*100:.2f}% {metric_name}). Chap nhan dua vao Pipeline!")
        return True
    elif diff < -0.005:
        print(f"-> BAC BO: Y tuong lam giam diem ({diff*100:.2f}% {metric_name}). LLM co the da bi ao giac hoac khong hop voi dataset nay!")
        return False
    else:
        print(f"-> CAN NHAC: Khac biet khong dang ke. Kiem tra xem co lam tang do phuc tap hay khong truoc khi dung.")
        return False

# Vi du kiem chung mot dac trung moi do LLM de xuat
verify_llm_suggestion(baseline_score=0.8210, improved_score=0.8355)
""",
            """Khối code thiết lập nguyên tắc kiểm chứng thực nghiệm (Verification Sandbox):
- Mọi ý tưởng do LLM gợi ý (dù nghe có vẻ rất thuyết phục) đều phải được kiểm nghiệm trên bảng số liệu điểm số thực tế.
- Xây dựng thói quen khoa học: Nói có sách, mách có chứng — chỉ đưa vào bài nộp những cải tiến có số liệu chứng minh."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Thiết kế Prompt giải thích lỗi CUDA
Viết một prompt bằng `RTPromptBuilder` yêu cầu LLM giải thích nguyên nhân lỗi `RuntimeError: CUDA out of memory` và đưa ra 4 giải pháp khắc phục có phân tích đánh đổi (trade-off) về thời gian và dung lượng.

### Bài 2 (Easy): Đặt câu hỏi theo phương pháp Socratic
Chuyển đổi một câu hỏi thụ động: *"Viết code phân loại văn bản cho tôi"* thành một prompt chủ động yêu cầu LLM hướng dẫn bạn từng bước tự xây dựng mô hình DAN (Deep Averaging Network) mà không đưa trước code đáp án.

### Bài 3 (Medium): Brainstorm đặc trưng với bài toán thương mại điện tử
Tạo một prompt chuyên sâu yêu cầu LLM brainstorm 5 đặc trưng mới từ dữ liệu lịch sử mua hàng của người dùng (gồm: thời gian đặt hàng, giá tiền, địa chỉ nhận). Yêu cầu LLM phân tích xem đặc trưng nào có nguy cơ bị Data Leakage cao nhất.

### Bài 4 (Medium): Phản biện tính đúng đắn của công thức toán học
Gửi cho LLM một công thức toán học tính Metric bị cố ý viết sai (ví dụ công thức tính F1-Score bị đổi mẫu số thành phép trừ). Yêu cầu LLM tìm ra lỗi sai và chứng minh bằng một ví dụ số cụ thể.

### Bài 5 (Challenge): Xây dựng Bộ lọc Phát hiện "Code ảo giác" (Hallucination Detector)
Liệt kê 5 dấu hiệu nhận biết một đoạn mã do LLM sinh ra đang bị ảo giác (ví dụ: bịa đặt ra một tham số không tồn tại trong hàm của Scikit-Learn, hoặc gọi một phương thức đã bị xóa trong phiên bản PyTorch mới nhất).
"""
    voai_connection_content = """
Trong một kỳ thi AI mở cho phép dùng Internet hoặc trợ lý AI, **tốc độ và chất lượng của câu hỏi bạn đặt ra cho LLM sẽ quyết định năng suất của bạn**:
Thí sinh nghiệp dư sẽ gõ: *"Giải bài này giúp tôi"* và nhận về một đoạn code lỗi thời không chạy được. Thí sinh đẳng cấp sẽ dùng LLM như một công cụ tra cứu tài liệu siêu tốc độ: *"Cú pháp chuẩn của StratifiedGroupKFold trong Scikit-Learn 1.3 là gì?", "Gợi ý cho tôi 3 phép biến đổi toán học phi tuyến cho cột dữ liệu bị lệch phải"*. Tận dụng LLM để tiết kiệm thời gian gõ tài liệu giúp bạn tập trung 100% trí tuệ vào chiến thuật thi đấu.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh copy nguyên si một đoạn code tiền xử lý do LLM sinh ra trên mạng và đưa vào bài thi, dẫn tới việc mã nguồn bị lỗi sập vì LLM tự ý bịa ra một tham số không có thật:

```python
# CODE CÓ LỖI DO ẢO GIÁC CỦA LLM (HALLUCINATED API PARAMETER):
# LLM tu bia ra tham so normalize_weights='auto' trong RandomForestClassifier!
# clf = RandomForestClassifier(n_estimators=100, normalize_weights='auto')
# TypeError: RandomForestClassifier.__init__() got an unexpected keyword argument 'normalize_weights'
```

**Nhiệm vụ của bạn:**
1. Giải thích hiện tượng ảo giác API (API Hallucination) của các mô hình ngôn ngữ lớn.
2. Nêu quy tắc bắt buộc: Luôn kiểm tra lại tài liệu chính thức (`help(ClassName)` hoặc Docstrings trong Python) trước khi chạy bất kỳ đoạn code lạ nào do AI sinh ra.
"""
    knowledge_check_content = """
1. Khung cấu trúc RTCC trong kỹ thuật đặt Prompt có phương pháp viết tắt của 4 từ nào?
   - A. Read, Type, Compile, Check
   - B. Role, Task, Context, Constraint
   - C. Real, True, Clear, Complete
   - D. Random, Time, Code, Case

2. Nguy cơ lớn nhất của việc sinh viên sao chép mã nguồn mù quáng từ các trợ lý AI là gì?
   - A. Tốn dung lượng ổ cứng.
   - B. Đánh mất khả năng tư duy độc lập, không hiểu luồng thực thi bên dưới, và hoàn toàn bất lực khi mã nguồn bị lỗi ảo giác hoặc khi đối mặt với đề thi mới lạ.
   - C. Khiến cho màn hình máy tính bị đơ.
   - D. Làm cho bàn phím bị hỏng.

3. Khi muốn yêu cầu LLM giúp bạn học và hiểu bản chất của một thuật toán khó, chiến thuật Prompt nào là ĐÚNG ĐẮN NHẤT?
   - A. Yêu cầu LLM viết hộ toàn bộ bài tập lớn.
   - B. Yêu cầu LLM đóng vai người thầy phản biện (Socratic Tutor), giải thích trực giác hình học và đặt ra các câu hỏi gợi mở để bạn tự lập trình.
   - C. Hỏi câu ngắn nhất có thể.
   - D. Chỉ copy đáp án cuối cùng.

4. Hiện tượng "Ảo giác" (Hallucination) của mô hình ngôn ngữ lớn trong lập trình biểu hiện như thế nào?
   - A. Mô hình tự động tắt máy tính.
   - B. Mô hình sinh ra câu trả lời có vẻ rất tự tin, trôi chảy và chuyên nghiệp nhưng thực chất lại bịa đặt ra các thư viện, hàm số hoặc tham số hoàn toàn không tồn tại trong thực tế.
   - C. Mô hình gõ chữ rất chậm.
   - D. Mô hình chỉ nói tiếng Latinh.

5. Nguyên tắc tối thượng khi ứng dụng bất kỳ gợi ý cải tiến nào của LLM vào bài thi Olympic AI là gì?
   - A. Tin tưởng 100% vào AI không cần kiểm tra.
   - B. Bắt buộc phải kiểm chứng định lượng bằng thực nghiệm trên hệ thống Validation nội bộ (Empirical Verification Sandbox).
   - C. Đổi đề thi khác.
   - D. Xóa tập dữ liệu huấn luyện.
"""
    session_summary_content = """
* **What I learned:**
  - Bản chất của việc sử dụng LLM như một Research Co-pilot có phương pháp.
  - Khung đặt lệnh chuẩn kỹ sư RTCC (Role - Task - Context - Constraint).
  - Các kỹ thuật Prompting nâng cao: Socratic Explanation, Feature Brainstorming, Code Review.
  - Nguyên tắc kiểm chứng thực nghiệm và nhận diện ảo giác (Hallucination).
* **What I can now do:**
  - Điều khiển LLM phục vụ đắc lực cho quá trình học tập và nghiên cứu AI mà không bị phụ thuộc.
  - Thiết kế các prompt chuẩn mực giúp khai phá tối đa sức mạnh của các mô hình ngôn ngữ.
* **What I still need to practice:**
  - Xây dựng quy trình gỡ lỗi có cấu trúc: **LLM Debugging Workflow** ở Buổi 2.
"""
    next_session_content = """
Trong **Buổi M6_S02**, chúng ta sẽ tấn công vào kỹ năng thực chiến gay cấn nhất: **Quy Trình Debug Code Thi Đấu Bằng LLM & Đối Soát Ảo Giác (Hallucination)** — Học cách cung cấp Traceback chuẩn xác, tạo các bộ kiểm thử biên (Edge Cases), và thiết lập môi trường cô lập an toàn!
"""
    trace = {
        "items": [
            "Tư duy sử dụng LLM như Research Co-pilot",
            "Khung cấu trúc RTCC Prompt (Role, Task, Context, Constraint)",
            "Prompting có phương pháp: Giải thích code, Brainstorm & Phản biện"
        ],
        "deliverable": "Mã nguồn Framework sinh Prompt RTCC chuẩn hóa cho kỹ sư AI"
    }
    nb = create_notebook(
        module_id="M6",
        module_name="DÙNG LLM NHƯ TRỢ LÝ HỌC VÀ THI",
        session_id="S01",
        topic="Kỹ Thuật Đặt Prompt Chuyên Sâu Cho Lập Trình & Ôn Luyện AI",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook xây dựng Framework RTCC Prompting chuyên nghiệp cho AI",
        competition_relevance="Tối ưu hóa hiệu suất nghiên cứu và giải quyết bài toán thi đấu có trợ lực AI.",
        syllabus_traceability=trace,
        why_content=why_content,
        concept_content=concept_content,
        visual_intuition_content=visual_intuition_content,
        code_cells=code_cells,
        practice_content=practice_content,
        voai_connection_content=voai_connection_content,
        debugging_content=debugging_content,
        knowledge_check_content=knowledge_check_content,
        session_summary_content=session_summary_content,
        next_session_content=next_session_content
    )
    save_notebook(nb, "course/M6_LLM_ASSISTED/M6_S01.ipynb")


def build_m6_s02():
    objectives = [
        "Làm chủ quy trình 4 bước chuẩn mực khi yêu cầu LLM hỗ trợ gỡ lỗi: Báo cáo vết lỗi (Traceback) + Mã nguồn tối thiểu (MRE) + Bối cảnh môi trường + Giả thuyết ban đầu.",
        "Sử dụng LLM để tự động sinh các Bộ dữ liệu kiểm thử biên (Edge Cases & Synthetic Test Cases) nhằm phát hiện lỗi tiềm ẩn.",
        "Thiết lập quy trình 3 bước Đối soát Ảo giác (Hallucination Cross-Verification Protocol).",
        "Tự tay cô lập và sửa chữa các lỗi phức tạp trong pipeline PyTorch bằng quy trình gỡ lỗi có trợ lực AI."
    ]
    prerequisites = [
        "Hoàn thành Buổi M6_S01 (Khung RTCC Prompt).",
        "Hiểu cách đọc thông báo lỗi (Error Traceback) trong Python."
    ]
    why_content = """
Khi gặp lỗi code trong phòng thi, phản xạ thông thường của một thí sinh hoảng loạn là: Chụp một bức ảnh màn hình mờ căm hoặc ném nguyên một file code 500 dòng vào ChatGPT và gõ: *"Code bị lỗi, sửa đi!"*. Kết quả bạn nhận lại sẽ là một mớ hỗn độn bịa đặt, cắt xén mất các hàm quan trọng, và bạn tốn thêm 1 tiếng đồng hồ nữa trong tuyệt vọng.

Gỡ lỗi (Debugging) là một môn khoa học chính xác. Một kỹ sư chuyên nghiệp biết cách **đóng gói lỗi thành một Mẫu tái lập tối thiểu (Minimal Reproducible Example - MRE)** và dẫn dắt LLM khoanh vùng đúng nguyên nhân gốc rễ trong chưa đầy 30 giây. Làm chủ quy trình Debugging bằng LLM giúp bạn biến sự cố thành cơ hội và tiết kiệm hàng giờ quý giá trong phòng thi.
"""
    concept_content = r"""
### 1. Cấu trúc Gói Báo cáo Lỗi Chuẩn (Structured Bug Report Packet)
Khi gửi lỗi cho LLM, gói thông tin bắt buộc phải gồm 4 phần:
1. **Full Traceback:** Toàn bộ thông báo lỗi từ dòng đầu đến dòng cuối (chứa tên file, dòng bị lỗi và loại ngoại lệ `Exception`).
2. **Minimal Reproducible Example (MRE):** Đoạn mã ngắn nhất có thể tự chạy độc lập để tái lập đúng lỗi đó (loại bỏ toàn bộ các phần code râu ria không liên quan).
3. **Environment & Tensor Shapes:** Phiên bản thư viện (PyTorch 2.0, Python 3.10) và kích thước thực tế của các biến đầu vào ngay trước dòng bị lỗi.
4. **My Hypothesis (Giả thuyết cá nhân):** *"Tôi nghi ngờ lỗi này do chiều kênh màu của Tensor bị đảo lộn ở bước Permute"*.

### 2. Sinh Test Cases nhân tạo (Synthetic Edge-Case Generation)
Hầu hết các lỗi sập code trong phòng thi xảy ra do các trường hợp biên dị thường:
* Batch chỉ có đúng 1 mẫu ($N=1$ làm sập BatchNorm).
* Câu văn bản rỗng (Độ dài bằng 0).
* Ảnh có toàn bộ pixel bằng nhau (Độ tương phản bằng 0).
* Dữ liệu chứa toàn giá trị âm hoặc giá trị cực lớn ($10^9$).
*Yêu cầu LLM sinh ra các bộ test biên này giúp bạn kiểm thử độ bền (Robustness) của pipeline trước khi nộp bài!*
"""
    visual_intuition_content = """
```
Quy trình LLM Debugging Workflow khép kín:
[Code bị Lỗi Runtime / Crash]
             │
             ▼
[Thu nhỏ thành MRE (Mẫu Tái lập Tối thiểu)] ──> [Đóng gói: Traceback + Shape + Environment]
                                                                │
                                                                ▼
                                                    [Gửi gói Debug cho LLM]
                                                                │
                                                                ▼
[Đối soát Ảo giác (Docstring / Sandbox Run)] ◄────── [LLM đề xuất Giải pháp]
             │
             ▼
[Kiểm nghiệm trên Synthetic Edge Cases] ──> [Áp dụng an toàn vào Bài thi chính!]
```
"""
    code_cells = [
        (
            """# 1. MO PHONG TINH HUONG DEBUG THUC TE: LOI LECH CHIEU TRONG PYTORCH
# Doan code gay loi duoc dong goi thanh Minimal Reproducible Example (MRE)

import torch
import torch.nn as nn

# Tinh huong loi: Kich thuoc dau vao khong khop voi kien truc mang
def reproduce_buggy_code():
    try:
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        # Dau vao bi sai chieu: Chi co 8 dac trung thay vi 10
        wrong_input = torch.randn(4, 8) 
        output = model(wrong_input)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return error_trace

traceback_output = reproduce_buggy_code()
print("===== THONG BAO LOI THUC TE (TRACEBACK THU THAP DUOC) =====")
print(traceback_output)
""",
            """Khối code mô phỏng việc trích xuất vết lỗi (Traceback) thực tế một cách chuẩn xác:
- Sử dụng module `traceback` để thu thập đầy đủ thông tin lỗi hệ thống.
- Cung cấp dữ liệu thô trung thực sẵn sàng nạp vào khung gỡ lỗi có cấu trúc."""
        ),
        (
            """# 2. FRAMEWORK TAO GOI DEBUG CHUAN MUC GUI CHO LLM (STRUCTURED BUG PACKET)
def create_debug_prompt(code_snippet, error_trace, env_info, hypothesis):
    prompt = f\"\"\"### SYSTEM CONTEXT:
Ban la Chuyen gia Debugging He thong AI cap cao. Nhiem vu cua ban la phan tich nguyen nhan goc re cua loi sau va de xuat huong sua chua an toan nhat.

### 1. MINIMAL REPRODUCIBLE EXAMPLE (MRE):
```python
{code_snippet}
```

### 2. FULL ERROR TRACEBACK:
```
{error_trace}
```

### 3. ENVIRONMENT & RUNTIME METADATA:
- Thiet bi & Phien ban: {env_info}

### 4. MY HYPOTHESIS & OBSERVATION:
{hypothesis}

### 5. DEBUGGING INSTRUCTIONS FOR LLM:
- Hay chi ra dong code cu the gay ra loi va ban chat toan hoc ben duoi.
- Giai thich tai sao ma tran dau vao lai khong the nhan voi ma tran trong so.
- De xuat dung 2 dong code sua loi ma khong can viet lai toan bo chuong trinh.
- Cung cap 1 doan code kiem thu bien (Edge-case test) de phong ngua loi nay lap lai.
\"\"\"
    return prompt

bug_code = '''model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
wrong_input = torch.randn(4, 8)
output = model(wrong_input)'''

debug_prompt = create_debug_prompt(
    code_snippet=bug_code,
    error_trace=traceback_output,
    env_info="PyTorch 2.1.0, CPU, OS Windows",
    hypothesis="Toi thay kich thuoc dac trung dau vao (8) khong khop voi in_features (10) cua tang Linear dau tien."
)

print("===== GOI PROMPT DEBUG HOAN HAO SAN SANG GUI CHO LLM =====")
print(debug_prompt)
""",
            """Khối code tự động đóng gói toàn bộ thông tin gỡ lỗi thành một tài liệu kỹ thuật hoàn chỉnh:
- Giúp LLM nhìn thấy chính xác 100% bối cảnh kỹ thuật mà không cần đoán mò.
- Yêu cầu sửa lỗi cục bộ (không xáo trộn kiến trúc toàn cục) và yêu cầu cung cấp test case kiểm thử biên."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Debug lỗi kích thước nhãn trong CrossEntropyLoss
Trong PyTorch, `nn.CrossEntropyLoss` yêu cầu nhãn mục tiêu phải là tensor 1D kiểu số nguyên (`torch.long`) có kích thước `[Batch]`. Nếu thí sinh truyền vào tensor 2D kiểu float `[Batch, 1]`, lỗi gì sẽ xảy ra? Hãy viết một gói Debug Prompt hoàn chỉnh để nhờ LLM giải thích lỗi này.

### Bài 2 (Easy): Tạo Synthetic Edge-Case kiểm thử hàm tiền xử lý
Viết một hàm `generate_edge_case_texts()` trả về danh sách các câu văn bản hiểm hóc:
- Chuỗi hoàn toàn rỗng `""`
- Chuỗi chỉ chứa toàn dấu cách `"   "`
- Chuỗi chỉ chứa toàn ký tự đặc biệt `@#$%^&*`
- Chuỗi có độ dài 10,000 từ
Chạy thử qua pipeline của bạn để xem hàm có bị sập (crash) không.

### Bài 3 (Medium): Phát hiện lỗi chia cho 0 trong chuẩn hóa dữ liệu
Khi một cột dữ liệu có tất cả các giá trị bằng nhau (độ lệch chuẩn $\sigma = 0$), công thức $x_{norm} = \frac{x - \mu}{\sigma}$ sẽ gây ra lỗi `ZeroDivisionError` hoặc sinh ra các giá trị `NaN`/`inf`. Viết một prompt yêu cầu LLM đề xuất cách xử lý an toàn bằng epsilon số học ($\epsilon = 10^{-7}$).

### Bài 4 (Medium): Kiểm chứng chéo phương thức bằng `help()` và Docstrings
Khi LLM gợi ý một tham số mới trong PyTorch (ví dụ `weights_only=True` trong `torch.load`), hãy viết câu lệnh Python dùng `help(torch.load)` hoặc `torch.load.__doc__` để tự mình kiểm tra xem tham số đó có thực sự tồn tại trong phiên bản bạn đang cài hay không.

### Bài 5 (Challenge): Xây dựng Bộ kiểm tra Tự động Không có NaN (Zero-NaN Assurance Sandbox)
Viết một hàm bọc (Decorator) `@assert_no_nan` áp dụng cho bất kỳ hàm huấn luyện nào. Nếu sau khi chạy hàm mà trọng số mô hình hoặc giá trị mất mát xuất hiện bất kỳ phần tử `NaN` hay `Inf` nào, chương trình sẽ lập tức ngắt và in ra cảnh báo đỏ cùng trạng thái của batch dữ liệu hiện tại.
"""
    voai_connection_content = """
Trong áp lực phòng thi Olympic AI, khi bạn gặp một thông báo lỗi dài 40 dòng, điều tồi tệ nhất là ngồi đọc mò từng chữ một cách vô định. Một thí sinh có trang bị **Quy trình LLM Debugging Workflow** sẽ chỉ mất 15 giây: Copy Traceback -> Thu gọn thành MRE -> Đưa vào Prompt Template -> Nhận giải pháp sửa lỗi -> Kiểm thử trên Edge Case -> Tiếp tục thi đấu! Kỹ năng này có thể cứu vãn cả một kỳ thi của bạn.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh copy giải pháp sửa lỗi của LLM nhưng không đọc kỹ, khiến LLM âm thầm thay thế hàm mất mát `nn.CrossEntropyLoss` thành `nn.MSELoss` mà thí sinh không hề hay biết:

```python
# CODE CÓ LỖI DO CHẤP NHẬN SỬA ĐỔI BỪA BÃI CỦA LLM:
# De bai yeu cau: Phan loai da lop (can CrossEntropyLoss)
# LLM thay loi Shape nen tu y doi sang MSELoss de "khong bi bao loi shape nua"!
# Hau qua: Mo hinh chay khong bao loi nhung do chinh xac rot ve 0%!
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao việc một đoạn code "chạy không báo lỗi" hoàn toàn khác với việc "mô hình học đúng bản chất toán học".
2. Khẳng định nguyên tắc: Tuyệt đối không cho phép LLM tự ý thay đổi hàm mất mát và mục tiêu của bài thi!
"""
    knowledge_check_content = """
1. Thuật ngữ MRE trong kỹ nghệ gỡ lỗi phần mềm viết tắt của cụm từ nào?
   - A. Maximum Resource Estimation
   - B. Minimal Reproducible Example (Mẫu Tái Lập Tối Thiểu)
   - C. Machine Recognition Engine
   - D. Model Regularization Error

2. Khi gửi thông báo lỗi cho LLM, hành động nào sau đây là TỐI KỴ và KÉM HIỆU QUẢ NHẤT?
   - A. Cung cấp toàn bộ Traceback chi tiết.
   - B. Chỉ ném mỗi câu: "Code bị lỗi, sửa đi!" mà không cung cấp thông báo lỗi, không cung cấp code tối thiểu và không nêu kích thước dữ liệu.
   - C. Cung cấp thông tin phiên bản hệ điều hành và thư viện.
   - D. Nêu giả thuyết cá nhân về nguyên nhân lỗi.

3. "Synthetic Edge Cases" (Các trường hợp kiểm thử biên nhân tạo) được thiết kế nhằm mục đích gì?
   - A. Làm cho máy tính chạy chậm lại.
   - B. Kiểm tra độ bền bỉ và khả năng chống chịu lỗi của mã nguồn trước các tình huống dữ liệu dị thường (như chuỗi rỗng, batch 1 mẫu, chia cho 0) trước khi nộp bài chính thức.
   - C. Tăng số lượng dòng code của notebook.
   - D. Thay thế cho việc thu thập dữ liệu thi đấu.

4. Phương thức nào trong Python giúp lập trình viên kiểm tra tài liệu và tham số chính thức của một hàm để đối soát xem gợi ý của LLM có phải là ảo giác hay không?
   - A. `print(math)`
   - B. `help(function_name)` hoặc `function_name.__doc__`
   - C. `delete(function_name)`
   - D. `exit()`

5. Khi LLM đề xuất một giải pháp sửa lỗi làm mất đi thông báo lỗi màu đỏ nhưng làm sai lệch bản chất toán học của bài toán (ví dụ đổi hàm mất mát), thái độ đúng đắn của thí sinh là gì?
   - A. Vui mừng nộp bài ngay lập tức.
   - B. Bác bỏ giải pháp đó, giữ vững hàm mục tiêu chuẩn mực và yêu cầu LLM tìm giải pháp sửa lỗi đúng đắn ở tầng kích thước dữ liệu (Tensor Reshaping).
   - C. Bỏ thi.
   - D. Tắt máy tính.
"""
    session_summary_content = """
* **What I learned:**
  - Quy trình 4 bước đóng gói báo cáo lỗi có cấu trúc (Structured Bug Report Packet).
  - Tầm quan trọng của Mẫu Tái Lập Tối Thiểu (Minimal Reproducible Example - MRE).
  - Cách dùng LLM sinh các bộ kiểm thử biên hiểm hóc (Synthetic Edge Cases).
  - Giao thức đối soát ảo giác API (Hallucination Cross-Verification).
* **What I can now do:**
  - Cô lập và giải quyết mọi lỗi sập code trong phòng thi một cách khoa học, bình tĩnh trong chưa đầy 60 giây.
  - Kiểm tra độ bền của pipeline trước khi bước vào giờ nộp bài chính thức.
* **What I still need to practice:**
  - Hoàn thiện sản phẩm cuối cùng của Module 6: **Personal Prompt Kit & Quy Trình Trợ Lực Thi Đấu** ở Buổi 3!
"""
    next_session_content = """
Buổi tiếp theo (**M6_S03**) là buổi tổng kết đỉnh cao của Module 6: **Xây Dựng Personal Prompt Kit & Quy Trình Trợ Lực Thi Đấu Không Lệ Thuộc**! Bạn sẽ hoàn thành bộ vũ khí Prompt Kit cá nhân hóa cho toàn bộ 5 giai đoạn làm bài thi, hoàn thiện quy trình Debugging Workflow và nhận Chứng chỉ **Certificate of AI-assisted Problem Solving**!
"""
    trace = {
        "items": [
            "Quy trình Debug code thi đấu có cấu trúc (Traceback + MRE)",
            "Sinh Synthetic Edge-cases kiểm thử độ bền pipeline",
            "Đối soát ảo giác API (Hallucination cross-verification protocol)"
        ],
        "deliverable": "Mã nguồn Framework đóng gói lỗi và kiểm thử biên có trợ lực AI"
    }
    nb = create_notebook(
        module_id="M6",
        module_name="DÙNG LLM NHƯ TRỢ LÝ HỌC VÀ THI",
        session_id="S02",
        topic="Quy Trình Debug Code Thi Đấu Bằng LLM & Đối Soát Ảo Giác",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook quy trình Debugging bằng LLM và kiểm thử biên an toàn",
        competition_relevance="Rút ngắn thời gian xử lý sự cố trong phòng thi từ hàng giờ xuống dưới 1 phút.",
        syllabus_traceability=trace,
        why_content=why_content,
        concept_content=concept_content,
        visual_intuition_content=visual_intuition_content,
        code_cells=code_cells,
        practice_content=practice_content,
        voai_connection_content=voai_connection_content,
        debugging_content=debugging_content,
        knowledge_check_content=knowledge_check_content,
        session_summary_content=session_summary_content,
        next_session_content=next_session_content
    )
    save_notebook(nb, "course/M6_LLM_ASSISTED/M6_S02.ipynb")


def build_m6_s03():
    objectives = [
        "Hoàn thành sản phẩm đầu ra chính thức của Module 6: Bộ công cụ Prompt cá nhân hóa (Personal Prompt Kit) bao quát 5 giai đoạn giải quyết bài toán AI.",
        "Thiết lập quy trình chuẩn 'LLM Debugging Workflow' được ghi nhận thành tài liệu quy chuẩn (Standard Operating Procedure).",
        "Hiểu và cam kết tuân thủ ranh giới đạo đức thi đấu (Competition Ethics & Academic Integrity): Dùng AI làm đòn bẩy gia tốc tư duy, không gian lận sao chép.",
        "Thực hành ứng dụng toàn bộ Personal Prompt Kit trên một tình huống thi đấu giả lập và nhận Chứng chỉ hoàn thành Module 6."
    ]
    prerequisites = [
        "Hoàn thành Buổi M6_S01 và M6_S02.",
        "Đã làm chủ toàn bộ kỹ năng từ M0 đến M5."
    ]
    why_content = """
Một người thợ lành nghề luôn có một hộp đồ nghề được sắp xếp ngăn nắp: cần cờ-lê nào là rút ra ngay trong tích tắc, không cần lục lọi tìm kiếm. Tương tự như vậy, một vận động viên Olympic AI xuất sắc trong kỷ nguyên mới luôn mang theo một **Bộ Công Cụ Prompt Cá Nhân Hóa (Personal Prompt Kit)** đã được tinh chỉnh, gọt giũa và kiểm chứng qua hàng trăm giờ huấn luyện thực tế.

Khi bước vào kỳ thi, bạn không có thời gian để ngồi gõ lại từng câu lệnh nhắc dài dòng. Bạn chỉ cần mở Personal Prompt Kit của mình ra, áp dụng cho bài toán cụ thể, và kích hoạt toàn bộ sức mạnh trợ lực của AI một cách kỷ luật, chuẩn xác và đạo đức. Đây là buổi học giúp bạn đóng gói vũ khí đó thành tài sản kỹ thuật trọn đời!
"""
    concept_content = r"""
### 1. Kiến trúc Personal Prompt Kit 5 Giai đoạn
Bộ công cụ được module hóa theo đúng chu trình thi đấu thực tế:
* **Giai đoạn 1: Giải mã Đề bài & Phân tích Rủi ro (Problem Deconstruction):**
  - Prompt trích xuất mục tiêu toán học, phát hiện các bẫy rò rỉ dữ liệu tiềm ẩn từ mô tả bài toán.
* **Giai đoạn 2: Brainstorm EDA & Kỹ nghệ Đặc trưng (Feature Ideation):**
  - Prompt gợi mở các đặc trưng tương tác, đặc trưng tỷ lệ và mã hóa danh mục dựa trên ngữ cảnh dữ liệu.
* **Giai đoạn 3: Phản biện Kiến trúc & Khảo sát Mô hình (Model Architecture Review):**
  - Prompt rà soát sự phù hợp của thuật toán, kiểm tra sự cân bằng giữa độ phức tạp và nguy cơ Overfitting.
* **Giai đoạn 4: Gỡ lỗi & Kiểm thử Biên An toàn (Debugging & Edge-Case Testing):**
  - Mẫu báo cáo lỗi MRE chuẩn mực kết hợp kiểm tra tính toàn vẹn của tensor và kiểm tra rò rỉ target.
* **Giai đoạn 5: Chuẩn hóa Mã nguồn & Vệ sinh Nộp bài (Refactoring & Submission Hygiene):**
  - Prompt làm sạch mã nguồn, bổ sung chú thích chuyên nghiệp và rà soát danh mục Checklist nộp bài.

### 2. Quy chuẩn Vận hành Gỡ lỗi bằng LLM (LLM Debugging Workflow - SOP)
Quy trình 5 bước bất di bất dịch:
`Phát hiện Lỗi -> Cô lập MRE -> Đóng gói Prompt RTCC -> Kiểm chứng Thực nghiệm -> Lưu trữ Tri thức Phòng ngừa`.
"""
    visual_intuition_content = """
```
Cấu trúc Bộ Công Cụ Personal Prompt Kit 5 Giai đoạn:
┌─────────────────────────────────────────────────────────────────────────────┐
│ [PHASE 1: PROBLEM DECONSTRUCTION] ──> Giải mã Thể lệ, Metric, Rủi ro        │
├─────────────────────────────────────────────────────────────────────────────┤
│ [PHASE 2: FEATURE IDEATION]       ──> Brainstorm Đặc trưng Tương tác & Tỷ lệ│
├─────────────────────────────────────────────────────────────────────────────┤
│ [PHASE 3: ARCHITECTURE REVIEW]    ──> Phản biện Khách quan, Chống Overfitting│
├─────────────────────────────────────────────────────────────────────────────┤
│ [PHASE 4: DEBUGGING & EDGE CASES] ──> Đóng gói MRE, Soát Ảo giác, Test Biên  │
├─────────────────────────────────────────────────────────────────────────────┤
│ [PHASE 5: CODE HYGIENE & AUDIT]   ──> Dọn sạch Code, Khóa Seed, Soát Nộp bài│
└─────────────────────────────────────────────────────────────────────────────┘
```
"""
    code_cells = [
        (
            """import json
import os

# 1. XAY DUNG VA XUAT BAN PERSONAL PROMPT KIT (OFFICIAL DELIVERABLE)
personal_prompt_kit = {
    "metadata": {
        "title": "VOAI Olympic AI Personal Prompt Kit",
        "version": "1.0",
        "author": "Freshman Olympic AI Athlete",
        "core_principle": "AI-Assisted, Human-Governed, Empirically-Verified"
    },
    "prompts": {
        "phase_1_problem_deconstruction": {
            "name": "Giai ma De bai & Metric Ranh gioi",
            "template": (
                "ROLE: Senior Competitive AI Architect.\\n"
                "CONTEXT: De thi Olympic AI, thoi gian {time_limit} phut, du lieu gom cac cot {feature_list}. Target la {target_col}.\\n"
                "TASK: Hay phan tich de bai va tra loi 3 cau hoi:\\n"
                "1. Metric {official_metric} phat nhung loai sai so nao nang nhat? Ham loss nao trong PyTorch/Sklearn tuong thich 100%?\\n"
                "2. Voi cac cot dinh danh {id_cols}, so do Validation nao la bat buoc de triet tieu Data Leakage?\\n"
                "3. Ba rui ro lon nhat co the khien mo hinh bi rot hang (Shakeup) tren Private Leaderboard la gi?\\n"
                "CONSTRAINT: Khong viet code giai san, chi phan tich chien thuat duoi 250 tu."
            )
        },
        "phase_2_feature_ideation": {
            "name": "Brainstorm Dac trung Tuong tac & Ty le",
            "template": (
                "ROLE: Feature Engineering Master in Competitive Machine Learning.\\n"
                "CONTEXT: Tap du lieu bang ve linh vuc {domain}, gom cac dac trung so hoc {numeric_cols} va danh muc {cat_cols}.\\n"
                "TASK: De xuat dung 5 dac trung moi co tinh dot pha ve mat tin hieu toan hoc:\\n"
                "- 2 dac trung ty le hoac nhan cheo (Ratios/Interactions).\\n"
                "- 2 dac trung thong ke gom nhom (Groupby Aggregations) giua danh muc va so hoc.\\n"
                "- 1 dac trung ma hoa tan suat (Frequency Encoding).\\n"
                "CONSTRAINT: Nêu ro truc giac toan hoc tai sao dac trung do giup tang diem, canh bao neu co nguy co Target Leakage."
            )
        },
        "phase_3_architecture_review": {
            "name": "Phan bien Kien truc & Chong Overfitting",
            "template": (
                "ROLE: Adversarial Code Reviewer & Deep Learning Specialist.\\n"
                "CONTEXT: Kien truc mang PyTorch ben duoi duoc thiet ke cho tap du lieu {num_samples} mau.\\n"
                "CODE:\\n{code_snippet}\\n"
                "TASK: Hay soi xet khong khoan nhuong va chi ra:\\n"
                "1. Kien truc co bi qua nang (Over-parameterized) dan toi Overfitting hay khong?\\n"
                "2. Vi tri cac tang Dropout va BatchNorm da dat chuan thu tu khoa hoc chua?\\n"
                "3. Goi y 2 giai phap toi uu hoa toc do chay ma khong lam giam do chinh xac."
            )
        },
        "phase_4_debugging_mre": {
            "name": "Dong goi Go loi MRE & Doi soat Ao giac",
            "template": (
                "ROLE: Python & PyTorch Systems Debugging Specialist.\\n"
                "MRE CODE:\\n{mre_code}\\n"
                "TRACEBACK:\\n{traceback_str}\\n"
                "METADATA: Tensor shapes: {tensor_shapes}, OS: {os_info}, Library: {lib_versions}.\\n"
                "TASK: Khoanh vung dung dong code gay loi va ban chat toan hoc.\\n"
                "CONSTRAINT: Chi de xuat sua doi toi thieu trong pham vi 3 dong code, kem 1 doan code test bien (Edge-case)."
            )
        },
        "phase_5_submission_hygiene": {
            "name": "Ve sinh Notebook & Sanity Check Nop bai",
            "template": (
                "ROLE: Quality Assurance & Reproducibility Auditor.\\n"
                "TASK: Hay dong vai giam khao kiem tra danh muc nop bai cho doan script sau:\\n"
                "1. Co dinh random seed cho numpy, random, torch chua?\\n"
                "2. File CSV xuat ra co bi loi cot index, co bi chua NaN nao khong?\\n"
                "3. Toan bo du doan co nam trong mien gia tri hop le {valid_range} khong?"
            )
        }
    }
}

# Xuat ban ra file JSON luu tru vinh vien
output_file = "personal_prompt_kit.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(personal_prompt_kit, f, indent=4, ensure_ascii=False)

print(f"-> Da xuat ban thanh cong Personal Prompt Kit v1.0 vao file: {output_file}")
print(f"Tong cong co {len(personal_prompt_kit['prompts'])} Module Prompt chuyen dung cho ca 5 giai doan!")
""",
            """Khối code tạo lập và xuất bản sản phẩm chính thức: `personal_prompt_kit.json`:
- Đóng gói đầy đủ 5 mẫu Prompt chuẩn mực cho toàn bộ các giai đoạn thi đấu.
- Sẵn sàng xuất khẩu, tái sử dụng và tích hợp vào bất kỳ môi trường lập trình nào."""
        ),
        (
            """# 2. XAY DUNG QUY CHUAN OPERATING PROCEDURE (LLM DEBUGGING WORKFLOW SOP)
workflow_sop = \"\"\"=============================================================================
             QUY TRÌNH VẬN HÀNH GỠ LỖI BẰNG LLM (LLM DEBUGGING WORKFLOW SOP)
=============================================================================
BƯỚC 1: CÔ LẬP VÀ XÁC ĐỊNH PHẠM VI (ISOLATION)
- Dừng ngay việc chạy toàn bộ notebook.
- Trích xuất đúng khối code gây lỗi và tạo một Minimal Reproducible Example (MRE) < 15 dòng.
- Ghi nhận chính xác shape của các tensor đầu vào/đầu ra ngay trước khi lỗi xảy ra.

BƯỚC 2: ĐÓNG GÓI BÁO CÁO LỖI CẤU TRÚC (PACKAGING)
- Nạp MRE, Full Traceback, Metadata môi trường vào Phase 4 Prompt Template của Personal Kit.
- Nêu rõ giả thuyết của bản thân để định hướng tư duy phản biện của LLM.

BƯỚC 3: ĐỐI SOÁT ẢO GIÁC (CROSS-VERIFICATION)
- Tuyệt đối không copy-paste ngay đề xuất của LLM.
- Kiểm tra tính xác thực của các hàm/tham số được đề xuất bằng `help()` hoặc Official Docs.
- Đảm bảo LLM không tự ý thay đổi hàm mất mát hay mục tiêu bài toán.

BƯỚC 4: THỬ NGHIỆM TRÊN TẬP BIÊN (EDGE-CASE TESTING)
- Chạy thử giải pháp sửa lỗi trên các dữ liệu biên (Empty string, Single sample, All-zero).
- Đảm bảo giải pháp giải quyết tận gốc nguyên nhân chứ không phải chỉ "che giấu" lỗi.

BƯỚC 5: TÍCH HỢP VÀ GHI NHỚ TRI THỨC (INTEGRATION & KNOWLEDGE LOG)
- Cập nhật giải pháp an toàn vào notebook chính.
- Ghi chú 1 dòng nhận xét: Nguyên nhân gây lỗi và cách phòng ngừa cho các bài thi sau.
=============================================================================\"\"\"

with open("LLM_DEBUGGING_WORKFLOW.txt", "w", encoding="utf-8") as f:
    f.write(workflow_sop)

print(workflow_sop)
print("\\n-> Da xuat ban tai lieu quy trinh LLM_DEBUGGING_WORKFLOW.txt thanh cong!")
""",
            """Khối code xuất bản tài liệu quy chuẩn vận hành: `LLM_DEBUGGING_WORKFLOW.txt`:
- Định hình quy trình 5 bước gỡ lỗi khép kín, có tính kỷ luật và bảo mật cao.
- Hoàn thành trọn vẹn cả hai sản phẩm đầu ra bắt buộc của Module 6."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Tùy biến Personal Prompt Kit
Đọc file `personal_prompt_kit.json` và bổ sung thêm một prompt mới thuộc `Phase 6: Final Presentation & Defense` (Prompt hỗ trợ tóm tắt giải pháp thành 3 ý chính để thuyết trình trước ban giám khảo).

### Bài 2 (Easy): Kích hoạt Prompt Phase 1 cho một bài toán cụ thể
Sử dụng phương thức chuỗi `.format(...)` để điền các thông tin thực tế của bài toán phát hiện gian lận tín dụng (Credit Fraud) vào template của Phase 1. In prompt hoàn chỉnh ra màn hình.

### Bài 3 (Medium): Kiểm thử quy trình Debugging Workflow trên lỗi CUDA OOM
Tạo một kịch bản giả định: Mô hình bị tràn bộ nhớ VRAM khi tăng batch size lên 256. Áp dụng tuần tự 5 bước của `LLM Debugging Workflow SOP` để mô tả cách bạn khoanh vùng và khắc phục sự cố (ví dụ: áp dụng Gradient Accumulation).

### Bài 4 (Medium): Phân tích Ranh giới Đạo đức Thi đấu (Competition Ethics)
Nêu rõ ranh giới phân định giữa:
1. Hành vi hợp lệ: Dùng LLM giải thích lỗi cú pháp, gợi ý công thức toán, hoặc tối ưu hóa tốc độ code.
2. Hành vi gian lận (Disqualification): Sử dụng LLM để giải hộ toàn bộ bài thi, nhờ người khác thi hộ qua tài khoản AI, hoặc cố tình dịch ngược dữ liệu kiểm tra ẩn.

### Bài 5 (Challenge): Xây dựng Ứng dụng Trợ lý Dòng lệnh (CLI Prompt Kit Assistant)
Viết một script Python ngắn `prompt_assistant.py` cho phép người dùng chạy từ terminal:
`python prompt_assistant.py --phase 2 --domain "Y tế"`
và tự động in ra mẫu prompt tương ứng đã được điền sẵn thông số.
"""
    voai_connection_content = """
Trong thi đấu Olympic AI, sự tự tin của một thí sinh không đến từ việc cầu may đề thi sẽ trúng tủ, mà đến từ **hệ thống quy trình làm việc chuẩn hóa (Disciplined Workflow)**. Khi bạn có trong tay Personal Prompt Kit và làm chủ quy trình LLM Debugging, bạn sẽ không bao giờ bị bối rối hay mất phương hướng dù đề thi có khó khăn đến đâu. Bạn biết chính xác từng bước mình cần làm từ phút thứ 1 đến phút thứ 180 của trận đấu.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh sử dụng Prompt Kit nhưng quên điền các thông tin ràng buộc cụ thể của môi trường thi (ví dụ: máy thi không có kết nối Internet để tải các thư viện ngoài), khiến LLM đề xuất cài thêm một package lạ:

```python
# CODE CÓ LỖI VI PHẠM QUY CHẾ MÔI TRƯỜNG PHÒNG THI:
# LLM de xuat: pip install optuna # Phong thi Olympic AI thuong cam ket noi Internet tai runtime!
# Lenh nay se gay loi connection timeout va lam gian doan qua trinh thi dau!
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao trong mọi Prompt gửi cho AI, bối cảnh môi trường (Offline / No-Internet / Fixed Libraries) phải luôn được nêu rõ ở phần CONTEXT.
2. Cập nhật ràng buộc này vào toàn bộ các mẫu template trong Personal Prompt Kit.
"""
    knowledge_check_content = """
1. Hai sản phẩm thực hành bắt buộc phải hoàn thành khi kết thúc Module M6 là gì?
   - A. Một bài thơ ca ngợi trí tuệ nhân tạo.
   - B. Personal Prompt Kit (Bộ công cụ Prompt cá nhân hóa) và LLM Debugging Workflow (Quy trình vận hành gỡ lỗi có cấu trúc).
   - C. Một mạng nơ-ron Transformer 1 tỷ tham số.
   - D. Một bài kiểm tra thể lực.

2. Tại sao Personal Prompt Kit lại được chia thành 5 giai đoạn riêng biệt khớp với chu trình thi đấu?
   - A. Để làm cho tài liệu dài hơn.
   - B. Giúp thí sinh có thể tra cứu và áp dụng ngay lập tức mẫu prompt tương ứng với từng giai đoạn làm bài mà không mất thời gian suy nghĩ câu hỏi dưới áp lực phòng thi.
   - C. Vì ban tổ chức yêu cầu chia làm 5 phần.
   - D. Do giới hạn của file JSON.

3. Bước thứ 3 trong Quy trình Gỡ lỗi Chuẩn (LLM Debugging Workflow SOP) là gì?
   - A. Nộp bài ngay lập tức.
   - B. Đối soát ảo giác (Cross-Verification): Kiểm tra tính xác thực của các đề xuất của LLM bằng tài liệu chính thức trước khi áp dụng.
   - C. Xóa toàn bộ code cũ.
   - D. Tắt máy tính đi ngủ.

4. Hành vi nào sau đây được coi là vi phạm nghiêm trọng Đạo đức Thi đấu và Quy chế Kỳ thi Olympic AI?
   - A. Dùng LLM giải thích thông báo lỗi traceback của PyTorch.
   - B. Dùng LLM để gian lận sao chép toàn bộ lời giải từ nguồn cấm hoặc chia sẻ tài khoản làm bài cho bên thứ ba trong thời gian thi đấu chính thức.
   - C. Dùng LLM để kiểm tra lỗi chính tả trong file báo cáo.
   - D. Đọc tài liệu Scikit-Learn.

5. Sau khi hoàn thành xuất sắc Buổi M6_S03, học viên chính thức đạt được chứng chỉ cột mốc nào?
   - A. Certificate of AI Orientation
   - B. Certificate of Machine Learning Basics
   - C. Certificate of AI-assisted Problem Solving
   - D. Final Project Certificate
"""
    session_summary_content = """
* **What I learned:**
  - Cấu trúc hoàn chỉnh của Personal Prompt Kit 5 giai đoạn thi đấu đỉnh cao.
  - Quy chuẩn vận hành gỡ lỗi khép kín (LLM Debugging Workflow SOP).
  - Các nguyên tắc đạo đức thi đấu và giữ vững tư duy làm chủ của con người.
* **What I can now do:**
  - Sử dụng thành thạo bộ Personal Prompt Kit như một vũ khí gia tốc giải quyết bài toán AI.
  - Tự tin gỡ lỗi, phản biện và tối ưu hóa giải pháp thi đấu có sự trợ lực thông minh của LLM.
* **What I still need to practice:**
  - Mang toàn bộ hành trang của 6 module trước bước vào chặng đường cuối cùng: **Module M7: Dự án AI thật để đưa vào Portfolio**!
"""
    next_session_content = """
Xin nhiệt liệt chúc mừng bạn đã chính thức tốt nghiệp Module M6 và nhận Chứng chỉ **Certificate of AI-assisted Problem Solving**!
Trong buổi tiếp theo (**M7_S01**), chúng ta sẽ bước vào đỉnh núi cuối cùng của khóa học: **Module M7: DỰ ÁN AI THẬT ĐỂ ĐƯA VÀO PORTFOLIO** — Bắt đầu khởi động Final Project: Xác định đề tài thế giới thực, thu thập và kiểm toán dữ liệu đạt chuẩn công nghiệp!
"""
    module_assessment = """
---

## 🏆 MODULE 6 ASSESSMENT & CERTIFICATION

### Phần A: Đánh giá Năng lực Tương tác & Phản biện AI (Theory & Prompt Mastery)
- Nắm vững cấu trúc RTCC (Role - Task - Context - Constraint) và cách kiểm soát ảo giác (Hallucination).
- Hiểu sâu sắc quy trình 5 bước của LLM Debugging Workflow SOP và nguyên tắc kiểm chứng thực nghiệm.
- Nắm vững ranh giới đạo đức thi đấu và nguyên tắc giữ vững quyền làm chủ tư duy (Human Agency).

### Phần B: Sản phẩm Thực hành Bắt buộc (Practical Deliverable)
- **Personal Prompt Kit (File JSON / Markdown):** Bộ công cụ hoàn chỉnh gồm tối thiểu 5 mẫu Prompt chuyên dụng cho 5 giai đoạn (Problem Deconstruction, Feature Ideation, Architecture Review, MRE Debugging, Submission Hygiene).
- **Quy trình Gỡ lỗi Chuẩn (LLM Debugging Workflow SOP):** Tài liệu quy chuẩn vận hành gỡ lỗi có cấu trúc kèm mã nguồn kiểm thử thực nghiệm trên các tình huống lỗi thực tế.

### Tiêu chí Đạt chuẩn (Rubric):
| Tiêu chí | Trọng số | Yêu cầu đạt |
| :--- | :--- | :--- |
| **Cấu trúc Personal Prompt Kit** | 35% | Đủ 5 giai đoạn, chuẩn RTCC, có ràng buộc định hướng tư duy |
| **Quy trình Debugging SOP**       | 25% | Đủ 5 bước khép kín, có MRE và đối soát ảo giác |
| **Kiểm thử Thực nghiệm (Sandbox)** | 25% | Đo lường hiệu quả bằng số liệu Benchmark khách quan |
| **Đạo đức & Tư duy Làm chủ**     | 15% | Hiểu rõ ranh giới đạo đức, không sao chép mù quáng |

**Chứng chỉ hoàn thành:**
🏅 **Certificate of AI-assisted Problem Solving** — Đủ điều kiện tiến vào Module M7 (Dự án AI thật Portfolio).
"""
    trace = {
        "items": [
            "Xây dựng Personal Prompt Kit 5 giai đoạn thi đấu",
            "Thiết lập chuẩn mực quy trình LLM Debugging Workflow SOP",
            "Đạo đức thi đấu & Tự chủ tư duy (Human Agency)",
            "Nhận chứng chỉ Certificate of AI-assisted Problem Solving"
        ],
        "deliverable": "Personal Prompt Kit (JSON) + LLM Debugging Workflow (SOP) hoàn chỉnh"
    }
    nb = create_notebook(
        module_id="M6",
        module_name="DÙNG LLM NHƯ TRỢ LÝ HỌC VÀ THI",
        session_id="S03",
        topic="Xây Dựng Personal Prompt Kit & Quy Trình Trợ Lực Thi Đấu",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Personal Prompt Kit (JSON) + LLM Debugging Workflow (SOP)",
        competition_relevance="Vũ khí gia tốc tư duy và bảo hiểm gỡ lỗi thần tốc trong phòng thi Olympic AI.",
        syllabus_traceability=trace,
        why_content=why_content,
        concept_content=concept_content,
        visual_intuition_content=visual_intuition_content,
        code_cells=code_cells,
        practice_content=practice_content,
        voai_connection_content=voai_connection_content,
        debugging_content=debugging_content,
        knowledge_check_content=knowledge_check_content,
        session_summary_content=session_summary_content,
        next_session_content=next_session_content,
        module_assessment=module_assessment
    )
    save_notebook(nb, "course/M6_LLM_ASSISTED/M6_S03.ipynb")

if __name__ == "__main__":
    build_m6_s01()
    build_m6_s02()
    build_m6_s03()
    print("M6 build complete! 3/3 notebooks generated.")
