"""
builder/build_m0.py
Generates M0_S01.ipynb and M0_S02.ipynb for Module 0.
"""

from utils import create_notebook, save_notebook

def build_m0_s01():
    objectives = [
        "Phân biệt rạch ròi bản chất kỹ thuật giữa Trí tuệ Nhân tạo (AI), Học máy (ML) và Học sâu (DL).",
        "Nắm vững bức tranh tổng thể của một AI Pipeline hoàn chỉnh trong nghiên cứu và thi đấu.",
        "Hiểu rõ cấu trúc đề thi, tiêu chuẩn đánh giá và phổ năng lực bắt buộc của kỳ thi Olympic AI (VOAI).",
        "Thực hiện self-assessment ban đầu thông qua công cụ trực quan hóa năng lực bằng Python."
    ]
    prerequisites = [
        "Tư duy logic cơ bản của sinh viên năm nhất ngành AI.",
        "Kỹ năng cài đặt và chạy môi trường Python / Jupyter Notebook cơ bản."
    ]
    
    why_content = """
Trong làn sóng bùng nổ công nghệ, phần lớn sinh viên năm nhất tiếp cận AI theo cách "học vẹt": sao chép mã nguồn, gọi thư viện mà không hiểu bản chất toán học hay luồng dữ liệu bên dưới. Khi bước vào các kỳ thi học thuật đỉnh cao như **Olympic AI (VOAI)** hoặc các đấu trường quốc tế, cách tiếp cận này sẽ thất bại hoàn toàn ngay khi đề bài thay đổi phân phối dữ liệu hoặc yêu cầu tối ưu hóa tài nguyên.

AI không phải là ma thuật, mà là sự kết hợp chặt chẽ giữa **Toán học (Đại số tuyến tính, Giải tích, Xác suất thống kê)**, **Khoa học dữ liệu (Data Engineering)** và **Kỹ thuật phần mềm (Software Engineering)**. Hiểu đúng vị trí của bản thân trong bức tranh này là bước đầu tiên để bạn không bị ngợp và xây dựng được lộ trình bứt phá.
"""

    concept_content = """
### 1. Phân định AI vs Machine Learning vs Deep Learning
* **Artificial Intelligence (AI - Trí tuệ Nhân tạo):** Khái niệm bao trùm từ thập niên 1950, chỉ bất kỳ hệ thống nào thể hiện hành vi thông minh (từ hệ chuyên gia rule-based, thuật toán tìm kiếm A*, heuristic đến học máy).
* **Machine Learning (ML - Học máy):** Một tập con của AI trong đó hệ thống tự động học các quy luật toán học (hàm ánh xạ $y = f(x)$) từ dữ liệu quan sát mà không cần lập trình tường minh từng quy tắc `if-else`.
* **Deep Learning (DL - Học sâu):** Một tập con của ML sử dụng các mạng nơ-ron nhân tạo sâu (Deep Neural Networks) với nhiều tầng biểu diễn phân cấp, có khả năng tự động trích xuất đặc trưng (representation learning) từ dữ liệu phi cấu trúc như ảnh, âm thanh, văn bản.

```
+--------------------------------------------------------+
| Artificial Intelligence (AI)                           |
|  +--------------------------------------------------+  |
|  | Machine Learning (ML)                            |  |
|  |   +--------------------------------------------+ |  |
|  |   | Deep Learning (DL)                         | |  |
|  |   |  (CNN, RNN, Transformers, PyTorch)         | |  |
|  |   +--------------------------------------------+ |  |
|  |   (Scikit-Learn, Random Forest, XGBoost)         |  |
|  +--------------------------------------------------+  |
|  (Search, Logic, Heuristics, Expert Systems)           |
+--------------------------------------------------------+
```

### 2. Chu trình AI Pipeline trong thi đấu Olympic
Một bài toán Olympic AI tiêu chuẩn đòi hỏi thí sinh phải làm chủ toàn bộ chuỗi mắt xích:
1. **Problem Formulation:** Đọc hiểu đề bài, xác định mục tiêu toán học và metric chấm điểm (F1, RMSE, MAP...).
2. **Data Exploration & Cleaning (EDA):** Khám phá bất thường, xử lý missing value, phân tích phân phối nhãn.
3. **Validation Strategy:** Thiết lập cơ chế kiểm chứng (K-Fold, Stratified Split) để tránh Overfitting và Data Leakage.
4. **Feature Engineering / Representation:** Biến đổi dữ liệu thô thành vectơ đặc trưng giàu thông tin.
5. **Model Architecture & Training:** Lựa chọn thuật toán nền tảng, huấn luyện và kiểm soát quá trình hội tụ.
6. **Ensembling & Post-processing:** Kết hợp mô hình, tinh chỉnh ngưỡng quyết định để tối ưu điểm số.
7. **Submission Hygiene:** Đóng gói mã nguồn sạch, đảm bảo tính tái lập (reproducibility) trong thời gian quy định.
"""

    visual_intuition_content = """
Hãy hình dung năng lực thi Olympic AI như một **Biểu đồ Radar đa chiều (Spider/Radar Chart)** gồm 6 trục chính:
1. **Nền tảng Toán (Math Foundation):** Đại số tuyến tính, Đạo hàm, Xác suất thống kê.
2. **Lập trình Python & Data Wrangling:** NumPy, Pandas, tối ưu hóa tốc độ tính toán.
3. **Tư duy Mô hình (ML/DL Foundations):** Hiểu rõ cơ chế tối ưu hàm mất mát, bias-variance tradeoff.
4. **Kỹ năng Thực chiến (Applied CV & NLP):** Khả năng xử lý ảnh số và văn bản thực tế.
5. **Chiến thuật Thi đấu (Competition Strategy):** Đọc đề, tạo baseline nhanh, validation chặt chẽ.
6. **Kỹ năng Tự lực & Công cụ (LLM-Assisted & Debugging):** Sử dụng AI để tăng tốc chứ không bị lệ thuộc.
"""

    code_cells = [
        (
            """import numpy as np
import matplotlib.pyplot as plt

# 1. Khai báo 6 trục năng lực cốt lõi của Olympic AI
categories = [
    'Math Foundation\\n(Toan hoc)',
    'Python & Data\\n(NumPy/Pandas)',
    'ML/DL Concepts\\n(Ban chat mo hinh)',
    'Applied CV/NLP\\n(Ung dung)',
    'Competition Tactics\\n(Chien thuat thi)',
    'Debugging & Tools\\n(Kiem loi & LLM)'
]
num_vars = len(categories)

# 2. Thiet lap moc nang luc mau: Sinh vien nam nhat khoi diem vs Muc tieu Olympic VOAI
baseline_freshman = [2.5, 3.0, 1.5, 1.0, 1.0, 2.0]
target_voai = [4.5, 4.5, 4.5, 4.0, 4.5, 4.0]

# Tinh goc cho radar chart
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
baseline_freshman += baseline_freshman[:1]
target_voai += target_voai[:1]
angles += angles[:1]

# 3. Ve bieu do danh gia nang luc
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.plot(angles, baseline_freshman, color='#1f77b4', linewidth=2, linestyle='solid', label='Hien tai (Freshman Baseline)')
ax.fill(angles, baseline_freshman, color='#1f77b4', alpha=0.25)

ax.plot(angles, target_voai, color='#d62728', linewidth=2, linestyle='dashed', label='Muc tieu Olympic AI (VOAI Ready)')
ax.fill(angles, target_voai, color='#d62728', alpha=0.15)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
ax.set_ylim(0, 5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(['1: So cap', '2: Co ban', '3: Trung cap', '4: Thao thao', '5: Chuyen gia'], color='gray', size=8)
plt.title('VOAI COMPETENCY RADAR CHART\\nKhoang cach nang luc va Muc tieu huan luyen', size=13, y=1.1, fontweight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()
""",
            """Đoạn code trên khởi tạo một cấu trúc dữ liệu mô tả 6 trục năng lực của vận động viên Olympic AI và vẽ biểu đồ Radar chuẩn hóa bằng `matplotlib`.
- **Input:** Mảng điểm đánh giá (thang 1-5) trên 6 chiều năng lực.
- **Output:** Biểu đồ cực (polar plot) thể hiện trực quan khoảng cách (gap) giữa năng lực hiện tại của tân sinh viên và chuẩn đầu ra Olympic.
- **Tại sao viết như vậy:** Giúp học viên định hình ngay lập tức "Tôi đang ở đâu?" và "Khóa học 38 buổi này đưa tôi đi đến đâu?".
- **Lỗi thường gặp:** Quên đóng vòng tròn dữ liệu (`angles += angles[:1]`), dẫn đến biểu đồ bị hở cạnh cuối cùng."""
        )
    ]

    practice_content = """
### Bài 1 (Easy): Tự hiệu chỉnh Radar Chart
Thay đổi các giá trị trong biến `baseline_freshman` dựa trên tự đánh giá thành thật của bạn về kỹ năng cá nhân (thang điểm 1.0 đến 5.0) và chạy lại code để quan sát diện mạo ban đầu của mình.

### Bài 2 (Easy): Phân loại bài toán thực tế
Với mỗi bài toán sau, hãy chỉ ra bài toán đó thuộc phạm vi **AI truyền thống (Rule-based/Search)**, **Classical Machine Learning**, hay **Deep Learning**:
1. Lập lịch phân công phòng thi tránh xung đột thời gian của các đội.
2. Dự đoán giá căn hộ dựa trên diện tích, số phòng và khoảng cách tới trung tâm.
3. Nhận diện biển số xe ô tô từ camera giám sát giao thông ban đêm.
4. Trả lời câu hỏi tự do từ tài liệu quy chế thi dài 200 trang.

### Bài 3 (Medium): Ánh xạ AI Pipeline vào thời gian phòng thi
Một kỳ thi Olympic AI diễn ra trong 4 giờ liên tục. Hãy phân bổ tỷ lệ thời gian (%) hợp lý cho 7 bước của AI Pipeline đã nêu trong phần Concept. Giải thích tại sao việc dành 80% thời gian chỉ để chạy mô hình Deep Learning phức tạp mà bỏ qua EDA và Validation là chiến lược sai lầm.

### Bài 4 (Medium): Phân tích rủi ro Metric
Nếu một bài thi sử dụng nhãn bị mất cân bằng trầm trọng (99% dữ liệu là lớp 0, chỉ 1% là lớp 1). Nếu một thí sinh nộp mô hình luôn đoán là 0 thì độ chính xác (Accuracy) là bao nhiêu? Tại sao ban tổ chức Olympic AI hiếm khi chọn Accuracy làm metric chính cho bài toán phân loại?

### Bài 5 (Challenge): Thiết kế kịch bản phát hiện Overfitting trong phòng thi
Viết một đoạn mã ngắn giả lập trường hợp: Điểm đánh giá trên tập huấn luyện (Train Score) đạt 99.5%, nhưng khi nộp lên hệ thống chấm ẩn (Public/Private Leaderboard) thì điểm rớt xuống 62%. Nêu 3 nguyên nhân cốt lõi dẫn tới hiện tượng "gãy mô hình" này trong phòng thi thực tế.
"""

    voai_connection_content = """
Trong đề thi Olympic AI, ban tổ chức không bao giờ kiểm tra bạn định nghĩa lý thuyết "AI là gì" như một câu hỏi trắc nghiệm phổ thông. Thay vào đó, họ sẽ đưa ra một bài toán mở với dữ liệu có nhiễu, phân phối nhãn dị biệt hoặc tài nguyên tính toán bị giới hạn (ví dụ: tối đa 2 giờ chạy trên CPU/GPU quy định).

Thí sinh thiếu định hướng sẽ vội vàng import các thư viện "khủng" hoặc copy-paste code từ các diễn đàn mà không đọc kỹ dữ liệu, dẫn tới hiện tượng tràn RAM, nộp trễ giờ, hoặc mô hình bị học vẹt (overfitting). Nắm vững tư duy tổng quan giúp bạn luôn giữ được sự bình tĩnh: **Hiểu bài toán -> Xây baseline đơn giản trong 30 phút đầu -> Liên tục cải tiến có kiểm chứng.**
"""

    debugging_content = """
**Tình huống thực tế:** Một sinh viên năm nhất muốn chạy thử nghiệm phân loại văn bản, tải về một file dữ liệu 500MB và viết vòng lặp `for` lồng nhau bằng Python thuần để đếm từ khóa. Sau 45 phút, Jupyter Notebook bị đơ hoàn toàn (Kernel Dead / Memory Out).

```python
# CODE CÓ LỖI TƯ DUY HIỆU NĂNG:
# Gia su data la danh sach 100,000 cau van ban
def slow_feature_extractor(corpus, vocab):
    features = []
    for doc in corpus:
        doc_vec = []
        for word in vocab:
            # Dem so lan word xuat hien trong doc
            count = doc.split().count(word)
            doc_vec.append(count)
        features.append(doc_vec)
    return features
```

**Nhiệm vụ của bạn:**
1. Chỉ ra độ phức tạp thuật toán (Big-O) của đoạn code trên khi `N = 100,000` văn bản và `V = 20,000` từ vựng.
2. Giải thích vì sao việc dùng Python thuần lặp qua từng phần tử mà không dùng Vectorization hoặc thư viện chuyên dụng là "án tử" trong phòng thi Olympic AI có giới hạn thời gian.
"""

    knowledge_check_content = """
1. Sự khác biệt cốt lõi nhất giữa Machine Learning cổ điển và Deep Learning là gì?
   - A. Deep Learning luôn luôn chính xác hơn Machine Learning trên mọi tập dữ liệu nhỏ.
   - B. Deep Learning có khả năng tự động học trích xuất đặc trưng (representation learning), trong khi ML cổ điển phụ thuộc nhiều vào kỹ nghệ đặc trưng thủ công (manual feature engineering).
   - C. Machine Learning không sử dụng toán học giải tích.
   - D. Deep Learning không thể huấn luyện trên CPU.

2. Tại sao một hệ chuyên gia gồm 10,000 câu lệnh `if-else` phức tạp vẫn được coi là AI nhưng KHÔNG PHẢI là Machine Learning?
   - A. Vì nó chạy trên máy tính thế hệ cũ.
   - B. Vì nó không tự học tham số từ dữ liệu thực nghiệm mà dựa vào tri thức đóng cứng của con người.
   - C. Vì nó có độ chính xác 100%.
   - D. Vì nó không có mã nguồn mở.

3. Trong AI Pipeline của một kỳ thi, bước nào quyết định trực tiếp tới việc bạn có đánh giá trung thực năng lực của mô hình trước khi nộp bài hay không?
   - A. Bước chọn màu sắc cho biểu đồ hiển thị.
   - B. Thiết lập chiến lược chia tập kiểm chứng (Validation Strategy) chuẩn mực, không rò rỉ thông tin.
   - C. Cài đặt phiên bản mới nhất của thư viện PyTorch.
   - D. Sử dụng máy tính có nhiều luồng CPU nhất.

4. Hiện tượng "Data Leakage" (Rò rỉ dữ liệu) thường xảy ra khi nào?
   - A. Khi máy tính bị mất kết nối mạng Internet.
   - B. Khi thông tin từ tập kiểm tra (Test set) vô tình bị rò rỉ vào quá trình huấn luyện hoặc xử lý dữ liệu của tập Train.
   - C. Khi file CSV bị hỏng ổ đĩa.
   - D. Khi mô hình bị tràn bộ nhớ RAM.

5. Mục tiêu tối thượng của việc tham gia kỳ thi Olympic AI đối với một sinh viên năm nhất là gì?
   - A. Thuộc lòng mọi công thức và dòng lệnh của Scikit-Learn.
   - B. Rèn luyện tư duy giải quyết vấn đề bằng dữ liệu, làm chủ luồng thực thi, và xây dựng bản lĩnh kỹ thuật vững chắc cho chặng đường dài hạn.
   - C. Chỉ tập trung lấy chứng chỉ mà không cần chạy code.
   - D. Copy code nhanh nhất có thể từ các bài thi mẫu.
"""

    session_summary_content = """
* **What I learned:**
  - Nắm vững ranh giới kỹ thuật giữa AI, ML và DL.
  - Hiểu trọn vẹn 7 bước của một AI Pipeline trong nghiên cứu và thi đấu.
  - Nhận diện 6 trụ cột năng lực bắt buộc để chinh phục Olympic AI (VOAI).
* **What I can now do:**
  - Tự phân loại chính xác các bài toán AI trong thực tế vào đúng nhóm thuật toán.
  - Sử dụng Python và Matplotlib để trực quan hóa bản đồ năng lực cá nhân.
* **What I still need to practice:**
  - Chuẩn bị kiến thức nền tảng để thực hiện bài thi chẩn đoán (Diagnostic Test) ở buổi 2.
"""

    next_session_content = """
Trong **Buổi M0_S02**, bạn sẽ chính thức làm bài **Diagnostic Test Đầu Vào** gồm 20 câu hỏi đánh giá toàn diện về Toán, Lập trình Python, Tư duy Dữ liệu và Problem Solving. Kết quả bài test sẽ tự động tổng hợp để xuất ra **Personal AI Learning Map** cá nhân hóa cho bạn suốt 38 buổi học!
"""

    trace = {
        "items": [
            "Hiểu AI là gì và ranh giới với ML/DL",
            "Olympic AI cần gì và cấu trúc đánh giá năng lực",
            "Xác định điểm xuất phát của học viên qua radar chart"
        ],
        "deliverable": "Bản đồ năng lực ban đầu (Baseline Radar Competency Chart)"
    }

    nb = create_notebook(
        module_id="M0",
        module_name="BẮT ĐẦU ĐÚNG HƯỚNG VỚI AI",
        session_id="S01",
        topic="AI là gì & Olympic AI cần gì?",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Bản đồ năng lực ban đầu & Bảng phân tích AI Pipeline",
        competition_relevance="Định hình tư duy giải quyết vấn đề, hiểu luật chơi và tiêu chuẩn phòng thi Olympic AI.",
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
    save_notebook(nb, "course/M0_AI_ORIENTATION/M0_S01.ipynb")


def build_m0_s02():
    objectives = [
        "Hoàn thành bài kiểm tra chẩn đoán năng lực đầu vào (Diagnostic Test) gồm 4 phân vùng kiến thức.",
        "Sử dụng công cụ tính điểm tự động bằng Python để phân tích điểm mạnh, điểm yếu.",
        "Xây dựng và xuất bản Bản đồ học tập cá nhân (Personal AI Learning Map) trải dài qua 38 buổi học.",
        "Thiết lập mục tiêu và cam kết kỷ luật học tập đạt chuẩn Olympic AI."
    ]
    prerequisites = [
        "Đã hoàn thành Buổi M0_S01.",
        "Kỹ năng chạy code cell và nhập dữ liệu trong Jupyter Notebook."
    ]
    
    why_content = """
Không có một lộ trình học tập nào hiệu quả cho tất cả mọi người nếu không biết rõ **điểm xuất phát**. Một sinh viên năm nhất có thể rất vững tư duy toán giải tích từ cấp 3 nhưng chưa từng chạm vào cú pháp Python; ngược lại, một bạn khác có thể thành thạo viết script lập trình nhưng hoàn toàn bối rối trước các phép nhân ma trận và vi phân đạo hàm.

Nếu bạn nhảy ngay vào huấn luyện mạng nơ-ron sâu mà không biết mình đang hổng kiến thức nền tảng nào, bạn sẽ liên tục gặp lỗi cú pháp, bế tắc khi điều chỉnh siêu tham số, và dễ dàng nản chí. Bài test đầu vào và Bản đồ học tập cá nhân hôm nay được thiết kế để bạn "soi gương" kỹ thuật một cách trung thực nhất.
"""

    concept_content = """
### 1. Cấu trúc bài Diagnostic Test Đầu Vào
Bài kiểm tra gồm 20 câu hỏi được chuẩn hóa theo 4 trụ cột cốt lõi:
1. **Phân vùng A: Toán học cho AI (Math Foundations - 5 câu):**
   - Đại số tuyến tính (Vector, Ma trận, Phép nhân vô hướng Dot Product).
   - Giải tích (Đạo hàm, Tiếp tuyến, Quy tắc chuỗi Chain Rule).
   - Xác suất thống kê (Kỳ vọng, Phương sai, Định lý Bayes cơ bản).
2. **Phân vùng B: Nền tảng Lập trình Python (Python Core - 5 câu):**
   - Cấu trúc dữ liệu (List, Dict, Set, Tuple), List Comprehension.
   - Hàm, Tham số, Phạm vi biến (Scope), Lambda function.
   - Tư duy viết code tối ưu tránh lặp vô ích.
3. **Phân vùng C: Tư duy Dữ liệu & Thống kê (Data Reasoning - 5 câu):**
   - Đọc bảng dữ liệu, phân biệt Biến liên tục (Continuous) vs Phân loại (Categorical).
   - Xử lý giá trị ngoại lai (Outliers), Mean vs Median trong phân phối lệch.
   - Trực giác về tương quan (Correlation) và Nhân quả (Causation).
4. **Phân vùng D: Tư duy Thuật toán & Problem Solving (5 câu):**
   - Phân tích độ phức tạp thời gian $O(N), O(N^2), O(\log N)$.
   - Tư duy chia để trị, tìm kiếm nhị phân.
   - Khả năng mô hình hóa một vấn đề đời thực thành bài toán tối ưu hóa.

### 2. Nguyên lý Xây dựng Bản đồ Học tập Cá nhân (Learning Map)
Dựa trên trọng số điểm của từng phân vùng, thuật toán sẽ tự động phân loại bạn vào 1 trong 3 Profile học tập:
* **Profile 1: Strong Coder / Math Gap:** Cần tập trung sâu vào trực giác toán học ở M2, M3.
* **Profile 2: Strong Math / Coding Gap:** Cần cày sâu kỹ năng thực hành Python/NumPy/Pandas ở M1 và viết PyTorch loop ở M3.
* **Profile 3: Dual Explorer (Khởi đầu đồng đều):** Bám sát tuần tự từng buổi, không nhảy cóc, hoàn thành 100% bài tập thực hành.
"""

    visual_intuition_content = """
Sơ đồ chuyển hóa từ kết quả Diagnostic Test thành Bản đồ học tập:

```
[Diagnostic Test: 20 Câu Hỏi]
         │
         ▼
[Python Diagnostic Scorer] ───> Tính điểm theo 4 trụ cột (Toán, Code, Data, Logic)
         │
         ▼
[Profile Classification] ────> Xác định vùng trũng kiến thức (Knowledge Deficit)
         │
         ▼
[Personal AI Learning Map] ──> Lộ trình 38 buổi được cá nhân hóa trọng tâm ôn luyện
         │
         ▼
[Certificate of AI Orientation] (Mở khóa bước vào M1)
```
"""

    code_cells = [
        (
            """# HE THONG DANH GIA DIAGNOSTIC TEST VA TAO PERSONAL LEARNING MAP
import json

# 1. Ngan hang cau hoi chuan doan dau vao (Trich doan 4 cau tieu bieu tu 4 phan vung)
diagnostic_questions = [
    {
        "id": 1,
        "pillar": "Math",
        "question": "Cho hai vector u = [1, 2] va v = [3, -1]. Tich vo huong (dot product) u . v bang bao nhieu?",
        "options": ["A. 1", "B. 5", "C. -1", "D. 7"],
        "correct": "A",
        "explanation": "Dot product = 1*3 + 2*(-1) = 3 - 2 = 1."
    },
    {
        "id": 2,
        "pillar": "Python",
        "question": "Trong Python, do phuc tap thoi gian de tra cuu mot phan tu trong set (set lookup) trung binh la gi?",
        "options": ["A. O(N)", "B. O(log N)", "C. O(1)", "D. O(N^2)"],
        "correct": "C",
        "explanation": "Set trong Python duoc cai dat bang Hash Table nen lookup trung binh la O(1)."
    },
    {
        "id": 3,
        "pillar": "Data",
        "question": "Khi tap du lieu luong thu nhap co vai ty phu keo lech rat xa ve ben phai (Right-skewed), dai luong nao phan anh trung tam tot hon?",
        "options": ["A. Mean (Trung binh cong)", "B. Median (Trung vi)", "C. Standard Deviation", "D. Maximum"],
        "correct": "B",
        "explanation": "Median khong bi anh huong boi cac gia tri ngoai lai cuc doan (robust to outliers)."
    },
    {
        "id": 4,
        "pillar": "Logic",
        "question": "Neu mot thuat toan chay mat 1 giay voi N = 1,000 phan tu va mat 4 giay voi N = 2,000 phan tu, do phuc tap xap xi la gi?",
        "options": ["A. O(N)", "B. O(N^2)", "C. O(log N)", "D. O(2^N)"],
        "correct": "B",
        "explanation": "Khi N tang gap 2 lan, thoi gian tang 4 = 2^2 lan => Do phuc tap bac hai O(N^2)."
    }
]

print(f"Da nap thanh cong {len(diagnostic_questions)} cau hoi chuan doan co ban.")
""",
            """Khối code khởi tạo cấu trúc dữ liệu của bài trắc nghiệm chẩn đoán.
- **Input:** Danh sách câu hỏi kèm phân vùng (Math, Python, Data, Logic), các lựa chọn và đáp án giải thích chi tiết.
- **Output:** Bộ dữ liệu kiểm tra sẵn sàng nạp vào động cơ chấm điểm.
- **Tại sao viết như vậy:** Dữ liệu có cấu trúc giúp việc chấm điểm, phân loại và sinh khuyến nghị học tập diễn ra hoàn toàn tự động."""
        ),
        (
            """# DONG CO CHAM DIEM VA TAO BAN DO HOC TAP CA NHAN
def evaluate_diagnostic(answers):
    scores = {"Math": 0, "Python": 0, "Data": 0, "Logic": 0}
    totals = {"Math": 0, "Python": 0, "Data": 0, "Logic": 0}
    
    for q in diagnostic_questions:
        pillar = q["pillar"]
        totals[pillar] += 1
        user_ans = answers.get(q["id"], "").strip().upper()
        if user_ans == q["correct"]:
            scores[pillar] += 1
            
    print("===== KET QUA DIAGNOSTIC TEST DAU VAO =====")
    for p in scores:
        pct = (scores[p] / totals[p]) * 100 if totals[p] > 0 else 0
        print(f"* Tru cot {p:8s}: {scores[p]}/{totals[p]} ({pct:.1f}%)")
        
    # Xac dinh Profile hoc tap
    total_score = sum(scores.values())
    max_score = sum(totals.values())
    overall_pct = (total_score / max_score) * 100
    
    print(f"\\nTong diem: {total_score}/{max_score} ({overall_pct:.1f}%)")
    
    # Sinh khuyen nghi Personalized Learning Map
    print("\\n===== PERSONAL AI LEARNING MAP (38 BUOI) =====")
    if scores["Python"] < totals["Python"]:
        print("-> [CANH BAO VUNG TRUNG]: Can tap trung cao do o Module M1 (Sessions S01-S05) de luyen chac ky nang mang NumPy va DataFrame.")
    if scores["Math"] < totals["Math"]:
        print("-> [CANH BAO VUNG TRUNG]: Can doc ky phan giai thich truc giac dao ham va dot product o Module M2, M3 truoc khi viet code.")
    if scores["Data"] < totals["Data"]:
        print("-> [CANH BAO VUNG TRUNG]: Chu y ky nang phat hien outlier va tranh data leakage o Module M1_S05 va M5_S01.")
        
    print("-> KE HOACH HANH DONG: Hoan thanh day du 38/38 notebook, khong dot chay giai doan.")
    return scores

# Gia lap bai lam mau cua mot hoc vien
sample_answers = {1: "A", 2: "C", 3: "A", 4: "B"}  # Sai cau 3 (chon Mean thay vi Median)
user_scores = evaluate_diagnostic(sample_answers)
""",
            """Khối code thực thi thuật toán phân tích kết quả bài test.
- **Input:** Dictionary chứa đáp án của học viên `{question_id: selected_option}`.
- **Output:** Bảng điểm chi tiết theo 4 trụ cột kiến thức và khuyến nghị trọng tâm học tập cụ thể cho từng module trong 38 buổi.
- **Tại sao viết như vậy:** Biến một kết quả thi thô thành một bản kế hoạch hành động thực tế có tính định hướng cá nhân cao."""
        )
    ]

    practice_content = """
### Bài 1 (Easy): Thực hiện bài kiểm tra với câu trả lời của chính bạn
Hãy tạo một dictionary `my_answers = {1: '...', 2: '...', 3: '...', 4: '...'}` với câu trả lời bạn tự chọn trước khi xem giải thích, sau đó gọi hàm `evaluate_diagnostic(my_answers)` để xem kết quả thật của mình.

### Bài 2 (Easy): Mở rộng ngân hàng câu hỏi
Viết thêm ít nhất 2 câu hỏi mới vào danh sách `diagnostic_questions` (1 câu về xác suất thống kê cho phân vùng `Math`, 1 câu về kiểu dữ liệu danh sách cho phân vùng `Python`). Chạy lại bộ chấm điểm.

### Bài 3 (Medium): Viết hàm tự động xuất file kế hoạch học tập
Viết hàm `export_learning_map_to_markdown(scores, filename="MY_LEARNING_MAP.md")` để lưu trữ bản đồ học tập cá nhân ra một file markdown lưu trong thư mục làm việc của bạn.

### Bài 4 (Medium): Tính chỉ số cân bằng kiến thức (Knowledge Balance Index)
Dựa trên điểm số 4 trụ cột, tính độ lệch chuẩn giữa điểm các trụ cột. Nếu độ lệch chuẩn cao (học lệch), in ra lời cảnh báo: *"Cảnh báo: Bạn đang có sự chênh lệch lớn giữa các kỹ năng nền tảng, cần bổ sung gấp trụ cột yếu nhất!"*.

### Bài 5 (Challenge): Xây dựng bộ đếm thời gian và giới hạn làm bài (Timer Simulator)
Giả lập một hàm kiểm tra thời gian thực thi làm bài: Mỗi câu hỏi chỉ cho phép suy nghĩ tối đa 60 giây. Nếu vượt quá, câu hỏi sẽ tự động đánh dấu là `Bỏ qua (Unanswered)`.
"""

    voai_connection_content = """
Trong một cuộc thi Olympic, ban tổ chức thường cung cấp đề bài kèm theo một tập dữ liệu mẫu. Những thí sinh đạt huy chương vàng luôn là những người có khả năng tự đánh giá chính xác:
1. Bài này thuộc dạng dữ liệu nào (bảng, chuỗi thời gian, ảnh hay văn bản)?
2. Điểm mạnh kỹ thuật của mình có thể triển khai giải pháp nào nhanh nhất trong 45 phút đầu tiên?
3. Nếu gặp bế tắc về toán, mình có thể dựa vào kỹ thuật feature engineering để bù đắp hay không?

Biết người biết ta, lượng định chính xác khả năng của mình trong từng bài toán là bí quyết giữ vững tâm lý thi đấu đỉnh cao.
"""

    debugging_content = """
**Tình huống thực tế:** Một bạn học viên viết hàm tính tỷ lệ phần trăm điểm số như sau:

```python
def calculate_percentage(score, total):
    # Bug: Gay ra ZeroDivisionError neu total = 0, hoac tra ve sai kieu du lieu
    return score / total * 100

# Kiem thu khi gap tru cot khong co cau hoi nao:
print(calculate_percentage(0, 0))
```

**Nhiệm vụ của bạn:**
1. Chạy thử đoạn code trên và quan sát mã lỗi.
2. Viết lại hàm `calculate_percentage` có cơ chế kiểm tra điều kiện biên (boundary check) an toàn, trả về `0.0` nếu `total <= 0` và làm tròn kết quả đến 2 chữ số thập phân.
"""

    knowledge_check_content = """
1. Tại sao giá trị Trung vị (Median) lại là thước đo trung tâm đáng tin cậy hơn Trung bình cộng (Mean) khi dữ liệu có nhiều ngoại lai cực đoan?
   - A. Vì Median luôn luôn lớn hơn Mean trong mọi trường hợp.
   - B. Vì Median là giá trị đứng giữa sau khi sắp xếp, không bị kéo dạt bởi một vài giá trị cực lớn hoặc cực nhỏ.
   - C. Vì Median tính toán nhanh hơn Mean trên máy tính.
   - D. Vì Median chỉ áp dụng được cho dữ liệu số nguyên.

2. Phép nhân vô hướng (Dot product) giữa hai vector vuông góc (orthogonal) bằng bao nhiêu?
   - A. 1
   - B. -1
   - C. 0
   - D. Vô cùng lớn

3. Khi gặp một phân vùng kiến thức bị điểm kém trong bài Diagnostic Test, chiến lược học tập đúng đắn là gì?
   - A. Bỏ qua module đó và chỉ học phần mình đã giỏi.
   - B. Xem lại kỹ lưỡng các buổi học tương ứng trong lộ trình 38 buổi, chủ động hoàn thành bài tập thực hành mức Easy và Medium.
   - C. Chờ đến gần ngày thi mới học.
   - D. Tìm giải pháp sao chép code để đối phó.

4. Cấu trúc dữ liệu nào trong Python cho phép kiểm tra sự tồn tại của một phần tử với độ phức tạp trung bình $O(1)$?
   - A. List (Danh sách tuần tự)
   - B. Tuple (Bộ dữ liệu bất biến)
   - C. Set (Tập hợp băm)
   - D. String (Chuỗi ký tự)

5. Sau khi hoàn thành M0_S02, học viên chính thức đạt được chứng chỉ cột mốc nào?
   - A. Certificate of Machine Learning Basics
   - B. Certificate of AI Orientation
   - C. Final Project Certificate
   - D. Certificate of Deep Learning Practice
"""

    session_summary_content = """
* **What I learned:**
  - Hoàn thành bài kiểm tra năng lực đầu vào 4 trụ cột (Toán, Python, Dữ liệu, Thuật toán).
  - Tự động phân tích điểm mạnh, điểm yếu thông qua script Python.
  - Thiết lập Bản đồ học tập cá nhân (Personal AI Learning Map) định hướng cho 38 buổi học.
* **What I can now do:**
  - Tự tin định vị bản thân trên lộ trình ôn luyện Olympic AI.
  - Nhận diện chính xác các chủ đề cần dành nhiều thời gian hơn trong các module tiếp theo.
* **What I still need to practice:**
  - Chuẩn bị tinh thần và môi trường sẵn sàng bước vào Module M1: Xử lý dữ liệu thực tế cùng Python và NumPy/Pandas.
"""

    next_session_content = """
Chào mừng bạn đã chính thức mở khóa **Module M1: NÓI CHUYỆN VỚI DỮ LIỆU BẰNG PYTHON**!
Trong buổi tiếp theo (**M1_S01**), chúng ta sẽ bắt đầu trực tiếp với: **Python cho Data & Mảng đa chiều NumPy** — giải phóng sức mạnh tính toán song song, tạm biệt các vòng lặp `for` chậm chạp!
"""

    module_assessment = """
---

## 🏆 MODULE 0 ASSESSMENT & CERTIFICATION

### Phần A: Đánh giá Lý thuyết & Nhận thức (Theory & Awareness)
- Học viên đã nắm rõ ranh giới AI/ML/DL và cấu trúc 7 bước của một AI Pipeline.
- Học viên hiểu rõ tiêu chuẩn kỹ thuật và tinh thần của đấu trường Olympic AI (VOAI).

### Phần B: Sản phẩm Thực hành (Practical Deliverable)
- **Bài test đầu vào:** Đã chạy và hoàn thành bài kiểm tra chẩn đoán trên notebook.
- **Bản đồ học tập cá nhân (Personal AI Learning Map):** Đã phân tích kết quả và xuất ra lộ trình cá nhân hóa.

### Tiêu chí Đạt chuẩn (Rubric):
| Tiêu chí | Trọng số | Yêu cầu đạt |
| :--- | :--- | :--- |
| **Hoàn thành Diagnostic Test** | 40% | Trả lời đầy đủ và chạy code chấm điểm không lỗi |
| **Bản đồ học tập cá nhân** | 40% | Xác định rõ ít nhất 1 vùng trũng và kế hoạch khắc phục |
| **Thực hành & Debugging** | 20% | Hoàn thành các bài tập mẫu và sửa lỗi an toàn |

**Chứng chỉ hoàn thành:**
🏅 **Certificate of AI Orientation** — Đủ điều kiện tiến vào Module M1.
"""

    trace = {
        "items": [
            "Đánh giá năng lực đầu vào qua Diagnostic Test",
            "Xây dựng bản đồ học tập cá nhân hóa (Personal AI Learning Map)",
            "Chính thức hoàn thành Module M0 và nhận Certificate of AI Orientation"
        ],
        "deliverable": "Bài test đầu vào có chấm điểm + Bản đồ học tập cá nhân (Personal AI Learning Map)"
    }

    nb = create_notebook(
        module_id="M0",
        module_name="BẮT ĐẦU ĐÚNG HƯỚNG VỚI AI",
        session_id="S02",
        topic="Diagnostic Test Đầu Vào & Bản Đồ Học Tập Cá Nhân",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Bài test đầu vào có chấm điểm + Bản đồ học tập cá nhân",
        competition_relevance="Đo lường năng lực chuẩn xác, thiết lập kỷ luật và chiến lược ôn luyện dài hạn.",
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
    save_notebook(nb, "course/M0_AI_ORIENTATION/M0_S02.ipynb")

if __name__ == "__main__":
    build_m0_s01()
    build_m0_s02()
    print("M0 build complete!")
