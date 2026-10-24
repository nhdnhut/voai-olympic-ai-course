"""
builder/build_course_map.py
Generates course/COURSE_MAP.ipynb with 38-session complete roadmap and progression graphs.
"""

import nbformat as nbf
import os
from utils import save_notebook

def build_course_map():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "language_info": {"name": "python", "version": "3.10"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"}
    }

    # 1. Course Overview & Introduction
    overview_md = """# 🗺️ COURSE MAP: LỘ TRÌNH 38 BUỔI HUẤN LUYỆN OLYMPIC AI (VOAI)
### Khóa học Chuyên sâu Dành riêng cho Sinh viên Năm nhất Ngành Trí tuệ Nhân tạo

---

## 1. TỔNG QUAN KHÓA HỌC (COURSE OVERVIEW)
Khóa học **Ôn tập Olympic AI (VOAI)** được thiết kế với phương pháp luận **Chuẩn mực - Khoa học - Thực chiến**. Chương trình bám sát 100% **Syllabus Gốc (Source of Truth M0 → M7)**, trải qua đúng **38 buổi học**, tương ứng với **38 Jupyter Notebooks độc lập**.

### Mục tiêu Tối thượng:
1. **Nền tảng vững chắc:** Chuyển hóa tư duy từ "người chép code" thành "Kỹ sư AI hiểu bản chất toán học và luồng dữ liệu".
2. **Kỹ năng thực chiến:** Làm chủ toàn bộ chuỗi mắt xích AI Pipeline (Data Wrangling, Machine Learning, Deep Learning PyTorch, Computer Vision, Natural Language Processing).
3. **Chiến thuật thi đấu:** Làm chủ nghệ thuật đọc đề nhanh trong 15 phút, thiết lập Validation chống rò rỉ dữ liệu (Data Leakage), Feature Engineering kéo điểm và Ensembling đỉnh cao.
4. **Sản phẩm Portfolio:** Sở hữu một Dự án AI hoàn chỉnh có mã nguồn GitHub chuẩn công nghiệp, ứng dụng Demo trực tiếp và Báo cáo khoa học.

---

## 2. OFFICIAL SYLLABUS — SOURCE OF TRUTH (M0 → M7)
| Module | Tên Module | Số buổi | Sản phẩm Đầu ra (Deliverable) | Chứng chỉ Cột mốc |
| :---: | :--- | :---: | :--- | :--- |
| **M0** | Bắt đầu đúng hướng với AI | 2 | Bản đồ học tập cá nhân & Bài test đầu vào | Certificate of AI Orientation |
| **M1** | Nói chuyện với dữ liệu bằng Python | 5 | Phân tích dữ liệu đầu tiên (First EDA Report) | Certificate of Python and Data Basics |
| **M2** | Dạy máy học từ dữ liệu | 5 | Model ML đầu tiên & Báo cáo kết quả | Certificate of Machine Learning Basics |
| **M3** | Xây model AI sâu hơn bằng PyTorch | 5 | Notebook Training Loop & Model Checkpoint (.pt) | Certificate of Deep Learning Practice |
| **M4** | AI nhìn và hiểu dữ liệu (CV & NLP) | 10 | Mini Project thực tế ảnh hoặc văn bản | Certificate of CV/NLP Applied AI |
| **M5** | Chiến thuật thi Olympic AI | 3 | Bài thi thử có chấm điểm & Submission Checklist | Certificate of AI Competition Practice |
| **M6** | Dùng LLM như trợ lý học và thi | 3 | Personal Prompt Kit & LLM Debugging Workflow | Certificate of AI-assisted Problem Solving |
| **M7** | Dự án AI thật để đưa vào Portfolio | 5 | GitHub Repo + Demo App + Báo cáo + Slide | **Final Project Certificate** |
| **TỔNG** | **8 MODULES TOÀN KHÓA** | **38** | **ĐỦ 38 NOTEBOOKS THỰC CHIẾN ĐỘC LẬP** | **TỐT NGHIỆP OLYMPIC AI** |
"""
    nb.cells.append(nbf.v4.new_markdown_cell(overview_md))

    # 2. 38-Session Master Roadmap Table
    sessions_data = [
        # M0
        ("01", "M0", "AI là gì & Olympic AI cần gì?", "Phân định AI/ML/DL, AI Pipeline, Phổ năng lực thi đấu", "Bản đồ năng lực ban đầu (Radar Chart)", "Định vị tư duy, hiểu tiêu chuẩn đánh giá Olympic"),
        ("02", "M0", "Diagnostic Test Đầu Vào & Lộ Trình Cá Nhân", "Diagnostic Scorer 4 trụ cột, Phân loại Profile học tập", "Bài test đầu vào + Personal Learning Map", "Lượng định xuất phát điểm, lập kỷ luật 38 buổi"),
        # M1
        ("03", "M1", "Python cho Data & Vectorization cùng NumPy", "NumPy ndarray, C contiguous memory, SIMD, Broadcasting", "Notebook tối ưu hóa tính toán số học C-speed", "Xử lý mảng dữ liệu cực nhanh, không nghẽn loop"),
        ("04", "M1", "Pandas Core: DataFrame, Series & Indexing", "Đọc CSV, info(), describe(), .loc vs .iloc, Boolean mask", "Notebook truy vấn lọc dữ liệu phức tạp", "Khám phá nhanh đề thi, trích lọc tập con chuẩn xác"),
        ("05", "M1", "Làm Sạch Dữ Liệu: NaNs, Dtypes & Duplicates", "Missing Imputation (Mean/Median/Mode), astype, drop_duplicates", "Mã nguồn Data Cleaning Pipeline chống rò rỉ", "Ngăn chặn lỗi sập code do thiếu dữ liệu khi thi"),
        ("06", "M1", "Thống Kê Mô Tả, GroupBy & Data Visualization", "IQR, Skewness, GroupBy Split-Apply-Combine, Boxplot, Heatmap", "Notebook trực quan hóa đa biến và tương quan", "Phát hiện mô thức ẩn, thiết kế đặc trưng thông minh"),
        ("07", "M1", "Data Cleaning Tổng Hợp & First Data Analysis", "Quy tắc biên IQR, Z-score, Capping, Data Storytelling", "Phân tích dữ liệu đầu tiên (First EDA Report)", "Hoàn thiện kỹ năng EDA thực tế trước khi train model"),
        # M2
        ("08", "M2", "Bản Chất Học Máy: Feature, Target & Split", "Supervised Learning, y = f(X), Generalization, Stratified Split", "Mã nguồn phân tách tập dữ liệu chuẩn hóa", "Nền móng thi đấu, ngăn chặn rò rỉ dữ liệu Train/Test"),
        ("09", "M2", "Baseline Models & Hồi Quy Tuyến Tính (Linear)", "Dummy Regressor, MSE Loss, Gradient Descent, fit/predict", "Notebook xây Baseline và Hồi quy Tuyến tính", "Xây mốc quy chiếu tối thiểu (First Working Submission)"),
        ("10", "M2", "Phân Loại Nhị Phân & Cây Quyết Định (Trees)", "Sigmoid, Log-Loss, Gini Impurity, max_depth, Random Forest", "Notebook huấn luyện Decision Tree & Feature Importance", "Mô hình nền tảng cho bảng dữ liệu, tìm feature quan trọng"),
        ("11", "M2", "Đọc & Hiểu Toàn Diện Metric Đánh Giá Mô Hình", "Accuracy Paradox, Confusion Matrix, Precision, Recall, F1, RMSE", "Notebook đánh giá metric đa chiều và phân tích tradeoff", "Căn chỉnh tối ưu đúng theo tiêu chí chấm của Ban giám khảo"),
        ("12", "M2", "Huấn Luyện Model Đầu Tiên & Báo Cáo Thi Đấu", "End-to-End ML Pipeline, StandardScaler, Benchmark Arena", "Model ML đầu tiên & Báo cáo kết quả chuẩn thi đấu", "Hoàn thiện năng lực huấn luyện và báo cáo so sánh"),
        # M3
        ("13", "M3", "PyTorch Tensors, Autograd & Đồ Thị Tính Toán", "Tensors, GPU transfer, requires_grad, .backward(), Autograd", "Notebook tối ưu hóa tham số bằng Tensor thuần", "Làm chủ tầng vi phân PyTorch, sẵn sàng viết Custom Loss"),
        ("14", "M3", "PyTorch Data Pipeline: Dataset & DataLoader", "Custom Dataset (__init__, __len__, __getitem__), Batching", "Notebook Data Pipeline nạp Mini-batch an toàn", "Loại bỏ triệt để nghẽn cổ chai nạp dữ liệu vào GPU"),
        ("15", "M3", "Thiết Kế Kiến Trúc Mạng Nơ-ron với nn.Module", "MLP, nn.Linear, ReLU, Sigmoid, parameters(), shape tracking", "Notebook thiết kế kiến trúc mạng nơ-ron MLP", "Khả năng xây dựng kiến trúc mạng tùy biến cho đề thi"),
        ("16", "M3", "Tự Viết Training Loop Chuẩn Mực Bằng PyTorch", "5 bước training step, model.train() vs eval(), no_grad()", "Notebook Training Loop & Validation Loop hoàn chỉnh", "Kiểm soát 100% luồng thực thi huấn luyện mô hình sâu"),
        ("17", "M3", "Chống Overfitting, Learning Curves & Checkpoint", "Learning curves, Dropout, Weight Decay, Save/Load state_dict", "Notebook PyTorch có Training Loop + Checkpoint (.pt)", "Bảo tồn mô hình tối ưu, ngăn ngừa rớt hạng do học vẹt"),
        # M4
        ("18", "M4", "Biểu Diễn Ảnh Số: Pixel, RGB & Image Tensors", "Ma trận pixel [0, 255], [C, H, W] PyTorch, Normalization", "Notebook xử lý và trực quan hóa Tensor ảnh số", "Nền tảng toàn bộ nhánh Computer Vision trong Olympic AI"),
        ("19", "M4", "Tiền Xử Lý Ảnh & Data Augmentation", "torchvision.transforms, RandomFlip, Rotation, Jitter, ImageNet stats", "Notebook Pipeline Data Augmentation chuyên nghiệp", "Gia tăng độ chính xác tổng quát hóa khi ít dữ liệu ảnh"),
        ("20", "M4", "Trực Giác Mạng Tích Chập (CNN): Conv & Pooling", "Weight Sharing, Conv2d, Kernel, Stride, Padding, MaxPool2d", "Notebook cài đặt tích chập và theo dõi Feature Maps", "Làm chủ khối kiến trúc cốt lõi của mọi mô hình thị giác"),
        ("21", "M4", "End-to-End Image Classification với PyTorch CNN", "Conv-BN-ReLU-Pool blocks, Flatten, Classifier Head, Training", "Notebook huấn luyện CNN phân loại ảnh hoàn chỉnh", "Mô hình thị giác tự viết bắt buộc phải thuần thục"),
        ("22", "M4", "Transfer Learning CV: Pretrained Models & Errors", "Freezing backbone, ResNet-18, Classifier replacement, Error analysis", "Notebook ứng dụng Transfer Learning kèm phân tích lỗi", "Vũ khí tối thượng giành huy chương phần Computer Vision"),
        ("23", "M4", "Biểu Diễn Văn Bản: Tokenization & Từ Điển", "Regex cleaning, Lowercase, Word/Subword Tokenization, Vocabulary", "Mã nguồn Vocabulary Indexer chuẩn xử lý OOV an toàn", "Nền móng xử lý văn bản cho mọi bài toán NLP thi đấu"),
        ("24", "M4", "Trích Xuất Đặc Trưng: Bag-of-Words & TF-IDF", "Túi từ, Công thức toán TF-IDF, Sparse Matrix, TfidfVectorizer", "Notebook trích xuất TF-IDF & phân loại văn bản Baseline", "Mô hình Baseline số 1 cho bài toán văn bản thi đấu"),
        ("25", "M4", "Word Embeddings & nn.Embedding Trong PyTorch", "Dense Vectors, Không gian ngữ nghĩa, Cosine Similarity, Lookup table", "Notebook thực hành Word Embedding và đo khoảng cách Cosine", "Nền tảng kiến trúc sâu của mọi mô hình ngôn ngữ"),
        ("26", "M4", "Text Classification Pipeline với PyTorch", "Variable Sequence Length, pad_sequence, Custom collate_fn, DAN", "Notebook Deep NLP phân loại văn bản bằng PyTorch", "Hoàn thiện năng lực xử lý văn bản cho đề thi Olympic NLP"),
        ("27", "M4", "Mini Project Thực Tế: So Sánh & Báo Cáo CV/NLP", "End-to-End Capstone, Comparative Benchmark, Error Analysis", "Mini project ảnh hoặc văn bản (Applied AI Report)", "Tổng duyệt năng lực ứng dụng trước khi luyện thi Olympic"),
        # M5
        ("28", "M5", "Chiến Thuật 15 Phút Đọc Đề & Chống Data Leakage", "15-minute Playbook, Target analysis, Loss-to-Metric, GroupKFold", "Notebook kiểm toán đề thi và sơ đồ Validation an toàn", "Bảo hiểm số 1 chống lại hiện tượng tụt hạng (Shakeup)"),
        ("29", "M5", "Feature Engineering Thực Chiến & K-Fold Blend", "Ratios, Groupby Aggregations, Frequency Encoding, OOF, Blending", "Notebook K-Fold Ensembling Harness tối ưu điểm số", "Động cơ kéo điểm cốt lõi giúp bứt phá bảng tổng sắp"),
        ("30", "M5", "Mock Olympic AI Competition & Submission Check", "Giả lập phòng thi 3 giờ, 4 giai đoạn timeboxing, Submission check", "Bài thi thử có chấm điểm + Checklist nộp bài", "Đợt diễn tập tổng lực bắt buộc trước kỳ thi chính thức"),
        # M6
        ("31", "M6", "Kỹ Thuật Đặt Prompt Chuyên Sâu Cho Lập Trình AI", "Khung RTCC (Role, Task, Context, Constraint), Socratic prompt", "Notebook Framework RTCC Prompting chuyên nghiệp", "Tối ưu hóa hiệu suất nghiên cứu có trợ lực AI an toàn"),
        ("32", "M6", "Quy Trình Debug Code Bằng LLM & Soát Ảo Giác", "Structured Bug Packet (MRE + Traceback), Edge Cases, Cross-check", "Notebook quy trình Debugging bằng LLM và test biên", "Rút ngắn thời gian xử lý sự cố phòng thi xuống < 1 phút"),
        ("33", "M6", "Xây Dựng Personal Prompt Kit & SOP Gỡ Lỗi", "Personal Prompt Kit 5 giai đoạn, LLM Debugging SOP, Đạo đức thi", "Personal Prompt Kit (JSON) + LLM Debugging SOP", "Vũ khí gia tốc tư duy và bảo hiểm gỡ lỗi thần tốc"),
        # M7
        ("34", "M7", "Project Scoping, Problem Framing & Data Audit", "Định vị bài toán AI đời thực, Cấu trúc GitHub repo, Schema validator", "Cấu trúc Repository hoàn chỉnh + Module kiểm toán", "Tiêu chuẩn quản lý mã nguồn và dữ liệu chuẩn công nghiệp"),
        ("35", "M7", "EDA Chuyên Sâu, Data Pipeline & Baseline Sạch", "In-depth EDA, ColumnTransformer, Modular Pipeline, Clean Baseline", "Notebook Preprocessing Pipeline + Clean Baseline Benchmark", "Xây dựng Pipeline dữ liệu sạch cấp độ sản phẩm"),
        ("36", "M7", "Thử Nghiệm Nâng Cao & Ablation Study", "Ablation Study bóc tách đóng góp, Tinh chỉnh siêu tham số", "Notebook thực nghiệm Ablation Study và bảng theo dõi", "Chứng minh giá trị khoa học định lượng trước hội đồng"),
        ("37", "M7", "Đánh Giá Chi Tiết, Phân Tích Lỗi & Live Demo", "Slice-based evaluation, High-confidence errors, Interactive Demo", "Notebook phân tích lỗi chi tiết + Interactive AI Demo", "Biến thuật toán thành sản phẩm trực quan trình diễn"),
        ("38", "M7", "Hoàn Thiện Portfolio: GitHub, Báo Cáo & Slide", "Polished README, Technical Report, 5-Minute Pitch, Package audit", "Bộ Tứ Sản Phẩm: GitHub + README + Demo + Report", "Đích đến vinh quang: Portfolio tuyển dụng và thi đấu")
    ]

    table_header = """---

## 3. BẢNG CHI TIẾT 38 BUỔI HỌC (38-SESSION MASTER ROADMAP)

| Buổi | Module | Tên Buổi Học (Topic) | Kỹ Năng Trọng Tâm (Skills) | Sản Phẩm Đầu Ra (Output) | Giá Trị Thi Đấu (VOAI Relevance) |
| :---: | :---: | :--- | :--- | :--- | :--- |
"""
    table_rows = []
    for s in sessions_data:
        table_rows.append(f"| **S{s[0]}** | `{s[1]}` | **{s[2]}** | {s[3]} | *{s[4]}* | {s[5]} |")

    table_md = table_header + "\n".join(table_rows) + f"\n\n*Tổng số buổi: {len(sessions_data)}/38 sessions (100% Coverage).* \n"
    nb.cells.append(nbf.v4.new_markdown_cell(table_md))

    # 3. Progression Graphs & Diagrams
    progression_md = """---

## 4. BẢN ĐỒ TIẾN TRÌNH NĂNG LỰC & ĐIỀU KIỆN TIÊN QUYẾT (PREREQUISITE GRAPH)

```
[M0: AI Orientation] (Buổi 01-02) ──> Diagnostic Test & Bản đồ học tập
         │
         ▼
[M1: Python & Data Wrangling] (Buổi 03-07) ──> NumPy Vectorization, Pandas, EDA, First Data Report
         │
         ▼
[M2: Classical Machine Learning] (Buổi 08-12) ──> Supervised, Linear/Tree, Metrics, First ML Model
         │
         ▼
[M3: PyTorch Deep Learning] (Buổi 13-17) ──> Tensors, Autograd, DataLoader, Training Loop, Checkpoint
         │
         ▼
[M4: Applied AI (CV & NLP)] (Buổi 18-27) ──> CNN, Transfer Learning, Tokenization, TF-IDF, Mini Project
         │
         ▼
[M5: Olympic Strategy] (Buổi 28-30) ──> 15p Đọc đề, Chống Leakage, Feature K-Fold, Mock Competition
         │
         ▼
[M6: LLM-Assisted Problem Solving] (Buổi 31-33) ──> RTCC Prompts, Synthetic Edge Cases, Prompt Kit SOP
         │
         ▼
[M7: Final Portfolio Project] (Buổi 34-38) ──> End-to-End Capstone, GitHub Repo, Demo App, Graduation!
```

---

## 5. BẢN ĐỒ SẢN PHẨM ĐẦU RA (OUTPUT PROGRESSION)
Học viên tích lũy liên tục các sản phẩm hữu hình qua từng chặng đường:
1. **M0:** Bản đồ năng lực cá nhân & Kết quả Diagnostic Test đầu vào.
2. **M1:** Báo cáo phân tích dữ liệu đầu tiên (First EDA Report).
3. **M2:** Mô hình Học máy đầu tiên & Báo cáo kết quả so sánh Baseline.
4. **M3:** Notebook PyTorch có Training Loop tự viết & File Checkpoint trọng số (`best_model.pt`).
5. **M4:** Báo cáo Mini Project thực tế chuyên sâu (Ảnh hoặc Văn bản).
6. **M5:** Bài thi thử có chấm điểm tự động & Bảng Checklist nộp bài an toàn.
7. **M6:** Bộ công cụ Personal Prompt Kit (JSON) & Quy chuẩn vận hành gỡ lỗi (SOP).
8. **M7:** **Bộ Tứ Sản Phẩm Tốt Nghiệp:** GitHub Repository chuẩn công nghiệp + File Demo tương tác (`app.py`) + Báo cáo Kỹ thuật toàn diện + Slide thuyết trình bảo vệ 5 phút!
"""
    nb.cells.append(nbf.v4.new_markdown_cell(progression_md))

    # Python code cell to inspect curriculum programmatically
    code_text = """# KIEM TRA VA TRUY XUAT THONG TIN KHOA HOC 38 BUOI
import pandas as pd

# Nap danh sach 38 buoi vao DataFrame de tra cuu nhanh
curriculum_df = pd.DataFrame([
    {"Session": f"S{i+1:02d}", "Module": s[1], "Topic": s[2], "Deliverable": s[4]}
    for i, s in enumerate(sessions_data)
])

print(f"Tong so buoi trong Course Map: {len(curriculum_df)} buoi")
print("\\nPhan bo so luong buoi theo Module:")
print(curriculum_df['Module'].value_counts().sort_index())
"""
    # Replace sessions_data inside code cell
    code_text = f"sessions_data = {repr(sessions_data)}\n" + code_text
    nb.cells.append(nbf.v4.new_code_cell(code_text))

    summary_md = """---
## 🎯 Hướng Dẫn Sử Dụng Course Map
- Hãy mở từng thư mục `M0` tới `M7` theo đúng thứ tự tiến trình.
- Không nhảy cóc các buổi học vì mỗi buổi đều là điều kiện tiên quyết trực tiếp cho buổi tiếp theo.
- Sau khi hoàn thành mỗi buổi, đánh dấu tích vào sản phẩm đầu ra tương ứng.
"""
    nb.cells.append(nbf.v4.new_markdown_cell(summary_md))

    save_notebook(nb, "course/COURSE_MAP.ipynb")
    print("[OK] Created course/COURSE_MAP.ipynb with 38 rows and full progression.")

if __name__ == "__main__":
    build_course_map()
