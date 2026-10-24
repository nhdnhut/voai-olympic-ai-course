# VOAI Olympic AI Comprehensive Training Course
### Khóa Học Ôn Tập Toàn Diện Olympic AI & Applied Machine Learning Dành Cho Sinh Viên Năm Nhất

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626.svg)](https://jupyter.org/)
[![Syllabus Locked](https://img.shields.io/badge/Syllabus-STRICTLY%20LOCKED-success.svg)](#official-syllabus)

---

## 1. Tổng Quan Khóa Học (Course Overview)
Khóa học **Ôn tập Olympic AI (VOAI)** là chương trình đào tạo chuyên sâu kéo dài **38 buổi học**, được đóng gói dưới dạng **38 Jupyter Notebooks (.ipynb) thực chiến độc lập**.

Chương trình được thiết kế chuẩn mực theo triết lý:
- **Không học lan man:** Bám sát 100% từng từ khóa trong **Syllabus Gốc (M0 → M7)**.
- **Hiểu bản chất toán học:** Không sao chép mã nguồn mù quáng, làm chủ luồng tính toán từ vi tích phân đến tensor GPU.
- **Thực hành thực tế:** 100% notebook đều có code chạy được độc lập, dữ liệu tự tái lập, không phụ thuộc kết nối bên ngoài.
- **Thi đấu đỉnh cao:** Tích hợp chiến thuật phòng thi Olympic AI (đọc đề 15 phút, chống rò rỉ dữ liệu, K-Fold Ensembling, vệ sinh submission).

---

## 2. Đối Tượng Người Học (Who is this for?)
- **Sinh viên năm nhất ngành Trí tuệ Nhân tạo (AI), Khoa học Dữ liệu (Data Science), Khoa học Máy tính (CS)** muốn xây dựng nền tảng vững chắc và bứt phá ngay từ năm đầu đại học.
- **Đội tuyển Ôn thi Olympic Tin học / Olympic AI (VOAI)** cần một giáo trình thực chiến, kỷ luật và bám sát tiêu chuẩn thi đấu quốc tế.
- **Các bạn trẻ muốn xây dựng Portfolio AI thật** với mã nguồn GitHub chuẩn mực, ứng dụng Demo trực tiếp và báo cáo kỹ thuật khoa học.

---

## 3. Điều Kiện Tiên Quyết (Prerequisites)
- Tư duy logic toán học cơ bản ở bậc phổ thông (Đại số, Giải tích đạo hàm, Xác suất thống kê cơ bản).
- Kỹ năng cài đặt và chạy môi trường Python / Jupyter Notebook cơ bản.
- Tinh thần kỷ luật: Cam kết hoàn thành tuần tự 38 buổi học, không nhảy cóc giai đoạn.

---

## 4. Môi Trường Khuyến Nghị & Cài Đặt (Environment & Setup)

### Cấu hình Tối thiểu:
- **Hệ điều hành:** Windows 10/11, macOS, hoặc Linux (Ubuntu 20.04+).
- **Python:** Phiên bản 3.10 trở lên.
- **Phần cứng:** Chạy tốt trên laptop CPU thông thường (RAM 8GB+). Không bắt buộc GPU cao cấp.

### Cài đặt trên máy cá nhân (Local Setup):
```bash
# 1. Clone repository hoặc tải thư mục mã nguồn
git clone https://github.com/nhdnhut/voai-olympic-ai-course.git
cd voai-olympic-ai-course

# 2. Tạo môi trường ảo Python
python -m venv venv
# Kích hoạt trên Windows:
venv\Scripts\activate
# Kích hoạt trên macOS/Linux:
source venv/bin/activate

# 3. Cài đặt toàn bộ thư viện cần thiết
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision nbformat jupyterlab

# 4. Khởi động Jupyter Lab / Notebook
jupyter lab
```

### Chạy trên Google Colab (Google Colab Ready):
- Toàn bộ 38 notebooks được thiết kế độc lập (Standalone). Bạn chỉ cần tải file `.ipynb` lên Google Colab là có thể thực thi ngay lập tức mà không cần cài đặt thêm dữ liệu phụ trợ.

---

## ️ 5. Lộ Trình 38 Buổi & Official Syllabus (Source of Truth)

Khóa học gồm đúng **38 buổi** phân bố trên **8 Modules (M0 → M7)**:

```
course/
├── README.md
├── COURSE_MAP.ipynb
├── M0_AI_ORIENTATION/     (2 Buổi: S01 - S02)
├── M1_PYTHON_DATA/        (5 Buổi: S01 - S05)
├── M2_MACHINE_LEARNING/   (5 Buổi: S01 - S05)
├── M3_PYTORCH/            (5 Buổi: S01 - S05)
├── M4_APPLIED_AI/         (10 Buổi: S01 - S10: 5 CV + 5 NLP)
├── M5_OLYMPIC_STRATEGY/   (3 Buổi: S01 - S03)
├── M6_LLM_ASSISTED/       (3 Buổi: S01 - S03)
└── M7_FINAL_PROJECT/      (5 Buổi: S01 - S05)
```

### Bảng Phân Bổ 38 Buổi Chi Tiết:
| STT | Module | Buổi | Tên Buổi Học | Sản Phẩm Đầu Ra |
| :---: | :---: | :---: | :--- | :--- |
| 1 | M0 | `M0_S01` | AI là gì & Olympic AI cần gì? | Bản đồ năng lực ban đầu (Radar Chart) |
| 2 | M0 | `M0_S02` | Diagnostic Test Đầu Vào & Lộ Trình Cá Nhân | Bài test đầu vào + Personal Learning Map |
| 3 | M1 | `M1_S01` | Python cho Data & Vectorization cùng NumPy | Mã nguồn tối ưu hóa tính toán số học C-speed |
| 4 | M1 | `M1_S02` | Pandas Core: DataFrame, Series & Indexing | Notebook truy vấn lọc dữ liệu phức tạp |
| 5 | M1 | `M1_S03` | Làm Sạch Dữ Liệu: NaNs, Dtypes & Duplicates | Data Cleaning Pipeline chuẩn chống rò rỉ |
| 6 | M1 | `M1_S04` | Thống Kê Mô Tả, GroupBy & Data Visualization | Bộ biểu đồ trực quan hóa đa biến và tương quan |
| 7 | M1 | `M1_S05` | Data Cleaning Tổng Hợp & First Data Analysis | **Phân tích dữ liệu đầu tiên (First EDA Report)** |
| 8 | M2 | `M2_S01` | Bản Chất Học Máy: Feature, Target & Split | Mã nguồn phân tách tập dữ liệu chuẩn hóa |
| 9 | M2 | `M2_S02` | Baseline Models & Hồi Quy Tuyến Tính (Linear) | Notebook xây Baseline và Hồi quy Tuyến tính |
| 10 | M2 | `M2_S03` | Phân Loại Nhị Phân & Cây Quyết Định (Trees) | Huấn luyện Decision Tree & Feature Importance |
| 11 | M2 | `M2_S04` | Đọc & Hiểu Toàn Diện Metric Đánh Giá Mô Hình | Đánh giá metric đa chiều & Confusion Matrix |
| 12 | M2 | `M2_S05` | Huấn Luyện Model Đầu Tiên & Báo Cáo Thi Đấu | **Model ML đầu tiên & Báo cáo kết quả chuẩn thi đấu** |
| 13 | M3 | `M3_S01` | PyTorch Tensors, Autograd & Đồ Thị Tính Toán | Mã nguồn tối ưu hóa tham số bằng Tensor thuần |
| 14 | M3 | `M3_S02` | PyTorch Data Pipeline: Dataset & DataLoader | Data Pipeline nạp Mini-batch an toàn bộ nhớ |
| 15 | M3 | `M3_S03` | Thiết Kế Kiến Trúc Mạng Nơ-ron với nn.Module | Thiết kế kiến trúc mạng nơ-ron MLP chuẩn mực |
| 16 | M3 | `M3_S04` | Tự Viết Training Loop Chuẩn Mực Bằng PyTorch | Training Loop & Validation Loop hoàn chỉnh |
| 17 | M3 | `M3_S05` | Chống Overfitting, Learning Curves & Checkpoint | **Notebook PyTorch có Training Loop + Checkpoint (.pt)** |
| 18 | M4 | `M4_S01` | Biểu Diễn Ảnh Số: Pixel, RGB & Image Tensors | Xử lý và trực quan hóa Tensor ảnh số chuẩn PyTorch |
| 19 | M4 | `M4_S02` | Tiền Xử Lý Ảnh & Data Augmentation | Pipeline Data Augmentation chuyên nghiệp |
| 20 | M4 | `M4_S03` | Trực Giác Mạng Tích Chập (CNN): Conv & Pooling | Cài đặt tích chập và theo dõi Feature Maps |
| 21 | M4 | `M4_S04` | End-to-End Image Classification với PyTorch CNN | Huấn luyện CNN phân loại ảnh hoàn chỉnh |
| 22 | M4 | `M4_S05` | Transfer Learning CV: Pretrained Models & Errors | Ứng dụng Transfer Learning kèm phân tích lỗi |
| 23 | M4 | `M4_S06` | Biểu Diễn Văn Bản: Tokenization & Từ Điển | Vocabulary Indexer chuẩn xử lý OOV an toàn |
| 24 | M4 | `M4_S07` | Trích Xuất Đặc Trưng: Bag-of-Words & TF-IDF | Trích xuất TF-IDF & phân loại văn bản Baseline |
| 25 | M4 | `M4_S08` | Word Embeddings & nn.Embedding Trong PyTorch | Tầng Word Embedding & đo khoảng cách Cosine |
| 26 | M4 | `M4_S09` | Text Classification Pipeline với PyTorch | Deep NLP phân loại văn bản bằng PyTorch thuần |
| 27 | M4 | `M4_S10` | Mini Project Thực Tế: So Sánh & Báo Cáo CV/NLP | **Mini project ảnh hoặc văn bản (Applied AI Report)** |
| 28 | M5 | `M5_S01` | Chiến Thuật 15 Phút Đọc Đề & Chống Data Leakage | Kiểm toán đề thi và sơ đồ Validation an toàn |
| 29 | M5 | `M5_S02` | Feature Engineering Thực Chiến & K-Fold Blend | K-Fold Ensembling Harness tối ưu điểm số |
| 30 | M5 | `M5_S03` | Mock Olympic AI Competition & Submission Check | **Bài thi thử có chấm điểm + Checklist nộp bài** |
| 31 | M6 | `M6_S01` | Kỹ Thuật Đặt Prompt Chuyên Sâu Cho Lập Trình AI | Framework RTCC Prompting chuyên nghiệp cho AI |
| 32 | M6 | `M6_S02` | Quy Trình Debug Code Bằng LLM & Soát Ảo Giác | Quy trình Debugging bằng LLM và test biên |
| 33 | M6 | `M6_S03` | Xây Dựng Personal Prompt Kit & SOP Gỡ Lỗi | **Personal Prompt Kit (JSON) + LLM Debugging SOP** |
| 34 | M7 | `M7_S01` | Project Scoping, Problem Framing & Data Audit | Cấu trúc Repository hoàn chỉnh + Schema Validator |
| 35 | M7 | `M7_S02` | EDA Chuyên Sâu, Data Pipeline & Baseline Sạch | Preprocessing Pipeline hoàn chỉnh + Clean Baseline |
| 36 | M7 | `M7_S03` | Thử Nghiệm Nâng Cao & Ablation Study | Thực nghiệm Ablation Study và bảng theo dõi mô hình |
| 37 | M7 | `M7_S04` | Đánh Giá Chi Tiết, Phân Tích Lỗi & Live Demo | Phân tích lỗi chi tiết + Ứng dụng Interactive AI Demo |
| 38 | M7 | `M7_S05` | Hoàn Thiện Portfolio: GitHub, Báo Cáo & Slide | **Bộ Tứ Sản Phẩm: GitHub + README + Demo + Report** |

---

## ️ 6. Cấu Trúc Sư Phạm Trong Từng Notebook (Pedagogical Design)
Mỗi notebook trong 38 buổi đều được cấu trúc thống nhất và nghiêm ngặt qua 12 phần:
1. **0. Session Header & Metadata:** Mục tiêu, Điều kiện tiên quyết, Thời lượng, Sản phẩm đầu ra, Mức độ liên quan thi đấu.
2. **1. WHY?:** Tại sao kiến thức này tồn tại? Nằm ở đâu trong AI Pipeline? Liên quan VOAI thế nào?
3. **2. CONCEPT:** Nền tảng lý thuyết và trực giác toán học sâu sắc, không giả định người học đã biết.
4. **3. VISUAL INTUITION:** Sơ đồ ASCII, biểu đồ minh họa trực quan.
5. **4. CODE IMPLEMENTATION:** Mã nguồn Python thực tế, sạch sẽ, không dùng pseudocode.
6. **5. CODE WALKTHROUGH:** Phân tích chi tiết từng khối code (làm gì, input/output, tại sao viết như vậy, lỗi thường gặp).
7. **6. PRACTICE:** Ít nhất 5 bài tập thực hành (2 Easy, 2 Medium, 1 Challenge).
8. **7. VOAI CONNECTION:** Chiến lược phòng thi Olympic AI thực chiến.
9. **8. DEBUGGING CHALLENGE:** Tình huống lỗi thực tế và cách khoanh vùng, khắc phục.
10. **9. KNOWLEDGE CHECK:** 5 câu hỏi trắc nghiệm kiểm tra độ hiểu bản chất.
11. **10. SESSION SUMMARY:** Tổng kết (What I learned, What I can now do, What I still need to practice).
12. **11. NEXT SESSION & SYLLABUS TRACEABILITY:** Đối chiếu trực tiếp với Syllabus Lock.

---

## 7. Hệ Thống Đánh Giá & Chứng Chỉ Cột Mốc (Certificates)
Sau khi hoàn thành bài kiểm tra cuối mỗi Module, học viên được cấp chứng chỉ cột mốc:
- **M0:** *Certificate of AI Orientation*
- **M1:** *Certificate of Python and Data Basics*
- **M2:** *Certificate of Machine Learning Basics*
- **M3:** *Certificate of Deep Learning Practice*
- **M4:** *Certificate of CV/NLP Applied AI*
- **M5:** *Certificate of AI Competition Practice*
- **M6:** *Certificate of AI-assisted Problem Solving*
- **M7:** **FINAL PROJECT CERTIFICATE — TỐT NGHIỆP TOÀN KHÓA OLYMPIC AI**

---

## 8. Giấy Phép & Bản Quyền (License)
Toàn bộ khóa học được phát hành vì mục đích giáo dục cho sinh viên và các vận động viên Olympic AI. Mọi nội dung tuân thủ nghiêm ngặt chuẩn mực liêm chính học thuật.
"""

    with open("course/README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("[OK] Created course/README.md and root README.md.")

if __name__ == "__main__":
    pass
