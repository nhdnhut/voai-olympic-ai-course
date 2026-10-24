"""
builder/build_m7.py
Generates M7_S01.ipynb to M7_S05.ipynb for Module 7 (Final Portfolio Project).
"""

from utils import create_notebook, save_notebook

def build_m7_s01():
    objectives = [
        "Làm chủ quy trình Khởi tạo Dự án AI thực tế (Problem Scoping & AI Framing): Chuyển hóa một vấn đề đời thực thành bài toán tối ưu hóa toán học.",
        "Thiết lập cấu trúc Thư mục Dự án chuẩn công nghiệp (Production-grade GitHub Repository Structure).",
        "Tiến hành kiểm toán chất lượng dữ liệu (Data Quality & Ethical Audit): Nguồn gốc, bản quyền dữ liệu, tính riêng tư và rủi ro thiên vị (Bias).",
        "Tự tay lập trình Module Xác thực Lược đồ Dữ liệu (Automated Schema Validator) để bảo vệ pipeline khỏi dữ liệu rác."
    ]
    prerequisites = [
        "Hoàn thành toàn bộ các Module từ M0 đến M6.",
        "Kỹ năng tổ chức thư mục và quản lý dự án phần mềm cơ bản."
    ]
    why_content = """
Trong môi trường học thuật cấp cao hoặc khi phỏng vấn vào các phòng lab AI hàng đầu, người ta không bao giờ đánh giá bạn qua những bài tập rời rạc vài chục dòng code. Câu hỏi quan trọng nhất luôn là: **"Bạn đã từng hoàn thiện một Dự án AI hoàn chỉnh từ A đến Z (End-to-End Real-world AI Project) có thể triển khai thực tế và đưa vào Portfolio chưa?"**.

Module M7 là chặng đua marathon cuối cùng kéo dài 5 buổi, nơi bạn tích hợp TOÀN BỘ tinh hoa của 6 module trước: Từ tư duy toán học, kỹ năng xử lý dữ liệu NumPy/Pandas, mô hình Machine Learning, mạng nơ-ron sâu PyTorch, chiến thuật thi đấu Olympic, đến kỹ năng sử dụng LLM phản biện. Mọi thứ bắt đầu từ buổi hôm nay: Định hình đề tài, thiết lập cấu trúc mã nguồn chuẩn GitHub và kiểm toán dữ liệu nghiêm ngặt!
"""
    concept_content = r"""
### 1. Khuôn khổ Định vị Dự án AI (AI Problem Framing Framework)
Để một dự án AI có giá trị thực tiễn cao đưa vào Portfolio, bạn phải trả lời rõ 4 câu hỏi:
1. **Bài toán thực tế là gì?** (Ví dụ: Hệ thống phát hiện sớm bất thường nhịp tim từ tín hiệu cảm biến đeo tay).
2. **Loại bài toán AI:** Phân loại nhị phân (Binary Classification) hay Hồi quy đa biến?
3. **Metric đo lường thành công:** Macro-F1 $\ge 0.85$, độ trễ suy diễn (Inference Latency) $< 50\text{ms}$ trên CPU.
4. **Dữ liệu đến từ đâu?** Nguồn cấp, giấy phép sử dụng (License CC-BY / Open Source), tính ẩn danh (Anonymization).

### 2. Cấu trúc Repository chuẩn GitHub của một Kỹ sư AI
```
final_portfolio_project/
├── data/                  # Dữ liệu (raw, processed)
├── src/                   # Mã nguồn chính (.py modules)
│   ├── data_loader.py     # Pipeline nạp & tiền xử lý
│   ├── models.py          # Kiến trúc mô hình PyTorch/Sklearn
│   └── utils.py           # Hàm bổ trợ & metric
├── notebooks/             # Các Jupyter Notebooks thực nghiệm
├── models/                # File checkpoint trọng số (.pt)
├── reports/               # Báo cáo kỹ thuật & Slide thuyết trình
├── app.py                 # File demo tương tác người dùng
├── README.md              # Trang giới thiệu dự án chuẩn quốc tế
└── requirements.txt       # Danh sách thư viện và phiên bản
```
"""
    visual_intuition_content = """
```
Luồng Tiến độ 5 Buổi của Module M7 (Final Portfolio Project):
[M7_S01: Scoping & Data Audit] ──► [M7_S02: EDA & Clean Baseline] ──► [M7_S03: Advanced Model & Tuning]
                                                                                │
                                                                                ▼
[M7_S05: GitHub, Report & Presentation] ◄── [M7_S04: Error Analysis & Interactive Demo]
```
"""
    code_cells = [
        (
            """import os
import json
import pandas as pd
import numpy as np

# 1. KHOI TAO CAU TRUC REPOSITORY CHUAN GITHUB CHO FINAL PROJECT
project_dirs = [
    "portfolio_project/data/raw",
    "portfolio_project/data/processed",
    "portfolio_project/src",
    "portfolio_project/notebooks",
    "portfolio_project/models",
    "portfolio_project/reports/figures"
]

for d in project_dirs:
    os.makedirs(d, exist_ok=True)

print("===== DA KHOI TAO CAU TRUC PROJECT REPOSITORY CHUAN GITHUB =====")
for d in project_dirs:
    print(f"[DIR CREATED] -> {d}")
""",
            """Khối code tự động khởi tạo toàn bộ cây thư mục chuẩn mực cho một dự án AI chuyên nghiệp:
- Phân định rõ ràng giữa dữ liệu thô (`raw`) và dữ liệu sạch (`processed`).
- Tách bạch giữa mã nguồn sản phẩm (`src`), sổ tay nghiên cứu (`notebooks`), file checkpoint (`models`) và báo cáo (`reports`)."""
        ),
        (
            """# 2. XAY DUNG MODULE KIEM TOAN VA XAC THUC LUOC DO DU LIEU (AUTOMATED SCHEMA VALIDATOR)
# Dinh nghia ban hop dong du lieu (Data Contract / Schema)
EXPECTED_SCHEMA = {
    "Patient_Age": {"type": "int64", "min": 18, "max": 100},
    "Heart_Rate": {"type": "float64", "min": 40.0, "max": 220.0},
    "Blood_Sugar": {"type": "float64", "min": 50.0, "max": 400.0},
    "Target_Diagnosis": {"type": "int64", "valid_values": [0, 1]}
}

def validate_dataset_schema(df, schema):
    print("===== KHOI DONG KIEM TOAN LUOC DO DU LIEU (DATA SCHEMA AUDIT) =====")
    audit_passed = True
    
    # Kiem tra su hien dien cua cac cot
    for col, specs in schema.items():
        if col not in df.columns:
            print(f"[FAIL] Thieu cot bat buoc: '{col}'!")
            audit_passed = False
            continue
            
        # Kiem tra mien gia tri (Min - Max)
        if "min" in specs and "max" in specs:
            out_of_bounds = df[(df[col] < specs["min"]) | (df[col] > specs["max"])]
            if len(out_of_bounds) > 0:
                print(f"[FAIL] Cot '{col}' co {len(out_of_bounds)} gia tri nam ngoai mien hop le [{specs['min']}, {specs['max']}]!")
                audit_passed = False
                
        # Kiem tra gia tri hop le
        if "valid_values" in specs:
            invalid_vals = set(df[col].unique()) - set(specs["valid_values"])
            if len(invalid_vals) > 0:
                print(f"[FAIL] Cot '{col}' chua cac gia tri bat hop ly: {invalid_vals}!")
                audit_passed = False
                
    if audit_passed:
        print("[PASS] 100% Du lieu thoat qua kiem toan Luoc do! Du lieu an toan cho Modeling.")
    return audit_passed

# Tao tap du lieu gia lap nghien cuu de kiem thu
np.random.seed(42)
demo_df = pd.DataFrame({
    "Patient_Age": [25, 45, 60, 35],
    "Heart_Rate": [72.0, 85.5, 110.0, 68.0],
    "Blood_Sugar": [95.0, 120.0, 140.0, 88.0],
    "Target_Diagnosis": [0, 1, 1, 0]
})

validate_dataset_schema(demo_df, EXPECTED_SCHEMA)
""",
            """Khối code xây dựng một hệ thống kiểm toán lược đồ dữ liệu tự động (Schema Validator):
- Ngăn chặn triệt để dữ liệu lỗi hoặc dữ liệu bị phá hủy cấu trúc (Corrupted Data) lọt vào pipeline huấn luyện.
- Thể hiện tư duy Kỹ sư Dữ liệu vững vàng của một ứng viên chuẩn công nghiệp."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Viết Project Charter tóm tắt
Soạn thảo một tài liệu `PROJECT_CHARTER.md` ngắn gọn (dưới 300 từ) gồm 4 mục:
1. Tên dự án và Lĩnh vực ứng dụng (Y tế, Tài chính, Giáo dục hoặc Giao thông).
2. Tuyên bố bài toán (Problem Statement).
3. Metric đánh giá thành công chính thức.
4. Môi trường công nghệ (Python, Scikit-Learn, PyTorch).

### Bài 2 (Easy): Khởi tạo file `requirements.txt`
Tạo một file `requirements.txt` trong thư mục gốc của project chứa đầy đủ danh sách các thư viện và phiên bản chuẩn xác cần thiết để chạy dự án (ví dụ: `torch>=2.0.0`, `scikit-learn>=1.3.0`, `pandas>=2.0.0`).

### Bài 3 (Medium): Kiểm toán Đạo đức & Quyền riêng tư (AI Ethics Audit)
Nêu rõ 3 rủi ro về mặt đạo đức và xã hội mà hệ thống AI của bạn có thể gây ra nếu mô hình dự đoán sai (ví dụ: thiên vị phân biệt đối xử theo độ tuổi hoặc vùng miền). Đề xuất biện pháp kỹ thuật giảm thiểu rủi ro này.

### Bài 4 (Medium): Bổ sung kiểm tra Missing Data vào Schema Validator
Nâng cấp hàm `validate_dataset_schema` để tự động kiểm tra tỷ lệ missing value của từng cột. Nếu một cột có tỷ lệ rỗng lớn hơn 30%, in ra cảnh báo màu vàng: `[WARNING] High Missingness`.

### Bài 5 (Challenge): Viết Script Tự động Tải và Phân vùng Dữ liệu (Data Ingestion Script)
Tạo file `src/data_loader.py` chứa một hàm `download_and_cache_dataset(url, dest_path)` tự động tải tập dữ liệu về thư mục `data/raw/`, kiểm tra mã băm toàn vẹn (MD5 / SHA-256 Checksum), và phân chia thành `train_raw.csv` và `test_raw.csv`.
"""
    voai_connection_content = """
Trong các cuộc thi Olympic AI và hackathon quốc tế lớn, việc xây dựng cấu trúc repo rõ ràng, có hợp đồng dữ liệu (Data Contract/Schema) là điểm cộng khổng lồ giúp bài thi của bạn ghi điểm tuyệt đối trong mắt ban giám khảo công nghệ. Một dự án có cấu trúc sạch sẽ chứng minh bạn không chỉ là một người biết bấm chạy notebook, mà là một **Nhà phát triển AI có tư duy hệ thống (Production-Minded AI Engineer)**.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh lưu dữ liệu thô đè lên dữ liệu đã làm sạch hoặc chỉnh sửa trực tiếp trên file gốc, khiến khi code bị lỗi không thể phục hồi lại trạng thái ban đầu:

```python
# CODE CÓ LỖI QUẢN LÝ DỮ LIỆU BẤT KHẢ ĐẢO NGƯỢC:
# df = pd.read_csv("data.csv")
# df.dropna(inplace=True) # Xoa truc tiep
# df.to_csv("data.csv") # Ghi de mat hoan toan file goc!
```

**Nhiệm vụ của bạn:**
1. Khẳng định nguyên tắc bất di bất dịch của Kỹ thuật dữ liệu: **Dữ liệu thô (Raw Data) là Bất biến (Immutable) — Không bao giờ được phép ghi đè!**
2. Thiết lập quy trình: Đọc từ `data/raw/`, xử lý và luôn xuất ra một file mới tại `data/processed/`.
"""
    knowledge_check_content = """
1. Mục tiêu cốt lõi của Buổi M7_S01 (Khởi động Final Project) là gì?
   - A. Hoàn thành ngay video thuyết trình.
   - B. Chuyển hóa bài toán đời thực thành bài toán AI có thể đo lường được, thiết lập cấu trúc repository chuẩn GitHub và kiểm toán chất lượng dữ liệu ban đầu.
   - C. Nộp bài lấy chứng chỉ ngay lập tức.
   - D. Cài đặt lại card đồ họa GPU.

2. Tại sao trong cấu trúc thư mục dự án AI chuyên nghiệp, chúng ta luôn phải tách riêng thư mục `data/raw/` và `data/processed/`?
   - A. Để máy tính có nhiều thư mục hơn.
   - B. Bảo vệ tính toàn vẹn bất biến (Immutability) của dữ liệu gốc, cho phép tái lập quy trình xử lý bất kỳ lúc nào mà không sợ mất mát dữ liệu nguồn.
   - C. Do quy định của ngôn ngữ Python.
   - D. Để tiết kiệm bộ nhớ RAM.

3. "Data Contract" hoặc "Schema Validation" có vai trò gì trong một pipeline AI?
   - A. Định nghĩa các quy tắc ràng buộc chặt chẽ về kiểu dữ liệu, miền giá trị và tính hợp lệ của các cột, bảo đảm ngăn chặn dữ liệu rác làm sập mô hình.
   - B. Hợp đồng mua bán card đồ họa.
   - C. Chứng chỉ tốt nghiệp của sinh viên.
   - D. File nén của mô hình.

4. File `requirements.txt` trong repository GitHub đóng vai trò gì đối với người khác khi muốn chạy lại dự án của bạn?
   - A. Tóm tắt nội dung bài toán.
   - B. Liệt kê chính xác tên và phiên bản của tất cả các thư viện phần mềm cần thiết, giúp người khác tái lập môi trường thực thi chỉ bằng một câu lệnh `pip install -r requirements.txt`.
   - C. Chứa mật khẩu máy tính.
   - D. Là file dữ liệu kiểm tra.

5. Khi phát hiện một cột dữ liệu trong đề tài y tế có các giá trị nhịp tim bằng âm (-15 bpm), hành động đúng đắn là gì?
   - A. Bỏ qua và đưa vào huấn luyện mô hình.
   - B. Xác định đây là lỗi nhập liệu/nhiễu thiết bị, gắn cờ vi phạm lược đồ và tiến hành loại bỏ hoặc điền khuyết hợp lý ở bước tiền xử lý.
   - C. Tắt phần mềm.
   - D. Đổi đề tài khác.
"""
    session_summary_content = """
* **What I learned:**
  - Quy trình định vị bài toán AI thế giới thực (AI Problem Framing).
  - Cấu trúc thư mục Repository chuẩn mực cho một dự án AI đưa vào Portfolio.
  - Tầm quan trọng của kiểm toán dữ liệu và đạo đức AI.
  - Cách xây dựng công cụ xác thực lược đồ dữ liệu tự động (Schema Validator).
* **What I can now do:**
  - Khởi tạo một dự án AI chuyên nghiệp đạt chuẩn công nghiệp từ con số 0.
  - Bảo vệ pipeline dữ liệu an toàn trước các dữ liệu rác và sai lệch lược đồ.
* **What I still need to practice:**
  - Tiến hành bước tiếp theo: **EDA Chuyên Sâu & Xây Dựng Baseline Sạch** ở Buổi 2.
"""
    next_session_content = """
Trong **Buổi M7_S02**, chúng ta sẽ tiến tới giai đoạn khai phá: **EDA Chuyên Sâu, Data Preprocessing Pipeline & Xây Dựng Baseline Sạch** — Phân tích tương quan đa chiều, thiết lập pipeline tiền xử lý khép kín và dựng mốc so sánh hiệu năng đầu tiên cho Final Project!
"""
    trace = {
        "items": [
            "Problem Scoping & AI Framing cho đề tài thế giới thực",
            "Thiết lập cấu trúc Production-grade GitHub Repository",
            "Kiểm toán chất lượng dữ liệu & Automated Schema Validator"
        ],
        "deliverable": "Cấu trúc Repository hoàn chỉnh + Module kiểm toán lược đồ dữ liệu"
    }
    nb = create_notebook(
        module_id="M7",
        module_name="DỰ ÁN AI THẬT ĐỂ ĐƯA VÀO PORTFOLIO",
        session_id="S01",
        topic="Project Scoping, Problem Framing & Data Audit",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Cấu trúc Repository hoàn chỉnh + Module kiểm toán dữ liệu tự động",
        competition_relevance="Tiêu chuẩn quản lý mã nguồn và dữ liệu chuẩn mực của các dự án AI chuyên nghiệp.",
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
    save_notebook(nb, "course/M7_FINAL_PROJECT/M7_S01.ipynb")


def build_m7_s02():
    objectives = [
        "Thực hiện Khám phá dữ liệu chuyên sâu (In-depth EDA) cho bài toán Final Project: Phân tích phân phối, tương quan và phát hiện ngoại lai.",
        "Xây dựng một Pipeline Tiền xử lý dữ liệu hoàn chỉnh, tự động hóa và có khả năng đóng gói (Modular Data Preprocessing Pipeline).",
        "Thiết lập mô hình cơ sở sạch sẽ (Clean Baseline Model) đầu tiên cho dự án Portfolio.",
        "Đo lường và ghi nhận mốc hiệu năng cơ sở (Baseline Performance Benchmark) làm cột mốc định lượng cho các cải tiến tiếp theo."
    ]
    prerequisites = [
        "Hoàn thành Buổi M7_S01 (Đã khởi tạo Repository và kiểm toán dữ liệu).",
        "Nắm vững kỹ năng EDA từ Module M1 và xây dựng Baseline từ Module M2."
    ]
    why_content = """
Sau khi đã có bài toán và dữ liệu thô sạch sẽ từ buổi 1, cám dỗ lớn nhất của một sinh viên non kinh nghiệm là ngay lập tức lao vào tải các mô hình phức tạp về chạy thử. Đó là con đường ngắn nhất dẫn đến thất bại!

Một Kỹ sư AI đẳng cấp luôn bắt đầu bằng việc **lắng nghe dữ liệu**: Khám phá các mối quan hệ tiềm ẩn, tìm ra quy luật phân phối và xây dựng một **Pipeline Tiền xử lý dữ liệu khép kín, mô-đun hóa**. Đồng thời, việc dựng một mô hình Baseline sạch sẽ ngay tại buổi hôm nay sẽ cung cấp cho bạn một thước đo khoa học để chứng minh giá trị của toàn bộ những cải tiến phức tạp ở các buổi sau.
"""
    concept_content = r"""
### 1. Nguyên tắc Xây dựng Preprocessing Pipeline mô-đun hóa
* Toàn bộ các bước: Điền khuyết (Imputation), Mã hóa danh mục (Encoding), Chuẩn hóa số học (Scaling) phải được đóng gói thành một đối tượng Pipeline duy nhất (ví dụ sử dụng `sklearn.pipeline.Pipeline` hoặc `ColumnTransformer`).
* **Không rò rỉ dữ liệu:** `pipeline.fit(X_train)` học các tham số thống kê chỉ từ tập Train, sau đó áp dụng sang tập Validation/Test.
* Có thể xuất bản (Serialize) toàn bộ pipeline thành file nhị phân (`preprocessor.joblib`) để sẵn sàng phục vụ cho ứng dụng Demo ở Buổi 4!

### 2. Thiết lập Clean Baseline cho Final Project
* Baseline phải là mô hình đơn giản nhất, dễ giải thích nhất nhưng hoạt động ổn định:
  - Phân loại: `DummyClassifier` kết hợp `LogisticRegression` hoặc `DecisionTreeClassifier(max_depth=3)`.
  - Hồi quy: `DummyRegressor` kết hợp `LinearRegression`.
* Ghi lại đầy đủ: Train Score, Validation Score (Accuracy, Macro-F1, RMSE) và thời gian thực thi.
"""
    visual_intuition_content = """
```
Kiến trúc Preprocessing Pipeline & Baseline Benchmark:
[Dữ liệu thô: X_train, y_train]
              │
              ▼
    [ColumnTransformer Pipeline]
    ├── Cột Số Học: Imputer (Median) ──> StandardScaler()
    └── Cột Danh Mục: Imputer (Mode) ──> OneHotEncoder(handle_unknown='ignore')
              │
              ▼
[Tập Đặc Trưng Đã Chuẩn Hóa] ──> [Huấn luyện Clean Baseline Model] ──> [BASELINE SCORE MỐC]
```
"""
    code_cells = [
        (
            """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. KHOI TAO TAP DU LIEU NGHIEP VU CHO FINAL PROJECT (HEALTHCARE PREDICTION)
np.random.seed(42)
N = 600

data = {
    'Age': np.random.randint(20, 80, size=N),
    'Blood_Pressure': np.random.normal(130, 18, size=N).round(1),
    'Cholesterol': np.random.normal(210, 35, size=N).round(1),
    'Smoking_Status': np.random.choice(['Never', 'Former', 'Current', np.nan], size=N, p=[0.4, 0.3, 0.2, 0.1]),
    'Heart_Disease': np.random.choice([0, 1], size=N, p=[0.75, 0.25]) # Target (25% Duong tinh)
}

df_project = pd.DataFrame(data)

# Chia Train / Test doc lap ngay tu dau de chong Leakage
X = df_project.drop(columns=['Heart_Disease'])
y = df_project['Heart_Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Tap Train: {len(X_train)} mau | Tap Test: {len(X_test)} mau")
""",
            """Khối code thiết lập bài toán Final Project với đầy đủ đặc trưng hỗn hợp:
- Dữ liệu gồm cả biến số học (`Age`, `Blood_Pressure`, `Cholesterol`) và biến danh mục có missing value (`Smoking_Status`).
- Phân chia Train/Test ngay từ đầu có Stratification bảo đảm tính nguyên vẹn khoa học."""
        ),
        (
            """# 2. XAY DUNG MODULAR PREPROCESSING PIPELINE BANG COLUMNTRANSFORMER
num_cols = ['Age', 'Blood_Pressure', 'Cholesterol']
cat_cols = ['Smoking_Status']

# Pipeline con cho bien so hoc
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline con cho bien danh muc
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Hop nhat thanh Preprocessor tong the
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# 3. HUAN LUYEN VA DANH GIA CLEAN BASELINE MODEL
baseline_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(random_state=42))
])

# Huan luyen toan bo luong chi voi 1 lenh fit duy nhat!
baseline_pipeline.fit(X_train, y_train)

# Du doan va Danh gia tren tap Test doc lap
test_preds = baseline_pipeline.predict(X_test)
base_acc = accuracy_score(y_test, test_preds)
base_f1 = f1_score(y_test, test_preds, average='macro')

print("===== KET QUA CLEAN BASELINE PERFORMANCE (MILESTONE) =====")
print(f"Baseline Model: Logistic Regression with Scaled & One-Hot Features")
print(f"Test Accuracy : {base_acc*100:.2f}%")
print(f"Test Macro-F1 : {base_f1:.4f}")
print("-> MOC CHUAN: Moi mo hinh nang cao o buoi M7_S03 bat buoc phai vuot qua diem Macro-F1 nay!")
""",
            """Khối code xây dựng một Pipeline Scikit-Learn chuẩn mực cấp độ sản phẩm:
- `ColumnTransformer`: Tự động áp dụng tiền xử lý chuyên biệt cho từng loại cột dữ liệu.
- Đóng gói toàn bộ từ tiền xử lý đến mô hình vào `baseline_pipeline`.
- Đo lường và thiết lập cột mốc điểm số Macro-F1 chính thức cho dự án."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Lưu trữ Preprocessor đã huấn luyện
Sử dụng thư viện `joblib` để lưu đối tượng `baseline_pipeline` vào file `portfolio_project/models/baseline_pipeline.joblib`. Viết đoạn code nạp lại file và chạy thử dự đoán trên 1 mẫu mới.

### Bài 2 (Easy): So sánh với Dummy Classifier
Huấn luyện một `DummyClassifier(strategy='most_frequent')` trên cùng tập dữ liệu. Tính điểm Macro-F1 và xác nhận xem mô hình Logistic Regression của bạn đã đánh bại được Dummy Classifier bao nhiêu phần trăm.

### Bài 3 (Medium): Bổ sung biểu đồ phân tích EDA vào thư mục báo cáo
Viết mã nguồn vẽ biểu đồ phân phối điểm số tương quan giữa `Blood_Pressure` và `Cholesterol` theo nhãn bệnh `Heart_Disease`. Tự động lưu biểu đồ chất lượng cao (DPI=300) vào đường dẫn `portfolio_project/reports/figures/eda_correlation.png`.

### Bài 4 (Medium): Kiểm tra khả năng xử lý danh mục lạ (Unseen Categories)
Tạo một mẫu dữ liệu mới có giá trị `Smoking_Status = 'Occasional'` (giá trị chưa từng xuất hiện trong tập Train). Chạy qua `baseline_pipeline` và giải thích tại sao tham số `handle_unknown='ignore'` trong `OneHotEncoder` giúp mô hình không bị sập (crash).

### Bài 5 (Challenge): Viết Custom Scikit-Learn Transformer
Tự viết một class `OutlierCapper(lower_quantile=0.01, upper_quantile=0.99)` kế thừa từ `BaseEstimator` và `TransformerMixin` của Scikit-Learn. Tích hợp Transformer này vào đầu của `num_pipeline` để tự động cắt tỉa các điểm dị biệt.
"""
    voai_connection_content = """
Trong các buổi bảo vệ dự án tốt nghiệp hoặc phỏng vấn tuyển dụng AI, nhà tuyển dụng sẽ đánh giá rất cao việc bạn sử dụng **`sklearn.pipeline.Pipeline` thay vì viết các đoạn code tiền xử lý rời rạc**. Một pipeline duy nhất nhận dữ liệu thô và xuất ra dự đoán thể hiện tư duy đóng gói sản phẩm sạch sẽ, không có lỗi rò rỉ dữ liệu và sẵn sàng đưa lên môi trường Server thực tế.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh tiền xử lý dữ liệu bằng cách gọi `preprocessor.fit_transform()` riêng cho tập Train và `preprocessor.fit_transform()` riêng cho tập Test, dẫn đến số lượng cột của hai tập bị lệch nhau do tập Test thiếu một giá trị danh mục:

```python
# CODE CÓ LỖI LỆCH CỘT DO FIT LẠI TẬP TEST:
# X_train_clean = preprocessor.fit_transform(X_train) # Sinh ra 6 cot
# X_test_clean = preprocessor.fit_transform(X_test)   # Chi sinh ra 5 cot vi thieu nhan!
# ValueError: Number of features of the model must match the input!
```

**Nhiệm vụ của bạn:**
1. Khẳng định quy tắc cốt tử: **Tuyệt đối KHÔNG BAO GIỜ gọi `fit` hoặc `fit_transform` trên tập Test!**
2. Sử dụng cấu trúc `Pipeline` như đã làm trong code cell trên để ngăn chặn hoàn toàn lỗi này.
"""
    knowledge_check_content = """
1. Tại sao trong một dự án AI chuyên nghiệp, chúng ta nên đóng gói toàn bộ quy trình tiền xử lý và mô hình vào một đối tượng `Pipeline` duy nhất?
   - A. Để máy tính chạy tốn nhiều RAM hơn.
   - B. Giúp quy trình xử lý dữ liệu trở nên mô-đun hóa, triệt tiêu hoàn toàn nguy cơ rò rỉ dữ liệu (Data Leakage) và dễ dàng đóng gói để triển khai lên sản phẩm thực tế chỉ bằng một hàm `predict()`.
   - C. Vì Scikit-Learn cấm viết code thông thường.
   - D. Để giấu mã nguồn không cho người khác xem.

2. Tham số `handle_unknown='ignore'` trong `OneHotEncoder` có tác dụng gì khi đưa vào triển khai thực tế?
   - A. Tự động xóa các hàng dữ liệu lạ.
   - B. Khi gặp một giá trị danh mục mới chưa từng xuất hiện trong tập huấn luyện (ví dụ một thành phố lạ), bộ mã hóa sẽ gán toàn bộ các cột one-hot tương ứng bằng số 0 thay vì báo lỗi sập chương trình.
   - C. Biến giá trị đó thành số âm.
   - D. Đổi tên cột dữ liệu.

3. Trong buổi M7_S02, việc thiết lập mốc "Clean Baseline Performance" có mục đích gì?
   - A. Để kết thúc dự án sớm.
   - B. Thiết lập chuẩn đối sánh tối thiểu có cơ sở khoa học để đánh giá xem các mô hình Deep Learning hoặc kỹ thuật tinh chỉnh phức tạp ở buổi sau có thực sự mang lại hiệu quả vượt trội hay không.
   - C. Để nộp báo cáo cho ban giám khảo chấm điểm ngay.
   - D. Để giảm thời gian huấn luyện.

4. Khi chuẩn hóa các cột số học (Scaling), tại sao giá trị trung bình $\mu$ và độ lệch chuẩn $\sigma$ chỉ được học từ tập Train?
   - A. Vì tập Train có nhiều mẫu hơn.
   - B. Để bảo đảm tính trung thực tuyệt đối: Mô hình không được phép biết trước phân phối thống kê của dữ liệu kiểm tra tương lai.
   - C. Vì tập Test không có số học.
   - D. Do giới hạn của thuật toán.

5. File định dạng `.joblib` hoặc `.pkl` thường được sử dụng trong hệ sinh thái Python AI để làm gì?
   - A. Lưu trữ file video demo.
   - B. Tuần tự hóa (Serialize) và lưu trữ trạng thái của các đối tượng mô hình và pipeline tiền xử lý đã huấn luyện xuống đĩa cứng để tái sử dụng.
   - C. Soạn thảo văn bản báo cáo.
   - D. Nén mã nguồn Python.
"""
    session_summary_content = """
* **What I learned:**
  - Quy trình EDA chuyên sâu và phân tích tương quan cho Final Project.
  - Cách xây dựng `ColumnTransformer` và `Pipeline` khép kín cấp độ sản xuất.
  - Kỹ thuật phòng ngừa lệch số lượng cột với `handle_unknown='ignore'`.
  - Thiết lập cột mốc hiệu năng Clean Baseline chính thức cho dự án.
* **What I can now do:**
  - Tự tin đóng gói mọi quy trình tiền xử lý phức tạp thành một đối tượng duy nhất.
  - Đo lường và lưu trữ mô hình Baseline sẵn sàng cho giai đoạn bứt phá hiệu năng.
* **What I still need to practice:**
  - Nâng cấp mô hình, tinh chỉnh siêu tham số và thực hiện **Ablation Study** ở Buổi 3.
"""
    next_session_content = """
Trong **Buổi M7_S03**, chúng ta sẽ bước vào giai đoạn tăng tốc hiệu năng: **Thử Nghiệm Mô Hình Nâng Cao, Tinh Chỉnh Siêu Tham Số & Ablation Study** — Đưa Random Forest, Gradient Boosting và Deep MLP vào so găng, tinh chỉnh siêu tham số và đo lường đóng góp của từng thành phần!
"""
    trace = {
        "items": [
            "Khám phá dữ liệu chuyên sâu (In-depth EDA) & Data storytelling",
            "Xây dựng Modular Preprocessing Pipeline với ColumnTransformer",
            "Thiết lập Clean Baseline model & Benchmark mốc hiệu năng ban đầu"
        ],
        "deliverable": "Mã nguồn Modular Data Preprocessing Pipeline + Clean Baseline Benchmark"
    }
    nb = create_notebook(
        module_id="M7",
        module_name="DỰ ÁN AI THẬT ĐỂ ĐƯA VÀO PORTFOLIO",
        session_id="S02",
        topic="EDA Chuyên Sâu, Data Pipeline & Xây Dựng Baseline Sạch",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook Preprocessing Pipeline hoàn chỉnh và Clean Baseline Model",
        competition_relevance="Tiêu chuẩn xây dựng Pipeline dữ liệu sạch cấp độ sản phẩm và mốc đối sánh ban đầu.",
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
    save_notebook(nb, "course/M7_FINAL_PROJECT/M7_S02.ipynb")


def build_m7_s03():
    objectives = [
        "Đưa các kiến trúc mô hình nâng cao vào thực nghiệm: Rừng ngẫu nhiên (Random Forest) và Mạng nơ-ron sâu (PyTorch MLP).",
        "Thực hiện Tinh chỉnh Siêu tham số có hệ thống (Systematic Hyperparameter Tuning) bằng Random Search hoặc Grid Search.",
        "Tiến hành Nghiên cứu loại trừ (Ablation Study) để chứng minh một cách khoa học đóng góp định lượng của từng thành phần trong mô hình.",
        "Xây dựng Bảng Theo dõi Thực nghiệm (Experiment Tracking Table) ghi nhận chi tiết lịch sử thử nghiệm đạt chuẩn nghiên cứu."
    ]
    prerequisites = [
        "Hoàn thành Buổi M7_S02 (Đã có Preprocessing Pipeline và mốc Baseline).",
        "Nắm vững kỹ năng huấn luyện PyTorch và Sklearn từ các module trước."
    ]
    why_content = """
Trong giới nghiên cứu và phát triển AI chuyên nghiệp, khi bạn tuyên bố: *"Mô hình của tôi đạt độ chính xác 92%"*, câu hỏi chất vấn đầu tiên của các chuyên gia phản biện sẽ là: **"Nhờ đâu mà bạn đạt được con số đó? Do bạn thêm đặc trưng mới, do kiến trúc mạng sâu hơn, hay đơn giản chỉ là may mắn ngẫu nhiên?"**.

Nếu bạn không thể trả lời câu hỏi này, giải pháp của bạn sẽ bị coi là thiếu cơ sở khoa học. **Nghiên cứu loại trừ (Ablation Study)** là phương pháp tiêu chuẩn vàng của các bài báo khoa học quốc tế đỉnh cao: Bạn lần lượt tháo bỏ từng thành phần (bỏ feature scaling, bỏ dropout, bỏ đặc trưng tương tác) để đo lường chính xác "giá trị gia tăng" của từng bộ phận. Buổi học hôm nay biến dự án của bạn thành một công trình nghiên cứu thực thụ!
"""
    concept_content = r"""
### 1. Phương pháp Nghiên cứu Loại trừ (Ablation Study)
* **Khái niệm:** Là chuỗi thực nghiệm có kiểm soát nhằm cô lập và định lượng tác động của một thành phần cụ thể trong hệ thống AI.
* **Quy trình chuẩn:**
  1. Mô hình Đầy đủ (Full Model): Có tất cả các đặc trưng và kỹ thuật.
  2. Thử nghiệm A: Full Model **loại bỏ** Feature Engineering.
  3. Thử nghiệm B: Full Model **loại bỏ** Dropout / Regularization.
  4. Thử nghiệm C: Full Model **thay thế** bộ tối ưu hóa Adam bằng SGD.
* Kết quả được tổng hợp thành bảng để chứng minh tính cần thiết của từng khối kỹ thuật!

### 2. Tinh chỉnh Siêu tham số (Hyperparameter Optimization)
* Không tinh chỉnh mò mẫm bằng tay!
* Sử dụng `RandomizedSearchCV` hoặc `GridSearchCV`:
  - Khám phá không gian siêu tham số: `n_estimators`, `max_depth`, `learning_rate`, `weight_decay`.
  - Luôn đo lường trên sơ đồ K-Fold Cross-Validation để tránh Overfitting trên tập kiểm thử.
"""
    visual_intuition_content = """
```
Bản đồ Nghiên cứu Loại trừ (Ablation Study Workflow):
[MÔ HÌNH ĐẦY ĐỦ: Deep MLP + Features + Dropout] ──────────► [F1: 0.885] (Đỉnh cao nhất)
                 │
                 ├── [Loại bỏ Dropout] ───────────────────► [F1: 0.842] (Mất -4.3% do Overfit)
                 ├── [Loại bỏ Feature Engineering] ───────► [F1: 0.810] (Mất -7.5% do thiếu tín hiệu)
                 └── [Thay Deep MLP bằng Baseline] ───────► [F1: 0.745] (Mất -14.0%)
```
"""
    code_cells = [
        (
            """import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 1. KHOI TAO BANG THEO DOI THUC NGHIEP (EXPERIMENT TRACKING TABLE)
# Gia lap tap du lieu dac trung phong phu (N=500, 8 dac trung)
np.random.seed(42)
N = 500
X_raw = np.random.randn(N, 5)
# Tao 3 dac trung tuong tac bo sung tu Feature Engineering
feat_interaction_1 = (X_raw[:, 0] * X_raw[:, 1]).reshape(-1, 1)
feat_ratio_2 = (X_raw[:, 2] / (np.abs(X_raw[:, 3]) + 0.1)).reshape(-1, 1)
feat_poly_3 = (X_raw[:, 4] ** 2).reshape(-1, 1)

X_full = np.hstack([X_raw, feat_interaction_1, feat_ratio_2, feat_poly_3])
y = (X_full[:, 0] * 1.5 + X_full[:, 5] * 2.0 - X_full[:, 6] + np.random.randn(N) > 0).astype(int)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"Tap du lieu nghien cuu: {X_full.shape[0]} mau, {X_full.shape[1]} dac trung tong cong.")
""",
            """Khối code thiết lập môi trường thực nghiệm có đối chứng:
- Tách bạch rõ ràng giữa 5 đặc trưng gốc (`X_raw`) và 3 đặc trưng tương tác sinh ra từ Feature Engineering.
- Sẵn sàng cho chuỗi thực nghiệm Ablation Study bóc tách đóng góp của từng nhóm."""
        ),
        (
            """# 2. TIEN HANH ABLATION STUDY & BENCHMARK CAC MO HINH
ablation_experiments = []

# Thuc nghiem 1: Baseline Co ban (Logistic Regression tren dac trung goc)
scores_exp1 = cross_val_score(LogisticRegression(), X_raw, y, cv=skf, scoring='f1_macro')
ablation_experiments.append({
    'Experiment': '1. Baseline (Logistic Reg + Raw Features)',
    'Features Used': '5 Raw Features',
    'Model Type': 'Linear Model',
    'CV Macro-F1': f"{scores_exp1.mean():.4f} (+/- {scores_exp1.std():.3f})",
    'Delta vs Baseline': '0.00% (Moc chuan)'
})

# Thuc nghiem 2: Random Forest tren dac trung goc (Kiem tra suc manh phi tuyen)
rf_raw = RandomForestClassifier(n_estimators=80, max_depth=4, random_state=42)
scores_exp2 = cross_val_score(rf_raw, X_raw, y, cv=skf, scoring='f1_macro')
ablation_experiments.append({
    'Experiment': '2. Non-linear Tree (RF + Raw Features)',
    'Features Used': '5 Raw Features',
    'Model Type': 'Random Forest',
    'CV Macro-F1': f"{scores_exp2.mean():.4f} (+/- {scores_exp2.std():.3f})",
    'Delta vs Baseline': f"+{(scores_exp2.mean() - scores_exp1.mean())*100:.2f}%"
})

# Thuc nghiem 3: FULL MODEL (Random Forest + Full Engineered Features)
rf_full = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
scores_exp3 = cross_val_score(rf_full, X_full, y, cv=skf, scoring='f1_macro')
ablation_experiments.append({
    'Experiment': '3. FULL SOLUTION (RF + Feature Engineering)',
    'Features Used': '8 Full Features',
    'Model Type': 'Tuned Random Forest',
    'CV Macro-F1': f"{scores_exp3.mean():.4f} (+/- {scores_exp3.std():.3f})",
    'Delta vs Baseline': f"+{(scores_exp3.mean() - scores_exp1.mean())*100:.2f}%"
})

df_ablation = pd.DataFrame(ablation_experiments)
print("===== BANG KET QUA NGHIEN CUU LOAI TRU (ABLATION STUDY REPORT) =====")
print(df_ablation.to_string(index=False))
print("\\n-> KET LUAN KHOA HOC: Feature Engineering dong gop +{:.2f}% F1, trong khi thuat toan phi tuyen dong gop +{:.2f}% F1!".format(
    (scores_exp3.mean() - scores_exp2.mean())*100, (scores_exp2.mean() - scores_exp1.mean())*100
))
""",
            """Khối code thực thi chuỗi thực nghiệm Ablation Study đạt chuẩn nghiên cứu khoa học:
- Đo lường phương sai ổn định qua 5-Fold Cross Validation (`mean +/- std`).
- Bóc tách rạch ròi bằng số liệu: Thuật toán phi tuyến đóng góp bao nhiêu phần trăm, và Kỹ nghệ đặc trưng đóng góp bao nhiêu phần trăm vào chiến thắng chung cuộc."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Thử nghiệm với thuật toán Gradient Boosting
Import `HistGradientBoostingClassifier` từ `sklearn.ensemble`. Thêm thực nghiệm số 4 vào bảng theo dõi và quan sát xem GBDT có vượt qua Random Forest hay không.

### Bài 2 (Easy): Tinh chỉnh siêu tham số bằng RandomizedSearchCV
Sử dụng `sklearn.model_selection.RandomizedSearchCV` để tìm bộ siêu tham số tối ưu cho Random Forest với không gian tìm kiếm: `n_estimators: [50, 100, 150]`, `max_depth: [3, 5, 7]`. In ra `best_params_`.

### Bài 3 (Medium): Đo lường chi phí tính toán (Computational Cost Tracking)
Thêm cột `Inference Time (ms)` vào bảng theo dõi thực nghiệm. Viết đoạn code đo thời gian thực hiện dự đoán cho 1000 mẫu của từng mô hình và thảo luận về sự đánh đổi giữa độ chính xác và độ trễ.

### Bài 4 (Medium): Bóc tách tác động của từng đặc trưng (Individual Feature Ablation)
Lần lượt loại bỏ từng đặc trưng một trong 3 đặc trưng tương tác và đo lường sự sụt giảm F1-Score. Cho biết đặc trưng nào trong 3 đặc trưng đó đóng vai trò "gánh team" quan trọng nhất.

### Bài 5 (Challenge): Xuất báo cáo Ablation sang định dạng Bảng LaTeX / Markdown
Viết một hàm Python `export_ablation_latex(df_ablation, filename="ablation_table.tex")` tự động chuyển đổi bảng kết quả thành mã nguồn bảng LaTeX chuẩn của các hội nghị AI quốc tế (như NeurIPS hoặc CVPR).
"""
    voai_connection_content = """
Một trong những tiêu chí chấm điểm khắt khe nhất của Ban giám khảo Olympic AI và các chuyên gia hội đồng thẩm định là **Tính Thuyết Phục Kỹ Thuật (Technical Justification)**:
Một bài thuyết trình chỉ nói: *"Em thử linh tinh và thấy mô hình này điểm cao nhất"* sẽ bị đánh giá rất thấp. Ngược lại, thí sinh trình bày một bảng **Ablation Study** rõ ràng, chỉ ra chính xác quyết định kỹ thuật nào mang lại bao nhiêu phần trăm cải thiện điểm số, sẽ ngay lập tức chinh phục hoàn toàn hội đồng và khẳng định đẳng cấp vượt trội.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh tiến hành Ablation Study nhưng mỗi lần chạy lại đổi `random_state` khác nhau hoặc đổi kích thước fold, khiến sự chênh lệch điểm số thực chất chỉ do nhiễu ngẫu nhiên chứ không phải do mô hình:

```python
# CODE CÓ LỖI THỰC NGHIỆM KHÔNG CÙNG MẶT BẰNG ĐỐI CHỨNG:
# Exp 1: cv = StratifiedKFold(n_splits=5, random_state=42)
# Exp 2: cv = StratifiedKFold(n_splits=5, random_state=999) # Doi seed -> Sai lech mat bang so sanh!
```

**Nhiệm vụ của bạn:**
1. Khẳng định nguyên tắc thực nghiệm khoa học: **Mọi thí nghiệm trong Ablation Study bắt buộc phải chia sẻ CHUNG MỘT SƠ ĐỒ FOLD DUY NHẤT** để đảm bảo sự so sánh là tuyệt đối công bằng.
2. Cố định đối tượng `skf` làm chuẩn quy chiếu chung cho tất cả các cuộc so găng.
"""
    knowledge_check_content = """
1. Mục tiêu tối thượng của phương pháp "Ablation Study" (Nghiên cứu loại trừ) trong AI là gì?
   - A. Xóa toàn bộ mã nguồn của dự án.
   - B. Lần lượt cô lập và tháo bỏ từng thành phần kỹ thuật để đo lường định lượng đóng góp thực tế của từng bộ phận, chứng minh tính khoa học của giải pháp.
   - C. Tự động dịch chuyển dữ liệu sang máy chủ khác.
   - D. Giảm kích thước file nộp bài.

2. Tại sao việc tinh chỉnh siêu tham số (Hyperparameter Tuning) luôn phải được đánh giá qua Cross-Validation thay vì chỉ đo trên một tập kiểm tra duy nhất?
   - A. Vì Cross-Validation chạy nhanh hơn.
   - B. Để ngăn ngừa hiện tượng mô hình bị Overfitting gián tiếp lên tập kiểm tra đó (Tuning Leakage) và bảo đảm tính tổng quát hóa ổn định trên nhiều phân vùng dữ liệu khác nhau.
   - C. Vì luật thi đấu bắt buộc.
   - D. Do Scikit-Learn yêu cầu.

3. Điểm khác biệt cơ bản giữa `GridSearchCV` và `RandomizedSearchCV` là gì?
   - A. `GridSearchCV` duyệt qua tất cả các tổ hợp có thể (chậm nhưng vét cạn), trong khi `RandomizedSearchCV` lấy mẫu ngẫu nhiên một số tổ hợp (nhanh hơn rất nhiều và hiệu quả trong không gian tham số lớn).
   - B. `GridSearchCV` chỉ dùng được cho Deep Learning.
   - C. `RandomizedSearchCV` luôn cho kết quả kém hơn.
   - D. Không có sự khác biệt nào.

4. Trong bảng theo dõi thực nghiệm (Experiment Tracking Table), giá trị độ lệch chuẩn `+/- std` bên cạnh điểm trung bình F1 biểu thị điều gì?
   - A. Số lượng tham số của mô hình.
   - B. Mức độ ổn định và độ biến thiên của mô hình qua các fold dữ liệu khác nhau: Độ lệch chuẩn càng nhỏ thì mô hình càng đáng tin cậy.
   - C. Thời gian chạy tính bằng phút.
   - D. Số lượng epochs đã hoàn thành.

5. Một giải pháp AI đạt chuẩn nghiên cứu khoa học bắt buộc phải chứng minh được điều gì trước hội đồng thẩm định?
   - A. Đã sử dụng máy tính đắt tiền nhất.
   - B. Điểm số cải tiến vượt bậc so với Baseline và có số liệu thực nghiệm minh chứng cho đóng góp của từng thành phần kỹ thuật.
   - C. Đã viết hơn 10,000 dòng code.
   - D. Đã dùng tất cả các thư viện có sẵn trên mạng.
"""
    session_summary_content = """
* **What I learned:**
  - Bản chất và phương pháp luận của Nghiên cứu Loại trừ (Ablation Study).
  - Kỹ thuật tinh chỉnh siêu tham số có hệ thống (Hyperparameter Optimization).
  - Cách thiết lập Bảng Theo dõi Thực nghiệm (Experiment Tracking Table) chuẩn quốc tế.
* **What I can now do:**
  - Chứng minh một cách khoa học giá trị của từng kỹ thuật trong hệ thống AI của mình.
  - Tự tin bảo vệ giải pháp kỹ thuật trước mọi hội đồng phản biện khó tính nhất.
* **What I still need to practice:**
  - Phân tích lỗi sai sâu sắc và xây dựng **Giao diện Demo Tương tác (Interactive Demo)** ở Buổi 4!
"""
    next_session_content = """
Trong **Buổi M7_S04**, chúng ta sẽ biến mô hình thành một sản phẩm có thể tương tác: **Đánh Giá Chi Tiết, Phân Tích Lỗi Sai (Error Analysis) & Xây Dựng Interactive Demo** — Khám phá các ca thất bại nặng nề nhất, đánh giá hiệu năng trên từng phân nhóm và tự tay lập trình ứng dụng Demo tương tác trực tiếp!
"""
    trace = {
        "items": [
            "Thực nghiệm mô hình nâng cao & Tinh chỉnh siêu tham số",
            "Nghiên cứu loại trừ (Ablation Study) bóc tách đóng góp kỹ thuật",
            "Bảng theo dõi thực nghiệm (Experiment tracking table) chuẩn khoa học"
        ],
        "deliverable": "Mã nguồn Ablation Study Framework + Bảng theo dõi thực nghiệm có đối chứng"
    }
    nb = create_notebook(
        module_id="M7",
        module_name="DỰ ÁN AI THẬT ĐỂ ĐƯA VÀO PORTFOLIO",
        session_id="S03",
        topic="Thử Nghiệm Mô Hình Nâng Cao & Ablation Study",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook thực nghiệm Ablation Study và bảng theo dõi mô hình chuẩn mực",
        competition_relevance="Chứng minh giá trị khoa học định lượng của giải pháp trước hội đồng giám khảo.",
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
    save_notebook(nb, "course/M7_FINAL_PROJECT/M7_S03.ipynb")


def build_m7_s04():
    objectives = [
        "Tiến hành Phân tích Lỗi sai Chuyên sâu (Granular Error Analysis): Bóc tách các ca mô hình tự tin sai lầm nghiêm trọng (High-Confidence Failures).",
        "Đánh giá hiệu năng đa chiều theo từng phân nhóm đối tượng (Slice-based Performance Evaluation) để phát hiện rủi ro thiên vị.",
        "Xây dựng một Ứng dụng Demo Tương tác Trực tiếp (Interactive Prediction Demo) gọn nhẹ, trực quan và chạy độc lập.",
        "Thiết kế cơ chế giải thích quyết định dự đoán (Prediction Explainability) kèm thanh phần trăm độ tự tin cho người dùng cuối."
    ]
    prerequisites = [
        "Hoàn thành Buổi M7_S03 (Đã có mô hình tối ưu từ Ablation Study).",
        "Nắm vững các metric đánh giá chi tiết từ Module M2 và M4."
    ]
    why_content = """
Một mô hình AI dù có đạt độ chính xác 95% trên tập dữ liệu kiểm tra thì nó vẫn hoàn toàn "vô hình" và vô giá trị đối với người dùng cuối, nhà đầu tư hoặc ban giám khảo nếu nó chỉ nằm im lìm dưới dạng các dòng code trong một cuốn notebook. Người ta không thể hình dung mô hình của bạn hoạt động thế nào trong đời sống thực!

Hơn thế nữa, 5% sai số còn lại là những trường hợp nào? Mô hình có xu hướng đoán sai ở người cao tuổi hay người trẻ tuổi? Buổi học hôm nay giải quyết trọn vẹn hai câu hỏi sống còn đó: **Mổ xẻ tận gốc các ca dự đoán sai lầm (Error Analysis)** và **Thổi hồn vào mô hình bằng một Ứng dụng Demo Tương tác Trực tiếp (Interactive Demo)** — biến thuật toán phức tạp thành một sản phẩm công nghệ sống động mà bất kỳ ai cũng có thể tự tay thử nghiệm và trầm trồ!
"""
    concept_content = r"""
### 1. Phương pháp Phân tích Hiệu năng theo Lát cắt (Slice-based Evaluation)
* Không chỉ nhìn vào con số trung bình toàn cục! Một mô hình có Accuracy 90% toàn diện nhưng có thể chỉ đạt 45% trên nhóm người cao tuổi ($Age \ge 65$) $\implies$ Rủi ro y tế chết người!
* **Cách thực hiện:** Chia nhỏ tập kiểm tra thành các lát cắt nhân khẩu học/đặc trưng (Sub-cohort slices) và đo lường metric độc lập trên từng lát cắt để phát hiện vùng trũng hiệu năng.

### 2. Kiến trúc Ứng dụng Demo Tương tác (Interactive Inference Engine)
Một ứng dụng demo đạt chuẩn phải hội tụ đủ 3 yếu tố:
1. **User Input Interface:** Cho phép người dùng nhập thông số tùy ý (hoặc chọn các mẫu thử có sẵn).
2. **End-to-End Inference Pipeline:** Tự động tiền xử lý dữ liệu nhập vào qua Pipeline đã huấn luyện từ Buổi 2.
3. **Visual Output with Confidence & Explanation:** Xuất kết quả kèm theo thanh phần trăm xác suất tự tin ($P \in [0\%, 100\%]$) và cảnh báo trực quan (Xanh / Đỏ).
"""
    visual_intuition_content = """
```
Kiến trúc Ứng dụng Demo Tương tác (Interactive Demo Engine):
[GIAO DIỆN NGƯỜI DÙNG: Nhập Tuổi, Huyết áp, Cholesterol, Tình trạng Hút thuốc]
                               │
                               ▼
            [Tự động gọi: preprocessor.transform()]
                               │
                               ▼
                 [Mô hình Tối ưu: model.predict_proba()]
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ KẾT QUẢ DỰ ĐOÁN: BỆNH NHÂN CÓ NGUY CƠ TIM MẠCH CAO (NGUY HIỂM!)            │
│ Độ tự tin của mô hình: [████████████████████░░] 87.5%                       │
│ Yếu tố cảnh báo chính: Huyết áp vượt ngưỡng (165 mmHg) & Tuổi cao (68 tuổi) │
└─────────────────────────────────────────────────────────────────────────────┘
```
"""
    code_cells = [
        (
            """import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# 1. PHAN TICH HIEU NANG THEO LAT CAT (SLICE-BASED ERROR ANALYSIS)
np.random.seed(42)
n_test = 150

# Gia lap tap test gom Tuoi, Huyet ap va Nhan thuc te
test_age = np.random.randint(20, 80, size=n_test)
test_bp = np.random.normal(130, 20, size=n_test).round(1)
true_target = (test_age * 0.04 + test_bp * 0.03 + np.random.randn(n_test) > 5.5).astype(int)

# Gia lap du doan cua mo hinh
pred_probs = 1 / (1 + np.exp(-(test_age * 0.04 + test_bp * 0.03 - 5.3)))
pred_target = (pred_probs >= 0.5).astype(int)

df_analysis = pd.DataFrame({
    'Age': test_age,
    'Blood_Pressure': test_bp,
    'True_Label': true_target,
    'Pred_Label': pred_target,
    'Confidence': np.maximum(pred_probs, 1 - pred_probs)
})

# Phan chia thanh 2 lat cat: Nguoi tre (< 50 tuoi) vs Nguoi cao tuoi (>= 50 tuoi)
young_mask = df_analysis['Age'] < 50
senior_mask = df_analysis['Age'] >= 50

acc_young = (df_analysis[young_mask]['True_Label'] == df_analysis[young_mask]['Pred_Label']).mean()
acc_senior = (df_analysis[senior_mask]['True_Label'] == df_analysis[senior_mask]['Pred_Label']).mean()

print("===== PHAN TICH HIEU NANG THEO LAT CAT NHAN KHAU HOC (SLICING) =====")
print(f"Tong so mau Test: {n_test} mau")
print(f"Do chinh xac nhom Nguoi Tre (< 50 tuoi)      : {acc_young*100:.1f}% ({young_mask.sum()} mau)")
print(f"Do chinh xac nhom Nguoi Cao Tuoi (>= 50 tuoi): {acc_senior*100:.1f}% ({senior_mask.sum()} mau)")
print("-> PHAT HIEN: Mo hinh co su chenh lech hieu nang giua cac nhom tuoi. Can them du lieu cho nhom yeu hon!")
""",
            """Khối code thực thi kỹ thuật phân tích hiệu năng theo lát cắt (Slice-based Evaluation):
- Chia nhỏ tập dữ liệu kiểm thử theo độ tuổi để phát hiện điểm mù của mô hình.
- Cung cấp bằng chứng định lượng rõ ràng giúp thí sinh đưa ra các giải pháp cải tiến chính xác."""
        ),
        (
            """# 2. XAY DUNG UNG DUNG DEMO TUONG TAC TRUC TIEP (INTERACTIVE PREDICTOR APP)
# Day la ma nguon san sang tich hop vao CLI hoac giao dien Web (Streamlit/Gradio)

def run_interactive_ai_demo(patient_age, blood_pressure, cholesterol, smoking_status):
    print("\\n==================================================================")
    print("      HE THONG AI HO TRO CHUAN DOAN NGUY CO TIM MACH (DEMO)      ")
    print("==================================================================")
    print(f"Thong tin dau vao: Tuoi = {patient_age} | Huyet ap = {blood_pressure} mmHg | Mo mau = {cholesterol} mg/dL | Hut thuoc = {smoking_status}")
    
    # Mo phong pipeline tien xu ly va suy dien (Inference)
    # Logits gia lap tinh toan dua tren cac trong so da hoc
    smoke_score = 1.2 if smoking_status == 'Current' else (0.5 if smoking_status == 'Former' else 0.0)
    risk_score = (patient_age - 45) * 0.05 + (blood_pressure - 120) * 0.04 + (cholesterol - 180) * 0.02 + smoke_score
    prob_risk = 1 / (1 + np.exp(-risk_score))
    
    # Dinh dang thanh tien do truc quan
    bar_length = 20
    filled_length = int(bar_length * prob_risk)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    
    print("\\n--- KET QUA PHAN TICH TU MO HINH AI ---")
    print(f"Do tin cay nguy co: [{bar}] {prob_risk*100:.1f}%")
    
    if prob_risk >= 0.65:
        print("🚨 KET LUAN: NGUY CO CAO (HIGH RISK) - Can can thiep y te som!")
        print("💡 Loi khuyen AI: Kiem soat huyet ap va ngung hut thuoc la ngay lap tuc.")
    elif prob_risk >= 0.35:
        print("⚠️ KET LUAN: NGUY CO TRUNG BINH (MODERATE RISK) - Can theo doi them.")
        print("💡 Loi khuyen AI: Cai thien che do an uong va tap the duc thuong xuyen.")
    else:
        print("✅ KET LUAN: NGUY CO THAP (LOW RISK) - Chi so tim mach on dinh.")
        print("💡 Loi khuyen AI: Duy tri loi song lanh manh hien tai.")
    print("==================================================================")

# Chay thu nghiem Demo tren 2 ca benh nhan thuc te
run_interactive_ai_demo(patient_age=65, blood_pressure=155.0, cholesterol=240.0, smoking_status='Current')
run_interactive_ai_demo(patient_age=28, blood_pressure=115.0, cholesterol=165.0, smoking_status='Never')
""",
            """Khối code lập trình một ứng dụng Demo tương tác hoàn chỉnh:
- Nhận thông số đầu vào linh hoạt của bất kỳ người dùng nào.
- Trực quan hóa phần trăm độ tự tin bằng thanh tiến độ (Visual ASCII Bar).
- Xuất kết luận phân loại rõ ràng kèm lời khuyên hành động thực tế, sẵn sàng trình diễn trực tiếp trước hội đồng giám khảo."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Đóng gói Demo thành file `app.py` độc lập
Trích xuất đoạn mã hàm `run_interactive_ai_demo` và lưu thành file `portfolio_project/app.py` độc lập. Chạy thử từ dòng lệnh terminal bằng lệnh `python portfolio_project/app.py` để xác minh chương trình chạy trơn tru.

### Bài 2 (Easy): Mở rộng giao diện nhập liệu từ bàn phím
Sửa đổi hàm demo để sử dụng hàm `input()` của Python cho phép người dùng tự gõ các chỉ số từ bàn phím tại thời điểm runtime thay vì gán cứng tham số.

### Bài 3 (Medium): Trích xuất 5 ca dự đoán sai tồi tệ nhất (Top-5 Worst Failures)
Viết hàm lọc ra 5 trường hợp trong bảng `df_analysis` mà mô hình có độ tự tin cao nhất ($P > 90\%$) nhưng lại dự đoán sai hoàn toàn nhãn thực tế. Phân tích đặc điểm chung của 5 trường hợp này.

### Bài 4 (Medium): Đánh giá tính giải thích được với SHAP / LIME Intuition
Tìm hiểu khái niệm SHAP (SHapley Additive exPlanations). Viết một đoạn văn ngắn giải thích làm thế nào SHAP có thể bóc tách đóng góp của từng đặc trưng (ví dụ: Huyết áp đóng góp $+35\%$ vào nguy cơ bệnh) cho từng bệnh nhân cụ thể.

### Bài 5 (Challenge): Xây dựng Web Demo bằng Streamlit hoặc Gradio
Tìm hiểu thư viện Gradio hoặc Streamlit. Viết một file `web_demo.py` ngắn dưới 30 dòng code tạo ra một trang web tương tác cục bộ có các thanh trượt (Sliders) và nút bấm (Button) để dự đoán trực quan trên trình duyệt.
"""
    voai_connection_content = """
Trong phần thi chung kết Olympic AI hoặc triển lãm sản phẩm nghiên cứu, **một ứng dụng Demo chạy mượt mà trước mắt Ban giám khảo có giá trị thuyết phục gấp 100 lần những lời nói lý thuyết suông**. Giám khảo luôn yêu cầu: *"Hãy nhập thử một dữ liệu bất kỳ do tôi đưa ra xem mô hình của bạn phản hồi thế nào?"*. Chuẩn bị sẵn một Demo Engine tương tác hoàn hảo giúp bạn làm chủ hoàn toàn sân khấu và nắm chắc điểm số tối đa.
"""
    debugging_content = """
**Tình huống thực tế:** Hàm demo nhận dữ liệu đầu vào từ người dùng nhưng bị lỗi sập chương trình khi người dùng nhập sai kiểu dữ liệu (ví dụ gõ chữ "sáu mươi" vào ô tuổi, hoặc nhập số âm):

```python
# CODE CÓ LỖI THIẾU KIỂM TRA ĐẦU VÀO NGƯỜI DÙNG:
# age = int(user_input) # Neu user go "abc" -> ValueError: invalid literal for int()!
```

**Nhiệm vụ của bạn:**
1. Viết khối lệnh bọc an toàn `try-except` và kiểm tra miền giá trị hợp lệ trước khi đưa vào mô hình.
2. Trả về thông báo hướng dẫn thân thiện: *"Vui lòng nhập tuổi là số nguyên từ 1 đến 120"*.
"""
    knowledge_check_content = """
1. Hoạt động "Slice-based Performance Evaluation" (Đánh giá hiệu năng theo lát cắt) có ý nghĩa gì đối với một dự án AI chuẩn mực?
   - A. Cắt nhỏ file code thành nhiều phần.
   - B. Đánh giá chất lượng mô hình trên từng phân nhóm dữ liệu cụ thể (như độ tuổi, giới tính, vùng miền) để phát hiện các điểm mù và bảo đảm tính công bằng, an toàn trước khi triển khai.
   - C. Giảm độ phân giải của hình ảnh.
   - D. Tăng tốc độ quạt chip CPU.

2. Tại sao một dự án AI hoàn chỉnh đưa vào Portfolio bắt buộc phải có một sản phẩm Demo tương tác (Interactive Demo)?
   - A. Để làm cho bài tập dài hơn.
   - B. Biến mô hình toán học trừu tượng thành một sản phẩm công nghệ trực quan, cho phép người dùng và nhà tuyển dụng trực tiếp thử nghiệm và kiểm chứng năng lực giải quyết vấn đề thực tế.
   - C. Vì luật thi bắt buộc phải dùng web.
   - D. Để tải thêm nhiều file từ Internet.

3. Trong ứng dụng demo dự đoán, thông tin "Độ tự tin" (Confidence Score) mang lại giá trị gì cho người dùng cuối?
   - A. Cho biết tốc độ đường truyền mạng.
   - B. Cung cấp mức độ chắc chắn của mô hình về quyết định đưa ra (ví dụ 95% chắc chắn vs 51% phân vân), giúp con người biết khi nào nên tin tưởng và khi nào cần kiểm tra lại cẩn trọng.
   - C. Cho biết dung lượng pin còn lại.
   - D. Đo lường nhiệt độ máy chủ.

4. Trường hợp nào sau đây được coi là lỗi thất bại nghiêm trọng nhất của mô hình trong quá trình phân tích lỗi sai (Error Analysis)?
   - A. Mô hình đoán sai nhưng độ tự tin thấp ($P \approx 50\%$).
   - B. Mô hình đoán sai hoàn toàn nhưng lại có độ tự tin cực cao ($P > 95\%$ - High-Confidence Failure).
   - C. Mô hình chạy mất 2 giây.
   - D. Mô hình in ra kết quả màu xanh.

5. Sản phẩm mà học viên xây dựng và kiểm thử thành công trong Buổi M7_S04 là gì?
   - A. Một bài báo in trên giấy.
   - B. Báo cáo phân tích lỗi sai chuyên sâu (Error Analysis) kèm Ứng dụng Demo Tương tác Trực tiếp (Interactive Prediction Demo).
   - C. Một chứng chỉ giả định.
   - D. Một bài hát về AI.
"""
    session_summary_content = """
* **What I learned:**
  - Kỹ thuật đánh giá hiệu năng theo lát cắt phân nhóm (Slice-based Evaluation).
  - Phương pháp mổ xẻ các ca thất bại có độ tự tin cao (High-Confidence Failures).
  - Kiến trúc xây dựng Ứng dụng Demo Tương tác Trực tiếp (Interactive Demo Engine).
* **What I can now do:**
  - Biến mô hình AI tĩnh trong notebook thành một ứng dụng tương tác sống động có thanh đo độ tự tin.
  - Tự tin phân tích điểm yếu và ranh giới an toàn của mô hình trước người dùng.
* **What I still need to practice:**
  - Đóng gói toàn bộ dự án vào **GitHub Repo, Báo cáo Kỹ thuật & Slide Thuyết trình** ở Buổi 5!
"""
    next_session_content = """
Buổi tiếp theo (**M7_S05**) là buổi tổng kết đỉnh cao và là đích đến cuối cùng của toàn bộ khóa học 38 buổi: **Hoàn Thiện Portfolio: Đóng Gói GitHub Repo, Viết Báo Cáo Kỹ Thuật & Thuyết Trình**! Bạn sẽ hoàn thiện trang README đẳng cấp, đóng gói mã nguồn, diễn tập thuyết trình và chính thức nhận Chứng chỉ tốt nghiệp toàn khóa **Final Project Certificate**!
"""
    trace = {
        "items": [
            "Đánh giá hiệu năng theo lát cắt (Slice-based evaluation)",
            "Phân tích lỗi sai chi tiết (High-confidence error analysis)",
            "Xây dựng Interactive Prediction Demo có đo lường độ tự tin"
        ],
        "deliverable": "Mã nguồn Interactive Prediction Demo hoàn chỉnh kèm phân tích lỗi"
    }
    nb = create_notebook(
        module_id="M7",
        module_name="DỰ ÁN AI THẬT ĐỂ ĐƯA VÀO PORTFOLIO",
        session_id="S04",
        topic="Đánh Giá Chi Tiết, Phân Tích Lỗi & Interactive Demo",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook phân tích lỗi chi tiết và ứng dụng Interactive AI Demo",
        competition_relevance="Biến thuật toán thành sản phẩm thực tế có khả năng trình diễn trực tiếp trước giám khảo.",
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
    save_notebook(nb, "course/M7_FINAL_PROJECT/M7_S04.ipynb")


def build_m7_s05():
    objectives = [
        "Hoàn thiện và đóng gói toàn diện Sản phẩm Đầu ra Cuối cùng của khóa học (Final Project Deliverable): GitHub Repository chuẩn công nghiệp.",
        "Thiết kế trang giới thiệu dự án chuyên nghiệp (Polished README.md) với đầy đủ Badges, Sơ đồ kiến trúc, Hướng dẫn cài đặt và Kết quả thực nghiệm.",
        "Soạn thảo Báo cáo Kỹ thuật Toàn diện (Comprehensive Technical Report) theo quy chuẩn nghiên cứu khoa học.",
        "Chuẩn bị Cấu trúc Bài Thuyết trình Bảo vệ Dự án (Slide Deck & Oral Defense Presentation) và nhận Chứng chỉ Tốt nghiệp Toàn khóa (Final Project Certificate)."
    ]
    prerequisites = [
        "Hoàn thành trọn vẹn 4 buổi trước của Module M7 (Từ S01 đến S04).",
        "Đã hoàn thành toàn bộ 37 buổi học trước của chương trình 38 buổi."
    ]
    why_content = """
Xin nhiệt liệt chúc mừng bạn! Bạn đã đi qua một hành trình phi thường: 37 buổi học đầy thử thách từ những dòng code NumPy đầu tiên, qua các mô hình Machine Learning cổ điển, mạng nơ-ron sâu PyTorch, các bài toán thị giác và ngôn ngữ, chiến thuật thi đấu Olympic đỉnh cao, đến kỹ năng làm chủ trợ lý AI.

Hôm nay là **Buổi 38 — Vạch đích vinh quang của khóa học**. Một công trình vĩ đại chỉ thực sự hoàn tất khi nó được đóng gói chỉn chu, có tài liệu kỹ thuật rõ ràng và được trình bày tự tin trước thế giới. Buổi học hôm nay dành trọn vẹn để bạn hoàn thiện trang **GitHub Portfolio**, xuất bản **Báo cáo Kỹ thuật**, diễn tập bài **Thuyết trình Bảo vệ Dự án**, và chính thức bước lên bục vinh quang nhận tấm bằng tốt nghiệp danh giá!
"""
    concept_content = r"""
### 1. Cấu trúc Trang README.md chuẩn Quốc tế
Một trang GitHub README đẳng cấp tuyển dụng bắt buộc phải có 6 phần:
1. **Header & Badges:** Tên dự án, huy hiệu phiên bản Python, PyTorch, giấy phép License, trạng thái Build.
2. **Project Overview & Motivation:** Bài toán thực tế giải quyết điều gì và tại sao nó quan trọng?
3. **System Architecture Diagram:** Sơ đồ luồng dữ liệu từ Data Pipeline -> Preprocessor -> Model -> Demo.
4. **Benchmark Results:** Bảng so sánh khoa học giữa Baseline và Mô hình đề xuất (Accuracy, Macro-F1, Latency).
5. **Quickstart & Reproducibility Guide:** Hướng dẫn 3 bước để bất kỳ ai cũng có thể clone repo và chạy lại kết quả chỉ trong 1 dòng lệnh (`git clone`, `pip install -r requirements.txt`, `python app.py`).
6. **Contributors & Citation:** Tác giả và trích dẫn khoa học.

### 2. Cấu trúc Báo cáo Kỹ thuật (Technical Report) & Slide Thuyết trình
* **Tóm tắt điều hành (Executive Summary):** 1 đoạn văn tóm tắt toàn bộ bài toán, phương pháp và kết quả cốt lõi.
* **Cấu trúc Slide Thuyết trình 5 phút (5-Minute Pitch):**
  - Slide 1: Vấn đề đời thực & Nỗi đau (Pain Point).
  - Slide 2: Dữ liệu & Quy trình xử lý (Pipeline).
  - Slide 3: Giải pháp Mô hình & Kết quả Ablation Study.
  - Slide 4: Trình diễn Demo trực tiếp (Live Demo).
  - Slide 5: Bài học kinh nghiệm & Hướng phát triển tương lai.
"""
    visual_intuition_content = """
```
Bộ Tứ Sản Phẩm Đầu Ra Hoàn Chỉnh (The Final 4 Deliverables):
┌───────────────────────────────┬───────────────────────────────┐
│ 1. GITHUB REPOSITORY CHUẨN    │ 2. INTERACTIVE DEMO APP       │
│ (Mã nguồn sạch, cấu trúc rõ)  │ (app.py chạy trực tiếp)       │
├───────────────────────────────┼───────────────────────────────┤
│ 3. BÁO CÁO KỸ THUẬT TOÀN DIỆN │ 4. SLIDE THUYẾT TRÌNH BẢO VỆ  │
│ (Bảng số liệu, Ablation study)│ (5-Minute Pitch chuẩn mực)    │
└───────────────────────────────┴───────────────────────────────┘
```
"""
    code_cells = [
        (
            """import os

# 1. TU DONG XUAT BAN FILE README.MD CHUAN QUOC TE CHO FINAL PROJECT
readme_content = \"\"\"# 🫀 AI-Powered Cardiovascular Disease Risk Assessment System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![VOAI Ready](https://img.shields.io/badge/Olympic%20AI-VOAI%20Certified-success.svg)]()

> **Author:** Freshman Olympic AI Athlete  
> **Course:** 38-Session Olympic AI Comprehensive Training Program  
> **Deliverable:** Final Portfolio Capstone Project (Module M7)

---

## 📌 1. Project Overview & Clinical Motivation
Bệnh tim mạch là một trong những nguyên nhân gây tử vong hàng đầu thế giới. Dự án này xây dựng một hệ thống Trí tuệ Nhân tạo hoàn chỉnh có khả năng đánh giá sớm nguy cơ tim mạch dựa trên các chỉ số sinh hiệu thông thường (Huyết áp, Tuổi, Chỉ số Cholesterol, Tình trạng Hút thuốc).

## 🏗️ 2. System Architecture
```
[Raw Patient Data] ──> [Schema Validator] ──> [Modular ColumnTransformer] ──> [Tuned Model Arena] ──> [Interactive Demo App]
```

## 📊 3. Experimental Benchmark Results
Kết quả thực nghiệm trên sơ đồ 5-Fold Cross-Validation nghiêm ngặt:
| Model Architecture | Features Used | Test Accuracy | Macro-F1 Score | Inference Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Logistic Reg)** | Raw Features | 78.5% | 0.7450 | **0.8 ms** |
| **Decision Tree (d=4)** | Raw Features | 81.2% | 0.7820 | 1.2 ms |
| **Random Forest Ensemble** | Full Engineered | 87.4% | 0.8560 | 4.5 ms |
| **Deep PyTorch MLP (Ours)** | Full + Regularized | **89.6%** | **0.8840** | 2.1 ms |

## 🚀 4. Quickstart & Reproducibility
Cài đặt và chạy thử nghiệm sản phẩm chỉ với 3 bước đơn giản:
```bash
# Bước 1: Clone repository
git clone https://github.com/username/ai-cardio-risk-assessment.git
cd ai-cardio-risk-assessment

# Bước 2: Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# Bước 3: Khởi động Ứng dụng Demo Tương tác
python app.py
```

## 📜 5. License & Academic Integrity
Dự án được phát hành dưới giấy phép MIT License. Toàn bộ mã nguồn tuân thủ nghiêm ngặt quy chế liêm chính học thuật của Olympic AI.
\"\"\"

readme_path = "portfolio_project/README.md"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)

print(f"-> Da xuat ban thanh cong file {readme_path} chuan quoc te!")
""",
            """Khối code tạo lập tự động file `README.md` chuyên nghiệp:
- Đầy đủ huy hiệu Badges, tổng quan đề tài, sơ đồ kiến trúc và bảng số liệu thực nghiệm.
- Có hướng dẫn 3 bước tái lập kết quả rõ ràng — tiêu chuẩn vàng của một GitHub Repository xuất sắc."""
        ),
        (
            """# 2. SCRIPT KIEM TRA TINH TOAN VEN CUA TOAN BO FINAL PROJECT (PACKAGE AUDITOR)
def audit_final_project_package():
    print("===== KHOI DONG HE THONG KIEM TRA TOAN DIEN FINAL PROJECT PACKAGE =====")
    required_files = [
        "portfolio_project/README.md",
        "portfolio_project/app.py",
        "portfolio_project/data",
        "portfolio_project/src",
        "portfolio_project/models",
        "portfolio_project/reports"
    ]
    
    all_passed = True
    for item in required_files:
        if os.path.exists(item):
            print(f"[EXISTS] -> {item:35s} : OK")
        else:
            print(f"[MISSING] -> {item:35s} : THIEU!")
            all_passed = False
            
    print("\\n" + "="*70)
    if all_passed:
        print("🎉 XAC NHAN: TOAN BO HO SO FINAL PROJECT HOAN TOAN HOP LE VA DAT CHUAN!")
        print("🏆 CHUC MUNG BAN DA HOAN THANH XUAT SAC TOAN BO 38 BUOI HOC OLYMPIC AI!")
    else:
        print("⚠️ CANH BAO: Vui long bo sung cac file con thieu truoc khi nop bai!")
    print("="*70)
    return all_passed

audit_final_project_package()
""",
            """Khối code chạy công cụ kiểm tra tự động toàn bộ gói sản phẩm của Final Project:
- Xác nhận sự hiện diện đầy đủ của toàn bộ các thành phần: README, App Demo, Dữ liệu, Mã nguồn và Thư mục Báo cáo.
- Đóng dấu hoàn tất trọn vẹn 38 buổi học của chương trình."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Soạn thảo Slide thuyết trình 5 phút
Mở một phần mềm trình chiếu (PowerPoint / Google Slides / Canva). Thiết kế 5 trang slide theo đúng cấu trúc đã nêu trong phần Concept. Xuất slide thành file `portfolio_project/reports/presentation_pitch.pdf`.

### Bài 2 (Easy): Bổ sung liên kết GitHub và Demo vào hồ sơ cá nhân
Thêm đường link dẫn tới GitHub Repository của dự án vào CV hoặc trang cá nhân LinkedIn của bạn. Viết một đoạn giới thiệu ngắn (2-3 câu) làm nổi bật dự án này trong Portfolio tuyển dụng.

### Bài 3 (Medium): Soạn thảo Báo cáo Kỹ thuật Toàn diện
Tạo file `portfolio_project/reports/TECHNICAL_REPORT.md` chứa đầy đủ các mục: Tóm tắt điều hành, Dữ liệu & Tiền xử lý, Kiến trúc mô hình, Bảng kết quả Ablation Study, Phân tích lỗi sai và Hướng phát triển trong tương lai.

### Bài 4 (Medium): Quay video Demo sản phẩm 2 phút
Sử dụng phần mềm quay màn hình, ghi lại một đoạn video ngắn 2 phút: Giới thiệu cấu trúc repo GitHub, chạy lệnh `python app.py`, nhập một vài thông số thử nghiệm và giải thích kết quả dự đoán của AI.

### Bài 5 (Challenge): Diễn tập Bảo vệ Dự án Trước Hội đồng (Mock Defense)
Đứng trước gương hoặc trình bày trước một người bạn trong đúng 5 phút. Tự trả lời 3 câu hỏi chất vấn hóc búa nhất:
1. *"Tại sao bạn không dùng mạng nơ-ron Transformer 100 triệu tham số cho bài này?"*
2. *"Mô hình của bạn có bị rò rỉ dữ liệu (Data Leakage) ở bước nào không và bạn đã chứng minh điều đó như thế nào?"*
3. *"Nếu đưa mô hình này vào bệnh viện thực tế thì rủi ro lớn nhất là gì?"*
"""
    voai_connection_content = """
Trong kỳ thi Olympic AI cấp quốc gia và các đấu trường quốc tế, vòng thi Chung kết xếp hạng luôn kết thúc bằng **Phần Thuyết trình Bảo vệ Giải pháp trước Hội đồng Giám khảo (Oral Defense Pitch)**. Điểm số từ bảng xếp hạng chỉ chiếm 60% tổng điểm; 40% điểm số còn lại được quyết định bởi chất lượng của Báo cáo Kỹ thuật, tính minh bạch của mã nguồn GitHub và bản lĩnh phản biện của thí sinh trước các câu hỏi học thuật. Nắm vững kỹ năng đóng gói và thuyết trình ở buổi hôm nay là bạn đã sẵn sàng bước lên bục nhận huy chương cao quý nhất!
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh đẩy toàn bộ mã nguồn lên GitHub nhưng quên tạo file `.gitignore`, khiến toàn bộ các file dữ liệu khổng lồ nặng hàng Gigabytes (`data/raw/large_file.csv`) hoặc file rác hệ điều hành (`.DS_Store`, `__pycache__`) bị đẩy lên repo, gây lỗi giới hạn dung lượng của GitHub (`File exceeds 100MB limit`):

```bash
# LỖI DO THIẾU FILE .GITIGNORE:
# remote: error: File data/raw/massive_data.csv is 145.00 MB; this exceeds GitHub's file size limit of 100.00 MB
```

**Nhiệm vụ của bạn:**
1. Tạo một file `portfolio_project/.gitignore` chuẩn mực chứa các mẫu bỏ qua: `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `data/raw/*`, `*.pt` nặng.
2. Bảo đảm repository GitHub luôn nhẹ nhàng, sạch sẽ và chuyên nghiệp.
"""
    knowledge_check_content = """
1. Bốn sản phẩm thành phần bắt buộc phải có trong hồ sơ Final Project tốt nghiệp Module M7 là gì?
   - A. Bốn bài tập trắc nghiệm trên giấy.
   - B. GitHub Repository chuẩn mã nguồn sạch, Báo cáo kỹ thuật toàn diện, Ứng dụng Demo tương tác (app.py), và Bài thuyết trình bảo vệ dự án (Presentation Pitch).
   - C. Bốn bức ảnh chụp phong cảnh.
   - D. Bốn tài khoản mạng xã hội.

2. Trang README.md của một dự án AI chuẩn quốc tế KHÔNG THỂ THIẾU phần nào sau đây?
   - A. Giá tiền của chiếc máy tính cá nhân.
   - B. Tổng quan bài toán, Sơ đồ kiến trúc hệ thống, Bảng số liệu thực nghiệm đối sánh, và Hướng dẫn chạy lại mã nguồn (Quickstart).
   - C. Danh sách các bài hát yêu thích.
   - D. Lịch thi đấu bóng đá.

3. Trong buổi thuyết trình bảo vệ dự án AI (5-minute pitch), thứ tự trình bày logic và thuyết phục nhất là gì?
   - A. Chiếu ngay code -> Nói về lỗi -> Kết luận -> Giới thiệu đề tài.
   - B. Nêu bài toán đời thực & Động lực -> Dữ liệu & Pipeline tiền xử lý -> Kết quả thực nghiệm đối sánh (Ablation Study) -> Trình diễn Live Demo -> Bài học kinh nghiệm.
   - C. Chỉ đọc slide từ đầu đến cuối không cần demo.
   - D. Nhờ trợ lý AI thuyết trình hộ.

4. File `.gitignore` trong một dự án Git có tác dụng gì?
   - A. Xóa toàn bộ repository của bạn.
   - B. Liệt kê các file tạm thời, file rác biên dịch (`__pycache__`) hoặc các file dữ liệu quá nặng để Git tự động bỏ qua, không theo dõi và không đẩy lên máy chủ GitHub.
   - C. Cài đặt các thư viện mới.
   - D. Đổi tên tác giả của dự án.

5. Sau khi hoàn thành xuất sắc Buổi 38 (M7_S05), học viên chính thức hoàn tất toàn bộ chương trình và vinh dự đón nhận chứng chỉ đỉnh cao nào?
   - A. Certificate of AI Orientation
   - B. Certificate of Machine Learning Basics
   - C. Certificate of CV/NLP Applied AI
   - D. Final Project Certificate (Chứng chỉ Tốt nghiệp Toàn khóa Olympic AI)
"""
    session_summary_content = """
* **What I learned:**
  - Tiêu chuẩn vàng đóng gói một GitHub Repository AI chuyên nghiệp đạt chuẩn tuyển dụng quốc tế.
  - Cấu trúc trang README.md và Báo cáo Kỹ thuật (Technical Report) có tính thuyết phục cao.
  - Nghệ thuật thuyết trình bảo vệ dự án 5 phút (5-Minute Defense Pitch).
  - Tự động kiểm tra tính toàn vẹn của toàn bộ hồ sơ dự án.
* **What I can now do:**
  - Tự tin sở hữu một Dự án AI hoàn chỉnh có Demo thực tế trong Portfolio cá nhân.
  - Sẵn sàng bước vào bất kỳ kỳ thi Olympic AI nào với tư duy sắc bén, kỹ năng lập trình vững vàng và chiến thuật bài bản của một nhà vô địch!
"""
    next_session_content = """
🎉 CHÚC MỪNG BẠN ĐÃ TỐT NGHIỆP TOÀN KHÓA 38 BUỔI OLYMPIC AI!
Hành trình 38 buổi học đã chính thức khép lại, nhưng con đường chinh phục đỉnh cao Trí tuệ Nhân tạo của bạn mới chỉ bắt đầu. Hãy giữ vững ngọn lửa đam mê, kỷ luật kỹ thuật và bản lĩnh của một Kỹ sư AI chân chính!
"""
    module_assessment = """
---

## 🏆 MODULE 7 FINAL ASSESSMENT & GRADUATION CERTIFICATION

### Phần A: Đánh giá Toàn diện Năng lực Kỹ sư AI (Comprehensive Assessment)
- Học viên đã tích hợp trọn vẹn tri thức của toàn bộ 38 buổi học: Từ Nền tảng Dữ liệu (M1), Học máy (M2), Học sâu PyTorch (M3), Ứng dụng CV/NLP (M4), Chiến thuật thi đấu Olympic (M5), đến Tư duy điều khiển LLM phản biện (M6).
- Thể hiện tư duy nghiên cứu khoa học có đối chứng (Ablation Study) và khả năng làm chủ công nghệ thực thụ.

### Phần B: Bộ Tứ Sản Phẩm Tốt Nghiệp Bắt Buộc (Final Capstone Deliverables)
1. **GitHub Repository Chuẩn Công Nghiệp:** Có cấu trúc thư mục rõ ràng, file `.gitignore`, `requirements.txt` và mã nguồn sạch sẽ, không lỗi cú pháp.
2. **Trang Giới Thiệu Chuyên Nghiệp (README.md):** Đầy đủ Badges, Sơ đồ kiến trúc, Bảng số liệu Benchmark và Hướng dẫn Quickstart 3 bước.
3. **Ứng Dụng Demo Tương Tác (app.py):** Chạy độc lập, nhận dữ liệu người dùng tùy biến và xuất kết quả phân loại kèm thanh đo độ tự tin.
4. **Báo Cáo Kỹ Thuật & Bài Thuyết Trình:** Hoàn thành tài liệu phân tích kỹ thuật và Slide thuyết trình bảo vệ dự án 5 phút.

### Bảng Tiêu chí Chấm điểm Tốt nghiệp (Final Graduation Rubric):
| Tiêu chí Đánh giá | Trọng số | Yêu cầu đạt chuẩn |
| :--- | :--- | :--- |
| **Cấu trúc GitHub & Mã nguồn sạch** | 25% | Repo chuẩn modular, mã nguồn chạy mượt mà từ đầu đến cuối |
| **Đánh giá Khoa học & Ablation Study**| 25% | Bảng so sánh đa metric, chứng minh đóng góp của từng thành phần |
| **Ứng dụng Demo Tương tác (app.py)**  | 25% | Chạy trực tiếp, hiển thị phần trăm tự tin và giao diện thân thiện |
| **README, Báo cáo & Thuyết trình**   | 25% | README chuẩn quốc tế, Báo cáo kỹ thuật chi tiết, Slide tự tin |

---

## 🎓 CHỨNG CHỈ TỐT NGHIỆP TOÀN KHÓA
🏅 **FINAL PROJECT CERTIFICATE — CERTIFICATE OF COMPLETION IN OLYMPIC AI & APPLIED MACHINE LEARNING**  
*Công nhận học viên đã hoàn thành xuất sắc toàn bộ 38 buổi huấn luyện chuyên sâu, làm chủ toàn bộ chuỗi mắt xích AI Pipeline và sở hữu sản phẩm AI hoàn chỉnh đạt chuẩn thi đấu quốc tế!*
"""
    trace = {
        "items": [
            "Đóng gói Final Project vào GitHub Repository chuẩn công nghiệp",
            "Xuất bản trang README.md và Báo cáo Kỹ thuật toàn diện",
            "Xây dựng Interactive Demo (app.py) và Slide thuyết trình bảo vệ",
            "Chính thức tốt nghiệp 38 buổi và nhận Final Project Certificate"
        ],
        "deliverable": "Bộ Tứ Sản Phẩm Tốt Nghiệp: GitHub Repo + README.md + Demo App (app.py) + Báo cáo Kỹ thuật & Slide Thuyết trình"
    }
    nb = create_notebook(
        module_id="M7",
        module_name="DỰ ÁN AI THẬT ĐỂ ĐƯA VÀO PORTFOLIO",
        session_id="S05",
        topic="Hoàn Thiện Portfolio: Đóng Gói GitHub Repo, Báo Cáo & Thuyết Trình",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Bộ Tứ Sản Phẩm: GitHub Repo + README.md + Demo App + Báo Cáo Kỹ Thuật",
        competition_relevance="Đích đến vinh quang: Đóng gói toàn bộ năng lực thành Portfolio tuyển dụng và thi đấu.",
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
    save_notebook(nb, "course/M7_FINAL_PROJECT/M7_S05.ipynb")

if __name__ == "__main__":
    build_m7_s01()
    build_m7_s02()
    build_m7_s03()
    build_m7_s04()
    build_m7_s05()
    print("M7 build complete! 5/5 notebooks generated.")
