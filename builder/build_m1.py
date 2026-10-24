"""
builder/build_m1.py
Generates M1_S01.ipynb to M1_S05.ipynb for Module 1.
"""

from utils import create_notebook, save_notebook

def build_m1_s01():
    objectives = [
        "Hiểu sự khác biệt về cấu trúc bộ nhớ và tốc độ giữa Python List và NumPy ndarray.",
        "Làm chủ kỹ thuật Vectorization (vectơ hóa), loại bỏ hoàn toàn các vòng lặp for chậm chạp trong xử lý dữ liệu số.",
        "Nắm vững cơ chế Broadcasting (lan truyền chiều) trong các phép tính toán ma trận.",
        "Thực hành tính toán thống kê và trích xuất dữ liệu đa chiều bằng Boolean Masking."
    ]
    prerequisites = [
        "Hoàn thành Module M0.",
        "Biết cú pháp cơ bản của Python (biến, vòng lặp, hàm)."
    ]
    why_content = """
Trong thi đấu Olympic AI, bạn thường xuyên phải làm việc với các bảng dữ liệu hàng triệu dòng hoặc tensor hình ảnh có hàng chục triệu pixel. Nếu xử lý bằng vòng lặp `for` thuần trong Python (Python loop), thời gian thực thi có thể mất hàng chục phút, dẫn tới Time Limit Exceeded (TLE).

NumPy là "trái tim" tính toán số học của toàn bộ hệ sinh thái AI (Scikit-Learn, PyTorch, TensorFlow đều xây dựng trên nền tảng ý tưởng của NumPy). Bằng cách lưu trữ dữ liệu liên tục trong bộ nhớ C (contiguous memory) và tận dụng chỉ thị phần cứng SIMD (Single Instruction, Multiple Data), NumPy tăng tốc độ xử lý lên gấp **50 đến 100 lần** so với Python thuần.
"""
    concept_content = r"""
### 1. Python List vs NumPy ndarray
* **Python List:** Là một mảng chứa các con trỏ (pointers) trỏ tới các đối tượng Python phân tán rải rác trong bộ nhớ Heap. Mỗi phần tử phải mang theo metadata kiểu dữ liệu (overhead lớn, cache miss liên tục).
* **NumPy ndarray:** Là một khối bộ nhớ liền kề (contiguous block of memory) chứa các phần tử đồng nhất kiểu (homogeneous data type, ví dụ `float32` hoặc `int64`). Không tốn chi phí kiểm tra kiểu động tại runtime.

### 2. Nguyên lý Vectorization & Broadcasting
* **Vectorization:** Thay vì viết vòng lặp duyệt qua từng phần tử:
  ```python
  # Cach cu (Slow)
  c = [a[i] + b[i] for i in range(len(a))]
  # Vectorized (Fast - chay o tang C)
  c = a + b
  ```
* **Broadcasting Rules:** Cho phép thực hiện phép toán số học giữa hai mảng có kích thước khác nhau nếu:
  1. Hai chiều bằng nhau, HOẶC
  2. Một trong hai chiều có kích thước bằng 1.
"""
    visual_intuition_content = """
```
Broadcasting Intuition:
Ma trận A (3x3):         Vector B (1x3):       Kết quả A + B (3x3):
[[1, 2, 3],             [[10, 20, 30]]        [[11, 22, 33],
 [4, 5, 6],         +   (Tự nhân bản       =   [14, 25, 36],
 [7, 8, 9]]              thành 3 dòng)         [17, 28, 39]]
```
"""
    code_cells = [
        (
            """import numpy as np
import time

# 1. So sanh hieu nang: Python List vs NumPy Array tren 1,000,000 phan tu
N = 1_000_000
list_a = list(range(N))
list_b = list(range(N))

arr_a = np.arange(N, dtype=np.float64)
arr_b = np.arange(N, dtype=np.float64)

# Do thoi gian Python list
start_time = time.time()
list_c = [list_a[i] + list_b[i] for i in range(N)]
python_time = time.time() - start_time

# Do thoi gian NumPy Vectorization
start_time = time.time()
arr_c = arr_a + arr_b
numpy_time = time.time() - start_time

speedup = python_time / numpy_time
print(f"Thoi gian Python List : {python_time:.5f} giay")
print(f"Thoi gian NumPy Array : {numpy_time:.5f} giay")
print(f"-> NumPy nhanh hon gap : {speedup:.1f} lan!")
""",
            """Khối code đo lường thời gian thực thi phép cộng 1 triệu phần tử giữa danh sách Python thuần và mảng NumPy.
- **Input:** 2 mảng $10^6$ số nguyên.
- **Output:** Thời gian thực thi và hệ số tăng tốc (Speedup factor).
- **Tại sao viết như vậy:** Chứng minh bằng thực nghiệm trực quan rằng Vectorization không chỉ là thói quen viết code gọn, mà là bắt buộc để sống sót trong xử lý dữ liệu lớn."""
        ),
        (
            """# 2. Thao tac mang da chieu, Broadcasting va Boolean Masking
# Tao ma tran diem thi cua 5 thi sinh qua 3 bai thi Olympic
scores = np.array([
    [75, 82, 90],
    [60, 65, 70],
    [95, 88, 92],
    [50, 45, 55],
    [85, 90, 88]
])

print("Ma tran diem goc (Shape: 5x3):\\n", scores)

# Tinh diem trung binh cua tung thi sinh (tinh theo truc ngang - axis=1)
student_means = np.mean(scores, axis=1, keepdims=True)
print("\\nDiem trung binh tung thi sinh:\\n", student_means)

# Broadcasting: Chuan hoa diem bang cach tru di diem trung binh cua tung bai (axis=0)
exam_means = np.mean(scores, axis=0)
centered_scores = scores - exam_means
print("\\nDiem da tru trung vi mon thi (Centered):\\n", centered_scores)

# Boolean Masking: Loc ra tat ca cac diem >= 85
high_scores = scores[scores >= 85]
print("\\nCac diem dat loai Gioi (>= 85):", high_scores)
""",
            """Khối code minh họa các thao tác cốt lõi trên ma trận:
- `axis=1` (tính dọc theo cột để thu về trung bình từng dòng) và `axis=0` (tính dọc theo dòng để thu về trung bình từng cột).
- `keepdims=True` giữ nguyên số chiều 2D để sẵn sàng cho Broadcasting.
- **Boolean Masking:** `scores[scores >= 85]` lọc trực tiếp các phần tử thỏa mãn điều kiện mà không cần duyệt vòng lặp `for`."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Tạo mảng và biến đổi chiều
Tạo một mảng NumPy 1D gồm 24 số từ 1 đến 24 (`np.arange(1, 25)`), sau đó biến đổi thành ma trận 2D có kích thước `(4, 6)` và ma trận 3D có kích thước `(2, 3, 4)`.

### Bài 2 (Easy): Chuẩn hóa Min-Max bằng Vectorization
Cho một mảng 1D `x = np.array([12, 45, 67, 89, 34, 23, 90, 11])`. Hãy viết công thức vectơ hóa tính Chuẩn hóa Min-Max:
$$x_{norm} = \\frac{x - x_{min}}{x_{max} - x_{min}}$$
Đảm bảo kết quả nằm trong đoạn $[0, 1]$.

### Bài 3 (Medium): Thay thế phần tử theo điều kiện
Cho một ma trận dữ liệu số học ngẫu nhiên `data = np.random.randn(5, 5)`. Hãy thay thế tất cả các giá trị âm (`< 0`) bằng số `0` chỉ bằng một dòng lệnh duy nhất (tương đương hàm kích hoạt ReLU).

### Bài 4 (Medium): Tính khoảng cách Euclid giữa các điểm dữ liệu
Cho tập 3 điểm trong không gian 2D `points = np.array([[0, 0], [3, 4], [1, 1]])` và điểm truy vấn `query = np.array([1, 2])`. Dùng Broadcasting để tính khoảng cách Euclid từ điểm truy vấn tới tất cả các điểm trong `points`.

### Bài 5 (Challenge): Tối ưu hóa phép nhân ma trận thủ công
Viết phép nhân ma trận giữa hai ma trận $A (N \\times K)$ và $B (K \\times M)$ bằng Python loops, so sánh thời gian chạy với toán tử `@` (`np.dot(A, B)`) khi $N=500, K=500, M=500$.
"""
    voai_connection_content = """
Trong phần thi Competitive AI, khi cần tạo ra các đặc trưng tương tác (interaction features) như tỷ lệ giữa hai cột hay khoảng cách không gian giữa hàng nghìn tọa độ, việc viết vòng lặp `for` sẽ khiến notebook của bạn vượt ngưỡng giới hạn thời gian chạy (timeout). Làm chủ NumPy Vectorization là kỹ năng sống còn giúp bạn xử lý các phép biến đổi dữ liệu tức thời trong chớp mắt.
"""
    debugging_content = """
**Tình huống thực tế:** Một thí sinh cố gắng cộng một vector trọng số vào ma trận dữ liệu nhưng bị báo lỗi `ValueError: operands could not be broadcast together with shapes (4, 3) (4,)`:

```python
# CODE CÓ LỖI BROADCASTING:
X = np.ones((4, 3))   # 4 dong, 3 cot
bias = np.array([1, 2, 3, 4]) # 4 phan tu

# Thi sinh muon cong bias vao tung cot (moi dong duoc cong bias tuong ung):
# result = X + bias # Gay ra loi!
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao `bias` có shape `(4,)` không thể broadcast tự động với `(4, 3)`.
2. Sửa lỗi bằng cách sử dụng `np.reshape`, `bias[:, np.newaxis]` hoặc `.reshape(4, 1)` để thực hiện phép cộng chính xác.
"""
    knowledge_check_content = """
1. Lý do cơ bản nhất giúp NumPy nhanh hơn nhiều lần so với Python List thuần là gì?
   - A. NumPy chạy trên card đồ họa GPU theo mặc định.
   - B. NumPy lưu trữ dữ liệu liên tục trong bộ nhớ, kiểu dữ liệu đồng nhất và thực thi ở tầng ngôn ngữ C tối ưu.
   - C. NumPy bỏ qua việc kiểm tra độ chính xác số học.
   - D. NumPy tự động xóa các biến rác trong máy tính.

2. Phép toán `np.mean(matrix, axis=0)` trên một ma trận 2D có kích thước $(10, 5)$ sẽ cho ra kết quả có kích thước nào?
   - A. $(10, 1)$
   - B. Một số vô hướng duy nhất
   - C. $(5,)$
   - D. $(10, 5)$

3. Trong NumPy, biểu thức `arr[arr % 2 == 0]` được gọi là kỹ thuật gì?
   - A. Integer Fancy Indexing
   - B. Slicing đa chiều
   - C. Boolean Masking
   - D. Data Imputation

4. Hai mảng có kích thước $(8, 1, 6, 1)$ và $(7, 1, 5)$ có thể broadcast với nhau được không?
   - A. Có thể, vì kích thước tổng bằng nhau.
   - B. Có thể, theo quy tắc căn lề từ phải qua trái.
   - C. Không thể, vì các chiều tương ứng không bằng nhau và không có chiều nào bằng 1.
   - D. Chỉ broadcast được nếu mảng có kiểu float32.

5. Cú pháp nào là chuẩn mực trong Python hiện đại để thực hiện phép nhân ma trận giữa hai mảng 2D $A$ và $B$?
   - A. `A * B`
   - B. `A @ B`
   - C. `A.multiply(B)`
   - D. `A ^ B`
"""
    session_summary_content = """
* **What I learned:**
  - Bản chất tối ưu hóa bộ nhớ của NumPy so với Python List.
  - Sức mạnh của Vectorization và luật Broadcasting đa chiều.
  - Cách dùng Boolean Masking để trích xuất và lọc dữ liệu tốc độ cao.
* **What I can now do:**
  - Viết code xử lý mảng số học chuẩn C mà không cần viết vòng lặp `for`.
  - Tự tin thực hiện các phép biến đổi ma trận, tính toán thống kê theo trục `axis=0` và `axis=1`.
* **What I still need to practice:**
  - Chuyển giao tư duy mảng số học sang cấu trúc bảng dữ liệu có nhãn cột trong **Pandas**.
"""
    next_session_content = """
Trong **Buổi M1_S02**, chúng ta sẽ tiến lên một bậc: Sử dụng **Pandas Core** để nạp dữ liệu từ các file thực tế (CSV/Excel), làm quen với Series & DataFrame, và làm chủ nghệ thuật truy vấn lọc dữ liệu phức tạp (`loc`, `iloc`, boolean indexing).
"""
    trace = {
        "items": [
            "Python dành cho data & cấu trúc bộ nhớ",
            "NumPy basics & Vectorization",
            "Broadcasting & Boolean Masking"
        ],
        "deliverable": "Mã nguồn tính toán số học tốc độ cao trên mảng đa chiều"
    }
    nb = create_notebook(
        module_id="M1",
        module_name="NÓI CHUYỆN VỚI DỮ LIỆU BẰNG PYTHON",
        session_id="S01",
        topic="Python cho Data & Mảng Đa Chiều với NumPy",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook tối ưu hóa tính toán số học bằng NumPy Vectorization",
        competition_relevance="Xử lý nhanh dữ liệu mảng, chuẩn bị đặc trưng không nghẽn cổ chai thời gian.",
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
    save_notebook(nb, "course/M1_PYTHON_DATA/M1_S01.ipynb")


def build_m1_s02():
    objectives = [
        "Hiểu cấu trúc 2 tầng của Pandas: Series (1D có chỉ mục) và DataFrame (2D bảng dữ liệu).",
        "Đọc và kiểm tra nhanh chóng file dữ liệu CSV bằng `info()`, `describe()`, `head()`, `shape`.",
        "Phân biệt rạch ròi và làm chủ phương thức truy vấn vị trí `.iloc` vs nhãn `.loc`.",
        "Xây dựng các bộ lọc điều kiện phức tạp (Complex Conditional Filtering) bằng toán tử logic `&`, `|`, `~`."
    ]
    prerequisites = [
        "Hoàn thành Buổi M1_S01 (NumPy & Vectorization).",
        "Hiểu cơ bản về cấu trúc file bảng (CSV, hàng và cột)."
    ]
    why_content = """
Dữ liệu đời thực và dữ liệu thi Olympic AI không bao giờ đến dưới dạng các ma trận số sạch sẽ. Chúng nằm trong các file bảng (CSV, Parquet, SQL) chứa hỗn hợp nhiều kiểu dữ liệu: tên thí sinh (string), độ tuổi (int), mức lương (float), thời gian nộp bài (datetime).

NumPy chỉ làm việc tốt với dữ liệu đồng nhất một kiểu. Để "nói chuyện" được với dữ liệu bảng phức tạp này, bạn bắt buộc phải sử dụng **Pandas**. Trong thi đấu AI, 70% thời gian của một chuyên gia dữ liệu là thao tác với Pandas để bóc tách, chọn lọc và biến đổi các dòng dữ liệu tiềm năng.
"""
    concept_content = """
### 1. Series vs DataFrame trong Pandas
* **Pandas Series:** Là mảng một chiều có gán nhãn chỉ mục (Indexed 1D Array). Bạn có thể truy cập phần tử qua nhãn tên hoặc qua số thứ tự.
* **Pandas DataFrame:** Là bảng dữ liệu hai chiều có cấu trúc dạng bảng (2D Table), gồm nhiều Series có chung Index ghép lại với nhau. Mỗi cột là một Series đại diện cho một đặc trưng (Feature).

### 2. Hai quy tắc cốt tử khi truy xuất: `.loc` vs `.iloc`
* **`.loc[row_label, col_label]` (Label-based):** Truy xuất dựa trên **Tên nhãn** của hàng và cột.
  - Ví dụ: `df.loc[0:5, ['Age', 'Score']]` lấy các dòng có index mang tên từ 0 đến 5 (bao gồm cả 5) và các cột có tên 'Age', 'Score'.
* **`.iloc[row_pos, col_pos]` (Integer position-based):** Truy xuất dựa trên **Vị trí chỉ số nguyên** (từ 0 đến $N-1$), tuân theo quy tắc cắt lát chuẩn của Python (không bao gồm cận cuối).
  - Ví dụ: `df.iloc[0:5, 1:3]` lấy đúng 5 dòng đầu tiên (0, 1, 2, 3, 4) và 2 cột ở vị trí 1, 2.

### 3. Lọc dữ liệu theo điều kiện (Boolean Indexing)
Trong Pandas, ta dùng toán tử bitwise:
- `&` thay cho `and`
- `|` thay cho `or`
- `~` thay cho `not`
*Bắt buộc phải bọc từng biểu thức điều kiện trong dấu ngoặc đơn `()` để đảm bảo thứ tự ưu tiên toán tử!*
"""
    visual_intuition_content = """
```
Cấu trúc Pandas DataFrame:
                 Columns: ['Name', 'Age', 'Score']
                    │        │       │
      Index (Row)   ▼        ▼       ▼
           0     ['An',     19,     88.5]
           1     ['Bình',   20,     92.0]  <─── Row 1 (df.iloc[1])
           2     ['Cúc',    18,     79.0]
```
"""
    code_cells = [
        (
            """import pandas as pd
import numpy as np

# 1. Tao tap du lieu mau mo phong ho so thi sinh thi Olympic AI
np.random.seed(42)
n_samples = 10

data = {
    'Student_ID': [f'VOAI_{i:03d}' for i in range(1, n_samples + 1)],
    'Name': ['An', 'Binh', 'Chi', 'Dung', 'Em', 'Giang', 'Hoa', 'Khoa', 'Lan', 'Minh'],
    'Region': ['North', 'South', 'North', 'Central', 'South', 'North', 'Central', 'South', 'North', 'Central'],
    'Age': np.random.randint(18, 22, size=n_samples),
    'Math_Score': np.random.uniform(6.0, 10.0, size=n_samples).round(1),
    'Code_Score': np.random.uniform(5.0, 9.8, size=n_samples).round(1),
    'Passed_Round1': np.random.choice([True, False], size=n_samples, p=[0.7, 0.3])
}

df = pd.DataFrame(data)
print("===== 5 DONG DAU TIEN CUA DATAFRAME =====")
display_df = df.head()
print(display_df)

print("\\n===== THONG TIN CAU TRUC DU LIEU (INFO) =====")
df.info()
""",
            """Khối code tạo lập một DataFrame mẫu gồm 10 thí sinh với đa dạng kiểu dữ liệu (`object`, `int64`, `float64`, `bool`).
- Dùng `head()` để xem nhanh dữ liệu đại diện.
- Dùng `info()` để kiểm tra số lượng dòng, số cột non-null và kiểu dữ liệu (Dtype) của từng cột.
- **Tại sao viết như vậy:** Thói quen đầu tiên khi nhận bất kỳ dataset nào là phải chạy `head()` và `info()` để nắm cấu trúc tổng thể."""
        ),
        (
            """# 2. Thao tac truy van loc du lieu: .loc, .iloc va Boolean Indexing

# Truong hop 1: Dung .iloc de lay 3 dong dau tien va 2 cot: Student_ID, Name (vi tri 0 va 1)
subset_iloc = df.iloc[0:3, 0:2]
print("--- Truy van bang .iloc (3 dong dau, cot 0-1) ---")
print(subset_iloc)

# Truong hop 2: Loc cac thi sinh o mien Bac ('North') VA co Math_Score >= 8.0
filter_cond = (df['Region'] == 'North') & (df['Math_Score'] >= 8.0)
top_north_students = df.loc[filter_cond, ['Student_ID', 'Name', 'Math_Score', 'Code_Score']]

print("\\n--- Thi sinh mien Bac co diem Toan >= 8.0 (Boolean Indexing + .loc) ---")
print(top_north_students)

# Truong hop 3: Tinh cot moi: Tong diem (Total_Score) bang Vectorization trong Pandas
df['Total_Score'] = df['Math_Score'] + df['Code_Score']
print("\\n--- Bang du lieu sau khi them cot Total_Score ---")
print(df[['Student_ID', 'Name', 'Total_Score']].head())
""",
            """Khối code thực thi truy vấn chi tiết:
- Phân biệt rõ việc trích xuất bằng chỉ số vị trí `.iloc` và trích xuất bằng điều kiện kèm nhãn `.loc`.
- Thể hiện cú pháp lọc chuẩn mực với toán tử `&` và dấu ngoặc tròn `()`.
- Thao tác thêm một cột đặc trưng mới `Total_Score` được tính toán song song tức thì."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Đọc và kiểm tra tóm tắt thống kê
Chạy lệnh `df.describe()` trên DataFrame vừa tạo. Nêu rõ ý nghĩa của các dòng: `mean`, `std`, `50%` (trung vị).

### Bài 2 (Easy): Lọc theo điều kiện chuỗi
Lọc ra tất cả các thí sinh thuộc khu vực miền Nam (`'South'`) HOẶC có điểm lập trình (`Code_Score`) lớn hơn 8.5.

### Bài 3 (Medium): Truy vấn bằng phương thức `.query()`
Viết lại câu truy vấn ở Bài 2 bằng cách sử dụng cú pháp phương thức chuỗi `df.query(...)`. So sánh ưu nhược điểm về độ dễ đọc của `.query()` so với Boolean Masking truyền thống.

### Bài 4 (Medium): Sắp xếp thứ hạng (Sorting)
Sắp xếp bảng dữ liệu theo thứ tự giảm dần của `Total_Score`. Nếu điểm tổng bằng nhau, ưu tiên bạn có `Math_Score` cao hơn. Đặt lại chỉ mục (Reset Index) sau khi sắp xếp.

### Bài 5 (Challenge): Tạo cờ phân loại (Flag Feature)
Tạo thêm một cột mới tên là `'Grade'` trong DataFrame với quy tắc:
- `'Xuất sắc'` nếu `Total_Score >= 16.5`
- `'Khá'` nếu `14.0 <= Total_Score < 16.5`
- `'Trung bình'` nếu `Total_Score < 14.0`
*Yêu cầu:* Thực hiện bằng cách sử dụng `pd.cut()` hoặc hàm custom kết hợp `.apply()`.
"""
    voai_connection_content = """
Đề thi Olympic AI thường cung cấp file `train.csv` và `test.csv`. Một sai lầm phổ biến của thí sinh là dùng vòng lặp duyệt từng dòng để tính toán hoặc lọc dữ liệu, khiến notebook chạy mất 10 phút chỉ cho bước đọc dữ liệu ban đầu. Thành thạo `.loc`, `.iloc` và Boolean Filtering giúp bạn chỉ mất 2 dòng code để trích xuất tập dữ liệu con cần thiết trong chưa đầy 1 giây.
"""
    debugging_content = """
**Tình huống thực tế:** Một sinh viên viết điều kiện lọc dữ liệu và gặp lỗi `ValueError: The truth value of a Series is ambiguous`:

```python
# CODE CÓ LỖI SYNTAX LOGIC TRONG PANDAS:
# Thi sinh muon loc hoc vien co Age > 18 va Math_Score > 7.0
# bug_query = df[df['Age'] > 18 and df['Math_Score'] > 7.0] # Bao loi ngay lap tuc!
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao từ khóa `and` của Python thuần gây ra lỗi trên khi áp dụng cho hai Pandas Series.
2. Viết lại biểu thức lọc chuẩn xác bằng toán tử bitwise và cặp ngoặc đơn.
"""
    knowledge_check_content = """
1. Sự khác biệt căn bản giữa `.loc` và `.iloc` trong Pandas là gì?
   - A. `.loc` chạy nhanh hơn `.iloc` gấp 10 lần.
   - B. `.loc` dựa trên nhãn tên (Labels), còn `.iloc` dựa trên vị trí chỉ số nguyên (Integer positions).
   - C. `.iloc` chỉ dùng được cho cột, không dùng được cho dòng.
   - D. Không có sự khác biệt nào.

2. Cú pháp nào sau đây là ĐÚNG khi lọc các dòng có cột `'A' > 10` và cột `'B' < 5`?
   - A. `df[df['A'] > 10 and df['B'] < 5]`
   - B. `df[(df['A'] > 10) & (df['B'] < 5)]`
   - C. `df[(df['A'] > 10) && (df['B'] < 5)]`
   - D. `df[df['A'] > 10 , df['B'] < 5]`

3. Phương thức `df.info()` KHÔNG cung cấp thông tin nào sau đây?
   - A. Tổng số lượng hàng và cột.
   - B. Tên và kiểu dữ liệu (Dtype) của từng cột.
   - C. Số lượng giá trị không bị khuyết (Non-null count).
   - D. Giá trị dự đoán của mô hình AI.

4. Đoạn code `df.iloc[0:3]` sẽ trả về bao nhiêu hàng?
   - A. 2 hàng (hàng 0 và 1)
   - B. 3 hàng (hàng 0, 1 và 2)
   - C. 4 hàng (hàng 0, 1, 2 và 3)
   - D. Báo lỗi Indexing

5. Khi đọc một file dữ liệu lớn từ đĩa, phương thức nào của Pandas được sử dụng phổ biến nhất?
   - A. `pd.read_file()`
   - B. `pd.open_csv()`
   - C. `pd.read_csv()`
   - D. `pd.load_table()`
"""
    session_summary_content = """
* **What I learned:**
  - Nắm vững cấu trúc Series và DataFrame.
  - Sử dụng thành thạo `info()`, `describe()`, `head()` để kiểm tra dataset.
  - Phân biệt tuyệt đối `.loc` và `.iloc`.
  - Thực thi Boolean Masking với các toán tử logic `&`, `|`, `~`.
* **What I can now do:**
  - Tải file bảng dữ liệu và bóc tách các tập dữ liệu con theo yêu cầu nghiệp vụ.
  - Tạo các cột đặc trưng mới một cách an toàn và nhanh chóng.
* **What I still need to practice:**
  - Học cách đối mặt với dữ liệu bẩn: Missing values, sai kiểu dữ liệu và trùng lặp ở Buổi 3.
"""
    next_session_content = """
Trong **Buổi M1_S03**, chúng ta sẽ tấn công vào bài toán thực tế nhất của Data Science: **Làm sạch dữ liệu (Data Cleaning)** — Xử lý các giá trị khuyết (Missing Values / NaNs), ép đúng kiểu dữ liệu (Data Types Casting), và loại bỏ trùng lặp (Duplicates) để bảo toàn tính trung thực của dữ liệu thi đấu!
"""
    trace = {
        "items": [
            "Đọc dữ liệu CSV & kiểm tra DataFrame",
            "Cấu trúc Series & DataFrame",
            "Indexing & Filtering bằng .loc, .iloc và Boolean conditions"
        ],
        "deliverable": "Mã nguồn truy vấn, trích xuất và lọc tập con dữ liệu chuẩn xác"
    }
    nb = create_notebook(
        module_id="M1",
        module_name="NÓI CHUYỆN VỚI DỮ LIỆU BẰNG PYTHON",
        session_id="S02",
        topic="Pandas Core: Đọc Dữ Liệu, DataFrame & Indexing/Filtering",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook truy vấn và lọc dữ liệu bảng chuyên nghiệp bằng Pandas",
        competition_relevance="Khám phá cấu trúc đề thi, trích lọc dữ liệu nhanh không bị lỗi indexing.",
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
    save_notebook(nb, "course/M1_PYTHON_DATA/M1_S02.ipynb")


def build_m1_s03():
    objectives = [
        "Phát hiện và định lượng tỷ lệ dữ liệu khuyết (Missing Values / NaNs) trên từng cột.",
        "Hiểu và áp dụng đúng đắn các chiến lược điền khuyết (Imputation): Mean, Median, Mode, hoặc Constant.",
        "Nhận diện và sửa lỗi sai lệch kiểu dữ liệu (Data Types Mismatch) bằng `.astype()` và `pd.to_datetime()`.",
        "Phát hiện, phân tích và loại bỏ các dòng dữ liệu trùng lặp (Duplicate Records)."
    ]
    prerequisites = [
        "Hoàn thành Buổi M1_S02 (Pandas Core & Indexing).",
        "Hiểu khái niệm giá trị rỗng (None / NaN) trong lập trình."
    ]
    why_content = """
Trong phòng thi Olympic AI, ban tổ chức gần như không bao giờ cung cấp một bộ dữ liệu hoàn hảo. Thậm chí, họ cố tình đưa vào các giá trị bị khuyết (`NaN`), các cột số bị lưu nhầm dưới dạng chuỗi văn bản (`object`), hoặc các bản ghi bị sao chép trùng lặp.

Hầu hết các thuật toán Machine Learning (như Linear Regression, SVM, mạng nơ-ron PyTorch) sẽ **lập tức báo lỗi Crash** nếu dữ liệu đầu vào chứa dù chỉ một giá trị `NaN`. Nếu bạn xử lý ẩu bằng cách xóa sạch mọi dòng có NaN, bạn có thể làm mất đi 50% dữ liệu quý giá. Làm chủ kỹ thuật làm sạch dữ liệu là rào chắn đầu tiên bảo vệ mô hình của bạn.
"""
    concept_content = """
### 1. Cơ chế Missing Data trong Pandas
* Giá trị khuyết thường được biểu diễn bằng `np.nan` (Not a Number - kiểu float) hoặc `None`.
* Phát hiện: `df.isna()` hoặc `df.isnull()`.
* Đếm số lượng khuyết: `df.isna().sum()`.
* Tỷ lệ phần trăm khuyết: `df.isna().mean() * 100`.

### 2. Các chiến lược xử lý dữ liệu khuyết (Imputation Strategies)
1. **Xóa bỏ (Dropping):** `df.dropna()`.
   - *Khi nào nên dùng:* Khi tỷ lệ khuyết quá nhỏ (< 1-2%) và tập dữ liệu đủ lớn, HOẶC khi một cột bị khuyết trên 70% không còn giá trị thông tin.
2. **Điền giá trị trung tâm (Mean / Median Imputation):**
   - Dùng `Mean` khi dữ liệu có phân phối chuẩn (chuông đối xứng).
   - Dùng `Median` khi dữ liệu có phân phối lệch (Skewed) hoặc có ngoại lai (Outliers).
3. **Điền giá trị xuất hiện nhiều nhất (Mode Imputation):**
   - Bắt buộc dùng cho biến phân loại (Categorical data, ví dụ: Thành phố, Giới tính).
4. **Cảnh báo Data Leakage:** Khi điền khuyết, giá trị Mean/Median phải được tính **chỉ trên tập Train**, sau đó mới áp dụng giá trị đó để điền cho tập Test/Validation!

### 3. Xử lý kiểu dữ liệu (Type Casting) & Trùng lặp (Duplicates)
* Đổi kiểu dữ liệu: `df['col'] = df['col'].astype(int)` hoặc `pd.to_numeric(df['col'], errors='coerce')`.
* Kiểm tra trùng lặp: `df.duplicated().sum()`.
* Xóa trùng lặp: `df.drop_duplicates(keep='first', inplace=True)`.
"""
    visual_intuition_content = """
```
Quy trình làm sạch dữ liệu chuẩn (Cleaning Pipeline):
[Raw Data] ──> [Phát hiện NaNs] ──> [Imputation / Drop] 
                 │
                 ├──> [Ép đúng Dtype (str -> float)] 
                 │
                 └──> [Xóa Duplicates] ──> [Clean Dataset sẵn sàng cho ML]
```
"""
    code_cells = [
        (
            """import pandas as pd
import numpy as np

# 1. Tao tap du lieu "ban" mo phong thuc te
raw_data = {
    'Candidate_ID': ['C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C03'], # C03 bi trung lap
    'Age': [19, np.nan, 21, 20, np.nan, 22, 21],
    'Salary_Expectation': ['$1,200', '$1,500', '$1,100', 'MISSING', '$2,000', '$1,800', '$1,100'], # Co ky tu '$' va 'MISSING'
    'City': ['Hanoi', 'HCMC', np.nan, 'Danang', 'Hanoi', 'HCMC', np.nan],
    'Score': [85.0, 90.0, 78.0, np.nan, 88.0, 92.0, 78.0]
}

df_dirty = pd.DataFrame(raw_data)
print("===== DU LIEU THO BAN DAU =====")
print(df_dirty)

print("\\n===== THONG KE SO LUONG GIA TRI KHUYET (NANS) =====")
print(df_dirty.isna().sum())
""",
            """Khối code tạo dựng một DataFrame chứa đầy đủ các "bệnh" điển hình của dữ liệu thực tế:
- Cột `Age` và `Score` bị khuyết `NaN`.
- Cột `City` bị khuyết dữ liệu dạng chuỗi (Categorical).
- Cột `Salary_Expectation` chứa ký tự đặc biệt `$` và chữ `'MISSING'`, khiến Pandas nhận diện nhầm thành kiểu `object` (chuỗi).
- Bản ghi `C03` bị lặp lại 2 lần."""
        ),
        (
            """# 2. Pipeline lam sach du lieu toan dien
df_clean = df_dirty.copy()

# Buoc 1: Loai bo ban ghi trung lap (Duplicates)
before_dedup = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=['Candidate_ID'], keep='first')
print(f"Da loai bo {before_dedup - len(df_clean)} dong trung lap.")

# Buoc 2: Xu ly cot Salary_Expectation: Bo '$', thay 'MISSING' thanh NaN roi ep sang float
df_clean['Salary_Expectation'] = (
    df_clean['Salary_Expectation']
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False)
    .replace('MISSING', np.nan)
)
df_clean['Salary_Expectation'] = pd.to_numeric(df_clean['Salary_Expectation'])

# Buoc 3: Dien khuyet (Imputation)
# Dien Age bang Median
median_age = df_clean['Age'].median()
df_clean['Age'] = df_clean['Age'].fillna(median_age)

# Dien City bang Mode (Gia tri xuat hien nhieu nhat)
mode_city = df_clean['City'].mode()[0]
df_clean['City'] = df_clean['City'].fillna(mode_city)

# Dien Score bang Mean
mean_score = df_clean['Score'].mean()
df_clean['Score'] = df_clean['Score'].fillna(mean_score)

print("\\n===== BANG DU LIEU SAU KHI LAM SACH HOAN TOAN =====")
print(df_clean)
print("\\nKiem tra lai so luong NaNs con lai:", df_clean.isna().sum().sum())
""",
            """Khối code thực thi toàn bộ pipeline làm sạch một cách tuần tự:
- Khử trùng lặp dựa trên khóa chính `Candidate_ID`.
- Sử dụng chuỗi xử lý chuỗi (`.str.replace`) và `pd.to_numeric` để chuyển hóa một cột rác về đúng dạng số thực `float64`.
- Sử dụng chính xác chiến lược điền khuyết: `Median` cho tuổi, `Mode` cho địa phương, `Mean` cho điểm thi.
- Kiểm tra tính toàn vẹn ở bước cuối cùng (`isna().sum().sum() == 0`)."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Báo cáo tỷ lệ khuyết
Viết một hàm `missing_report(df)` nhận vào một DataFrame và trả về một bảng thống kê gồm 2 cột: `Missing_Count` (số lượng khuyết) và `Missing_Ratio_%` (tỷ lệ phần trăm khuyết sắp xếp giảm dần).

### Bài 2 (Easy): Xử lý khuyết bằng Forward Fill
Tìm hiểu và thực thi phương thức `df.fillna(method='ffill')` hoặc `df.ffill()`. Giải thích trong dạng dữ liệu nào (ví dụ: chuỗi thời gian - Time Series, giá chứng khoán) thì kỹ thuật này lại cực kỳ hữu ích.

### Bài 3 (Medium): Phát hiện giá trị rác ngầm định
Trong nhiều cuộc thi, người ra đề mã hóa giá trị khuyết thành các số dị thường như `-999`, `9999`, hoặc chuỗi `'?'`, `'N/A'`. Hãy viết hàm tự động tìm và thay thế tất cả các giá trị `-999` trong các cột số thành `np.nan`.

### Bài 4 (Medium): Đổi kiểu ngày tháng (Datetime Parsing)
Cho cột `Date = ['2026-01-15', '2026/02/20', '15-03-2026']`. Sử dụng `pd.to_datetime()` để chuẩn hóa toàn bộ về kiểu `datetime64[ns]`, sau đó trích xuất ra 2 cột mới: `Year` và `Month`.

### Bài 5 (Challenge): Viết Class Transformer làm sạch dữ liệu
Tạo một lớp Python `DataCleaner` có hai phương thức chuẩn phong cách Scikit-Learn:
- `fit(df)`: Học và lưu trữ lại Median của các cột số và Mode của các cột danh mục từ tập dữ liệu.
- `transform(df)`: Dùng các giá trị đã học ở bước `fit` để điền khuyết cho một DataFrame mới mà không gây ra Data Leakage.
"""
    voai_connection_content = """
Một trong những cái bẫy tinh vi nhất trong kỳ thi Olympic AI là **Data Leakage trong quá trình Imputation**:
Nhiều thí sinh gộp cả file `train.csv` và `test.csv` lại rồi mới tính Mean để điền khuyết. Điều này đồng nghĩa với việc bạn đã cho mô hình "nhìn trộm" phân phối của tập kiểm tra trước khi thi đấu, dẫn đến điểm kiểm thử cục bộ cực cao nhưng điểm nộp bài thực tế lại thảm hại. Quy tắc vàng: **Chỉ fit trên Train, áp dụng sang Test!**
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh cố gắng ép kiểu một cột dữ liệu sang số nguyên nhưng bị văng lỗi:

```python
# CODE CÓ LỖI CASTING KHI CHỨA NAN:
s = pd.Series([1, 2, np.nan, 4])
# s_int = s.astype(int) # Loi: ValueError: Cannot convert non-finite values (NA or inf) to integer
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao kiểu dữ liệu nguyên thủy `int` trong NumPy cũ không thể chứa giá trị `NaN`.
2. Sửa lỗi bằng cách sử dụng kiểu số nguyên mở rộng của Pandas (`Int64` với chữ I viết hoa) hoặc điền khuyết trước khi ép kiểu.
"""
    knowledge_check_content = """
1. Khi một cột dữ liệu bị khuyết nhiều giá trị và có sự xuất hiện của các số cực đoan (outliers), ta nên điền khuyết bằng đại lượng nào?
   - A. Giá trị 0
   - B. Giá trị trung bình cộng (Mean)
   - C. Giá trị trung vị (Median)
   - D. Luôn luôn xóa toàn bộ cột đó

2. Lệnh `df.drop_duplicates(keep='last')` có ý nghĩa gì?
   - A. Xóa tất cả các dòng bị trùng lặp, không giữ lại dòng nào.
   - B. Giữ lại bản ghi xuất hiện cuối cùng trong các bản ghi bị trùng, xóa các bản ghi phía trước.
   - C. Chỉ xóa dòng cuối cùng của bảng dữ liệu.
   - D. Báo lỗi nếu có bản ghi trùng lặp.

3. Phương thức nào của Pandas giúp chuyển đổi một chuỗi văn bản không chuẩn hóa sang số thực và biến các ký tự lỗi thành NaN?
   - A. `pd.to_numeric(s, errors='coerce')`
   - B. `s.to_float()`
   - C. `float(s)`
   - D. `pd.cast_number(s)`

4. Vì sao tính Mean trên toàn bộ tập dữ liệu (cả Train và Test) rồi điền khuyết lại bị coi là phạm luật kỹ thuật Data Leakage?
   - A. Vì làm tăng thời gian chạy của CPU.
   - B. Vì vi phạm nguyên tắc tập Test phải hoàn toàn vô hình đối với mô hình trong quá trình học.
   - C. Vì khiến cho Pandas bị tràn bộ nhớ.
   - D. Vì kết quả luôn luôn bằng 0.

5. Cột có kiểu dữ liệu là gì nếu nó chứa cả số thực lẫn chuỗi văn bản trong Pandas?
   - A. `float64`
   - B. `string`
   - C. `object`
   - D. `mixed`
"""
    session_summary_content = """
* **What I learned:**
  - Kỹ thuật phát hiện và đo lường tỷ lệ Missing Data.
  - Các chiến lược điền khuyết (Mean, Median, Mode) và phòng tránh rò rỉ dữ liệu.
  - Xử lý xung đột kiểu dữ liệu và khử trùng lặp dữ liệu bảng.
* **What I can now do:**
  - Tự xây dựng một Data Cleaning Pipeline hoàn chỉnh cho dữ liệu thô.
  - Đảm bảo dữ liệu không còn giá trị khuyết bất hợp lệ trước khi đưa vào mô hình.
* **What I still need to practice:**
  - Nâng cao kỹ năng phân tích nhóm dữ liệu (GroupBy) và biểu diễn bằng biểu đồ trực quan ở Buổi 4.
"""
    next_session_content = """
Trong **Buổi M1_S04**, chúng ta sẽ học cách "đọc vị" những câu chuyện ẩn sâu trong bảng dữ liệu: **Thống Kê Mô Tả, Phân Tích Nhóm (GroupBy) & Trực Quan Hóa Dữ Liệu (Data Visualization)** với Matplotlib và Seaborn!
"""
    trace = {
        "items": [
            "Làm sạch dữ liệu & phát hiện NaNs",
            "Xử lý missing values bằng Mean, Median, Mode",
            "Ép kiểu dữ liệu (Data Types) & loại bỏ Duplicates"
        ],
        "deliverable": "Mã nguồn Data Cleaning Pipeline chuẩn mực không rò rỉ dữ liệu"
    }
    nb = create_notebook(
        module_id="M1",
        module_name="NÓI CHUYỆN VỚI DỮ LIỆU BẰNG PYTHON",
        session_id="S03",
        topic="Làm Sạch Dữ Liệu: Missing Values, Data Types & Duplicates",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook làm sạch dữ liệu toàn diện loại bỏ NaNs và Duplicates",
        competition_relevance="Bảo đảm tính toàn vẹn dữ liệu, triệt tiêu lỗi sập code do thiếu dữ liệu khi thi đấu.",
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
    save_notebook(nb, "course/M1_PYTHON_DATA/M1_S03.ipynb")


def build_m1_s04():
    objectives = [
        "Tính toán và phân tích sâu các chỉ số thống kê mô tả: Mean, Median, Standard Deviation, Skewness, IQR.",
        "Sử dụng thành thạo cơ chế `Split-Apply-Combine` thông qua hàm `.groupby()` và `.agg()` trong Pandas.",
        "Trực quan hóa phân phối biến đơn lẻ (Univariate) bằng Histogram và Boxplot.",
        "Trực quan hóa mối quan hệ giữa các biến (Bivariate) bằng Scatter Plot và Heatmap ma trận tương quan."
    ]
    prerequisites = [
        "Hoàn thành Buổi M1_S02 và M1_S03.",
        "Hiểu các khái niệm thống kê cơ bản ở bậc phổ thông (trung bình, phương sai)."
    ]
    why_content = """
Một bảng dữ liệu 100,000 dòng là một "mớ bòng bong" nếu bạn chỉ nhìn vào các con số thô. Làm thế nào để biết khách hàng nam hay nữ chi tiêu nhiều hơn? Làm thế nào để biết thời gian làm bài thi có tỷ lệ nghịch với số lỗi sai hay không?

Não bộ con người tiến hóa để xử lý hình ảnh và mô thức (patterns), không phải để đọc từng con số. **Thống kê mô tả và Trực quan hóa dữ liệu (Visualization)** chính là chiếc kính hiển vi giúp bạn nhìn thấu bản chất dữ liệu: tìm ra quy luật phân phối, phát hiện ra các nhóm hành vi tiềm năng, và định hình hướng đi đúng đắn cho các bước kỹ nghệ đặc trưng tiếp theo.
"""
    concept_content = """
### 1. Cơ chế Split - Apply - Combine của GroupBy
* **Split:** Chia nhỏ DataFrame thành các nhóm con dựa trên một hoặc nhiều khóa phân loại (ví dụ: nhóm theo `City` hoặc `Gender`).
* **Apply:** Áp dụng một hàm tính toán độc lập lên từng nhóm con (ví dụ: tính trung bình điểm, đếm số lượng phần tử, tìm giá trị lớn nhất).
* **Combine:** Gộp kết quả tính toán từ tất cả các nhóm lại thành một cấu trúc dữ liệu tổng hợp duy nhất.

Cú pháp chuẩn mực trong thi đấu:
```python
df.groupby('Category_Col').agg({
    'Numeric_Col1': ['mean', 'median'],
    'Numeric_Col2': ['count', 'std']
})
```

### 2. Các loại biểu đồ trực quan hóa cốt lõi
1. **Histogram & KDE (Kernel Density Estimation):** Xem hình dạng phân phối của một biến số (có đối xứng chuẩn không, có bị lệch trái hay lệch phải không).
2. **Boxplot (Biểu đồ hộp Tukey):** Thể hiện trực quan 5 con số thống kê: $Min, Q1 (25\\%), Median (50\\%), Q3 (75\\%), Max$ và phát hiện các chấm ngoại lai (Outliers).
3. **Scatter Plot (Biểu đồ phân tán):** Quan sát tương quan phi tuyến hoặc tuyến tính giữa hai biến số liên tục.
4. **Heatmap Correlation Matrix:** Thể hiện ma trận hệ số tương quan Pearson ($r \\in [-1, 1]$) giữa tất cả các cặp đặc trưng số học.
"""
    visual_intuition_content = """
```
Cấu trúc Boxplot (Tukey's Box):
          Q1 - 1.5*IQR                Q1         Median (Q2)       Q3             Q3 + 1.5*IQR
               │                      ┌──────────────┬─────────────┐                   │
  [Outliers] * ├─── Lower Whisker ────┤              │             ├──── Upper Whisker ┤ * [Outliers]
                                      └──────────────┴─────────────┘
                                      │<──────── IQR ─────────────>│
```
"""
    code_cells = [
        (
            """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Khoi tao tap du lieu mo phong ve nang suat hoc tap cua 200 sinh vien
np.random.seed(42)
n = 200

study_hours = np.random.uniform(2, 20, size=n)
# Diem thi phu thuoc tuyen tinh vao so gio hoc + nhieu ngau nhien
exam_scores = np.clip(30 + study_hours * 3.2 + np.random.normal(0, 5, size=n), 0, 100)
majors = np.random.choice(['AI', 'Data_Science', 'Computer_Science'], size=n, p=[0.4, 0.3, 0.3])
sleep_hours = np.random.normal(7, 1.2, size=n).round(1)

df_students = pd.DataFrame({
    'Major': majors,
    'Study_Hours': study_hours.round(1),
    'Sleep_Hours': sleep_hours,
    'Exam_Score': exam_scores.round(1)
})

# Thong ke gom nhom theo chuyen nganh (GroupBy)
grouped_stats = df_students.groupby('Major').agg(
    Avg_Score=('Exam_Score', 'mean'),
    Median_Score=('Exam_Score', 'median'),
    Std_Score=('Exam_Score', 'std'),
    Total_Students=('Exam_Score', 'count')
).reset_index()

print("===== KET QUA PHAN TICH NHOM (GROUPBY AGGREGATION) =====")
print(grouped_stats)
""",
            """Khối code minh họa kỹ thuật gom nhóm dữ liệu chuyên nghiệp:
- Tạo dataset 200 sinh viên với mối quan hệ thực nghiệm giữa số giờ học và điểm thi.
- Sử dụng cú pháp Named Aggregation trong `.agg()` để gán nhãn rõ ràng cho các cột thống kê đầu ra (`Avg_Score`, `Median_Score`, `Total_Students`)."""
        ),
        (
            """# 2. Truc quan hoa da chieu bang Matplotlib va Seaborn
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Bieu do 1: Histogram + KDE phan phoi diem thi
sns.histplot(df_students['Exam_Score'], kde=True, ax=axes[0], color='#2b5c8f', bins=15)
axes[0].set_title('Phan phoi Diem thi (Histogram & KDE)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Exam Score')

# Bieu do 2: Boxplot so sanh diem thi giua cac chuyen nganh
sns.boxplot(x='Major', y='Exam_Score', data=df_students, ax=axes[1], palette='Set2')
axes[1].set_title('So sanh Diem thi giua cac Nganh (Boxplot)', fontsize=12, fontweight='bold')

# Bieu do 3: Scatter Plot the hien tuong quan giua Gio hoc va Diem thi
sns.scatterplot(x='Study_Hours', y='Exam_Score', hue='Major', data=df_students, ax=axes[2], alpha=0.8)
axes[2].set_title('Tuong quan: Gio hoc vs Diem thi', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Study Hours / Week')

plt.tight_layout()
plt.show()
""",
            """Khối code vẽ 3 biểu đồ trực quan hóa dữ liệu chuẩn mực:
- **Histogram & KDE:** Đánh giá độ lệch và mật độ phân phối điểm số.
- **Boxplot:** So sánh độ biến động và phát hiện outlier giữa 3 chuyên ngành.
- **Scatter Plot:** Quan sát trực quan tương quan dương mạnh mẽ giữa thời gian học và kết quả thi."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Phân tích phân vị (Quantiles)
Sử dụng hàm `np.quantile` hoặc `pd.Series.quantile` để tìm giá trị phân vị $25\\%, 50\\%, 75\\%$ của cột `Study_Hours`.

### Bài 2 (Easy): Tính ma trận tương quan Pearson
Tính ma trận tương quan giữa các cột số trong `df_students` bằng lệnh `df_students[['Study_Hours', 'Sleep_Hours', 'Exam_Score']].corr()`. Cho biết cặp biến nào có tương quan tuyến tính mạnh nhất.

### Bài 3 (Medium): Vẽ Heatmap tương quan
Sử dụng hàm `sns.heatmap(..., annot=True, cmap='coolwarm')` để trực quan hóa ma trận tương quan vừa tính ở Bài 2.

### Bài 4 (Medium): Gom nhóm nhiều tầng (Multi-level GroupBy)
Tạo thêm cột phân loại nhị phân `Well_Rested` (`True` nếu `Sleep_Hours >= 7.0`, ngược lại `False`). Thực hiện gom nhóm theo cả hai tiêu chí `['Major', 'Well_Rested']` và tính điểm thi trung bình của từng nhóm con.

### Bài 5 (Challenge): Phát hiện nghịch lý Simpson (Simpson's Paradox)
Tìm hiểu khái niệm Nghịch lý Simpson trong thống kê. Tạo một ví dụ nhỏ bằng Pandas chứng minh một xu hướng xuất hiện trong từng nhóm nhỏ nhưng lại biến mất hoặc đảo ngược khi gộp chung tất cả các nhóm lại.
"""
    voai_connection_content = """
Trong thi đấu Olympic AI, trực quan hóa không phải để "làm cảnh" mà là công cụ **thám tử dữ liệu**. Khi bạn vẽ phân phối của nhãn (Target variable) và phát hiện nhãn bị lệch cực đoan (Right-skewed, ví dụ bài toán dự đoán doanh số), động tác tiếp theo của một thí sinh đẳng cấp là ngay lập tức áp dụng biến đổi hàm Logarit $\\log(1 + x)$ để đưa phân phối về dạng hình chuông đối xứng, giúp các mô hình Linear Regression và Neural Network hội tụ tốt hơn gấp nhiều lần!
"""
    debugging_content = """
**Tình huống thực tế:** Một thí sinh gom nhóm dữ liệu bằng `.groupby()` và cố gắng lưu file CSV nhưng bị mất cột nhóm hoặc nhận file có cấu trúc index kỳ lạ:

```python
# CODE DỄ GÂY LỖI CẤU TRÚC:
res = df_students.groupby('Major')['Exam_Score'].mean()
# res bay gio la mot Series co index la 'Major', khong con la DataFrame dang bang thong thuong!
# Neu dem merge voi bang khac se de sinh loi mismatch index.
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao việc thiếu `.reset_index()` sau groupby có thể gây khó khăn khi thao tác ghép bảng dữ liệu (merge/join).
2. Sửa lại dòng lệnh trên để luôn nhận được một DataFrame chuẩn với các cột tên rõ ràng.
"""
    knowledge_check_content = """
1. Đại lượng nào sau đây đo lường mức độ bất đối xứng của một phân phối xác suất quanh giá trị trung bình của nó?
   - A. Standard Deviation (Độ lệch chuẩn)
   - B. Skewness (Hệ số bất đối xứng / Độ lệch)
   - C. Kurtosis (Độ nhọn)
   - D. Variance (Phương sai)

2. Khoảng tứ phân vị (Interquartile Range - IQR) được tính bằng công thức nào?
   - A. $IQR = Q3 - Q1$
   - B. $IQR = Max - Min$
   - C. $IQR = Mean - Median$
   - D. $IQR = Q2 - Q1$

3. Hệ số tương quan Pearson $r = -0.85$ giữa hai biến $X$ và $Y$ mang ý nghĩa gì?
   - A. $X$ và $Y$ không có bất kỳ mối quan hệ nào.
   - B. $X$ và $Y$ có mối quan hệ tương quan tuyến tính nghịch biến rất mạnh (khi $X$ tăng thì $Y$ giảm mạnh).
   - C. $X$ là nguyên nhân trực tiếp gây ra $Y$.
   - D. Dữ liệu bị lỗi tính toán.

4. Biểu đồ Boxplot sử dụng đường kẻ bên trong hộp (line inside box) để biểu diễn giá trị nào?
   - A. Giá trị trung bình (Mean)
   - B. Giá trị trung vị (Median / Q2)
   - C. Giá trị lớn nhất (Max)
   - D. Độ lệch chuẩn

5. Cơ chế hoạt động của hàm `.groupby()` trong Pandas tuân theo mô hình nào?
   - A. Map - Filter - Reduce
   - B. Split - Apply - Combine
   - C. Fit - Predict - Score
   - D. Load - Transform - Dump
"""
    session_summary_content = """
* **What I learned:**
  - Nắm vững các chỉ số thống kê mô tả nâng cao và ý nghĩa của chúng.
  - Sử dụng thành thạo mô hình `Split-Apply-Combine` với `.groupby()` và `.agg()`.
  - Thành thạo vẽ các biểu đồ phân phối và tương quan: Histogram, Boxplot, Scatter, Heatmap.
* **What I can now do:**
  - Tự tin phân tích mối tương quan giữa các nhóm đặc trưng trong bảng dữ liệu lớn.
  - Trực quan hóa dữ liệu sạch đẹp, phục vụ đắc lực cho việc ra quyết định kỹ thuật.
* **What I still need to practice:**
  - Kết hợp toàn bộ kỹ năng của M1 để thực hiện **Phân tích dữ liệu đầu tiên (First Data Analysis Project)** ở Buổi 5.
"""
    next_session_content = """
Buổi tiếp theo (**M1_S05**) là buổi tổng kết đỉnh cao của Module 1: **Data Cleaning Tổng Hợp, Phát Hiện Dị Biệt (Anomalies) & First Data Analysis Report**! Bạn sẽ tự tay thực hiện một bài toán phân tích dữ liệu thực tế từ A đến Z, xuất bản báo cáo phân tích đầu tiên và nhận Chứng chỉ **Certificate of Python and Data Basics**!
"""
    trace = {
        "items": [
            "Thống kê mô tả & số đo phân tán",
            "Phân tích nhóm bằng GroupBy & Aggregation",
            "Trực quan hóa dữ liệu (Visualization) bằng Matplotlib & Seaborn"
        ],
        "deliverable": "Mã nguồn phân tích thống kê nhóm và bộ biểu đồ trực quan hóa dữ liệu"
    }
    nb = create_notebook(
        module_id="M1",
        module_name="NÓI CHUYỆN VỚI DỮ LIỆU BẰNG PYTHON",
        session_id="S04",
        topic="Thống Kê Mô Tả, Phân Tích Nhóm (GroupBy) & Visualization",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook trực quan hóa đa chiều và phân tích thống kê nhóm",
        competition_relevance="Phát hiện mô thức ẩn trong dữ liệu thi đấu, thiết kế feature thông minh.",
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
    save_notebook(nb, "course/M1_PYTHON_DATA/M1_S04.ipynb")


def build_m1_s05():
    objectives = [
        "Làm chủ hai phương pháp phát hiện dị biệt (Anomalies / Outliers): Quy tắc biên IQR và Phương pháp Z-score.",
        "Thiết lập một End-to-End Data Cleaning & EDA Pipeline khép kín cho tập dữ liệu thực nghiệm.",
        "Hoàn thành sản phẩm 'Phân tích dữ liệu đầu tiên' (First Data Analysis Report) đạt chuẩn thi đấu.",
        "Rèn luyện kỹ năng kể chuyện bằng dữ liệu (Data Storytelling) và đề xuất giả thuyết mô hình hóa."
    ]
    prerequisites = [
        "Hoàn thành trọn vẹn 4 buổi trước (M1_S01 đến M1_S04).",
        "Có kỹ năng phối hợp NumPy, Pandas, Matplotlib và Seaborn."
    ]
    why_content = """
Trong thế giới thực và trong các bài toán Olympic AI, bạn không bao giờ được giao việc dưới dạng các câu lệnh đơn lẻ như "hãy tính mean" hay "hãy xóa dòng 5". Bạn sẽ được quẳng cho một tập dữ liệu thô kèm một bài toán kinh doanh hoặc khoa học: "Hãy tìm hiểu tại sao bệnh nhân tái nhập viện?", "Tại sao giao dịch này bị nghi ngờ gian lận?".

Nếu không có khả năng kết nối tất cả các kỹ năng đơn lẻ lại thành một **Quy trình Phân tích Dữ liệu hoàn chỉnh (End-to-End EDA Report)**, bạn sẽ mãi chỉ là một thợ gõ code rời rạc. Buổi học hôm nay là nơi bạn chứng minh năng lực "Nói chuyện với dữ liệu": biến các dòng số vô tri thành những phát hiện sâu sắc (Actionable Insights).
"""
    concept_content = r"""
### 1. Kỹ thuật phát hiện dị biệt (Outlier Detection)
* **Phương pháp 1: Quy tắc Biên IQR (Tukey's Fences):**
  - Tính khoảng tứ phân vị: $IQR = Q3 - Q1$.
  - Biên dưới (Lower Bound): $LB = Q1 - 1.5 \times IQR$.
  - Biên trên (Upper Bound): $UB = Q3 + 1.5 \times IQR$.
  - Bất kỳ điểm dữ liệu nào nằm ngoài khoảng $[LB, UB]$ được gắn cờ là Outlier.
* **Phương pháp 2: Phương pháp Z-Score:**
  - $Z = \frac{x - \mu}{\sigma}$ đo lường khoảng cách từ điểm dữ liệu tới giá trị trung bình theo đơn vị độ lệch chuẩn.
  - Nếu $|Z| > 3.0$ (nằm ngoài vùng 99.7% của phân phối chuẩn), điểm đó bị coi là ngoại lai cực đoan.

### 2. Cấu trúc một Báo cáo Phân tích Dữ liệu Đầu Tiên (EDA Report) chuẩn Olympic
1. **Tổng quan Dữ liệu (Data Overview):** Kích thước, kiểu dữ liệu, định dạng nhãn.
2. **Kiểm toán Chất lượng Dữ liệu (Data Quality Audit):** Missing data, duplicates, sai kiểu.
3. **Làm sạch & Xử lý Dị biệt (Cleaning & Outlier Treatment):** Imputation, lọc rác hoặc giới hạn giá trị (clipping/capping).
4. **Phân tích Đơn biến & Đa biến (Univariate & Bivariate Analysis):** Phân phối đặc trưng, tương quan với biến mục tiêu.
5. **Đề xuất Hành động (Feature Engineering & Modeling Hypotheses):** Danh sách các gợi ý để chuẩn bị bước vào giai đoạn Machine Learning ở Module M2!
"""
    visual_intuition_content = """
```
Cấu trúc Báo cáo Phân tích Dữ liệu Đầu tiên:
┌─────────────────────────────────────────────────────────────────┐
│ 1. DATA AUDIT: (N dòng, M cột, tỷ lệ NaNs, duplicates)          │
├─────────────────────────────────────────────────────────────────┤
│ 2. DATA CLEANING: (Imputation chuẩn, ép kiểu, khử trùng lặp)    │
├─────────────────────────────────────────────────────────────────┤
│ 3. OUTLIER DETECTION: (Tìm & xử lý điểm dị biệt bằng IQR)        │
├─────────────────────────────────────────────────────────────────┤
│ 4. STATISTICAL EDA: (Phân phối nhãn, tương quan đa biến)         │
├─────────────────────────────────────────────────────────────────┤
│ 5. ACTIONABLE INSIGHTS: (Kết luận cho bước huấn luyện Model)     │
└─────────────────────────────────────────────────────────────────┘
```
"""
    code_cells = [
        (
            """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Khoi tao tap du lieu nghien cuu: "Housing Market & Pricing Anomalies"
np.random.seed(101)
n_houses = 300

# Du lieu dien tich (m2) va gia nha (trieu VND)
area = np.random.normal(80, 25, size=n_houses).clip(30, 250)
base_price = area * 35 + np.random.normal(0, 300, size=n_houses)

# Tao mot so diem di biet co y (Anomalies / Outliers do nhap sai du lieu)
base_price[15] = 25000  # Nhap thua so 0 (25 ty thay vi 2.5 ty)
base_price[42] = -500   # Gia tri am bat hop ly
area[88] = 1200         # Dien tich dot bien

df_houses = pd.DataFrame({
    'House_ID': [f'H_{i:04d}' for i in range(1, n_houses + 1)],
    'Area_m2': area.round(1),
    'Price_M_VND': base_price.round(1),
    'Location': np.random.choice(['Center', 'Suburbs', 'Rural'], size=n_houses, p=[0.3, 0.5, 0.2])
})

print("===== DU LIEU NHA O MAU (300 CAN) =====")
print(df_houses.head())
""",
            """Khối code thiết lập bài toán phân tích thực tế: Thị trường bất động sản với các lỗi nhập liệu điển hình:
- Nhập thừa số 0 khiến giá nhà cao phi lý.
- Giá tiền bị âm do lỗi hệ thống.
- Diện tích đột biến phá vỡ phân phối thông thường."""
        ),
        (
            """# 2. Ham phat hien Outlier bang quy tac IQR (Tukey's Fences)
def detect_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers, lower_bound, upper_bound

# Phat hien outlier tren cot Price
outlier_mask, lb, ub = detect_outliers_iqr(df_houses['Price_M_VND'])
num_outliers = outlier_mask.sum()
print(f"Phat hien {num_outliers} diem di biet ve gia nha.")
print(f"Ranh gioi hop le (IQR Bounds): [{lb:.1f}, {ub:.1f}] trieu VND")

# Xu ly Outlier: Loc bo cac gia tri am va loai bo outlier cuc doan de lam sach tap du lieu
df_houses_clean = df_houses[~outlier_mask & (df_houses['Price_M_VND'] > 0)].copy()

# 3. Xuat bao cao tuong quan sau khi lam sach
plt.figure(figsize=(9, 5))
sns.scatterplot(x='Area_m2', y='Price_M_VND', hue='Location', data=df_houses_clean, alpha=0.8)
plt.title('Bao cao Tuong quan: Dien tich vs Gia nha (Da loai bo Outliers)', fontsize=12, fontweight='bold')
plt.xlabel('Dien tich (m2)')
plt.ylabel('Gia nha (Trieu VND)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print(f"So luong ban ghi con lai sau lam sach: {len(df_houses_clean)}/{len(df_houses)}")
""",
            """Khối code xây dựng hàm chuẩn phát hiện ngoại lai bằng IQR:
- Tính toán chính xác ngưỡng biên trên/dưới.
- Loại bỏ an toàn các điểm dị biệt và giá trị âm bất hợp lý.
- Trực quan hóa mối quan hệ thực sự giữa diện tích và giá nhà mà không bị bóp méo bởi các điểm cực đoan."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Viết hàm phát hiện Outlier bằng Z-score
Tự viết hàm `detect_outliers_zscore(series, threshold=3.0)` tính toán Z-score và trả về một Boolean Mask đánh dấu các điểm vượt quá ngưỡng 3 độ lệch chuẩn.

### Bài 2 (Easy): So sánh số lượng ngoại lai giữa IQR và Z-Score
Áp dụng cả hai phương pháp lên cột `Area_m2`. So sánh xem phương pháp nào nhạy hơn với các giá trị lệch.

### Bài 3 (Medium): Kỹ thuật Capping (Winsorization)
Thay vì xóa bỏ các dòng ngoại lai, hãy thay thế các giá trị lớn hơn `Upper Bound` bằng đúng giá trị `Upper Bound`, và các giá trị nhỏ hơn `Lower Bound` bằng đúng `Lower Bound`. Giải thích tại sao trong thi đấu Olympic AI, Capping thường được ưu tiên hơn xóa dòng khi tập dữ liệu có kích thước nhỏ.

### Bài 4 (Medium): Báo cáo thống kê phân nhóm theo địa điểm
Sau khi làm sạch dữ liệu, hãy dùng `groupby('Location')` để tính đơn giá trung bình trên mỗi mét vuông (`Price_per_m2 = Price_M_VND / Area_m2`) cho từng khu vực và nhận xét khu vực nào có giá đắt đỏ nhất.

### Bài 5 (Challenge): Viết hàm xuất bản báo cáo tóm tắt tự động
Viết một hàm Python `generate_eda_markdown_summary(df, target_col)` tự động in ra màn hình một bản báo cáo tóm tắt dạng Markdown gồm: số lượng mẫu, số lượng biến số, biến mục tiêu, số lượng missing value và danh sách các đặc trưng có độ tương quan cao nhất với biến mục tiêu.
"""
    voai_connection_content = """
Trong phần thi Olympic AI, nhiều đề thi cố tình "gài bẫy" bằng các giá trị ngoại lai ở tập Test. Nếu mô hình hồi quy của bạn được huấn luyện trên dữ liệu còn sót lại các điểm dị biệt khổng lồ (ví dụ giá nhà gấp 100 lần bình thường), đường hồi quy sẽ bị kéo lệch hoàn toàn (High Bias), khiến sai số RMSE trên bảng xếp hạng bùng nổ. Biết cách phát hiện và xử lý dị biệt khéo léo là tố chất của một thí sinh giành huy chương.
"""
    debugging_content = """
**Tình huống thực tế:** Một thí sinh tính Z-score trên một cột dữ liệu bị phân phối lệch nặng (Right-skewed) và áp dụng ngưỡng $|Z| > 3.0$, dẫn đến việc xóa nhầm rất nhiều dữ liệu hợp lệ:

```python
# CODE CÓ VẤN ĐỀ TƯ DUY TOÁN HỌC:
# Khi du lieu khong phai la phan phoi chuan (Gaussian/Normal Distribution), 
# viec gia dinh 99.7% du lieu nam trong khoang [-3*std, +3*std] la hoan toan sai lam!
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao phương pháp Z-score chỉ thực sự tối ưu khi dữ liệu tuân theo phân phối chuẩn.
2. Nêu lý do vì sao quy tắc IQR (Tukey's Fences) lại an toàn và bền vững (robust) hơn trong hầu hết các bài toán thực tế.
"""
    knowledge_check_content = """
1. Khi tính toán biên trên trong quy tắc IQR, hệ số chuẩn mực thường được nhân với IQR là bao nhiêu?
   - A. 1.0
   - B. 1.5
   - C. 2.0
   - D. 3.0

2. Kỹ thuật thay thế các giá trị ngoại lai bằng giá trị biên tối đa/tối thiểu hợp lệ mà không xóa bỏ hàng dữ liệu được gọi là gì?
   - A. Normalization
   - B. One-Hot Encoding
   - C. Winsorization / Capping
   - D. Dropdown Filtering

3. Tại sao trong các bài thi AI, việc tự ý xóa bỏ hàng loạt dòng chứa Outlier ở tập Test là điều KHÔNG ĐƯỢC PHÉP?
   - A. Vì tập Test là dữ liệu đánh giá của ban tổ chức, bạn bắt buộc phải nộp dự đoán cho 100% số dòng của tập Test.
   - B. Vì tập Test không có ngoại lai.
   - C. Vì làm giảm điểm số F1.
   - D. Vì hệ thống tự động khóa tài khoản.

4. Điểm Z-score $Z = +2.5$ của một mẫu thử biểu thị điều gì?
   - A. Mẫu thử đó gấp 2.5 lần giá trị trung bình.
   - B. Mẫu thử đó nằm cao hơn giá trị trung bình 2.5 lần độ lệch chuẩn.
   - C. Mẫu thử đó chắc chắn là một ngoại lai bị lỗi.
   - D. Mẫu thử đó có xác suất xuất hiện là 25%.

5. Sản phẩm cốt lõi mà học viên hoàn thành ở cuối Module M1 là gì?
   - A. Mạng nơ-ron tích chập ResNet-50.
   - B. Phân tích dữ liệu đầu tiên (First Data Analysis Report).
   - C. Ứng dụng Web Chatbot thông minh.
   - D. Mô hình Transformer dịch máy tự động.
"""
    session_summary_content = """
* **What I learned:**
  - Làm chủ hai phương pháp phát hiện ngoại lai: Biên IQR và Z-score.
  - Kỹ thuật Capping (Winsorization) giữ trọn vẹn cỡ mẫu khi thi đấu.
  - Quy trình 5 bước xây dựng Báo cáo Phân tích Dữ liệu Đầu tiên (First Data Analysis Report).
* **What I can now do:**
  - Tự tin nhận một tập dữ liệu thô, thực hiện kiểm toán, làm sạch toàn diện và xuất báo cáo chất lượng cao.
  - Phục hồi và xử lý an toàn các dị biệt trong dữ liệu bảng.
* **What I still need to practice:**
  - Sẵn sàng bước sang **Module M2: Dạy máy học từ dữ liệu** — nơi bạn sẽ chính thức huấn luyện mô hình Machine Learning đầu tiên!
"""
    next_session_content = """
Chúc mừng bạn đã hoàn thành xuất sắc Module M1!
Trong buổi tiếp theo (**M2_S01**), chúng ta sẽ bước vào **Module M2: DẠY MÁY HỌC TỪ DỮ LIỆU** — Bắt đầu giải mã bản chất toán học của Học Máy: Feature, Target, Supervised vs Unsupervised Learning và chiến thuật Train/Test Split chống rò rỉ dữ liệu!
"""
    module_assessment = """
---

## 🏆 MODULE 1 ASSESSMENT & CERTIFICATION

### Phần A: Đánh giá Lý thuyết & Tư duy Dữ liệu (Theory & Data Reasoning)
- Giải thích bản chất Vectorization của NumPy và cơ chế Broadcasting.
- Phân biệt `.loc` vs `.iloc`, giải thích rủi ro Data Leakage khi Imputation.
- Phân tích sự khác biệt giữa IQR và Z-score trong phát hiện Outlier.

### Phần B: Sản phẩm Thực hành Bắt buộc (Practical Deliverable)
- **Phân tích dữ liệu đầu tiên (First Data Analysis Report):** Học viên hoàn thành một notebook phân tích đầy đủ trên một tập dữ liệu có vấn đề (missing data, duplicates, outliers).
- **Mã nguồn chuẩn:** Không dùng vòng lặp for cho các phép tính số học, có đầy đủ biểu đồ trực quan hóa (Histogram, Boxplot, Correlation Heatmap).

### Tiêu chí Đạt chuẩn (Rubric):
| Tiêu chí | Trọng số | Yêu cầu đạt |
| :--- | :--- | :--- |
| **Data Cleaning Pipeline** | 30% | Xử lý triệt để NaNs, Duplicates, ép đúng kiểu dữ liệu |
| **Outlier Detection & Handling**| 25% | Phát hiện và xử lý dị biệt bằng IQR hoặc Capping |
| **Visualization & Storytelling** | 25% | Vẽ ít nhất 3 biểu đồ trực quan có phân tích ý nghĩa |
| **Chất lượng mã nguồn & Tối ưu** | 20% | Sử dụng NumPy/Pandas Vectorized, không lỗi runtime |

**Chứng chỉ hoàn thành:**
🏅 **Certificate of Python and Data Basics** — Đủ điều kiện tiến vào Module M2.
"""
    trace = {
        "items": [
            "Phát hiện dị biệt (Anomalies / Outliers) bằng IQR và Z-score",
            "End-to-End Data Cleaning & EDA Pipeline",
            "Hoàn thành sản phẩm: Phân tích dữ liệu đầu tiên (First Data Analysis Report)",
            "Nhận chứng chỉ Certificate of Python and Data Basics"
        ],
        "deliverable": "Phân tích dữ liệu đầu tiên (First Data Analysis Report) + Báo cáo xử lý dị biệt hoàn chỉnh"
    }
    nb = create_notebook(
        module_id="M1",
        module_name="NÓI CHUYỆN VỚI DỮ LIỆU BẰNG PYTHON",
        session_id="S05",
        topic="Data Cleaning Tổng Hợp, Phát Hiện Dị Biệt & First Data Analysis Report",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Báo cáo phân tích dữ liệu đầu tiên (First Data Analysis Report)",
        competition_relevance="Hoàn thiện năng lực EDA thực chiến, loại bỏ rác dữ liệu trước khi train model.",
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
    save_notebook(nb, "course/M1_PYTHON_DATA/M1_S05.ipynb")

if __name__ == "__main__":
    build_m1_s01()
    build_m1_s02()
    build_m1_s03()
    build_m1_s04()
    build_m1_s05()
    print("M1 build complete!")
