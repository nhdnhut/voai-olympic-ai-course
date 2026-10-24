"""
builder/build_m2.py
Generates M2_S01.ipynb to M2_S05.ipynb for Module 2.
"""

from utils import create_notebook, save_notebook

def build_m2_s01():
    objectives = [
        "Hiểu bản chất toán học của Học có giám sát (Supervised Learning): Ánh xạ hàm mục tiêu y = f(X).",
        "Phân biệt rõ ràng giữa Ma trận Đặc trưng (Feature Matrix X) và Vector Nhãn (Target Vector y).",
        "Phân biệt giữa Bài toán Hồi quy (Regression) và Bài toán Phân loại (Classification).",
        "Nắm vững nguyên lý Khả năng tổng quát hóa (Generalization) và thực hành phân chia tập huấn luyện/kiểm tra (Train/Test Split & Stratified Split)."
    ]
    prerequisites = [
        "Hoàn thành trọn vẹn Module M1 (Đã nắm vững NumPy và Pandas).",
        "Hiểu khái niệm biến số độc lập và biến số phụ thuộc."
    ]
    why_content = """
Tại sao chúng ta không tiếp tục viết các hàm `if-else` để giải quyết bài toán mà phải dùng Machine Learning? Hãy tưởng tượng bạn cần phân loại 1 triệu email xem có phải thư rác (Spam) hay không. Nếu viết rule bằng tay, bạn sẽ phải nghĩ ra hàng chục nghìn luật `if "khuyến mãi" in email...`, và người gửi spam chỉ cần đổi một chữ cái là toàn bộ luật của bạn sụp đổ!

Machine Learning ra đời để giải quyết chính xác sự bế tắc đó: Thay vì con người tự viết luật, chúng ta đưa **Dữ liệu mẫu (Input X)** và **Kết quả đúng (Target y)** vào máy tính, để thuật toán tự động tối ưu hóa và tìm ra quy luật toán học ánh xạ $y = f(X)$. Hiểu đúng bản chất này giúp bạn không bao giờ nhầm lẫn giữa việc "học vẹt dữ liệu" (Overfitting) và "học được quy luật tổng quát".
"""
    concept_content = r"""
### 1. Khuôn khổ toán học của Học máy có giám sát (Supervised Learning)
* **Ma trận đặc trưng (Feature Matrix $X \in \mathbb{R}^{N \times D}$):**
  - Gồm $N$ hàng (mẫu dữ liệu - samples) và $D$ cột (các thuộc tính quan sát được - features).
  - Ví dụ: Diện tích, số phòng, khoảng cách trường học...
* **Vector mục tiêu (Target Vector $y \in \mathbb{R}^N$):**
  - Nhãn mà mô hình cần học cách dự đoán.
  - **Hồi quy (Regression):** $y$ là số thực liên tục (Continuous), ví dụ: Giá nhà, nhiệt độ ngày mai, doanh số.
  - **Phân loại (Classification):** $y$ là nhãn rời rạc (Discrete), ví dụ: Bệnh/Không bệnh, Thư rác/Hợp lệ.

### 2. Sự tổng quát hóa (Generalization) vs Ghi nhớ vẹt (Memorization)
Mục tiêu tối thượng của Machine Learning không phải là đạt điểm 100% trên dữ liệu đã thấy, mà là **dự đoán chính xác trên dữ liệu mới chưa từng xuất hiện (Unseen Data)**.
* **Train Set (Tập huấn luyện):** Dữ liệu dùng để thuật toán học các tham số.
* **Test Set (Tập kiểm tra):** Dữ liệu tuyệt đối bí mật, chỉ dùng để đánh giá năng lực tổng quát hóa cuối cùng.

### 3. Nguyên tắc Phân chia tập (Train/Test Split & Stratification)
* Thông thường phân chia theo tỷ lệ $80/20$ hoặc $75/25$.
* **Stratified Split (Phân tầng):** Bắt buộc dùng cho bài toán phân loại mất cân bằng nhãn (Imbalanced Classes). Kỹ thuật này đảm bảo tỷ lệ phần trăm giữa các lớp ở tập Train và tập Test là hoàn toàn đồng nhất!
"""
    visual_intuition_content = """
```
Luồng phân tách dữ liệu chuẩn (Data Partitioning):
Toàn bộ dữ liệu bảng: [ Feature Matrix X (N x D) | Target y (N) ]
                              │
                    train_test_split()
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
[Train Set X_train, y_train]              [Test Set X_test, y_test]
     (Chiếm 80% dữ liệu)                       (Chiếm 20% dữ liệu)
         │                                         │
    DÙNG ĐỂ HỌC                               CHỈ DÙNG ĐỂ THI
(Model thấy cả X và y)                     (Giấu y_test, chỉ đưa X_test)
```
"""
    code_cells = [
        (
            """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Tao tap du lieu mo phong ve chuan doan benh tieu duong (Classification Problem)
np.random.seed(42)
n_samples = 500

# Cac dac trung (Features): Tuoi, Chi so khoi co the (BMI), Muc duong huyet (Glucose)
age = np.random.randint(20, 70, size=n_samples)
bmi = np.random.normal(26, 4, size=n_samples).round(1)
glucose = np.random.normal(110, 25, size=n_samples).round(1)

# Nhãn muc tieu (Target): 1 la Mac benh, 0 la Binh thuong (Ty le mac benh khoang 20% - Imbalanced)
logits = -8.0 + 0.04 * age + 0.08 * bmi + 0.04 * glucose
probs = 1 / (1 + np.exp(-logits))
diabetes = (np.random.rand(n_samples) < probs).astype(int)

df = pd.DataFrame({'Age': age, 'BMI': bmi, 'Glucose': glucose, 'Outcome': diabetes})
print("===== 5 DONG DAU TIEN CUA TAP DU LIEU =====")
print(df.head())

print("\\nPhan phoi nhan goc (Target Distribution):")
print(df['Outcome'].value_counts(normalize=True) * 100)
""",
            """Khối code mô phỏng bài toán phân loại y tế với sự mất cân bằng nhãn thực tế:
- Tách bạch rõ ràng 3 cột đặc trưng (Age, BMI, Glucose) và cột nhãn mục tiêu (`Outcome`).
- Kiểm tra phân phối nhãn cho thấy lớp 1 chỉ chiếm khoảng 20% tổng số mẫu."""
        ),
        (
            """# 2. Tach Feature Matrix X va Target Vector y
X = df[['Age', 'BMI', 'Glucose']] # Ma tran dac trung 2D
y = df['Outcome']                  # Vector nhan muc tieu 1D

# 3. Thuc hien Stratified Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y  # Phan tang de giu nguyen ty le nhan
)

print(f"Kich thuoc tap Train: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Kich thuoc tap Test : X_test  = {X_test.shape}, y_test  = {y_test.shape}")

print("\\nTy le nhan 1 trong tap Train: {:.2f}%".format(y_train.mean() * 100))
print("Ty le nhan 1 trong tap Test : {:.2f}%".format(y_test.mean() * 100))
""",
            """Khối code thực hiện quy chuẩn phân tách tập dữ liệu bằng Scikit-Learn:
- Sử dụng tham số `stratify=y` bảo đảm tỷ lệ người mắc bệnh ở tập Train và tập Test hoàn toàn khớp nhau (tránh hiện tượng tập Test không có mẫu dương tính nào).
- `random_state=42` bảo đảm tính tái lập (Reproducibility) — tiêu chuẩn bắt buộc trong thi đấu."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Phân biệt bài toán Hồi quy vs Phân loại
Với mỗi bài toán sau, xác định rõ đâu là **Feature $X$**, đâu là **Target $y$**, và thuộc loại **Regression** hay **Classification**:
1. Dự đoán số lượng xe đạp được thuê trong một ngày dựa trên thời tiết và nhiệt độ.
2. Xác định giao dịch thẻ tín dụng là hợp pháp hay gian lận.
3. Dự đoán điểm tổng kết môn học (thang điểm 10.0) của sinh viên.
4. Phân loại cảm xúc bình luận của người dùng (Tích cực / Tiêu cực / Trung tính).

### Bài 2 (Easy): Phân chia dữ liệu theo tỷ lệ 70/30
Viết lại lệnh `train_test_split` để chia dữ liệu thành $70\\%$ Train và $30\\%$ Test. In ra số dòng của từng tập.

### Bài 3 (Medium): So sánh phân phối khi KHÔNG dùng `stratify`
Thực hiện chia tập không có tham số `stratify` (`stratify=None`) với 5 giá trị `random_state` khác nhau. Đo sự biến động của tỷ lệ nhãn 1 trên tập Test và giải thích rủi ro nếu nhãn mục tiêu bị lệch nghiêm trọng (ví dụ chỉ có 1% mẫu hiếm).

### Bài 4 (Medium): Phân chia theo chuỗi thời gian (Temporal Split)
Tìm hiểu tại sao đối với dữ liệu chuỗi thời gian (ví dụ: giá cổ phiếu từng ngày), chúng ta **tuyệt đối không được dùng `train_test_split` ngẫu nhiên** mà phải cắt theo mốc thời gian (ví dụ: dùng quá khứ train, dùng tương lai test).

### Bài 5 (Challenge): Tự viết hàm `my_train_test_split` từ số 0
Viết hàm `my_train_test_split(X, y, test_size=0.2, shuffle=True, seed=None)` chỉ sử dụng thư viện NumPy và Pandas thuần (không import Scikit-Learn), trả về kết quả tương đương.
"""
    voai_connection_content = """
Trong thi đấu Olympic AI, nếu bạn vô tình để xảy ra hiện tượng **Data Leakage** trong bước phân chia dữ liệu (ví dụ: chuẩn hóa dữ liệu toàn bộ trước khi chia train/test, hoặc chia ngẫu nhiên trên dữ liệu có tính chất nhóm/thời gian), mô hình của bạn sẽ đạt điểm số ảo rất cao trên máy cá nhân nhưng khi nộp lên hệ thống chấm điểm độc lập thì điểm số rơi thẳng đứng. Hiểu sâu về `train_test_split` là tấm vé bảo hiểm đầu tiên cho bài thi của bạn.
"""
    debugging_content = """
**Tình huống thực tế:** Một thí sinh viết code chia dữ liệu và huấn luyện mô hình nhưng bị lỗi `Shape Mismatch`:

```python
# CODE CÓ LỖI SHAPE VECTOR/MATRIX:
X_wrong = df['Age'] # Day la Series 1D, shape (500,) khong phai Ma tran 2D (500, 1)!
y_target = df['Outcome']

# Khi goi model.fit(X_wrong, y_target), Scikit-Learn se văng loi:
# ValueError: Expected 2D array, got 1D array instead.
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao Scikit-Learn bắt buộc ma trận đặc trưng $X$ phải là mảng 2D kể cả khi chỉ có 1 đặc trưng duy nhất.
2. Nêu 2 cách sửa: dùng `df[['Age']]` (danh sách cột) hoặc `X_wrong.values.reshape(-1, 1)`.
"""
    knowledge_check_content = """
1. Sự khác biệt cốt lõi nhất giữa bài toán Hồi quy (Regression) và Phân loại (Classification) nằm ở đâu?
   - A. Thuật toán tối ưu hóa được sử dụng.
   - B. Bản chất của nhãn mục tiêu $y$: Số thực liên tục đối với Hồi quy, và Nhãn danh mục rời rạc đối với Phân loại.
   - C. Tốc độ thực thi của CPU.
   - D. Số lượng đặc trưng đầu vào $X$.

2. Tại sao chúng ta không được phép sử dụng tập Test để điều chỉnh siêu tham số (Hyperparameters) của mô hình?
   - A. Vì tập Test quá nhỏ.
   - B. Vì điều đó sẽ gây ra hiện tượng rò rỉ thông tin (Data Leakage), khiến mô hình bị Overfitting gián tiếp lên tập Test và mất khả năng tổng quát hóa khách quan.
   - C. Vì Scikit-Learn sẽ tự động khóa code.
   - D. Vì tập Test không có nhãn.

3. Tham số `stratify=y` trong `train_test_split` có tác dụng gì?
   - A. Sắp xếp lại thứ tự dữ liệu theo nhãn $y$.
   - B. Bảo đảm tỷ lệ phân phối giữa các lớp của nhãn $y$ ở tập Train và tập Test là hoàn toàn đồng nhất.
   - C. Tự động xóa các dòng có nhãn bị khuyết.
   - D. Nhân bản dữ liệu của lớp thiểu số.

4. Kích thước (shape) chuẩn mực của Ma trận đặc trưng $X$ trong bài toán có $N$ mẫu và $D$ biến số là gì?
   - A. $(N,)$
   - B. $(D, N)$
   - C. $(N, D)$
   - D. $(D,)$

5. Khi huấn luyện mô hình đạt độ chính xác 100% trên tập Train nhưng chỉ đạt 52% trên tập Test, hiện tượng này được gọi là gì?
   - A. Underfitting (Mô hình quá đơn giản)
   - B. Overfitting (Mô hình học vẹt)
   - C. Convergence (Mô hình hội tụ chuẩn)
   - D. High Bias
"""
    session_summary_content = """
* **What I learned:**
  - Bản chất toán học của Supervised Learning ($y = f(X)$).
  - Phân định rõ bài toán Hồi quy vs Phân loại.
  - Tầm quan trọng của Generalization và cơ chế Train/Test Split có Stratification.
* **What I can now do:**
  - Tách đúng ma trận $X$ và vector $y$ từ bất kỳ bảng dữ liệu nào.
  - Phân chia tập dữ liệu chuẩn mực, không rò rỉ thông tin trước khi huấn luyện.
* **What I still need to practice:**
  - Tìm hiểu cách thiết lập một mô hình cơ sở (Baseline) và thuật toán Hồi quy Tuyến tính ở Buổi 2.
"""
    next_session_content = """
Trong **Buổi M2_S02**, chúng ta sẽ chính thức bắt đầu huấn luyện mô hình: **Baseline Models & Hồi Quy Tuyến Tính (Linear Regression)** — Hiểu cặn kẽ hàm mất mát (MSE), gradient descent từ trực giác toán học đến các lệnh `fit` và `predict` của Scikit-Learn!
"""
    trace = {
        "items": [
            "Hiểu bản chất feature và target",
            "Supervised learning: Regression vs Classification",
            "Train/test split & Stratified partitioning"
        ],
        "deliverable": "Mã nguồn phân tách tập dữ liệu chuẩn hóa chống Data Leakage"
    }
    nb = create_notebook(
        module_id="M2",
        module_name="DẠY MÁY HỌC TỪ DỮ LIỆU",
        session_id="S01",
        topic="Bản Chất Của Học Máy: Feature, Target & Train/Test Split",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook phân chia dữ liệu huấn luyện và kiểm thử chuẩn hóa",
        competition_relevance="Nền tảng của quy trình thi đấu, bảo đảm tính trung thực và ngăn chặn rò rỉ dữ liệu.",
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
    save_notebook(nb, "course/M2_MACHINE_LEARNING/M2_S01.ipynb")


def build_m2_s02():
    objectives = [
        "Hiểu tại sao việc thiết lập một Baseline Model đơn giản là bước bắt buộc đầu tiên trong mọi dự án AI.",
        "Nắm vững công thức toán học và trực giác hình học của Hồi quy Tuyến tính (Linear Regression): y_hat = w^T * x + b.",
        "Hiểu trực quan Hàm mất mát Sai số Bình phương Trung bình (Mean Squared Error - MSE) và phần dư (Residuals).",
        "Thực thi thành thạo chu trình API chuẩn mực của Scikit-Learn: Khởi tạo -> `fit()` -> `predict()`."
    ]
    prerequisites = [
        "Hoàn thành Buổi M2_S01 (Feature, Target, Train/Test Split).",
        "Biết phương trình đường thẳng bậc nhất y = ax + b trong hình học giải tích."
    ]
    why_content = """
Khi bạn xây dựng một mô hình AI và đạt được sai số RMSE là 15.2, con số đó là TỐT hay XẤU? Bạn hoàn toàn không thể trả lời nếu không có một **Cột mốc so sánh (Baseline)**! Nếu một thuật toán ngây thơ chỉ đơn giản đoán giá trị trung bình (Mean) mà đã đạt sai số 15.5, thì mô hình phức tạp của bạn thực chất chẳng học được điều gì có giá trị.

Hồi quy Tuyến tính (Linear Regression) là nền tảng của toàn bộ ngành khoa học dữ liệu và học máy. Mọi khái niệm tối thượng sau này — từ Trọng số (Weights), Độ lệch (Bias), Hàm mất mát (Loss Function), đến Đạo hàm suy giảm (Gradient Descent) — đều bắt nguồn từ thuật toán thanh lịch này.
"""
    concept_content = r"""
### 1. Tư duy Baseline Model trong phòng thi
* **Baseline là gì?** Là mô hình đơn giản nhất có thể chạy được (Naive/Dummy Model), ví dụ: luôn đoán giá trị trung bình $\bar{y}$ (bài toán hồi quy) hoặc luôn đoán lớp chiếm đa số (bài toán phân loại).
* **Mục đích:** Thiết lập "mặt sàn" hiệu năng (Lower Bound). Bất kỳ mô hình Machine Learning thực thụ nào cũng bắt buộc phải đánh bại Baseline.

### 2. Bản chất Toán học của Hồi quy Tuyến tính (Linear Regression)
* **Phương trình dự đoán:**
  $$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_D x_D + b = \mathbf{w}^T \mathbf{x} + b$$
  Trong đó: $\mathbf{w}$ là vector trọng số (Weights/Coefficients), $b$ là hệ số tự do (Bias/Intercept).
* **Hàm mất mát (Loss Function - MSE):** Đo lường tổng bình phương khoảng cách sai lệch giữa giá trị thực tế $y_i$ và giá trị dự đoán $\hat{y}_i$:
  $$\mathcal{L}_{MSE}(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
* **Mục tiêu của quá trình `fit()`:** Tìm bộ tham số $(\mathbf{w}^*, b^*)$ sao cho hàm mất mát $\mathcal{L}_{MSE}$ đạt giá trị cực tiểu toàn cục!
"""
    visual_intuition_content = """
```
Trực quan hóa Hồi quy Tuyến tính & Phần dư (Residuals):
   y (Giá)
   ▲             • (Mẫu thực tế y_i)
   │            /│
   │           / ┆ <── Sai số phần dư e_i = y_i - y_hat_i
   │          /  ▼
   │         /───• (Dự đoán y_hat_i nằm trên đường thẳng)
   │        /
   │       /  Đường hồi quy: y_hat = w * x + b
   │      /
   └─────┴────────────────────────► x (Diện tích)
```
"""
    code_cells = [
        (
            """import numpy as np
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Khoi tao du lieu mau mo phong quan he giua Kinh nghiem lam viec va Muc luong
np.random.seed(42)
n_samples = 80

experience = np.random.uniform(1, 10, size=(n_samples, 1)) # Feature X (2D array)
# Luong thuc te = 15 trieu co ban + 4.5 trieu/nam kinh nghiem + nhieu ngau nhien
salary = 15.0 + 4.5 * experience.ravel() + np.random.normal(0, 3.0, size=n_samples) # Target y (1D)

# 2. Xay dung Baseline Model (Doan gia tri trung binh cua tap Train)
dummy = DummyRegressor(strategy='mean')
dummy.fit(experience, salary)
dummy_preds = dummy.predict(experience)
dummy_mse = mean_squared_error(salary, dummy_preds)

# 3. Huan luyen mo hinh Hoi quy Tuyen tinh thuc thu (Linear Regression)
lin_reg = LinearRegression()
lin_reg.fit(experience, salary)
lin_preds = lin_reg.predict(experience)
lin_mse = mean_squared_error(salary, lin_preds)

print("===== SO SANH HIEU NANG MO HINH =====")
print(f"Baseline (Dummy Mean) MSE : {dummy_mse:.2f}")
print(f"Linear Regression MSE     : {lin_mse:.2f}")
print(f"-> Linear Regression giam sai so: {((dummy_mse - lin_mse) / dummy_mse) * 100:.1f}%")

print(f"\\nTrong so hoc duoc (w): {lin_reg.coef_[0]:.2f} (Gia tri that: 4.50)")
print(f"He so bias hoc duoc (b) : {lin_reg.intercept_:.2f} (Gia tri that: 15.00)")
""",
            """Khối code minh họa chu trình chuẩn mực từ Baseline đến Model:
- Khởi tạo bài toán ước lượng lương dựa trên năm kinh nghiệm.
- `DummyRegressor(strategy='mean')`: Thiết lập chuẩn đối sánh cơ sở.
- `LinearRegression()`: Tối ưu hóa trọng số $w$ và bias $b$ qua phương thức `fit()`.
- Chứng minh mô hình tuyến tính giảm sai số vượt trội so với Baseline ngây thơ."""
        ),
        (
            """# 4. Truc quan hoa duong hoi quy va sai so phan du (Residuals)
plt.figure(figsize=(9, 5))
plt.scatter(experience, salary, color='#1f77b4', label='Du lieu thuc te (Data points)', alpha=0.7)
plt.plot(experience, dummy_preds, color='gray', linestyle='--', label=f'Baseline Mean (MSE={dummy_mse:.1f})')
plt.plot(experience, lin_preds, color='#d62728', linewidth=2, label=f'Linear Regression (MSE={lin_mse:.1f})')

# Ve cac duong phan du cho 5 diem dau tien
for i in range(5):
    plt.vlines(x=experience[i], ymin=min(salary[i], lin_preds[i]), ymax=max(salary[i], lin_preds[i]), 
               color='orange', linestyle=':', alpha=0.8)

plt.title('Truc quan hoa Hoi quy Tuyen tinh: Duong hoi quy va Phan du (Residuals)', fontsize=12, fontweight='bold')
plt.xlabel('So nam kinh nghiem (Years of Experience)')
plt.ylabel('Muc luong (Trieu VND)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
""",
            """Khối code trực quan hóa hình học của thuật toán:
- Đường thẳng màu xám biểu diễn Baseline ngây thơ.
- Đường thẳng màu đỏ biểu diễn nghiệm tối ưu của Linear Regression.
- Các vệt kẻ đứt màu cam minh họa trực quan phần dư $e_i = y_i - \hat{y}_i$ mà hàm mất mát MSE đang tìm cách thu nhỏ tối đa."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Phân tích hệ số học được
Nhìn vào giá trị `lin_reg.coef_[0]` và `lin_reg.intercept_`, hãy giải thích ý nghĩa kinh tế thực tế của 2 con số này (ví dụ: cứ mỗi năm kinh nghiệm tăng thêm, mức lương dự kiến tăng bao nhiêu triệu?).

### Bài 2 (Easy): Tính chỉ số Sai số Tuyệt đối Trung bình (MAE)
Sử dụng hàm `mean_absolute_error` từ Scikit-Learn để tính MAE cho cả Baseline và Linear Regression. Nêu điểm khác biệt trực giác giữa MAE và MSE.

### Bài 3 (Medium): Hồi quy tuyến tính đa biến (Multiple Linear Regression)
Tạo thêm một đặc trưng mới `Certifications` (số lượng chứng chỉ, từ 0 đến 5). Huấn luyện lại mô hình hồi quy với ma trận đặc trưng 2 chiều gồm cả `Experience` và `Certifications`. So sánh MSE xem có cải thiện không.

### Bài 4 (Medium): Kiểm tra tính ngẫu nhiên của phần dư (Residual Analysis)
Vẽ biểu đồ phân tán giữa giá trị dự đoán $\hat{y}$ (trục hoành) và phần dư $e = y - \hat{y}$ (trục tung). Giải thích tại sao nếu phần dư phân bố ngẫu nhiên quanh trục 0 thì mô hình được coi là phù hợp tốt (Good fit).

### Bài 5 (Challenge): Cài đặt Linear Regression từ số 0 bằng Công thức Giải tích (Normal Equation)
Nghiệm tối ưu của Hồi quy tuyến tính được tính bằng công thức ma trận đóng (Normal Equation):
$$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
Hãy dùng NumPy (`np.linalg.inv`, `@`) để tính toán trực tiếp trọng số mà không gọi thư viện Scikit-Learn và đối chiếu kết quả.
"""
    voai_connection_content = """
Trong 30 phút đầu tiên của bất kỳ cuộc thi AI nào, việc đầu tiên của một đội tuyển chuyên nghiệp là **dựng một Baseline hoàn chỉnh có thể nộp được (Working Submission)**. Đừng bao giờ bắt đầu bằng việc tinh chỉnh một mạng Transformer khổng lồ. Một pipeline Linear Regression sạch sẽ giúp bạn kiểm tra toàn bộ luồng nộp bài, kiểm tra metric chấm điểm và đảm bảo bạn có điểm số an toàn trên bảng xếp hạng trước khi thử nghiệm các kỹ thuật phức tạp hơn.
"""
    debugging_content = """
**Tình huống thực tế:** Một thí sinh huấn luyện mô hình trên dữ liệu chưa xử lý và nhận được cảnh báo lỗi `NaN` hoặc trọng số khổng lồ vô lý ($10^{15}$):

```python
# CODE CÓ LỖI ĐA CỘNG TUYẾN HOÀNG TOÀN (PERFECT MULTICOLLINEARITY):
# Thi sinh tao 2 dac trung giong het nhau ve mat toan hoc:
# Feature 1: Chieu cao tinh bang met (m)
# Feature 2: Chieu cao tinh bang centimet (cm) = 100 * Feature 1
# Khi do ma tran (X^T * X) khong kha nghich (Singular Matrix), dan toi loi so hoc!
```

**Nhiệm vụ của bạn:**
1. Giải thích hiện tượng Đa cộng tuyến hoàn hảo (Multicollinearity) làm sụp đổ phép nghịch đảo ma trận trong Linear Regression như thế nào.
2. Nêu cách phát hiện và khắc phục bằng cách loại bỏ một trong hai đặc trưng tương quan tuyệt đối ($r = 1.0$).
"""
    knowledge_check_content = """
1. Mục đích quan trọng nhất của việc xây dựng mô hình Baseline ở đầu dự án là gì?
   - A. Để nộp bài lấy điểm tuyệt đối ngay lập tức.
   - B. Để làm mốc quy chiếu tối thiểu (Benchmark), chứng minh mô hình phức tạp thực sự học được tri thức có ích.
   - C. Để thay thế hoàn toàn cho bước làm sạch dữ liệu.
   - D. Để kiểm tra tốc độ xung nhịp của vi xử lý.

2. Trong công thức Hồi quy Tuyến tính $\hat{y} = \mathbf{w}^T \mathbf{x} + b$, đại lượng $b$ (Bias/Intercept) đại diện cho điều gì?
   - A. Độ dốc của đường hồi quy.
   - B. Giá trị dự đoán của $\hat{y}$ khi tất cả các đặc trưng đầu vào $x$ đều bằng 0.
   - C. Sai số lớn nhất của mô hình.
   - D. Số lượng mẫu dữ liệu trong tập Train.

3. Hàm mất mát Sai số Bình phương Trung bình (MSE) phạt các sai số lớn như thế nào so với MAE?
   - A. Nhẹ hơn MAE.
   - B. Phạt nặng hơn rất nhiều (bình phương sai số), khiến mô hình rất nhạy cảm với các điểm ngoại lai (Outliers).
   - C. Bằng đúng MAE.
   - D. Bỏ qua hoàn toàn các sai số lớn.

4. Phương thức `.fit(X, y)` trong Scikit-Learn thực chất đang thực hiện thao tác gì?
   - A. Dự đoán nhãn cho dữ liệu mới.
   - B. Tối ưu hóa các tham số nội tại của mô hình (ví dụ: weights và bias) sao cho hàm mất mát đạt giá trị nhỏ nhất trên tập dữ liệu được cung cấp.
   - C. Xóa các hàng có dữ liệu khuyết.
   - D. Vẽ biểu đồ phân phối.

5. Nếu giá trị $R^2$ (Hệ số xác định) của mô hình hồi quy bằng $0.85$, điều đó có nghĩa là gì?
   - A. Mô hình bị sai $15\\%$ trên tập Test.
   - B. $85\\%$ sự biến thiên của biến mục tiêu $y$ đã được giải thích bởi các đặc trưng đầu vào $X$ trong mô hình.
   - C. Mô hình đã được huấn luyện qua 85 epochs.
   - D. Dữ liệu có 85 cột.
"""
    session_summary_content = """
* **What I learned:**
  - Tư duy bắt buộc về Baseline Model trong nghiên cứu và thi đấu AI.
  - Bản chất toán học của Hồi quy Tuyến tính và cơ chế tối ưu hàm mất mát MSE.
  - Thành thạo chu trình API của Scikit-Learn: Khởi tạo, `.fit()`, `.predict()`.
* **What I can now do:**
  - Huấn luyện mô hình hồi quy tuyến tính đầu tiên và phân tích ý nghĩa các trọng số.
  - Phân tích phần dư để đánh giá độ phù hợp của mô hình.
* **What I still need to practice:**
  - Chuyển sang bài toán phân loại và làm quen với mô hình phi tuyến: **Logistic Regression & Decision Trees** ở Buổi 3.
"""
    next_session_content = """
Trong **Buổi M2_S03**, chúng ta sẽ tiến sang phân loại nhãn: **Phân Loại Nhị Phân & Cây Quyết Định (Logistic Regression & Decision Trees)** — Khám phá hàm Sigmoid, Log-Loss, cơ chế chia nhánh theo độ tinh khiết (Gini Impurity) và trực giác về Random Forest!
"""
    trace = {
        "items": [
            "Baseline model & Dummy regressors",
            "Linear regression intuition & MSE loss",
            "Scikit-Learn fit/predict workflow & Residuals analysis"
        ],
        "deliverable": "Mã nguồn huấn luyện Linear Regression đầu tiên so sánh với Baseline"
    }
    nb = create_notebook(
        module_id="M2",
        module_name="DẠY MÁY HỌC TỪ DỮ LIỆU",
        session_id="S02",
        topic="Baseline Models & Hồi Quy Tuyến Tính (Linear Regression)",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook xây dựng Baseline và huấn luyện Linear Regression",
        competition_relevance="Xây dựng mốc quy chiếu điểm số đầu tiên (First Working Submission) trong phòng thi.",
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
    save_notebook(nb, "course/M2_MACHINE_LEARNING/M2_S02.ipynb")


def build_m2_s03():
    objectives = [
        "Hiểu cách chuyển hóa từ bài toán Hồi quy sang Phân loại nhị phân thông qua hàm kích hoạt Sigmoid và mô hình Logistic Regression.",
        "Nắm vững cơ chế hoạt động của Cây Quyết Định (Decision Tree): Chia nhánh dựa trên độ vẩn đục Gini (Gini Impurity) hoặc Entropy.",
        "Quan sát trực quan ranh giới quyết định (Decision Boundary) và hiện tượng quá khớp (Overfitting) khi cây phát triển quá sâu.",
        "Hiểu trực giác mở rộng sang Rừng Ngẫu Nhiên (Random Forest Ensemble) và trích xuất độ quan trọng đặc trưng (Feature Importance)."
    ]
    prerequisites = [
        "Hoàn thành Buổi M2_S02 (Hồi quy tuyến tính & Loss function).",
        "Hiểu khái niệm xác suất nằm trong khoảng [0, 1]."
    ]
    why_content = """
Trong thực tế, đa số các bài toán kinh doanh và thi đấu AI là bài toán **Phân loại (Classification)**: Dự đoán khách hàng có rời bỏ dịch vụ không, khối u là lành tính hay ác tính, người dùng có nhấp vào quảng cáo không. Đường thẳng của Hồi quy tuyến tính có thể cho ra kết quả âm hoặc lớn hơn 1, hoàn toàn không phù hợp để biểu diễn xác suất!

Logistic Regression và Decision Tree là hai trụ cột không thể thay thế: Một bên đại diện cho ranh giới phân tách tuyến tính êm dịu dựa trên xác suất, một bên đại diện cho cấu trúc suy luận dạng cây trực quan, có khả năng xử lý phi tuyến tính mà không cần giả định phân phối dữ liệu.
"""
    concept_content = r"""
### 1. Logistic Regression & Hàm Sigmoid
* Để đưa giá trị đầu ra $z = \mathbf{w}^T \mathbf{x} + b$ về khoảng xác suất $[0, 1]$, ta sử dụng hàm Sigmoid:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
* **Ngưỡng quyết định (Decision Threshold):** Thông thường, nếu $\sigma(z) \ge 0.5 \implies \hat{y} = 1$, ngược lại $\hat{y} = 0$.
* **Hàm mất mát (Binary Cross-Entropy / Log-Loss):** Phạt cực nặng nếu mô hình tự tin dự đoán xác suất gần 1 cho một mẫu thực tế có nhãn 0!

### 2. Cây Quyết Định (Decision Tree) & Chỉ số Gini Impurity
* Cây phân nhánh dữ liệu bằng các câu hỏi nhị phân dạng `Feature <= Threshold`.
* Tại mỗi nút (Node), thuật toán duyệt qua mọi đặc trưng và mọi ngưỡng chia để tìm ra nhát cắt làm giảm **Độ vẩn đục (Gini Impurity)** nhiều nhất:
  $$Gini(p) = 1 - \sum_{k=1}^K p_k^2$$
  (Gini = 0 khi nút hoàn toàn thuần khiết - tất cả các mẫu đều cùng một lớp).
* **Hiểm họa Overfitting:** Nếu để cây phát triển tự do không giới hạn độ sâu (`max_depth=None`), cây sẽ chia nhánh cho đến khi mỗi chiếc lá chỉ chứa đúng 1 mẫu dữ liệu -> Học vẹt 100% tập Train nhưng thất bại trên tập Test!
"""
    visual_intuition_content = """
```
Cấu trúc Cây Quyết Định (Decision Tree):
              [ Glucose <= 125.5 ]
                   /        \\
                 Yes         No
                 /            \\
        [ BMI <= 28.3 ]      [ Nút lá: 95% Mắc bệnh ]
           /       \\
         Yes        No
         /           \\
[ Lá: 5% Bệnh ]   [ Lá: 60% Bệnh ]
```
"""
    code_cells = [
        (
            """import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# 1. Tao tap du lieu phan loai 2 dac trung de de dang truc quan hoa mat phang
X, y = make_classification(
    n_samples=300, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, flip_y=0.05, class_sep=1.2, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 2. Huan luyen Logistic Regression
clf_lr = LogisticRegression()
clf_lr.fit(X_train, y_train)

# 3. Huan luyen Decision Tree (Co gioi han do sau de tranh Overfitting)
clf_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_dt.fit(X_train, y_train)

# 4. Huan luyen Random Forest
clf_rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
clf_rf.fit(X_train, y_train)

print(f"Logistic Regression Train Acc: {clf_lr.score(X_train, y_train):.3f} | Test Acc: {clf_lr.score(X_test, y_test):.3f}")
print(f"Decision Tree (d=3) Train Acc: {clf_dt.score(X_train, y_train):.3f} | Test Acc: {clf_dt.score(X_test, y_test):.3f}")
print(f"Random Forest (d=3) Train Acc: {clf_rf.score(X_train, y_train):.3f} | Test Acc: {clf_rf.score(X_test, y_test):.3f}")
""",
            """Khối code tạo lập bài toán phân loại 2D có nhiễu thực tế:
- Huấn luyện đồng thời 3 mô hình cốt lõi: Logistic Regression, Decision Tree (khống chế `max_depth=3`) và Random Forest.
- Đo lường và so sánh điểm số chính xác (Accuracy) trên cả tập Train và tập Test."""
        ),
        (
            """# 5. Truc quan hoa cau truc cay quyet dinh va Do quan trong dac trung
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# So do cay quyet dinh
plot_tree(clf_dt, feature_names=['Feature_1', 'Feature_2'], class_names=['Class_0', 'Class_1'], 
          filled=True, rounded=True, ax=axes[0], fontsize=9)
axes[0].set_title('Cau truc suy luan cua Decision Tree (max_depth=3)', fontweight='bold')

# Do quan trong cua dac trung (Feature Importances) tu Random Forest
importances = clf_rf.feature_importances_
axes[1].bar(['Feature_1', 'Feature_2'], importances, color=['#2b5c8f', '#d62728'], width=0.4)
axes[1].set_title('Do quan trong dac trung (Random Forest Feature Importances)', fontweight='bold')
axes[1].set_ylabel('Importance Score')
axes[1].set_ylim(0, 1.0)

plt.tight_layout()
plt.show()
""",
            """Khối code minh họa tính khả giải thích cao (Interpretability) của mô hình cây:
- `plot_tree`: Cho phép quan sát từng câu hỏi điều kiện tại mỗi nút nhánh và độ tinh khiết Gini.
- `feature_importances_`: Cho biết đặc trưng nào đóng góp nhiều nhất vào việc giảm độ vẩn đục toàn hệ thống."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Thử nghiệm Overfitting với `max_depth`
Huấn luyện một cây quyết định mới với `max_depth=None` (không giới hạn) và `min_samples_split=2`. Quan sát điểm số trên tập Train và tập Test. Nêu nhận xét về hiện tượng Overfitting.

### Bài 2 (Easy): Đọc xác suất dự đoán `predict_proba`
Sử dụng phương thức `clf_lr.predict_proba(X_test[:5])` để in ra xác suất dự đoán chi tiết cho 5 mẫu đầu tiên. Kiểm tra xem tổng xác suất của 2 cột có bằng 1.0 hay không.

### Bài 3 (Medium): Thay đổi ngưỡng quyết định (Threshold Tuning)
Mặc định ngưỡng phân loại là 0.5. Viết một đoạn code gán nhãn 1 nếu xác suất $P(y=1) \ge 0.3$. Thao tác này có lợi ích gì khi ta muốn bắt triệt để các ca bệnh nguy hiểm (ưu tiên Recall)?

### Bài 4 (Medium): So sánh tiêu chí chia nhánh Gini vs Entropy
Huấn luyện hai cây quyết định có cùng độ sâu `max_depth=4`, một cây dùng `criterion='gini'` và một cây dùng `criterion='entropy'`. So sánh độ chính xác và cấu trúc của hai cây.

### Bài 5 (Challenge): Vẽ đường ranh giới quyết định (Decision Boundary) 2D
Sử dụng `np.meshgrid` và `plt.contourf` để vẽ ranh giới quyết định 2D cho cả Logistic Regression (đường thẳng tuyến tính) và Decision Tree (các đường gấp khúc vuông góc theo trục).
"""
    voai_connection_content = """
Trong các bảng xếp hạng Olympic AI dạng dữ liệu bảng (Tabular Data), các mô hình họ cây (Decision Tree, Random Forest, GBDT) luôn là "ông hoàng" thống trị. Chúng không đòi hỏi chuẩn hóa đặc trưng (Feature Scaling), có khả năng tự động xử lý tương quan phi tuyến, và cung cấp ngay lập tức bảng **Feature Importance** giúp thí sinh biết đặc trưng nào đang mang lại giá trị cao nhất để tiếp tục đào sâu kỹ nghệ đặc trưng.
"""
    debugging_content = """
**Tình huống thực tế:** Một bạn sinh viên dùng Decision Tree huấn luyện trên tập dữ liệu bị mất cân bằng trầm trọng (95% nhãn 0, 5% nhãn 1) nhưng không đặt tham số `class_weight`, khiến cây luôn dự đoán ra nhãn 0 để đạt Gini thấp:

```python
# CODE CÓ VẤN ĐỀ TRỌNG SỐ LỚP:
# Khi du lieu mat can bang nang, cay quyet dinh se bo qua hoan toan lop thieu so!
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao việc tối ưu Gini Impurity thuần túy có thể bỏ rơi lớp thiểu số.
2. Sửa lỗi bằng cách thêm tham số `class_weight='balanced'` trong Scikit-Learn để tự động phạt nặng các lỗi sai trên lớp hiếm.
"""
    knowledge_check_content = """
1. Hàm Sigmoid biến đổi bất kỳ giá trị số thực nào $z \in (-\infty, +\infty)$ về khoảng giá trị nào?
   - A. $[-1, 1]$
   - B. $[0, 1]$
   - C. $[0, +\infty)$
   - D. $[-\pi, \pi]$

2. Độ tinh khiết Gini (Gini Impurity) của một nút lá bằng 0 khi nào?
   - A. Khi nút đó chứa số lượng mẫu bằng nhau giữa các lớp.
   - B. Khi tất cả các mẫu trong nút lá đó đều thuộc về cùng một lớp duy nhất (Pure node).
   - C. Khi nút đó không chứa mẫu nào.
   - D. Khi cây đạt độ sâu tối đa.

3. Kỹ thuật nào sau đây giúp kiểm soát trực tiếp hiện tượng Overfitting trong Cây Quyết Định?
   - A. Tăng số lượng mẫu lên vô hạn.
   - B. Cắt tỉa cây (Pruning) hoặc giới hạn độ sâu tối đa bằng siêu tham số `max_depth`.
   - C. Bỏ qua bước chia Train/Test.
   - D. Xóa nhãn của dữ liệu.

4. Tại sao ranh giới quyết định (Decision Boundary) của một Cây Quyết Định luôn có dạng các đường thẳng song song với các trục tọa độ?
   - A. Vì cây chỉ sử dụng hàm sin và cosin.
   - B. Vì tại mỗi bước phân nhánh, cây chỉ so sánh một đặc trưng đơn lẻ với một giá trị ngưỡng (`Feature_i <= Threshold`).
   - C. Do lỗi hiển thị của Matplotlib.
   - D. Vì dữ liệu đã được chuẩn hóa Min-Max.

5. Rừng Ngẫu Nhiên (Random Forest) cải thiện độ ổn định và giảm phương sai (Variance) của một cây quyết định đơn lẻ dựa trên nguyên lý cốt lõi nào?
   - A. Huấn luyện duy nhất một cây với kích thước khổng lồ.
   - B. Tạo ra nhiều cây quyết định độc lập trên các tập con dữ liệu ngẫu nhiên (Bootstrap Aggregating / Bagging) và lấy biểu quyết số đông (Ensemble Voting).
   - C. Bỏ qua hoàn toàn dữ liệu có nhiễu.
   - D. Luôn luôn chọn đặc trưng đầu tiên.
"""
    session_summary_content = """
* **What I learned:**
  - Bản chất hàm Sigmoid và mô hình Logistic Regression trong phân loại nhị phân.
  - Cơ chế phân nhánh theo độ vẩn đục Gini Impurity của Cây Quyết Định.
  - Nhận diện trực quan hiện tượng Overfitting qua độ sâu của cây (`max_depth`).
  - Trực giác về Random Forest và trích xuất độ quan trọng đặc trưng (Feature Importance).
* **What I can now do:**
  - Huấn luyện và trực quan hóa cây suy luận quyết định trên dữ liệu thực tế.
  - Điều chỉnh `max_depth` và trích xuất đặc trưng then chốt của bài toán.
* **What I still need to practice:**
  - Học cách đo lường độ chính xác đa chiều: **Confusion Matrix, Precision, Recall, F1-Score, ROC-AUC** ở Buổi 4.
"""
    next_session_content = """
Trong **Buổi M2_S04**, chúng ta sẽ đi sâu vào "trọng tài" của mọi cuộc thi: **Đọc & Hiểu Toàn Diện Metric Đánh Giá Mô Hình** — Bóc trần nghịch lý Accuracy, phân tích ma trận nhầm lẫn (Confusion Matrix), và làm chủ cán cân Precision-Recall!
"""
    trace = {
        "items": [
            "Logistic regression & Sigmoid activation",
            "Decision trees & Gini impurity",
            "Overfitting control via max_depth & Random Forest intuition"
        ],
        "deliverable": "Mã nguồn huấn luyện phân loại nhị phân và trích xuất Feature Importance"
    }
    nb = create_notebook(
        module_id="M2",
        module_name="DẠY MÁY HỌC TỪ DỮ LIỆU",
        session_id="S03",
        topic="Phân Loại Nhị Phân & Cây Quyết Định (Logistic Regression & Decision Trees)",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook huấn luyện Decision Tree và trích xuất Feature Importance",
        competition_relevance="Mô hình nền tảng cho dữ liệu bảng thi đấu, khám phá các đặc trưng quan trọng nhất.",
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
    save_notebook(nb, "course/M2_MACHINE_LEARNING/M2_S03.ipynb")


def build_m2_s04():
    objectives = [
        "Hiểu cặn kẽ 'Nghịch lý độ chính xác' (Accuracy Paradox) trên các tập dữ liệu mất cân bằng nhãn.",
        "Nắm vững 4 thành phần của Ma trận Nhầm lẫn (Confusion Matrix): TP (True Positive), FP, TN, FN.",
        "Phân biệt và tính toán chuẩn xác: Precision (Độ chuẩn xác), Recall (Độ thu hồi / Độ nhạy), và F1-Score.",
        "Hiểu sự đánh đổi giữa Precision và Recall (Precision-Recall Tradeoff) và các metric đánh giá hồi quy: MAE, MSE, RMSE, R²."
    ]
    prerequisites = [
        "Hoàn thành Buổi M2_S03 (Phân loại nhị phân, Logistic Regression).",
        "Hiểu khái niệm dương tính (Positive) và âm tính (Negative)."
    ]
    why_content = """
Hãy tưởng tượng bạn tham gia một cuộc thi Olympic AI phát hiện tế bào ung thư hiếm gặp, nơi chỉ có 0.5% mẫu dữ liệu là bệnh nhân thực sự. Một thí sinh ngây thơ nộp một đoạn mã chỉ gồm 1 dòng: `return 0` (luôn đoán tất cả mọi người khỏe mạnh). Mô hình đó đạt **Độ chính xác (Accuracy) lên tới 99.5%**, nhưng giá trị thực tế của nó hoàn toàn bằng 0 vì để lọt 100% bệnh nhân ung thư!

Nếu bạn chỉ biết nhìn vào duy nhất chỉ số Accuracy, bạn sẽ bị "mù" trước những sai lầm chết người của mô hình. Trong thi đấu chuyên nghiệp, ban tổ chức luôn sử dụng các metric khắt khe như **F1-Score, Macro-F1, ROC-AUC, hay Log-Loss**. Làm chủ hệ thống metric đánh giá giúp bạn đọc vị chính xác sức khỏe của mô hình và tối ưu hóa đúng mục tiêu ban tổ chức yêu cầu.
"""
    concept_content = r"""
### 1. Ma trận Nhầm lẫn (Confusion Matrix)
| Thực tế \ Dự đoán | Dự đoán Dương (Predicted Positive) | Dự đoán Âm (Predicted Negative) |
| :--- | :--- | :--- |
| **Thực tế Dương (Actual Positive)** | **True Positive (TP)**: Đoán đúng bệnh | **False Negative (FN)**: Bỏ sót bệnh (Lỗi loại II) |
| **Thực tế Âm (Actual Negative)** | **False Positive (FP)**: Báo động giả (Lỗi loại I) | **True Negative (TN)**: Đoán đúng người khỏe |

### 2. Các chỉ số đo lường cốt lõi trong Phân loại
* **Accuracy (Độ chính xác tổng thể):** $\frac{TP + TN}{TP + TN + FP + FN}$ (Chỉ đáng tin khi nhãn cân bằng).
* **Precision (Độ chuẩn xác):** $\frac{TP}{TP + FP}$
  - Trong số những mẫu mô hình **nói là Dương**, có bao nhiêu phần trăm là Dương thật?
  - Quan trọng khi cái giá của việc báo động giả (FP) là đắt (ví dụ: bộ lọc thư rác không được xóa nhầm email quan trọng của khách hàng).
* **Recall / Sensitivity (Độ thu hồi / Độ nhạy):** $\frac{TP}{TP + FN}$
  - Trong số những mẫu **thực sự là Dương**, mô hình đã bắt được bao nhiêu phần trăm?
  - Cực kỳ quan trọng trong y tế và an ninh: Bắt buộc không được để sót bệnh nhân (giảm thiểu tối đa FN).
* **F1-Score (Trung bình điều hòa giữa Precision và Recall):**
  $$F_1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$
  F1 chỉ cao khi cả Precision và Recall đều đồng thời cao!

### 3. Các metric trong bài toán Hồi quy (Regression Metrics)
* **MAE (Mean Absolute Error):** Trung bình sai số tuyệt đối, không phóng đại outlier.
* **MSE (Mean Squared Error):** Bình phương sai số, phạt rất nặng các sai lệch lớn.
* **RMSE (Root MSE):** Căn bậc hai của MSE, có cùng đơn vị đo với biến mục tiêu ban đầu.
"""
    visual_intuition_content = """
```
Cán cân Precision vs Recall Tradeoff:
Hạ ngưỡng quyết định (Threshold giảm):
  ==> Bắt được nhiều ca bệnh hơn (Recall TĂNG ⬆️)
  ==> Nhưng báo động giả nhiều hơn (Precision GIẢM ⬇️)

Tăng ngưỡng quyết định (Threshold tăng):
  ==> Chỉ báo bệnh khi cực kỳ chắc chắn (Precision TĂNG ⬆️)
  ==> Nhưng bỏ sót nhiều ca bệnh (Recall GIẢM ⬇️)
```
"""
    code_cells = [
        (
            """import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

# 1. Gia lap bai toan phat hien gian lan tin dung (Fraud Detection) mat can bang nang
np.random.seed(42)
n_test = 200

# Thuc te: 180 giao dich hop le (0), 20 giao dich gian lan (1)
y_true = np.array([0] * 180 + [1] * 20)

# Mo hinh A: "Mo hinh luoi bieng" (Luon doan la 0 de gianh Accuracy cao)
y_pred_dummy = np.zeros(n_test, dtype=int)

# Mo hinh B: "Mo hinh Machine Learning thuc thu"
# Bat duoc 16/20 ca gian lan, nhung bao dong nham 8 giao dich hop le
y_pred_ml = y_true.copy()
y_pred_ml[np.random.choice(np.where(y_true == 1)[0], size=4, replace=False)] = 0 # 4 ca FN (bo sot)
y_pred_ml[np.random.choice(np.where(y_true == 0)[0], size=8, replace=False)] = 1 # 8 ca FP (bao nham)

# 2. So sanh su danh doi giua Accuracy va F1-Score
print("===== NGHICH LY ACCURACY TREN DU LIEU MAT CAN BANG =====")
print(f"Mo hinh A (Luon doan 0) -> Accuracy: {(y_pred_dummy == y_true).mean() * 100:.1f}% | F1-Score: {f1_score(y_true, y_pred_dummy, zero_division=0):.3f}")
print(f"Mo hinh B (ML thuc thu) -> Accuracy: {(y_pred_ml == y_true).mean() * 100:.1f}% | F1-Score: {f1_score(y_true, y_pred_ml):.3f}")
print("-> Ket luan: Accuracy cao hon chua chac mo hinh da hoat dong tot hon!")
""",
            """Khối code mô phỏng chân thực "Nghịch lý độ chính xác":
- Mô hình ngây thơ luôn đoán 0 đạt Accuracy tới 90.0% nhưng có $F1 = 0$ (vô dụng).
- Mô hình ML thực thụ có Accuracy 94.0% nhưng đạt $F1 = 0.727$, bắt được 80% các vụ gian lận nguy hiểm."""
        ),
        (
            """# 3. Truc quan hoa Ma tran Nham lan (Confusion Matrix) va Bao cao chi tiet
cm = confusion_matrix(y_true, y_pred_ml)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
            xticklabels=['Doan Hop le (0)', 'Doan Gian lan (1)'],
            yticklabels=['Thuc te Hop le (0)', 'Thuc te Gian lan (1)'])
plt.title('Ma tran Nham lan (Confusion Matrix) - Mo hinh B', fontweight='bold')
plt.ylabel('Nhan Thuc te')
plt.xlabel('Nhan Du doan')
plt.show()

print("\\n===== CLASSIFICATION REPORT CHI TIET =====")
print(classification_report(y_true, y_pred_ml, target_names=['Hop le (0)', 'Gian lan (1)']))
""",
            """Khối code vẽ Heatmap của Confusion Matrix và xuất `classification_report`:
- Hiển thị rõ ràng $TN = 172, FP = 8, FN = 4, TP = 16$.
- `classification_report` cung cấp cái nhìn 360 độ: Precision, Recall, F1-Score cho từng lớp riêng biệt và điểm số trung bình vĩ mô (Macro Average)."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Tính toán thủ công từ Confusion Matrix
Dựa trên ma trận nhầm lẫn ở code cell trên ($TN=172, FP=8, FN=4, TP=16$), hãy tự viết công thức Python tính lại:
1. $Precision = \\frac{TP}{TP + FP}$
2. $Recall = \\frac{TP}{TP + FN}$
3. $F_1 = 2 \\times \\frac{Precision \\times Recall}{Precision + Recall}$
Đối chiếu xem kết quả có khớp với `classification_report` không.

### Bài 2 (Easy): Đánh giá bài toán chẩn đoán y tế
Trong bài toán phát hiện bệnh truyền nhiễm nguy hiểm, giữa việc tăng Precision (giảm báo động nhầm) và tăng Recall (giảm bỏ sót người bệnh), bạn sẽ ưu tiên metric nào hơn? Vì sao?

### Bài 3 (Medium): Macro F1 vs Weighted F1
Tìm hiểu sự khác biệt giữa `f1_score(y_true, y_pred, average='macro')` và `average='weighted'`. Khi tập dữ liệu có một lớp cực nhỏ (chiếm 1%), việc dùng Macro-F1 sẽ phạt mô hình như thế nào nếu mô hình dự đoán kém trên lớp nhỏ đó?

### Bài 4 (Medium): Tính toán các metric Hồi quy
Cho hai mảng: `y_real = np.array([100, 150, 200, 250])` và `y_hat = np.array([110, 140, 220, 240])`. Viết hàm tính:
1. Sai số tuyệt đối trung bình (MAE)
2. Căn bậc hai sai số bình phương trung bình (RMSE)

### Bài 5 (Challenge): Vẽ đường cong Precision-Recall Tradeoff
Cho mảng xác suất dự đoán $P(y=1)$ của một mô hình. Viết vòng lặp quét qua 10 ngưỡng quyết định khác nhau từ $0.1$ đến $0.9$. Với mỗi ngưỡng, tính Precision và Recall tương ứng, sau đó vẽ biểu đồ thể hiện sự đánh đổi (Precision-Recall Curve).
"""
    voai_connection_content = """
Trong thể lệ thi đấu Olympic AI, ban tổ chức luôn nêu rõ: *"Bài thi được đánh giá bằng chỉ số Macro-F1 (hoặc Log-Loss, hoặc RMSE)"*. Một lỗi sơ đẳng của thí sinh non kinh nghiệm là tối ưu hàm mất mát sai hướng: Đề bài yêu cầu tối ưu Macro-F1 nhưng thí sinh chỉ nhìn vào điểm Accuracy của Scikit-Learn. Khi bạn nắm vững cách đọc và tinh chỉnh ngưỡng theo đúng metric của ban tổ chức, thứ hạng của bạn trên bảng tổng sắp có thể tăng hàng chục bậc chỉ sau 1 dòng lệnh điều chỉnh ngưỡng quyết định!
"""
    debugging_content = """
**Tình huống thực tế:** Một thí sinh chạy hàm `f1_score` trên bài toán phân loại đa lớp (Multi-class Classification) và bị báo lỗi:

```python
# CODE CÓ LỖI METRIC KHI PHÂN LOẠI NHIỀU LỚP:
y_true_multi = [0, 1, 2, 1, 0, 2]
y_pred_multi = [0, 2, 2, 1, 0, 1]
# score = f1_score(y_true_multi, y_pred_multi) 
# Bao loi: ValueError: Target is multiclass but average='binary'
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao mặc định `f1_score` chỉ hoạt động cho nhị phân (`average='binary'`).
2. Sửa lỗi bằng cách truyền thêm đối số `average='macro'` hoặc `average='weighted'`.
"""
    knowledge_check_content = """
1. Trong Ma trận Nhầm lẫn (Confusion Matrix), trường hợp một bệnh nhân thực sự mắc bệnh nhưng mô hình lại kết luận là khỏe mạnh được gọi là gì?
   - A. True Positive (TP)
   - B. False Positive (FP - Báo động giả)
   - C. False Negative (FN - Bỏ sót bệnh)
   - D. True Negative (TN)

2. Khi nhãn mục tiêu bị mất cân bằng nghiêm trọng (ví dụ 99% âm tính, 1% dương tính), chỉ số nào sau đây là KÉM TIN CẬY NHẤT để đánh giá chất lượng mô hình?
   - A. Accuracy (Độ chính xác tổng thể)
   - B. Recall
   - C. F1-Score
   - D. Precision

3. Chỉ số F1-Score là loại trung bình nào giữa Precision và Recall?
   - A. Trung bình cộng (Arithmetic Mean)
   - B. Trung vị (Median)
   - C. Trung bình nhân (Geometric Mean)
   - D. Trung bình điều hòa (Harmonic Mean)

4. Metric hồi quy nào đo lường độ lệch trung bình giữa giá trị dự đoán và thực tế nhưng đồng thời giữ nguyên đơn vị đo lường gốc của biến mục tiêu?
   - A. Mean Squared Error (MSE)
   - B. Root Mean Squared Error (RMSE)
   - C. R-squared ($R^2$)
   - D. Log-Loss

5. Nếu bạn muốn hệ thống chống thư rác của mình tuyệt đối không được xóa nhầm các email quan trọng của người dùng vào hòm rác, bạn cần ưu tiên tối ưu chỉ số nào cao nhất?
   - A. Recall
   - B. Precision
   - C. Mean Squared Error
   - D. Gini Impurity
"""
    session_summary_content = """
* **What I learned:**
  - Bóc trần bản chất của Accuracy Paradox trên tập dữ liệu mất cân bằng.
  - 4 thành phần nền tảng của Confusion Matrix (TP, FP, TN, FN).
  - Công thức và ý nghĩa thực tế của Precision, Recall, F1-Score và Precision-Recall Tradeoff.
  - Các metric hồi quy cốt lõi: MAE, MSE, RMSE, R².
* **What I can now do:**
  - Đọc và phân tích sâu sắc bảng `classification_report` của Scikit-Learn.
  - Lựa chọn chính xác metric tối ưu cho từng bài toán thi đấu Olympic AI cụ thể.
* **What I still need to practice:**
  - Kết hợp toàn bộ kiến thức M2 để hoàn thành **Model ML Đầu Tiên & Báo Cáo Kết Quả Chuẩn Thi Đấu** ở Buổi 5.
"""
    next_session_content = """
Buổi tiếp theo (**M2_S05**) là buổi tổng kết đỉnh cao của Module 2: **Huấn Luyện Model Đầu Tiên & Báo Cáo Đánh Giá Kết Quả Chuẩn Thi Đấu**! Bạn sẽ xây dựng trọn vẹn một pipeline học máy từ Tiền xử lý, Baseline, Thử nghiệm nhiều mô hình đến xuất Báo cáo so sánh và nhận **Certificate of Machine Learning Basics**!
"""
    trace = {
        "items": [
            "Hiểu bản chất Accuracy Paradox & Confusion matrix",
            "Precision, Recall, F1-score & Cán cân Tradeoff",
            "Regression metrics: MAE, MSE, RMSE & Classification report"
        ],
        "deliverable": "Mã nguồn đánh giá ma trận nhầm lẫn và phân tích đa metric chuẩn thi đấu"
    }
    nb = create_notebook(
        module_id="M2",
        module_name="DẠY MÁY HỌC TỪ DỮ LIỆU",
        session_id="S04",
        topic="Đọc & Hiểu Toàn Diện Metric Đánh Giá Mô Hình (Evaluation Metrics)",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook đánh giá metric chuyên sâu và trực quan hóa Confusion Matrix",
        competition_relevance="Căn chỉnh mô hình theo đúng tiêu chí chấm điểm của ban giám khảo Olympic AI.",
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
    save_notebook(nb, "course/M2_MACHINE_LEARNING/M2_S04.ipynb")


def build_m2_s05():
    objectives = [
        "Xây dựng một End-to-End Machine Learning Pipeline khép kín: Tiền xử lý -> Chuẩn hóa đặc trưng -> Huấn luyện -> Đánh giá đa metric.",
        "So sánh định lượng hiệu năng giữa 3 cấp độ mô hình: Baseline ngây thơ vs Mô hình tuyến tính vs Mô hình cây (Decision Tree / Random Forest).",
        "Thực hiện phân tích lỗi sai (Error Analysis) để tìm ra điểm mù của mô hình.",
        "Hoàn thành sản phẩm đầu ra chính thức của Module 2: Huấn luyện model đầu tiên và xuất Báo cáo kết quả chuẩn thi đấu (ML Result Report)."
    ]
    prerequisites = [
        "Hoàn thành trọn vẹn 4 buổi trước (M2_S01 đến M2_S04).",
        "Nắm vững các bước làm sạch dữ liệu từ Module M1."
    ]
    why_content = """
Chào mừng bạn đến với vạch đích của Module 2! Trong mọi cuộc thi AI, người chiến thắng không phải là người gọi được nhiều thư viện nhất, mà là người có một **Quy trình Thực nghiệm khoa học, chặt chẽ và có thể tái lập (Disciplined ML Workflow)**.

Hôm nay, bạn sẽ không làm các bài tập manh mún nữa. Bạn sẽ đóng vai một thí sinh thực thụ: Nhận một bài toán thực tế, dựng Baseline, chuẩn hóa dữ liệu, đưa 3 thuật toán vào "đấu trường so găng", phân tích số liệu trung thực bằng các metric đã học, và xuất bản một **Báo cáo Kết quả (ML Result Report)** hoàn chỉnh. Đây chính là cột mốc biến bạn từ một người học code thành một Kỹ sư Machine Learning thực thụ!
"""
    concept_content = r"""
### 1. Chuẩn hóa đặc trưng (Feature Scaling)
Nhiều thuật toán nhạy cảm với thang đo của dữ liệu (ví dụ: Tuổi từ 18-60 nhưng Lương từ 10 triệu - 100 triệu). Nếu không chuẩn hóa, thuật toán tối ưu sẽ bị kéo lệch theo biến có biên độ lớn.
* **StandardScaler (Z-score Standardization):**
  $$x' = \frac{x - \mu}{\sigma}$$
  Đưa phân phối về trung bình bằng 0 và độ lệch chuẩn bằng 1.
* *Quy tắc sống còn:* Luôn `fit_transform` trên tập Train, nhưng **chỉ `transform` trên tập Test**!

### 2. Cấu trúc Báo cáo Kết quả Machine Learning (ML Result Report) chuẩn Olympic
1. **Bài toán & Dữ liệu (Problem Formulation):** Mục tiêu dự đoán, tỷ lệ phân chia Train/Test, kiểm tra mất cân bằng nhãn.
2. **Thiết lập Baseline:** Hiệu năng của Dummy Model (sàn kỹ thuật).
3. **Bảng so sánh mô hình (Model Benchmark Table):** Bảng tổng hợp so sánh các thuật toán (Baseline vs Logistic Regression vs Decision Tree vs Random Forest) trên cả Train và Test qua các metric cốt lõi (Accuracy, F1, Precision, Recall).
4. **Phân tích lỗi (Error Analysis):** Khám phá những trường hợp mô hình đoán sai nhiều nhất và lý giải nguyên nhân dữ liệu.
5. **Kết luận & Bước đi tiếp theo:** Khuyến nghị hướng cải tiến (ví dụ: cần thêm đặc trưng gì, hay cần chuyển sang mạng nơ-ron sâu ở Module M3).
"""
    visual_intuition_content = """
```
Quy trình End-to-End ML Pipeline:
[Raw Dataset]
      │
[Train / Test Split] (Stratified 80/20)
      │
[Feature Scaler] ──> fit_transform(X_train) ──> transform(X_test)
      │
[Model Arena] ───┬──> Model 1: Baseline (Dummy)
                 ├──> Model 2: Logistic Regression
                 ├──> Model 3: Decision Tree
                 └──> Model 4: Random Forest
      │
[Evaluation & Benchmark Table] (Accuracy, F1-Score, Confusion Matrix)
      │
[Official ML Result Report & Certification]
```
"""
    code_cells = [
        (
            """import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Khoi tao bai toan thuc te: Du doan kha nang roi bo dich vu (Customer Churn)
X, y = make_classification(
    n_samples=600, n_features=4, n_informative=3, n_redundant=1,
    weights=[0.75, 0.25], flip_y=0.03, random_state=101
)

feature_names = ['Account_Age', 'Monthly_Usage', 'Support_Tickets', 'Bill_Amount']
df_features = pd.DataFrame(X, columns=feature_names)

# Chia tap du lieu chuan hoa
X_train, X_test, y_train, y_test = train_test_split(
    df_features, y, test_size=0.25, random_state=42, stratify=y
)

# Chuan hoa dac trung (Chong Data Leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Tap Train: {X_train.shape[0]} mau | Tap Test: {X_test.shape[0]} mau")
print("Ty le khach hang roi bo (Churn=1) trong tap Test: {:.1f}%".format(y_test.mean() * 100))
""",
            """Khối code khởi tạo bài toán dự đoán rời bỏ dịch vụ thực tế:
- Bộ dữ liệu gồm 4 đặc trưng liên tục với thang đo khác biệt.
- Sử dụng `StandardScaler` đúng chuẩn chống rò rỉ dữ liệu (`fit_transform` trên Train, `transform` trên Test)."""
        ),
        (
            """# 2. Dau truong so sanh hieu nang (Model Benchmark Arena)
models = {
    'Baseline (Majority)': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree (d=3)': DecisionTreeClassifier(max_depth=3, random_state=42),
    'Random Forest (50 trees)': RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
}

results = []

for name, model in models.items():
    # Huan luyen tren tap Train da scale
    model.fit(X_train_scaled, y_train)
    
    # Du doan tren ca Train va Test de kiem tra Overfitting
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_rec = recall_score(y_test, test_preds, zero_division=0)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    
    results.append({
        'Model': name,
        'Train Acc': f"{train_acc:.3f}",
        'Test Acc': f"{test_acc:.3f}",
        'Test Precision': f"{test_prec:.3f}",
        'Test Recall': f"{test_rec:.3f}",
        'Test F1-Score': f"{test_f1:.3f}"
    })

df_results = pd.DataFrame(results)
print("===== BANG SO SANH HIEU NANG MO HINH (BENCHMARK TABLE) =====")
print(df_results.to_string(index=False))
""",
            """Khối code xây dựng một khung đánh giá tự động đa mô hình:
- Tự động lặp qua 4 mô hình, huấn luyện và đánh giá trên 5 chỉ số khác nhau.
- Xuất ra Bảng Benchmark trực quan thể hiện rõ ràng sự vượt trội của các mô hình học máy so với Baseline ngây thơ."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Phân tích bảng kết quả Benchmark
Dựa vào bảng kết quả trên, hãy cho biết:
1. Mô hình nào đạt điểm $F1-Score$ cao nhất trên tập Test?
2. Có hiện tượng Overfitting nghiêm trọng xảy ra ở mô hình nào không (so sánh chênh lệch giữa `Train Acc` và `Test Acc`)?

### Bài 2 (Easy): Thử nghiệm với thuật toán K-Láng giềng gần nhất (KNN)
Import `KNeighborsClassifier` từ `sklearn.neighbors`. Thêm mô hình KNN với `n_neighbors=5` vào dictionary `models` và chạy lại bảng đánh giá. Giải thích tại sao KNN bắt buộc phải chuẩn hóa đặc trưng (Scaling) trước khi chạy.

### Bài 3 (Medium): Xuất báo cáo Markdown tự động
Viết một hàm `export_ml_report(df_results, filename="ML_RESULT_REPORT.md")` tự động ghi bảng so sánh hiệu năng kèm nhận xét vào một file Markdown trong thư mục làm việc của bạn.

### Bài 4 (Medium): Phân tích trường hợp đoán sai (Error Analysis)
Lọc ra các chỉ số index của tập Test mà mô hình tốt nhất đưa ra dự đoán sai (`test_preds != y_test`). Quan sát giá trị các đặc trưng của những mẫu bị đoán sai đó và đưa ra nhận xét: Tại sao mô hình lại bị nhầm lẫn ở những trường hợp này?

### Bài 5 (Challenge): Tối ưu siêu tham số đơn giản (Simple Grid Search)
Viết một vòng lặp thử nghiệm tham số `max_depth` của Decision Tree từ 1 đến 10. Vẽ biểu đồ thể hiện điểm `Train F1` và `Test F1` theo `max_depth` để tìm ra độ sâu tối ưu nhất giúp mô hình tránh cả Underfitting lẫn Overfitting.
"""
    voai_connection_content = """
Khi nộp bài thi Olympic AI, một báo cáo kỹ thuật rõ ràng với bảng so sánh Baseline vs Candidate Models là minh chứng cho sự chuyên nghiệp của bạn. Ban giám khảo đánh giá rất cao những thí sinh hiểu rõ:
1. Tại sao mình chọn mô hình A mà không chọn mô hình B?
2. Mô hình có bị overfit hay không?
3. Tradeoff giữa tốc độ suy diễn và độ chính xác F1 là bao nhiêu?
Làm chủ kỹ năng lập báo cáo này là bạn đã tiến sát tới tiêu chuẩn của một vận động viên AI cấp quốc gia.
"""
    debugging_content = """
**Tình huống thực tế:** Một thí sinh chuẩn hóa dữ liệu bằng `StandardScaler` trước khi chia `train_test_split`:

```python
# CODE CÓ LỖI RÒ RỈ DỮ LIỆU NGHIÊM TRỌNG:
# scaler = StandardScaler()
# X_all_scaled = scaler.fit_transform(X) # Rò rỉ Mean và Std của tập Test vào tập Train!
# X_train, X_test, y_train, y_test = train_test_split(X_all_scaled, y)
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao việc tính `mean` và `std` trên toàn bộ tập dữ liệu trước khi chia lại vi phạm tính trung thực của kiểm thử (Data Leakage).
2. Viết lại quy trình chuẩn: Phân chia tập trước -> `fit_transform` trên `X_train` -> `transform` trên `X_test`.
"""
    knowledge_check_content = """
1. Thứ tự đúng đắn của một quy trình Machine Learning chuẩn mực là gì?
   - A. Huấn luyện mô hình -> Thu thập dữ liệu -> Đánh giá -> Chia Train/Test.
   - B. Chia Train/Test -> Chuẩn hóa đặc trưng (Fit trên Train) -> Xây dựng Baseline -> Thử nghiệm mô hình -> Đánh giá đa metric.
   - C. Chuẩn hóa toàn bộ dữ liệu -> Huấn luyện mô hình phức tạp nhất -> Bỏ qua Baseline.
   - D. Đánh giá trên tập Train -> Nộp bài.

2. Tại sao chúng ta chỉ được gọi `scaler.transform(X_test)` mà KHÔNG ĐƯỢC gọi `scaler.fit_transform(X_test)`?
   - A. Vì hàm `fit_transform` không tồn tại trong Scikit-Learn.
   - B. Vì tập Test phải được chuẩn hóa bằng chính các tham số thống kê ($\mu, \sigma$) đã học từ tập Train, nhằm mô phỏng dữ liệu mới đến từ thực tế.
   - C. Vì làm như vậy sẽ gây tràn bộ nhớ RAM.
   - D. Để tiết kiệm thời gian chạy code.

3. Dấu hiệu rõ ràng nhất của hiện tượng Overfitting trên bảng Benchmark là gì?
   - A. Cả điểm Train và điểm Test đều rất thấp.
   - B. Điểm trên tập Train rất cao (gần 100%), nhưng điểm trên tập Test tụt dốc thảm hại.
   - C. Thời gian huấn luyện mô hình quá nhanh.
   - D. Mô hình dự đoán ra toàn số 0.

4. Trong bảng so sánh các mô hình, vai trò của `DummyClassifier` là gì?
   - A. Là mô hình dự thi chính thức để giành huy chương.
   - B. Đóng vai trò là mốc cơ sở tối thiểu (Baseline) để chứng minh các mô hình học máy thực sự có giá trị tri thức.
   - C. Tự động sửa lỗi cú pháp trong code.
   - D. Tăng độ chính xác của Random Forest.

5. Sản phẩm đầu ra chính thức mà học viên hoàn thành khi kết thúc Module M2 là gì?
   - A. Một mạng nơ-ron sâu 100 lớp viết bằng PyTorch.
   - B. Model phân loại hoặc dự đoán đầu tiên kèm Báo cáo kết quả (First ML Model & Result Report).
   - C. Một ứng dụng di động AI.
   - D. Bản đồ học tập cá nhân.
"""
    session_summary_content = """
* **What I learned:**
  - Quy trình khép kín xây dựng một End-to-End Machine Learning Pipeline.
  - Kỹ thuật chuẩn hóa đặc trưng an toàn không rò rỉ dữ liệu (StandardScaler).
  - So sánh đối sánh định lượng giữa Baseline, Logistic Regression, Decision Tree và Random Forest.
  - Cách thực hiện Error Analysis và cấu trúc một Báo cáo Kết quả ML chuẩn mực.
* **What I can now do:**
  - Tự tin huấn luyện mô hình Machine Learning đầu tiên từ bảng dữ liệu thực tế.
  - Phân tích và báo cáo kết quả thực nghiệm đạt chuẩn thi đấu khoa học.
* **What I still need to practice:**
  - Chuẩn bị bước vào thế giới **Deep Learning và PyTorch** ở Module M3!
"""
    next_session_content = """
Xin chúc mừng bạn đã chính thức chinh phục thành công Module M2 và nhận Chứng chỉ **Certificate of Machine Learning Basics**!
Trong buổi tiếp theo (**M3_S01**), chúng ta sẽ bước lên một nấc thang công nghệ mới: **Module M3: XÂY MODEL AI SÂU HƠN BẰNG PYTORCH** — Bắt đầu làm chủ Tensors, Autograd và đồ thị tính toán tự động!
"""
    module_assessment = """
---

## 🏆 MODULE 2 ASSESSMENT & CERTIFICATION

### Phần A: Đánh giá Lý thuyết & Tư duy Mô hình (Theory & Model Reasoning)
- Nắm vững công thức ánh xạ $y = f(X)$, phân biệt Regression vs Classification.
- Giải thích bản chất hàm mất mát (MSE vs Log-Loss) và ranh giới phân tách của Decision Tree.
- Bóc trần bản chất của Accuracy Paradox và giải thích cán cân Precision-Recall.

### Phần B: Sản phẩm Thực hành Bắt buộc (Practical Deliverable)
- **Model phân loại hoặc dự đoán đầu tiên:** Huấn luyện thành công ít nhất 2 mô hình máy học (Linear/Logistic, Tree/Forest) có so sánh với Baseline.
- **Báo cáo kết quả chuẩn thi đấu (ML Result Report):** Bảng Benchmark đa metric (Accuracy, Precision, Recall, F1 hoặc MAE, RMSE) kèm phân tích hiện tượng Overfitting và phân tích lỗi sai (Error Analysis).

### Tiêu chí Đạt chuẩn (Rubric):
| Tiêu chí | Trọng số | Yêu cầu đạt |
| :--- | :--- | :--- |
| **Pipeline & Chống Leakage** | 30% | Phân chia tập chuẩn hóa, scaling đúng quy cách |
| **So sánh Baseline & Models** | 25% | Có mô hình Baseline ngây thơ và đánh bại được Baseline |
| **Đánh giá Đa Metric** | 25% | Đọc đúng Confusion Matrix, Precision, Recall, F1 |
| **Báo cáo Kết quả & Phân tích lỗi** | 20% | Xuất bảng Benchmark và chỉ ra nguyên nhân lỗi sai |

**Chứng chỉ hoàn thành:**
🏅 **Certificate of Machine Learning Basics** — Đủ điều kiện tiến vào Module M3 (PyTorch Deep Learning).
"""
    trace = {
        "items": [
            "Xây dựng End-to-End ML Pipeline hoàn chỉnh",
            "Model phân loại/dự đoán đầu tiên đánh bại Baseline",
            "Báo cáo kết quả chuẩn thi đấu (Benchmark table & Error analysis)",
            "Nhận chứng chỉ Certificate of Machine Learning Basics"
        ],
        "deliverable": "Model phân loại/dự đoán đầu tiên + Báo cáo kết quả (ML Result Report) hoàn chỉnh"
    }
    nb = create_notebook(
        module_id="M2",
        module_name="DẠY MÁY HỌC TỪ DỮ LIỆU",
        session_id="S05",
        topic="Huấn Luyện Model Đầu Tiên & Báo Cáo Kết Quả Chuẩn Thi Đấu",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Model phân loại/dự đoán đầu tiên & Báo cáo kết quả (ML Result Report)",
        competition_relevance="Hoàn thiện năng lực huấn luyện và báo cáo thử nghiệm mô hình theo chuẩn thi đấu.",
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
    save_notebook(nb, "course/M2_MACHINE_LEARNING/M2_S05.ipynb")

if __name__ == "__main__":
    build_m2_s01()
    build_m2_s02()
    build_m2_s03()
    build_m2_s04()
    build_m2_s05()
    print("M2 build complete!")
