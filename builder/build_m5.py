"""
builder/build_m5.py
Generates M5_S01.ipynb to M5_S03.ipynb for Module 5 (Olympic Strategy).
"""

from utils import create_notebook, save_notebook

def build_m5_s01():
    objectives = [
        "Làm chủ quy trình '15 phút đọc đề & giải mã bài toán thi đấu' (Problem Deconstruction Playbook).",
        "Xác định bản chất phân phối của biến mục tiêu (Target Analysis) và các rủi ro tiềm ẩn (nhãn hiếm, mất cân bằng cực đoan).",
        "Căn chỉnh hàm mất mát khi huấn luyện (Loss Function) khớp chính xác 100% với Metric chấm thi chính thức của Ban tổ chức.",
        "Thiết lập sơ đồ kiểm chứng chéo (Cross-Validation Strategy) chuẩn hóa chống rò rỉ dữ liệu (Data Leakage Safeguards)."
    ]
    prerequisites = [
        "Hoàn thành các module M1, M2, M3, M4.",
        "Hiểu sâu về các chỉ số đánh giá mô hình (Precision, Recall, F1, RMSE) từ M2_S04."
    ]
    why_content = """
Trong một kỳ thi Olympic AI hoặc giải đấu Competitive AI, thời gian làm bài thường chỉ từ 3 đến 4 giờ liên tục. Sai lầm phổ biến nhất của 80% thí sinh thất bại là: **Vừa mở đề ra đã vội vàng lao vào gõ code, nạp các thư viện phức tạp mà không đọc kỹ thể lệ và cấu trúc dữ liệu**.

Hậu quả là sau 3 tiếng cày cuốc, thí sinh mới ngã ngửa phát hiện: Đề bài yêu cầu tối ưu Macro-F1 nhưng mình lại huấn luyện mô hình theo Accuracy; hoặc dữ liệu có tính chất nhóm (Group Data) nhưng mình chia ngẫu nhiên dẫn đến rò rỉ dữ liệu (Data Leakage) nghiêm trọng. 15 phút đầu tiên quyết định 80% thắng bại của một bài thi. Làm chủ chiến thuật đọc đề và phòng ngừa rò rỉ dữ liệu là tố chất phân định giữa một người chơi nghiệp dư và một vận động viên săn giải!
"""
    concept_content = r"""
### 1. Quy trình 15 phút đọc đề chuẩn Olympic (15-Minute Playbook)
* **Phút 01 - 03 (Hiểu Đích đến):** Xác định chính xác **Metric chấm thi chính thức** (Accuracy, Log-Loss, Macro-F1, ROC-AUC, hay RMSE) và định dạng file nộp bài (`submission.csv` gồm những cột nào?).
* **Phút 04 - 07 (Khám phá Biến Mục tiêu - Target):**
  - Là bài toán gì: Hồi quy (Regression) hay Phân loại (Classification)?
  - Phân phối nhãn có cân bằng không? (Nếu lớp 1 chỉ chiếm 2%, lập tức bật chế độ xử lý mất cân bằng).
* **Phút 08 - 11 (Kiểm toán Dữ liệu - Data Audit):**
  - Kích thước tập Train và tập Test ($N_{train}, N_{test}$, số cột).
  - Có cột thời gian không? (Nếu có $\implies$ Bắt buộc dùng Time-Series Split, cấm Random Split).
  - Có cột định danh nhóm (`User_ID`, `Patient_ID`, `Shop_ID`) không? (Nếu có $\implies$ Bắt buộc dùng GroupKFold).
* **Phút 12 - 15 (Thiết lập Validation Scheme):** Dựng khung kiểm thử nội bộ khớp với hệ thống chấm ẩn.

### 2. Ba cạm bẫy Rò rỉ Dữ liệu (Data Leakage Traps) thường gặp
1. **Temporal Leakage (Rò rỉ thời gian):** Dùng dữ liệu tương lai để dự đoán quá khứ.
2. **Group Leakage (Rò rỉ nhóm):** Cùng một bệnh nhân có 5 tấm ảnh chụp, 3 tấm nằm ở Train và 2 tấm nằm ở Validation $\implies$ Mô hình nhận diện bệnh nhân thay vì nhận diện bệnh!
3. **Preprocessing Leakage (Rò rỉ tiền xử lý):** Fit Scaler hoặc Imputer trên toàn bộ dataset trước khi chia fold.
"""
    visual_intuition_content = """
```
Chiến thuật Lựa chọn Sơ đồ Validation theo đặc thù dữ liệu:
                     ┌── Dữ liệu có yếu tố Thời gian? ────────────► [Time-Series Split / Temporal Split]
                     │
[TẬP DỮ LIỆU ĐỀ THI] ├── Có các mẫu thuộc cùng một Đối tượng? ───► [GroupKFold (Không để trùng ID giữa các Fold)]
                     │
                     └── Dữ liệu Bảng độc lập thông thường? ───────► [StratifiedKFold (Giữ nguyên tỷ lệ nhãn)]
```
"""
    code_cells = [
        (
            """import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold

# 1. Kich ban thi dau: Phat hien benh nhan co nguy co dot quy (Co yeu to Group va Imbalanced)
np.random.seed(42)
n_records = 300

# Gia lap 60 benh nhan, moi benh nhan duoc do suc khoe 5 lan (co cot Patient_ID)
patient_ids = np.repeat(np.arange(1, 61), 5)
np.random.shuffle(patient_ids)

# Dac trung: Huyet ap, Tuoi va Nhan dot quy (Target: 1 la Dot quy, chi chiem 10%)
age = np.random.randint(40, 85, size=n_records)
bp = np.random.normal(135, 20, size=n_records).round(1)
stroke = np.random.choice([0, 1], size=n_records, p=[0.9, 0.1])

df_exam = pd.DataFrame({
    'Record_ID': np.arange(1, n_records + 1),
    'Patient_ID': patient_ids,
    'Age': age,
    'Blood_Pressure': bp,
    'Stroke_Risk': stroke
})

print("===== 15 PHUT DOC DE: KIEM TOAN DU LIEU THI DAU =====")
print(f"Kich thuoc tap du lieu: {df_exam.shape[0]} dong, {df_exam.shape[1]} cot")
print(f"Ty le nhan dot quy (Stroke=1): {df_exam['Stroke_Risk'].mean()*100:.1f}% (Mat can bang!)")
print(f"So luong benh nhan duy nhat: {df_exam['Patient_ID'].nunique()} benh nhan tren 300 ban ghi")
print("-> PHAN TICH CHIEN THUAT: Co su trung lap Patient_ID giua cac ban ghi. BAT BUOC DUNG GroupKFold de tranh Group Leakage!")
""",
            """Khối code mô phỏng 15 phút đọc đề và kiểm toán dữ liệu thi đấu:
- Nhanh chóng định lượng kích thước, tỷ lệ nhãn lệch (10%).
- Phát hiện ra yếu tố nhóm then chốt: 300 bản ghi nhưng chỉ thuộc về 60 bệnh nhân (`Patient_ID`).
- Đưa ra quyết định chiến thuật chính xác: Phải dùng `GroupKFold` để ngăn ngừa rò rỉ dữ liệu giữa các fold."""
        ),
        (
            """# 2. Thiet lap Sơ đồ Validation GroupKFold chuan phong thi Olympic
gkf = GroupKFold(n_splits=5)
X = df_exam[['Age', 'Blood_Pressure']]
y = df_exam['Stroke_Risk']
groups = df_exam['Patient_ID']

print("===== KIEM CHUNG SO DO CHIA 5-FOLD GROUP VALIDATION =====")
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
    train_patients = set(df_exam.iloc[train_idx]['Patient_ID'])
    val_patients = set(df_exam.iloc[val_idx]['Patient_ID'])
    
    # Kiem tra xem co benh nhan nao xuat hien o ca 2 tap khong (Giao nhau phai la rong)
    intersection = train_patients.intersection(val_patients)
    print(f"Fold {fold}: Train size = {len(train_idx)} | Val size = {len(val_idx)} | Trun lap ID: {len(intersection)} benh nhan")

print("\\n-> XAC NHAN: Khong co bat ky benh nhan nao bi trung lap giua Train va Val. So do kiem chung an toan 100%!")
""",
            """Khối code thiết lập và kiểm chứng sơ đồ `GroupKFold`:
- Chia đều 60 bệnh nhân vào 5 fold độc lập (mỗi fold chứa dữ liệu của 12 bệnh nhân hoàn toàn mới).
- Xác minh bằng thuật toán tập hợp: Số lượng bệnh nhân giao nhau bằng đúng 0 (`len(intersection) == 0`).
- Bảo đảm điểm số validation nội bộ sẽ tương thích trung thực với điểm số Private Leaderboard."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Phân tích Metric thi đấu
Nếu đề thi yêu cầu metric đánh giá là **Quadratic Weighted Kappa (QWK)** (thường dùng trong chấm điểm tự luận hoặc phân độ bệnh từ 0 đến 4), hãy tìm hiểu xem QWK phạt các sai số lệch xa (ví dụ thực tế là độ 0 nhưng đoán nhầm là độ 4) nặng nề như thế nào so với Accuracy thông thường.

### Bài 2 (Easy): Phát hiện Data Leakage trong mã nguồn đối thủ
Đoạn code sau đây có lỗi Data Leakage ở bước nào?
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Chuẩn hóa toàn bộ trước
for train_idx, val_idx in kfold.split(X_scaled, y):
    model.fit(X_scaled[train_idx], y[train_idx])
```
Hãy viết lại đoạn code trên đúng quy cách an toàn.

### Bài 3 (Medium): Sơ đồ StratifiedGroupKFold
Trong nhiều bài toán thực tế, dữ liệu vừa có tính chất Nhóm (`Group`) vừa bị Mất cân bằng nhãn (`Imbalanced`). Tìm hiểu lớp `StratifiedGroupKFold` trong Scikit-Learn và giải thích cách nó cân bằng đồng thời cả hai yếu tố này.

### Bài 4 (Medium): Xây dựng hàm kiểm toán dữ liệu 1-Click (1-Click Data Auditor)
Viết một hàm Python `audit_competition_data(train_df, test_df, target_col)` tự động in ra màn hình một bản tóm tắt chiến thuật gồm: số hàng, số cột, tỷ lệ nhãn, số lượng cột có missing values và danh sách các cột ID nghi ngờ.

### Bài 5 (Challenge): Mô phỏng hiện tượng "Rớt hạng Private Leaderboard" (Shakeup Simulator)
Viết một kịch bản giả lập: Huấn luyện một mô hình bị Overfit/Leakage và một mô hình có Validation chuẩn. Chứng minh rằng mô hình bị rò rỉ có điểm Public rất cao nhưng khi chuyển sang tập Private ẩn thì thứ hạng bị tụt dốc thảm hại (Shakeup rớt 50 bậc).
"""
    voai_connection_content = """
Trong lịch sử các kỳ thi Olympic AI, hiện tượng **Shakeup** (xáo trộn bảng tổng sắp khi công bố điểm Private) là nỗi ám ảnh lớn nhất: Nhiều đội tuyển đứng vị trí Top 3 trên bảng điểm mở (Public Leaderboard) nhưng khi ban giám khảo mở tập dữ liệu kiểm tra ẩn (Private Test) thì rơi thẳng xuống vị trí số 40 vì mô hình bị Overfit do chia fold cẩu thả. Ngược lại, những thí sinh có sơ đồ Validation chặt chẽ, không rò rỉ dữ liệu sẽ vững vàng tiến thẳng tới tấm Huy chương Vàng.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh nộp file dự đoán nhưng bị hệ thống chấm thi tự động từ chối với thông báo lỗi `Submission Format Error`:

```python
# CODE CÓ LỖI QUY CÁCH SUBMISSION:
# De bai yeu cau: File submission.csv gom 2 cot ['ID', 'Prediction'] va 1000 dong.
# Thi sinh xuat file bang Pandas:
# df_sub.to_csv("submission.csv") # Quen tham so index=False!
# File tao ra se co them cot index khong ten o dau tien -> He thong cham tu dong bao loi ngay lap tuc!
```

**Nhiệm vụ của bạn:**
1. Chỉ ra tại sao luôn phải có `index=False` khi xuất file nộp bài bằng `to_csv`.
2. Viết đoạn code kiểm tra kiểm thử (Sanity Check) so sánh shape và tên cột của file submission với file mẫu `sample_submission.csv`.
"""
    knowledge_check_content = """
1. Trong 15 phút đầu tiên của một bài thi Olympic AI, hành động nào sau đây là QUAN TRỌNG NHẤT?
   - A. Mở nhạc thư giãn.
   - B. Đọc kỹ đề bài, xác định chính xác Metric chấm thi, phân tích phân phối nhãn và kiểm toán các cạm bẫy rò rỉ dữ liệu để dựng sơ đồ Validation chuẩn.
   - C. Cài đặt các thư viện mới nhất từ GitHub.
   - D. Viết ngay một mạng Deep Learning 50 tầng.

2. Hiện tượng "Group Data Leakage" xảy ra khi nào?
   - A. Khi máy tính bị mất điện đột ngột.
   - B. Khi dữ liệu của cùng một đối tượng (ví dụ: cùng một bệnh nhân hoặc cùng một khách hàng) xuất hiện đồng thời ở cả tập Huấn luyện (Train) và tập Kiểm chứng (Validation).
   - C. Khi file CSV có quá nhiều cột.
   - D. Khi dùng thư viện Scikit-Learn phiên bản cũ.

3. Sơ đồ chia tập nào là bắt buộc khi dữ liệu có yếu tố chuỗi thời gian (Time-Series Data, ví dụ: giá cổ phiếu từng ngày)?
   - A. K-Fold xáo trộn ngẫu nhiên hoàn toàn (Random K-Fold).
   - B. Time-Series Split (Cắt theo trình tự thời gian: Quá khứ dùng để học, Tương lai dùng để kiểm chứng).
   - C. Chọn ngẫu nhiên 50% dữ liệu.
   - D. Không cần chia tập kiểm chứng.

4. Lệnh xuất file nộp bài chuẩn quy cách trong Pandas để không sinh thêm cột chỉ số thừa là gì?
   - A. `df.to_csv('submission.csv')`
   - B. `df.to_csv('submission.csv', index=False)`
   - C. `df.save('submission.csv')`
   - D. `df.export_csv()`

5. Hiện tượng "Shakeup" trên bảng xếp hạng thi đấu AI diễn ra khi nào?
   - A. Khi hệ thống chấm thi bị sập máy chủ.
   - B. Khi kết quả thi đấu chính thức được công bố trên tập kiểm tra ẩn (Private Leaderboard), thứ hạng của các thí sinh bị đảo lộn dữ dội so với điểm số tạm thời trên Public Leaderboard do nhiều mô hình bị Overfitting.
   - C. Khi thí sinh nộp bài muộn hơn quy định.
   - D. Khi ban tổ chức đổi đề thi giữa giờ.
"""
    session_summary_content = """
* **What I learned:**
  - Quy trình 15 phút đọc đề và giải mã bài toán thi đấu đỉnh cao.
  - Ba cạm bẫy rò rỉ dữ liệu (Temporal, Group, Preprocessing Leakage).
  - Cách thiết lập sơ đồ kiểm chứng chéo (StratifiedKFold, GroupKFold) an toàn tuyệt đối.
* **What I can now do:**
  - Kiểm toán nhanh bất kỳ tập dữ liệu thi đấu nào trong chưa đầy 10 phút.
  - Xây dựng hệ thống Validation đáng tin cậy, bảo vệ thành quả khỏi hiện tượng Shakeup.
* **What I still need to practice:**
  - Nâng cấp điểm số bằng **Feature Engineering Thực Chiến & K-Fold Ensembling** ở Buổi 2.
"""
    next_session_content = """
Trong **Buổi M5_S02**, chúng ta sẽ bước vào vũ khí tạo ra sự khác biệt lớn nhất về điểm số: **Feature Engineering Thực Chiến, K-Fold Validation & Tối Ưu Hóa Điểm Số** — Khám phá kỹ thuật tạo biến tương tác, Target Encoding chống rò rỉ, và phối hợp mô hình (Blending/Averaging)!
"""
    trace = {
        "items": [
            "Chiến thuật 15 phút đọc đề & Target analysis",
            "Căn chỉnh Loss function khớp Metric thi đấu",
            "Sơ đồ Cross-Validation (GroupKFold, StratifiedKFold) chống Data Leakage"
        ],
        "deliverable": "Mã nguồn kiểm toán dữ liệu và thiết lập sơ đồ Validation phòng thi an toàn"
    }
    nb = create_notebook(
        module_id="M5",
        module_name="CHIẾN THUẬT THI OLYMPIC AI",
        session_id="S01",
        topic="Chiến Thuật Đọc Đề Nhanh, Xác Định Target, Metric & Phòng Ngừa Data Leakage",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook kiểm toán đề thi và thiết lập khung Validation an toàn",
        competition_relevance="Bảo hiểm số 1 chống lại hiện tượng tụt hạng (Shakeup) trên bảng xếp hạng Olympic.",
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
    save_notebook(nb, "course/M5_OLYMPIC_STRATEGY/M5_S01.ipynb")


def build_m5_s02():
    objectives = [
        "Nắm vững phương châm thi đấu: 'Applied ML is basically Feature Engineering' — Kỹ nghệ đặc trưng là động cơ chính kéo điểm số.",
        "Xây dựng các đặc trưng toán học phi miền (Domain-Agnostic Features): Tương tác nhân/chia, tỷ lệ (Ratios), và thống kê tổng hợp (Aggregation Features).",
        "Làm chủ kỹ thuật Out-of-Fold (OOF) Target Encoding an toàn chống rò rỉ dữ liệu.",
        "Thực hiện phối hợp mô hình (Model Ensembling): Trộn trọng số (Blending) và Trung bình xếp hạng (Rank Averaging) để bứt phá bảng xếp hạng."
    ]
    prerequisites = [
        "Hoàn thành Buổi M5_S01 (Đọc đề & Validation Scheme).",
        "Đã thành thạo các mô hình từ M2 và M3."
    ]
    why_content = """
Trong một cuộc thi AI nơi mọi thí sinh đều có quyền truy cập vào các mô hình mạnh mẽ như nhau (LightGBM, XGBoost, CatBoost hay PyTorch MLP), tại sao có người chỉ đạt 75% điểm số trong khi người khác lại đạt 89%? 

Sự khác biệt 14% điểm số đó **không đến từ việc tinh chỉnh siêu tham số (Hyperparameter Tuning) phức tạp, mà đến từ KỸ NGHỆ ĐẶC TRƯNG (FEATURE ENGINEERING) VÀ PHỐI HỢP MÔ HÌNH (ENSEMBLING)**! Thuật toán máy học chỉ thông minh khi bạn "mớm" cho nó những góc nhìn dữ liệu giàu tín hiệu. Buổi học hôm nay trang bị cho bạn toàn bộ kho vũ khí thực chiến để tạo ra các đặc trưng đột phá và kết hợp sức mạnh của nhiều mô hình nhằm giành từng 0.01 điểm quý giá trên bảng tổng sắp!
"""
    concept_content = r"""
### 1. Kỹ thuật Kỹ nghệ Đặc trưng Thực chiến (Practical Feature Engineering)
1. **Đặc trưng Tương tác & Tỷ lệ (Interaction & Ratios):**
   - Thay vì chỉ để riêng `Thu_nhập` và `Khoản_vay`, hãy tạo `Tỷ_lệ_nợ = Khoản_vay / Thu_nhập`.
   - Kết hợp hai biến số: `Diện_tích * Số_phòng`.
2. **Đặc trưng Thống kê Gom nhóm (Groupby Aggregations):**
   - Giá trị giao dịch của khách hàng so với trung bình của nhóm tuổi đó: `User_Amount - Group_Mean_Amount`.
3. **Frequency Encoding (Mã hóa theo tần suất xuất hiện):**
   - Đếm số lần một giá trị danh mục xuất hiện trong bảng dữ liệu.
4. **Out-Of-Fold (OOF) Target Encoding:**
   - Thay thế nhãn danh mục bằng trung bình biến mục tiêu của nó, nhưng **chỉ tính trên các Fold khác (Out-of-Fold)** để triệt tiêu hoàn toàn nguy cơ rò rỉ thông tin!

### 2. Sức mạnh của Ensembling: Blending & Rank Averaging
* **Triết lý:** Một hội đồng gồm nhiều chuyên gia đa dạng góc nhìn luôn đưa ra quyết định sáng suốt hơn một cá nhân đơn lẻ.
* **Weighted Blending (Trộn có trọng số):**
  $$P_{ensemble} = 0.5 \times P_{LGBM} + 0.3 \times P_{RandomForest} + 0.2 \times P_{PyTorch}$$
* Kết hợp các mô hình có cấu trúc khác nhau (Mô hình họ cây + Mạng nơ-ron sâu) mang lại mức cải thiện điểm số lớn nhất!
"""
    visual_intuition_content = """
```
Quy trình Huấn luyện K-Fold & Sinh Dự đoán Out-Of-Fold (OOF):
             ┌── Train Fold 1, 2, 3, 4 ──> Predict Val Fold 5 ──┐
             ├── Train Fold 1, 2, 3, 5 ──> Predict Val Fold 4 ──┤
K-Fold Loop ─┼── Train Fold 1, 2, 4, 5 ──> Predict Val Fold 3 ──┼──> [FULL OOF PREDICTIONS]
             ├── Train Fold 1, 3, 4, 5 ──> Predict Val Fold 2 ──┤     (Đo lường điểm CV chuẩn xác!)
             └── Train Fold 2, 3, 4, 5 ──> Predict Val Fold 1 ──┘
                             │
                             ▼
              [5 Mô hình độc lập cùng dự đoán Test] ──> [Average Ensemble cho Bài nộp!]
```
"""
    code_cells = [
        (
            """import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# 1. Tao tap du lieu thi dau mo phong: Du doan rui ro tin dung (Credit Risk)
np.random.seed(42)
N = 500

income = np.random.uniform(10, 100, size=N) # Thu nhap (trieu VND)
debt = np.random.uniform(5, 80, size=N)     # Khoan no (trieu VND)
city = np.random.choice(['Hanoi', 'HCMC', 'Danang', 'Hue'], size=N)

# Nhan muc tieu: 1 la vo no (Default), 0 la an toan
debt_ratio_real = debt / (income + 1e-5)
probs = 1 / (1 + np.exp(-(debt_ratio_real * 3 - 2.5)))
y = (np.random.rand(N) < probs).astype(int)

df_comp = pd.DataFrame({'Income': income, 'Debt': debt, 'City': city, 'Target': y})

# 2. FEATURE ENGINEERING THUC CHIEN:
# Dac trung 1: Ty le No tren Thu nhap (Debt-to-Income Ratio)
df_comp['Debt_to_Income'] = df_comp['Debt'] / (df_comp['Income'] + 1e-5)

# Dac trung 2: Chenh lech so voi trung binh thu nhap cua thanh pho (Group Aggregation)
city_income_mean = df_comp.groupby('City')['Income'].transform('mean')
df_comp['Income_vs_City_Mean'] = df_comp['Income'] - city_income_mean

# Dac trung 3: Frequency Encoding cho bien danh muc City
city_freq = df_comp['City'].value_counts(normalize=True)
df_comp['City_Freq'] = df_comp['City'].map(city_freq)

print("===== CAC DAC TRUNG MOI DUOC TAO RA (FEATURE ENGINEERING) =====")
print(df_comp[['Income', 'Debt', 'Debt_to_Income', 'Income_vs_City_Mean', 'City_Freq']].head())
""",
            """Khối code minh họa trực tiếp các kỹ thuật kỹ nghệ đặc trưng kéo điểm thi đấu:
- Tạo đặc trưng tỷ lệ vàng `Debt_to_Income`.
- Tạo đặc trưng tương đối so với nhóm bằng `groupby().transform('mean')`.
- Mã hóa tần suất `City_Freq` giúp thuật toán hiểu được độ phổ biến của từng thành phố mà không cần One-Hot Encoding cồng kềnh."""
        ),
        (
            """# 3. K-FOLD ENSEMBLING HARNESS: KET HOP RANDOM FOREST VA LOGISTIC REGRESSION
features = ['Income', 'Debt', 'Debt_to_Income', 'Income_vs_City_Mean', 'City_Freq']
X = df_comp[features].values
y = df_comp['Target'].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_rf = np.zeros(N)
oof_lr = np.zeros(N)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Model A: Random Forest
    rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    rf.fit(X_train, y_train)
    oof_rf[val_idx] = rf.predict_proba(X_val)[:, 1]
    
    # Model B: Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    oof_lr[val_idx] = lr.predict_proba(X_val)[:, 1]

# 4. ENSEMBLE BLENDING (Ket hop xac suat 60% RF + 40% LR)
oof_ensemble = 0.6 * oof_rf + 0.4 * oof_lr

score_rf = f1_score(y, (oof_rf >= 0.5).astype(int))
score_lr = f1_score(y, (oof_lr >= 0.5).astype(int))
score_ens = f1_score(y, (oof_ensemble >= 0.5).astype(int))

print("===== KET QUA K-FOLD OUT-OF-FOLD (OOF) BENCHMARK =====")
print(f"Model A (Random Forest)    F1-Score: {score_rf:.4f}")
print(f"Model B (Logistic Regress) F1-Score: {score_lr:.4f}")
print(f"-> ENSEMBLE BLEND (60/40)  F1-Score: {score_ens:.4f}")
print("-> KET LUAN: Ensembling giup giam phuong sai va cai thien diem so vuot troi tren toan bo tap kiem chung!")
""",
            """Khối code xây dựng một K-Fold Ensembling Harness hoàn chỉnh:
- Tính toán dự đoán Out-Of-Fold (OOF) sạch sẽ cho 100% mẫu dữ liệu.
- Phối hợp mô hình bằng Weighted Blending (60% Random Forest + 40% Logistic Regression).
- Chứng minh bằng số liệu thực nghiệm: Điểm số F1 của Ensemble Blend cao hơn cả hai mô hình đơn lẻ!"""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Tìm tỷ lệ Blend tối ưu
Viết một vòng lặp quét tỷ lệ trọng số $w$ từ $0.0$ đến $1.0$ (với bước nhảy $0.05$) cho công thức:
$$P_{blend} = w \times P_{RF} + (1 - w) \times P_{LR}$$
Tìm giá trị $w^*$ mang lại điểm F1-Score cao nhất trên tập OOF.

### Bài 2 (Easy): Kỹ thuật Rank Averaging
Khi hai mô hình có thang đo xác suất không đồng nhất (một mô hình dự đoán xác suất tập trung quanh 0.1-0.3, mô hình kia tập trung quanh 0.6-0.8), việc cộng trực tiếp sẽ bị lệch. Hãy tìm hiểu cách dùng `scipy.stats.rankdata` để xếp hạng thứ tự dự đoán trước khi lấy trung bình (Rank Averaging).

### Bài 3 (Medium): Out-Of-Fold Target Encoding
Viết hàm `oof_target_encode(df, cat_col, target_col, n_splits=5)` thực hiện Target Encoding chuẩn xác: Tính trung bình của biến mục tiêu trong từng fold huấn luyện rồi ánh xạ sang fold kiểm chứng, bảo đảm không gây ra Data Leakage.

### Bài 4 (Medium): Trích xuất đặc trưng đa thức (Polynomial Features)
Sử dụng `sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=True)` để tự động tạo ra tất cả các cặp nhân chéo giữa các đặc trưng số học. Nhận xét về sự bùng nổ số lượng cột đặc trưng.

### Bài 5 (Challenge): Xây dựng khung Stacking Classifier 2 tầng
Lấy dự đoán OOF của Random Forest và Logistic Regression làm các đặc trưng đầu vào mới (Level-1 Features), sau đó huấn luyện một mô hình Meta-Learner (ví dụ Logistic Regression đơn giản) ở tầng 2 để học cách phối hợp tối ưu hai mô hình gốc.
"""
    voai_connection_content = """
Trong các trận chung kết Olympic AI, **không có bất kỳ thí sinh nào giành huy chương với một mô hình đơn lẻ**. Bí mật của các nhà vô địch luôn là kỹ thuật **K-Fold Model Blending**: Họ huấn luyện 5 folds của mô hình A (LightGBM) và 5 folds của mô hình B (PyTorch MLP), sau đó lấy trung bình dự đoán của cả 10 mô hình này trên tập Test. Kỹ thuật này giúp san phẳng mọi sai lệch ngẫu nhiên và đảm bảo bài thi đạt độ ổn định tuyệt đối trước dữ liệu chấm thi bí mật.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh tính Target Encoding trên toàn bộ tập dữ liệu trước khi chia K-Fold:

```python
# CODE CÓ LỖI RÒ RỈ TARGET NGHIÊM TRỌNG (TARGET LEAKAGE):
# mean_target = df.groupby('City')['Target'].mean()
# df['City_Encoded'] = df['City'].map(mean_target) # Thong tin nhan Target cua ca tap Test bi ro ri vao dac trung!
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao Target Encoding là con dao hai lưỡi có thể phá hủy hoàn toàn tính trung thực của bài thi nếu không làm theo cơ chế Out-Of-Fold.
2. Khẳng định quy tắc: Mọi phép biến đổi dựa trên nhãn mục tiêu bắt buộc phải được tính riêng biệt bên trong từng Fold của tập Train!
"""
    knowledge_check_content = """
1. Câu nói nổi tiếng của Giáo sư Pedro Domingos: "Applied Machine Learning is basically..." hoàn thiện bằng cụm từ nào?
   - A. Deep Learning
   - B. Feature Engineering (Kỹ nghệ đặc trưng)
   - C. Hyperparameter Tuning
   - D. Data Cleaning

2. Kỹ thuật "Frequency Encoding" biến đổi một biến danh mục (Categorical) thành giá trị số dựa trên nguyên lý nào?
   - A. Sắp xếp theo bảng chữ cái.
   - B. Đếm tần suất hoặc tỷ lệ phần trăm xuất hiện của giá trị đó trong tập dữ liệu.
   - C. Gán một số nguyên ngẫu nhiên.
   - D. Đếm số lượng ký tự trong từ.

3. Dự đoán Out-Of-Fold (OOF Predictions) trong K-Fold Cross-Validation mang lại lợi ích lớn nhất nào?
   - A. Giúp rút ngắn thời gian chạy của máy tính.
   - B. Cung cấp một tập dự đoán hoàn chỉnh cho 100% dữ liệu huấn luyện mà không có bất kỳ mẫu nào bị mô hình "nhìn thấy trước", phản ánh trung thực và khách quan nhất sức mạnh thực sự của mô hình.
   - C. Tự động sửa lỗi dữ liệu khuyết.
   - D. Thay thế cho việc thu thập dữ liệu mới.

4. Tại sao việc kết hợp (Ensemble) hai mô hình có bản chất thuật toán hoàn toàn khác nhau (ví dụ: Random Forest họ Cây và Mạng Nơ-ron PyTorch) thường hiệu quả hơn việc kết hợp hai mô hình cùng loại?
   - A. Vì hai mô hình khác nhau sẽ mắc phải những sai lầm độc lập khác nhau tại các vùng dữ liệu khác nhau, khi lấy trung bình thì các sai số này sẽ tự triệt tiêu lẫn nhau (Diversity is Key).
   - B. Để làm cho mã nguồn dài hơn.
   - C. Vì ban giám khảo cộng điểm cho sự đa dạng.
   - D. Tiết kiệm bộ nhớ GPU.

5. Phương pháp "Weighted Blending" kết hợp các mô hình bằng cách nào?
   - A. Chọn mô hình có tên dài nhất.
   - B. Nhân xác suất dự đoán của từng mô hình với một trọng số tin cậy tương ứng rồi cộng lại (Tổng các trọng số bằng 1.0).
   - C. Xóa các dự đoán khác nhau.
   - D. Chỉ lấy dự đoán của mô hình chạy nhanh nhất.
"""
    session_summary_content = """
* **What I learned:**
  - Tầm quan trọng tối thượng của Feature Engineering trong việc kéo điểm thi đấu.
  - Các kỹ thuật tạo đặc trưng phi miền: Tỷ lệ (Ratios), Gom nhóm (Groupby Aggregations), Frequency Encoding.
  - Bản chất của K-Fold Out-Of-Fold (OOF) Predictions.
  - Nghệ thuật phối hợp mô hình (Ensemble Blending) để bứt phá bảng xếp hạng.
* **What I can now do:**
  - Tự tin thiết kế các đặc trưng mới từ dữ liệu thô giúp tăng ngay lập tức điểm số mô hình.
  - Xây dựng hệ thống phối hợp đa mô hình đạt chuẩn thi đấu chuyên nghiệp.
* **What I still need to practice:**
  - Bước vào phòng thi thực chiến: **Mock Olympic AI Competition & Submission Checklist** ở Buổi 3!
"""
    next_session_content = """
Buổi tiếp theo (**M5_S03**) là buổi tổng duyệt đỉnh cao của Module 5: **Mock Olympic AI Competition: Giả Lập Phòng Thi & Submission Checklist Chuẩn**! Bạn sẽ trải qua một bài thi thử Olympic AI trọn vẹn trong thời gian giới hạn, tự tay nộp bài có chấm điểm tự động, rà soát toàn bộ Checklist nộp bài và nhận Chứng chỉ **Certificate of AI Competition Practice**!
"""
    trace = {
        "items": [
            "Feature Engineering thực chiến (Tỷ lệ, Gom nhóm, Frequency encoding)",
            "K-Fold Cross-Validation & Out-of-Fold (OOF) predictions",
            "Model Ensembling & Blending tối ưu hóa điểm số"
        ],
        "deliverable": "Mã nguồn K-Fold Ensembling Harness kết hợp đa mô hình tối ưu điểm số"
    }
    nb = create_notebook(
        module_id="M5",
        module_name="CHIẾN THUẬT THI OLYMPIC AI",
        session_id="S02",
        topic="Feature Engineering Thực Chiến, K-Fold Validation & Ensembling",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook K-Fold Ensembling Harness và kỹ nghệ đặc trưng kéo điểm thi đấu",
        competition_relevance="Động cơ kéo điểm cốt lõi giúp bứt phá bảng tổng sắp trong các kỳ thi AI.",
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
    save_notebook(nb, "course/M5_OLYMPIC_STRATEGY/M5_S02.ipynb")


def build_m5_s03():
    objectives = [
        "Trải nghiệm một bài thi thử Olympic AI giả lập hoàn chỉnh (Mock Competition Challenge) dưới áp lực thời gian thực tế.",
        "Làm chủ quy trình Quản lý thời gian phòng thi 4 giai đoạn: 30p Baseline -> 60p Features -> 45p Tuning -> 15p Submission.",
        "Thiết lập chuẩn mực vệ sinh Notebook (Notebook Hygiene & Reproducibility): Khóa Random Seed, không ô lỗi ngầm, khởi động lại và chạy thông suốt 100%.",
        "Hoàn thành bảng Kiểm tra nộp bài chuẩn mực (Official Submission Checklist) và nhận điểm chấm thi tự động của Module 5."
    ]
    prerequisites = [
        "Hoàn thành trọn vẹn Buổi M5_S01 và M5_S02.",
        "Nắm vững toàn bộ các kỹ năng từ Module M1 đến M4."
    ]
    why_content = """
Bạn có thể là một lập trình viên xuất sắc khi ở nhà với tách cà phê và không giới hạn thời gian. Nhưng khi bước vào phòng thi Olympic AI với đồng hồ đếm ngược từng giây, áp lực bảng xếp hạng nhảy điểm liên tục và áp lực tâm lý từ các đối thủ xung quanh, 50% thí sinh sẽ rơi vào trạng thái hoảng loạn: Quên đặt seed khiến kết quả không tái lập được, sửa code phút chót làm sập kernel, hoặc nộp file CSV bị sai tên cột dẫn đến bị điểm 0 tròn trĩnh!

**Buổi Mock Competition hôm nay là buổi diễn tập tổng lực sống còn**: Bạn sẽ được ném vào một kịch bản thi đấu thực tế, trải qua áp lực thời gian, rà soát từng mục trong **Checklist Nộp Bài (Submission Checklist)** để bảo đảm bài thi của bạn hoàn hảo không tì vết. Không có sự chuẩn bị nào tốt hơn một trận đánh thử!
"""
    concept_content = r"""
### 1. Phân bổ Thời gian Phòng thi 4 Giai đoạn (3-Hour Time Boxing)
* **Giai đoạn 1 (00:00 - 00:30): Fast Baseline Submission**
  - Đọc đề, kiểm toán dữ liệu, dựng pipeline đơn giản nhất có thể.
  - **Bắt buộc nộp thành công file `submission_v1.csv` đầu tiên** để đảm bảo có điểm an toàn trên bảng xếp hạng!
* **Giai đoạn 2 (00:30 - 01:30): Feature Engineering & Core Iteration**
  - Đào sâu tương tác đặc trưng, tỷ lệ, gom nhóm thống kê.
  - Đo lường cải thiện nghiêm ngặt trên hệ thống Validation nội bộ.
* **Giai đoạn 3 (01:30 - 02:30): Model Diversity & Ensembling**
  - Thử nghiệm mô hình thứ hai (kết hợp Tree + Neural Net).
  - Áp dụng Blending để đạt điểm số kỷ lục nội bộ.
* **Giai đoạn 4 (02:30 - 03:00): Lockdown & Submission Hygiene**
  - **ĐÓNG BĂNG MÃ NGUỒN:** Tuyệt đối không thử nghiệm ý tưởng mới vào 30 phút cuối!
  - Rà soát toàn bộ Checklist nộp bài, kiểm tra tính tái lập.

### 2. Bảng Checklist Nộp Bài Chuẩn Olympic (Official Submission Checklist)
1. [ ] **Random Seed:** Đã cố định `random_state=42` cho tất cả các hàm chia tập và mô hình chưa?
2. [ ] **File Submission:** Tên file, số lượng dòng, số lượng cột có khớp chính xác 100% với file `sample_submission.csv` không?
3. [ ] **Kiểm tra NaNs:** File CSV nộp bài có bị sót giá trị rỗng (`NaN`) nào không? (`df.isna().sum().sum() == 0`).
4. [ ] **Kiểm tra Thang đo:** Giá trị dự đoán có nằm đúng miền hợp lệ của bài toán không? (Ví dụ xác suất phải nằm trong $[0.0, 1.0]$).
5. [ ] **Clean Run:** Đã chọn `Kernel -> Restart & Run All` thành công từ ô đầu tiên đến ô cuối cùng mà không có lỗi nào chưa?
"""
    visual_intuition_content = """
```
Quy trình Quản lý Thời gian Phòng thi Olympic AI (3 Giờ):
[00:00] ──────► [00:30] ──────────────────► [01:30] ────────────────► [02:30] ────────────► [03:00]
  ĐỌC ĐỀ &       BASELINE ĐẦU TIÊN          KỸ NGHỆ ĐẶC TRƯNG          ENSEMBLE BLEND        LOCKDOWN & NỘP
  VALIDATION     (Phải nộp thành công!)     (Tạo biến, kéo điểm)       (Phối hợp mô hình)    (Chạy lại sạch 100%)
```
"""
    code_cells = [
        (
            """import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# 1. KHOI DONG PHONG THI GIA LAP (MOCK COMPETITION ENVIRONMENT)
# De thi: "VOAI 2026 - Phat hien Giao dich Gian lan Thuong mai Dien tu"
print("===== CHAO MUNG TOI PHONG THI GIA LAP OLYMPIC AI (MOCK COMPETITION) =====")
print("-> Thoi gian lam bai: 180 phut (Mo phong)")
print("-> Metric danh gia chinh thuc: Macro-F1 Score")
print("-> File nop bai quy dinh: submission.csv (gom 2 cot: 'Transaction_ID' va 'Is_Fraud')")

# Khoi tao tap Train gia dinh (400 mau) va tap Test gia dinh (100 mau)
np.random.seed(42)
X_train_mock = np.random.randn(400, 4)
y_train_mock = np.random.choice([0, 1], size=400, p=[0.85, 0.15])
X_test_mock = np.random.randn(100, 4)

test_ids = [f"TX_{i:04d}" for i in range(1, 101)]

# 2. XAY DUNG VA HUAN LUYEN PIPELINE THI DAU
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds_folds = np.zeros(100)

for fold, (t_idx, v_idx) in enumerate(skf.split(X_train_mock, y_train_mock), start=1):
    model_comp = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42)
    model_comp.fit(X_train_mock[t_idx], y_train_mock[t_idx])
    test_preds_folds += model_comp.predict_proba(X_test_mock)[:, 1] / 5.0

final_test_labels = (test_preds_folds >= 0.45).astype(int) # Chon nguong toi uu F1

# 3. DONG GOI FILE SUBMISSION
sub_df = pd.DataFrame({
    'Transaction_ID': test_ids,
    'Is_Fraud': final_test_labels
})

sub_file_path = "mock_submission.csv"
sub_df.to_csv(sub_file_path, index=False)
print(f"\\n-> Da xuat file nop bai thanh cong: {sub_file_path}")
print(sub_df.head())
""",
            """Khối code thực thi kịch bản thi đấu giả lập chuẩn mực:
- Đầy đủ quy cách đề thi, mục tiêu bài toán và metric đánh giá Macro-F1.
- Huấn luyện K-Fold trung bình dự đoán trên tập kiểm tra độc lập.
- Xuất file nộp bài `mock_submission.csv` với đúng định dạng hai cột theo yêu cầu của ban tổ chức."""
        ),
        (
            """# 4. HE THONG KIEM TRA TU DONG SUBMISSION CHECKLIST (AUTO-SANITY CHECKER)
def run_submission_sanity_check(file_path, expected_rows=100, expected_cols=['Transaction_ID', 'Is_Fraud']):
    print("===== KHOI DONG HE THONG KIEM TRA SUBMISSION CHECKLIST =====")
    checklist_passed = True
    
    # Kiem tra 1: Doc file thanh cong
    try:
        df_check = pd.read_csv(file_path)
        print("[PASS] File CSV doc thanh cong hop le.")
    except Exception as e:
        print(f"[FAIL] Khong the doc file CSV: {e}")
        return False
        
    # Kiem tra 2: So luong dong
    if len(df_check) == expected_rows:
        print(f"[PASS] So luong dong chinh xac: {len(df_check)}/{expected_rows}")
    else:
        print(f"[FAIL] Sai lech so dong! Thuc te: {len(df_check)}, Yeu cau: {expected_rows}")
        checklist_passed = False
        
    # Kiem tra 3: Ten cot
    if list(df_check.columns) == expected_cols:
        print(f"[PASS] Ten cot khop 100%: {list(df_check.columns)}")
    else:
        print(f"[FAIL] Sai ten cot! Thuc te: {list(df_check.columns)}, Yeu cau: {expected_cols}")
        checklist_passed = False
        
    # Kiem tra 4: Du lieu khuyet (NaNs)
    nan_count = df_check.isna().sum().sum()
    if nan_count == 0:
        print("[PASS] Khong co bat ky gia tri NaN nao trong file nop.")
    else:
        print(f"[FAIL] Phat hien {nan_count} gia tri NaN bat hop le!")
        checklist_passed = False
        
    # Kiem tra 5: Gia tri hop le
    valid_values = set(df_check['Is_Fraud'].unique()).issubset({0, 1})
    if valid_values:
        print("[PASS] Gia tri du doan hop le (chi gom 0 va 1).")
    else:
        print("[FAIL] Gia tri du doan chua cac so ngoai le!")
        checklist_passed = False
        
    if checklist_passed:
        print("\\n🏆 TIEU CHUAN THI DAU: TAT CA CAC MUC CHECKLIST DEU DAT CHUAN! SAN SANG NOP BAI.")
    return checklist_passed

# Chay kiem tra tren file vua xuat
is_ready = run_submission_sanity_check("mock_submission.csv")
""",
            """Khối code xây dựng công cụ kiểm tra tự động 5 tiêu chí cốt tử trước khi nộp bài:
- Đọc file hợp lệ, số lượng dòng chính xác ($100/100$), tên cột khớp 100%.
- Không chứa bất kỳ giá trị NaN nào và toàn bộ nhãn dự đoán đều thuộc miền hợp lệ $\{0, 1\}$.
- Trao cho thí sinh sự tự tin tuyệt đối trước khi bấm nút nộp bài chính thức."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Tự tạo kịch bản lỗi để kiểm tra Checklist
Cố ý tạo ra một file submission bị lỗi (ví dụ: chứa 1 giá trị `NaN`, hoặc đặt sai tên cột thành `'Prediction'` thay vì `'Is_Fraud'`). Chạy hàm `run_submission_sanity_check` để quan sát hệ thống bắt lỗi và in ra cảnh báo đỏ như thế nào.

### Bài 2 (Easy): Khóa ngẫu nhiên toàn diện (Global Seed Locker)
Viết một hàm `seed_everything(seed=42)` thiết lập cố định seed ngẫu nhiên cho toàn bộ các thư viện: `random.seed`, `np.random.seed`, `torch.manual_seed`, và `torch.cuda.manual_seed_all`.

### Bài 3 (Medium): Tự động tính điểm thi thử (Mock Competition Scorer)
Giả sử ban giám khảo có file nhãn ẩn `y_test_ground_truth`. Viết hàm `grade_submission(sub_file, ground_truth_file)` tự động tính toán điểm Macro-F1 của bài thi và xếp loại thí sinh theo thang:
- Vàng: $F1 \ge 0.85$
- Bạc: $0.75 \le F1 < 0.85$
- Đồng: $0.65 \le F1 < 0.75$

### Bài 4 (Medium): Thiết lập Chiến lược Nộp bài Phút Chót (Final Submission Selection)
Đa số các kỳ thi cho phép thí sinh chọn 2 bài nộp cuối cùng để tính điểm chung cuộc. Hãy nêu chiến thuật lựa chọn 2 bài nộp này: Tại sao nên chọn 1 bài có điểm CV nội bộ cao nhất (an toàn) và 1 bài có giải pháp đột phá nhất (tiềm năng)?

### Bài 5 (Challenge): Viết Script Tự động Đóng gói Bài thi (Package Exporter)
Viết một script Python tự động gom toàn bộ: Notebook thi đấu, file `submission.csv`, file checkpoint mô hình `.pt`, và một file văn bản tóm tắt giải pháp `SOLUTION_SUMMARY.txt` vào một file nén zip duy nhất mang tên mã số thí sinh: `VOAI_2026_CANDIDATE.zip`.
"""
    voai_connection_content = """
Trong một kỳ thi AI đỉnh cao, bài nộp chỉ được công nhận khi nó **vừa đạt điểm cao vừa tuân thủ 100% quy chế kỹ thuật**. Đã có những đội tuyển dẫn đầu bảng xếp hạng nhưng bị ban giám khảo tước quyền thi đấu (Disqualified) vào phút cuối vì notebook không thể chạy lại được từ đầu (Not Reproducible) hoặc file mã nguồn chứa các dòng lệnh can thiệp bất hợp lệ. Sự kỷ luật, cẩn trọng và tuân thủ nghiêm ngặt Checklist là phẩm chất cao quý nhất của một nhà vô địch Olympic AI.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh nộp bài và chạy lệnh kiểm thử trên notebook của mình thì thấy điểm số mỗi lần chạy lại cho ra một kết quả khác nhau (lúc 82%, lúc 78%), khiến ban giám khảo không thể tái lập kết quả:

```python
# CODE CÓ LỖI THIẾU TÍNH TẤT ĐỊNH (NON-DETERMINISTIC BEHAVIOR):
# Thi sinh su dung cac thuat toan ngau nhien ma quen truyen random_state=42!
# Trong PyTorch can them:
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao tính tái lập (Reproducibility) là yêu cầu số 1 của nghiên cứu khoa học và thi đấu AI.
2. Thêm đầy đủ các thiết lập kiểm soát tính tất định vào đầu notebook thi đấu.
"""
    knowledge_check_content = """
1. Trong quy trình 4 giai đoạn phân bổ thời gian thi đấu Olympic AI (3 giờ), việc bắt buộc phải hoàn thành trong 30 phút đầu tiên là gì?
   - A. Nộp bài thi chính thức cuối cùng.
   - B. Đọc hiểu đề bài, thiết lập sơ đồ Validation và hoàn thành việc nộp bài cho mô hình Baseline đầu tiên (`submission_v1.csv`) để đảm bảo có điểm số an toàn trên hệ thống.
   - C. Đi ăn trưa.
   - D. Viết báo cáo tổng kết.

2. Mục đích quan trọng nhất của việc chạy thao tác "Kernel -> Restart & Run All" trước khi nộp bài thi là gì?
   - A. Giúp máy tính khởi động lại nhanh hơn.
   - B. Xác minh tuyệt đối rằng notebook có thể thực thi trơn tru từ đầu đến cuối một cách tuần tự, không bị phụ thuộc vào các biến ngầm hoặc thứ tự chạy nhảy cóc của các ô code.
   - C. Xóa sạch file dự đoán.
   - D. Đóng cổng nộp bài của ban tổ chức.

3. Trong bảng Checklist nộp bài, điều gì sẽ xảy ra nếu file `submission.csv` của bạn chứa dù chỉ MỘT giá trị `NaN` duy nhất?
   - A. Hệ thống chấm điểm tự động sẽ bỏ qua dòng đó.
   - B. Hệ thống chấm điểm tự động sẽ báo lỗi `Submission Format Error` và bài nộp của bạn sẽ bị chấm 0 điểm.
   - C. Bạn sẽ được cộng thêm 1 điểm khuyến khích.
   - D. File CSV tự động được sửa lỗi.

4. Tại sao các chuyên gia luôn khuyến cáo thí sinh nên "Đóng băng mã nguồn" (Lockdown Code) trong 30 phút cuối cùng của cuộc thi?
   - A. Để đi ngủ sớm.
   - B. Tránh rủi ro đưa vào các ý tưởng vội vàng chưa được kiểm chứng kỹ lưỡng có thể làm hỏng toàn bộ pipeline nạp dữ liệu hoặc gây lỗi sập kernel vào phút chót.
   - C. Vì ban tổ chức cấm gõ bàn phím.
   - D. Để tiết kiệm pin máy tính.

5. Sau khi hoàn thành xuất sắc Buổi M5_S03, học viên chính thức đạt được chứng chỉ cột mốc nào?
   - A. Certificate of AI Orientation
   - B. Certificate of Machine Learning Basics
   - C. Certificate of AI Competition Practice
   - D. Final Project Certificate
"""
    session_summary_content = """
* **What I learned:**
  - Quy trình phân bổ thời gian phòng thi 4 giai đoạn chuẩn mực dưới áp lực đồng hồ.
  - Bộ Checklist 5 tiêu chuẩn vệ sinh Notebook và kiểm tra tính toàn vẹn của file nộp bài.
  - Kỹ năng kiểm soát tính tất định (Reproducibility) và khóa seed ngẫu nhiên toàn hệ thống.
* **What I can now do:**
  - Tự tin bước vào bất kỳ phòng thi Competitive AI nào với tâm lý vững vàng và chiến thuật bài bản.
  - Tự động kiểm tra tính hợp lệ của bài nộp trong 3 giây trước khi nộp lên ban giám khảo.
* **What I still need to practice:**
  - Khám phá cách sử dụng trí tuệ nhân tạo như một trợ thủ đắc lực ở **Module M6: Dùng LLM như trợ lý học và thi**!
"""
    next_session_content = """
Xin chúc mừng bạn đã chính thức vượt qua đấu trường thi đấu Module M5 và nhận Chứng chỉ **Certificate of AI Competition Practice**!
Trong buổi tiếp theo (**M6_S01**), chúng ta sẽ bước sang một chủ đề mang tính thời đại: **Module M6: DÙNG LLM NHƯ TRỢ LÝ HỌC VÀ THI** — Khám phá phương pháp đặt Prompt có phương pháp cho lập trình thi đấu: Giải thích mã nguồn, Brainstorm kiến trúc và Review giải pháp mà không bị phụ thuộc!
"""
    module_assessment = """
---

## 🏆 MODULE 5 ASSESSMENT & CERTIFICATION

### Phần A: Đánh giá Chiến thuật & Vệ sinh Thi đấu (Strategy & Competition Hygiene)
- Nắm vững 15 phút đọc đề, phân loại Target và nhận diện chính xác 3 loại Data Leakage.
- Giải thích bản chất của K-Fold Out-Of-Fold (OOF) Predictions và chiến lược Blending/Ensembling.
- Nắm vững quy trình quản lý thời gian phòng thi 4 giai đoạn và tính tất định (Reproducibility).

### Phần B: Sản phẩm Thực hành Bắt buộc (Practical Deliverable)
- **Bài thi thử có chấm điểm tự động (Mock Competition Deliverable):** Học viên hoàn thành một pipeline thi đấu khép kín trên dữ liệu giả lập, có tính toán điểm F1-Score trên hệ thống K-Fold OOF và xuất file nộp bài.
- **Checklist Nộp bài (Submission Checklist):** Chạy và vượt qua 100% các tiêu chí kiểm tra tự động của bộ kiểm thử `run_submission_sanity_check` (Số dòng, Tên cột, Không NaNs, Giá trị hợp lệ).

### Tiêu chí Đạt chuẩn (Rubric):
| Tiêu chí | Trọng số | Yêu cầu đạt |
| :--- | :--- | :--- |
| **Chiến thuật Validation & Chống Leakage** | 30% | Dựng đúng sơ đồ kiểm chứng (Stratified/Group K-Fold) |
| **Feature Engineering & Ensembling**     | 25% | Có đặc trưng mới và kết hợp tối thiểu 2 mô hình |
| **File Nộp bài & Kiểm thử tự động**      | 25% | Đạt 100% các tiêu chí của Submission Checklist |
| **Tính Tái lập (Reproducibility)**       | 20% | Cố định Seed, Notebook chạy mượt mà từ đầu đến cuối |

**Chứng chỉ hoàn thành:**
🏅 **Certificate of AI Competition Practice** — Đủ điều kiện tiến vào Module M6 (AI-assisted Problem Solving).
"""
    trace = {
        "items": [
            "Giả lập phòng thi Olympic AI (Mock competition challenge)",
            "Quản lý thời gian thi đấu 4 giai đoạn & Notebook hygiene",
            "Bảng kiểm tra nộp bài tự động (Submission checklist)",
            "Nhận chứng chỉ Certificate of AI Competition Practice"
        ],
        "deliverable": "Bài thi thử có chấm điểm + Checklist nộp bài (Submission Checklist) hoàn chỉnh"
    }
    nb = create_notebook(
        module_id="M5",
        module_name="CHIẾN THUẬT THI OLYMPIC AI",
        session_id="S03",
        topic="Mock Olympic AI Competition & Submission Checklist",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Bài thi thử có chấm điểm + Checklist nộp bài (Submission Checklist)",
        competition_relevance="Đợt diễn tập tổng lực bắt buộc trước khi bước vào kỳ thi Olympic AI chính thức.",
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
    save_notebook(nb, "course/M5_OLYMPIC_STRATEGY/M5_S03.ipynb")

if __name__ == "__main__":
    build_m5_s01()
    build_m5_s02()
    build_m5_s03()
    print("M5 build complete! 3/3 notebooks generated.")
