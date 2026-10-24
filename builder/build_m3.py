"""
builder/build_m3.py
Generates M3_S01.ipynb to M3_S05.ipynb for Module 3.
"""

from utils import create_notebook, save_notebook

def build_m3_s01():
    objectives = [
        "Làm chủ cấu trúc dữ liệu nền tảng PyTorch Tensor và sự tương đồng/khác biệt với NumPy ndarray.",
        "Hiểu cơ chế quản lý thiết bị tính toán: Chuyển đổi linh hoạt giữa CPU và GPU (CUDA/MPS).",
        "Nắm vững cơ chế Autograd (Tự động tính đạo hàm) và Đồ thị tính toán (Computational Graph).",
        "Tự tay lập trình thuật toán Hồi quy tuyến tính từ số 0 chỉ bằng PyTorch Tensors và cơ chế lan truyền ngược (`.backward()`)."
    ]
    prerequisites = [
        "Hoàn thành Module M2 (Đặc biệt là M2_S02 về Hồi quy Tuyến tính và hàm mất mát).",
        "Kiến thức đại số tuyến tính: Phép nhân ma trận và đạo hàm hàm hợp (Chain Rule)."
    ]
    why_content = """
Trong Machine Learning cổ điển (Scikit-Learn), bạn chỉ cần gọi `model.fit()`. Mọi phép toán đạo hàm tối ưu đều bị giấu kín trong một "hộp đen" (Black-box). Nhưng khi bước vào Học Sâu (Deep Learning) và thi đấu Olympic AI, các kiến trúc mạng nơ-ron có thể có hàng chục triệu tham số với các kết nối phi tuyến phức tạp, đòi hỏi bạn phải can thiệp trực tiếp vào từng tầng tính toán.

PyTorch không chỉ là một thư viện Deep Learning; nó là một **Nền tảng Tính toán Ma trận Đa chiều có Tự động Tính Đạo hàm (Automatic Differentiation Engine - Autograd) được tăng tốc bằng GPU**. Hiểu cách PyTorch theo dõi từng phép toán để tự động tính gradient là chìa khóa mở toang cánh cửa bước vào kỷ nguyên AI hiện đại!
"""
    concept_content = r"""
### 1. PyTorch Tensor vs NumPy ndarray
* Tương đồng: Đều là mảng đa chiều hỗ trợ tính toán số học vectơ hóa tốc độ cao.
* Khác biệt vượt trội của PyTorch Tensor:
  1. **Khả năng chạy trên GPU:** Tăng tốc tính toán song song hàng nghìn nhân CUDA (`tensor.to('cuda')`).
  2. **Tích hợp sẵn Autograd:** Tự động xây dựng đồ thị tính toán để đạo hàm ngược không cần con người tính tay.

### 2. Trực giác Đồ thị tính toán (Computational Graph) & Autograd
Khi bạn thiết lập `requires_grad=True` cho một tensor, PyTorch sẽ bắt đầu ghi nhớ mọi thao tác đại số tác động lên nó.
* **Forward Pass (Lan truyền xuôi):** Tính toán giá trị đầu ra từ đầu vào thông qua các phép toán nối tiếp.
* **Backward Pass (Lan truyền ngược):** Khi gọi `loss.backward()`, PyTorch sẽ tự động áp dụng **Quy tắc đạo hàm chuỗi (Chain Rule)**, đi ngược từ nút ngọn (Loss) về các nút gốc, và lưu giá trị đạo hàm riêng $\frac{\partial Loss}{\partial w}$ vào thuộc tính `w.grad`.

### 3. Tối ưu hóa Gradient Descent bằng PyTorch thuần
Với tốc độ học $\alpha$ (Learning Rate), tham số $w$ được cập nhật theo hướng ngược chiều dốc:
$$w \leftarrow w - \alpha \times \frac{\partial Loss}{\partial w}$$
*Lưu ý cốt tử:* Cần bọc bước cập nhật trọng số trong `with torch.no_grad():` để không làm ô nhiễm đồ thị tính toán!
"""
    visual_intuition_content = """
```
Đồ thị tính toán tự động (Computational Graph):
     x (Input) ──┐
                 ├──> [ Phép nhân: w * x ] ──> [ Phép cộng: + b ] ──> y_hat (Dự đoán)
     w (Weight) ─┘                                                           │
                                                                             ▼
     y (Target) ───────────────────────────────────────────────────> [ Loss: MSE ]
                                                                             │
                                                                     loss.backward()
                                                                             │
    w.grad = dLoss/dw  <─────────────────────────────────────────────────────┘
```
"""
    code_cells = [
        (
            """import torch
import numpy as np

# 1. Khoi tao PyTorch Tensor va quan ly thiet bi (Device Agnostic Code)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"He thong dang su dung thiet bi tinh toan: {device}")

# Tao tensor tu du lieu co san va kiem tra thuoc tinh
x_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device)
print("Tensor x_data:\\n", x_data)
print(f"Shape: {x_data.shape} | Dtype: {x_data.dtype} | Device: {x_data.device}")

# Chuyen doi qua lai 0-copy giua NumPy va PyTorch CPU
np_arr = np.array([10, 20, 30])
torch_from_np = torch.from_numpy(np_arr)
print("\\nChuyen tu NumPy sang PyTorch:", torch_from_np)
""",
            """Khối code minh họa thao tác cơ sở với PyTorch Tensor:
- Kỹ thuật viết code bất khả tri thiết bị (Device-agnostic) tự động nhận diện GPU nếu có.
- Khám phá các thuộc tính then chốt: `shape`, `dtype`, `device`.
- Phương thức `torch.from_numpy()` chia sẻ chung vùng nhớ bộ nhớ đệm giúp chuyển đổi không tốn chi phí copy."""
        ),
        (
            """# 2. Co che Autograd va Vong lap Toi uu hoa tu so 0 bang Pure PyTorch
torch.manual_seed(42)

# Tao du lieu mau: y = 3.5 * x + 2.0 + nhieu
X = torch.randn(100, 1)
y_true = 3.5 * X + 2.0 + 0.2 * torch.randn(100, 1)

# Khoi tao tham so mo hinh can hoc (Weights & Bias) voi requires_grad=True
w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, 1, requires_grad=True)

learning_rate = 0.05
epochs = 50

print("===== BAT DAU TRAINING LOOP PURE PYTORCH =====")
for epoch in range(1, epochs + 1):
    # Buoc 1: Forward Pass - Tinh toan du doan
    y_pred = X @ w + b
    
    # Buoc 2: Tinh ham mat mat (MSE Loss)
    loss = torch.mean((y_pred - y_true) ** 2)
    
    # Buoc 3: Backward Pass - Tinh dao ham tu dong
    loss.backward()
    
    # Buoc 4: Cap nhat tham so bang Gradient Descent (khong luu vao do thi)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # Buoc 5: Xoa trang gradient truoc luot tiep theo
        w.grad.zero_()
        b.grad.zero_()
        
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:2d}/{epochs} | MSE Loss: {loss.item():.4f} | w: {w.item():.3f} | b: {b.item():.3f}")

print(f"\\nGia tri toi uu hoc duoc: w = {w.item():.3f} (That: 3.50), b = {b.item():.3f} (That: 2.00)")
""",
            """Khối code tự tay viết một vòng lặp tối ưu hóa Gradient Descent thuần túy bằng PyTorch Autograd:
- `requires_grad=True`: Kích hoạt cơ chế theo dõi đạo hàm cho $w$ và $b$.
- `loss.backward()`: Tự động tính toàn bộ đạo hàm riêng.
- `with torch.no_grad()`: Tạm thời vô hiệu hóa autograd trong quá trình cập nhật trọng số.
- `w.grad.zero_()`: **Bắt buộc** xóa gradient cũ vì PyTorch mặc định sẽ cộng dồn đạo hàm (Accumulate Gradients)."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Phép toán đại số với Tensors
Tạo tensor $A$ có kích thước $(3, 4)$ và tensor $B$ có kích thước $(4, 2)$ với các giá trị ngẫu nhiên. Thực hiện phép nhân ma trận $A \times B$ bằng toán tử `@` hoặc hàm `torch.matmul`. In ra kích thước của tensor kết quả.

### Bài 2 (Easy): Đạo hàm hàm đa thức một biến
Cho hàm số $y = 3x^3 - 5x^2 + 7x - 2$. Hãy dùng PyTorch Autograd để tính giá trị đạo hàm $\\frac{dy}{dx}$ tại điểm $x = 2.0$. Đối chiếu với đạo hàm giải tích tính bằng tay ($y' = 9x^2 - 10x + 7$).

### Bài 3 (Medium): Tác động của Learning Rate
Thử nghiệm thay đổi `learning_rate` trong đoạn code trên thành $1.5$ (quá lớn) và $0.0001$ (quá nhỏ). Quan sát hiện tượng nổ mất mát (Exploding Loss / NaN) hoặc mô hình học quá chậm.

### Bài 4 (Medium): Ngắt nhánh đồ thị bằng `.detach()`
Tìm hiểu phương thức `tensor.detach()`. Viết một ví dụ nhỏ chứng minh rằng khi gọi `.detach()`, tensor mới sẽ không còn kết nối với đồ thị tính toán ban đầu và không thể lan truyền gradient ngược qua nó.

### Bài 5 (Challenge): Hồi quy phi tuyến bậc hai bằng Autograd
Tạo dữ liệu theo hàm bậc hai $y = 2x^2 - 3x + 1$. Sử dụng PyTorch Tensors với 3 tham số cần học: $w_2, w_1, b$ và viết vòng lặp tối ưu hóa để tìm lại bộ tham số trên.
"""
    voai_connection_content = """
Trong các bài thi Olympic AI nâng cao, bạn thường gặp các bài toán yêu cầu thiết kế hàm mất mát tùy biến (Custom Loss Function, ví dụ Focal Loss cho phân loại mất cân bằng, hoặc Metric Learning Loss). Nếu không hiểu Autograd, bạn sẽ không biết viết hàm mất mát sao cho PyTorch có thể tự động đạo hàm được. Quy tắc sống còn: **Mọi phép toán trong Custom Loss phải dùng toán tử của PyTorch (không dùng NumPy), nếu không đồ thị tính toán sẽ bị đứt gãy và `.backward()` sẽ không hoạt động!**
"""
    debugging_content = """
**Tình huống thực tế:** Một sinh viên quên dòng lệnh xóa gradient `w.grad.zero_()`, khiến mô hình chạy qua mỗi epoch thì Loss lại tăng vọt bất thường:

```python
# CODE CÓ LỖI TÍCH LŨY GRADIENT:
# PyTorch co tinh nang mac dinh la w.grad += new_grad (Accumulate Gradients)!
# Neu quen xoa ve 0 sau moi buoc, gradient se ngay cang khong lo!
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao PyTorch lại thiết kế tính năng cộng dồn gradient thay vì tự động xóa đè (gợi ý: hữu ích khi cần tích lũy gradient qua nhiều mini-batch khi GPU bị thiếu VRAM).
2. Kiểm chứng việc bỏ dòng `w.grad.zero_()` làm sai lệch kết quả tối ưu như thế nào.
"""
    knowledge_check_content = """
1. Thuộc tính nào của PyTorch Tensor cần được đặt thành `True` để hệ thống tự động theo dõi và ghi lại đồ thị tính toán đạo hàm?
   - A. `is_trainable=True`
   - B. `requires_grad=True`
   - C. `track_gradient=True`
   - D. `auto_diff=True`

2. Khối lệnh `with torch.no_grad():` thường được sử dụng trong trường hợp nào?
   - A. Khi muốn tăng độ chính xác của mạng nơ-ron lên gấp đôi.
   - B. Khi cập nhật tham số hoặc khi chạy kiểm thử (Inference / Evaluation) để tiết kiệm bộ nhớ RAM/VRAM và tăng tốc độ xử lý do không cần lưu đồ thị tính toán.
   - C. Khi muốn xóa dữ liệu tập Train.
   - D. Khi chuyển code sang chạy trên GPU.

3. Tại sao sau mỗi bước cập nhật trọng số trong PyTorch, chúng ta bắt buộc phải gọi `.grad.zero_()`?
   - A. Vì nếu không xóa, PyTorch sẽ báo lỗi cú pháp.
   - B. Vì PyTorch mặc định sẽ cộng dồn gradient mới vào gradient cũ từ các vòng lặp trước, dẫn tới việc cập nhật sai lệch hướng dốc.
   - C. Để giải phóng toàn bộ card đồ họa.
   - D. Vì gradient sẽ biến thành số âm.

4. Phương thức nào dùng để chuyển một PyTorch Tensor từ CPU sang card đồ họa GPU một cách an toàn nhất?
   - A. `tensor.gpu()`
   - B. `tensor.to(device)` với `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
   - C. `tensor.cuda_force()`
   - D. `tensor.convert_gpu()`

5. Khi ta gọi lệnh `loss.backward()`, thuật toán toán học cốt lõi nào được kích hoạt để tính toán đạo hàm ngược qua toàn bộ các tầng?
   - A. Thuật toán tìm kiếm nhị phân (Binary Search).
   - B. Quy tắc đạo hàm chuỗi hàm hợp (Chain Rule) trong giải tích vi phân.
   - C. Phép biến đổi Fourier nhanh (FFT).
   - D. Phương pháp khử Gauss-Jordan.
"""
    session_summary_content = """
* **What I learned:**
  - Bản chất của PyTorch Tensor và cơ chế quản lý phần cứng CPU/GPU.
  - Sức mạnh của Autograd và Đồ thị tính toán tự động (Computational Graph).
  - Tự tay cài đặt vòng lặp Gradient Descent thuần túy không dùng thư viện cao cấp.
* **What I can now do:**
  - Viết code tính toán đạo hàm tự động cho bất kỳ hàm mục tiêu phức tạp nào.
  - Tự tin kiểm soát và điều khiển dòng chảy của Tensor trong bộ nhớ.
* **What I still need to practice:**
  - Chuyển giao cách nạp dữ liệu thủ công sang hệ thống nạp dữ liệu theo Batch chuyên nghiệp: **Dataset & DataLoader** ở Buổi 2.
"""
    next_session_content = """
Trong **Buổi M3_S02**, chúng ta sẽ giải quyết bài toán nạp dữ liệu quy mô lớn vào bộ nhớ: **PyTorch Data Pipeline: Tùy Biến `Dataset` & `DataLoader`** — Học cách chia Mini-batch, xáo trộn dữ liệu (Shuffle), và xử lý đa luồng nạp dữ liệu!
"""
    trace = {
        "items": [
            "PyTorch tensors & GPU device transfer",
            "Autograd & Computational graph",
            "Manual optimization loop via backward() & grad.zero_()"
        ],
        "deliverable": "Mã nguồn tối ưu hóa tham số bằng PyTorch Tensor và Autograd từ số 0"
    }
    nb = create_notebook(
        module_id="M3",
        module_name="XÂY MODEL AI SÂU HƠN BẰNG PYTORCH",
        session_id="S01",
        topic="PyTorch Tensors, Autograd & Đồ Thị Tính Toán Tự Động",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook lập trình tối ưu hóa đạo hàm tự động bằng PyTorch Autograd",
        competition_relevance="Làm chủ tầng tính toán vi phân của PyTorch, sẵn sàng viết Custom Loss cho phòng thi.",
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
    save_notebook(nb, "course/M3_PYTORCH/M3_S01.ipynb")


def build_m3_s02():
    objectives = [
        "Hiểu lý do tại sao không thể nạp toàn bộ tập dữ liệu khổng lồ vào RAM/VRAM cùng một lúc (Out of Memory - OOM).",
        "Kế thừa và tùy biến lớp dữ liệu `torch.utils.data.Dataset` thông qua 3 phương thức chuẩn mực: `__init__`, `__len__`, `__getitem__`.",
        "Sử dụng `torch.utils.data.DataLoader` để tự động hóa việc chia Mini-batch, xáo trộn (Shuffle) và nạp dữ liệu đa tiến trình.",
        "Tích hợp tiền xử lý và chuyển đổi kiểu dữ liệu Tensor trực tiếp trong luồng nạp dữ liệu."
    ]
    prerequisites = [
        "Hoàn thành Buổi M3_S01 (PyTorch Tensors).",
        "Hiểu cú pháp Lập trình hướng đối tượng (OOP) cơ bản trong Python (kế thừa class)."
    ]
    why_content = """
Trong các bài thi Olympic AI hoặc các dự án thực tế, tập dữ liệu có thể chứa hàng triệu dòng dữ liệu bảng hoặc hàng trăm nghìn tấm ảnh độ phân giải cao (nhiều Gigabytes). Nếu bạn cố gắng ép toàn bộ tập dữ liệu thành một Tensor khổng lồ rồi đưa vào huấn luyện, card đồ họa GPU hoặc RAM máy tính sẽ lập tức nổ tung với thông báo lỗi kinh hoàng: `CUDA out of memory`!

PyTorch giải quyết triệt để vấn đề này bằng một cặp đôi hoàn hảo: **`Dataset`** (đóng vai trò như một thủ kho biết chính xác từng món hàng nằm ở đâu) và **`DataLoader`** (đóng vai trò như đội ngũ xe tải chuyên chở từng chuyến hàng nhỏ - Mini-batch - nạp vào GPU đều đặn theo thời gian thực).
"""
    concept_content = r"""
### 1. Kiến trúc tùy biến `torch.utils.data.Dataset`
Để tạo một Dataset tùy biến trong PyTorch, bạn bắt buộc phải kế thừa lớp `Dataset` và ghi đè (override) đúng 3 phương thức:
1. `__init__(self, ...)`: Nạp dữ liệu thô (từ DataFrame, file CSV, hoặc mảng NumPy) và lưu trữ các biến cấu hình.
2. `__len__(self)`: Trả về tổng số lượng mẫu trong tập dữ liệu (`len(self.data)`).
3. `__getitem__(self, idx)`: Trả về đúng một cặp mẫu thử $(X_{idx}, y_{idx})$ tại chỉ số `idx` cụ thể, đã được chuyển hóa sang kiểu `torch.Tensor`.

### 2. Các tham số cốt tử của `DataLoader`
* `batch_size`: Số lượng mẫu dữ liệu được gom lại trong một lần chuyển giao cho mô hình (ví dụ: 32, 64, 128).
* `shuffle`: Có xáo trộn ngẫu nhiên thứ tự các mẫu ở đầu mỗi epoch không? (Luôn đặt `True` cho tập Train để phá vỡ tương quan thứ tự, và đặt `False` cho tập Validation/Test để giữ nguyên thứ tự đánh giá).
* `num_workers`: Số lượng tiến trình con chạy song song dưới nền để đọc dữ liệu từ đĩa cứng (giúp GPU không bị đói dữ liệu).
* `drop_last`: Có bỏ qua batch cuối cùng nếu số lượng mẫu còn sót lại không đủ một full batch hay không.
"""
    visual_intuition_content = """
```
Luồng dữ liệu PyTorch Data Pipeline:
[Toàn bộ File Dữ liệu / Đĩa cứng]
               │
               ▼
   [Custom Dataset Class] ──> Cung cấp __getitem__(idx) cho từng mẫu đơn lẻ
               │
               ▼
        [DataLoader]
      ├──> Gom nhóm mẫu thành Mini-Batches (batch_size=32)
      ├──> Xáo trộn ngẫu nhiên (shuffle=True)
      └──> Nạp song song đa luồng (num_workers=2)
               │
               ▼
[Vòng lặp Training Loop] ──> for batch_X, batch_y in train_loader: ...
```
"""
    code_cells = [
        (
            """import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# 1. Khoi tao mot tap du lieu bang mo phong ve du doan suc khoe
np.random.seed(42)
n_records = 200

raw_df = pd.DataFrame({
    'Heart_Rate': np.random.normal(75, 12, size=n_records).round(1),
    'Blood_Pressure': np.random.normal(120, 15, size=n_records).round(1),
    'Cholesterol': np.random.normal(200, 30, size=n_records).round(1),
    'Has_Heart_Risk': np.random.choice([0, 1], size=n_records, p=[0.7, 0.3])
})

# 2. Dinh nghia lop Custom Tabular Dataset ke thua tu torch.utils.data.Dataset
class HealthTabularDataset(Dataset):
    def __init__(self, dataframe, is_train=True):
        self.features = dataframe[['Heart_Rate', 'Blood_Pressure', 'Cholesterol']].values
        self.labels = dataframe['Has_Heart_Risk'].values
        self.is_train = is_train
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        # Trich xuat mau tai chi so idx va ep kieu sang torch.float32
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32) # Co the la float32 neu dung BCEWithLogitsLoss
        return x, y

# Tao doi tuong Dataset
health_dataset = HealthTabularDataset(raw_df)
print(f"Tong so mau trong Dataset: {len(health_dataset)}")

# Kiem tra thu mau dau tien
sample_x, sample_y = health_dataset[0]
print(f"Mau thu dau tien (Index 0) -> x: {sample_x}, y: {sample_y}")
""",
            """Khối code minh họa việc tạo một lớp Custom Dataset chuẩn mực:
- Kế thừa `Dataset` từ `torch.utils.data`.
- Cài đặt đầy đủ 3 phương thức `__init__`, `__len__`, `__getitem__`.
- Đảm bảo dữ liệu trả về từ `__getitem__` luôn là các PyTorch Tensor với kiểu dữ liệu chuẩn (`float32`)."""
        ),
        (
            """# 3. Dong goi Dataset vao DataLoader va duyet qua cac Mini-Batches
batch_size = 32

train_loader = DataLoader(
    dataset=health_dataset,
    batch_size=batch_size,
    shuffle=True,       # Xao tron du lieu de mo hinh khong hoc theo thu tu
    drop_last=False     # Giu lai batch cuoi du co the nho hon 32
)

print(f"Tong so batch trong 1 Epoch: {len(train_loader)} (moi batch chua toi da {batch_size} mau)")

# Duyet thu qua Batch dau tien de kiem tra kich thuoc Tensor
for batch_idx, (batch_features, batch_labels) in enumerate(train_loader, start=1):
    print(f"Batch {batch_idx:2d} -> Features Shape: {batch_features.shape} | Labels Shape: {batch_labels.shape}")
    if batch_idx == 2:
        print("... Cac batch tiep theo nap tuong tu ...")
        break
""",
            """Khối code khởi tạo và kiểm tra `DataLoader`:
- Tự động chia 200 mẫu thành 7 batch (6 batch 32 mẫu và 1 batch cuối 8 mẫu).
- Kích thước tensor đầu ra của mỗi batch là `(batch_size, num_features)` — sẵn sàng đưa trực tiếp vào tầng đầu tiên của mạng nơ-ron."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Tích hợp Chuẩn hóa dữ liệu vào Dataset
Sửa lại class `HealthTabularDataset` để nhận thêm tham số `mean` và `std`. Trong `__getitem__`, áp dụng công thức chuẩn hóa $x_{norm} = \\frac{x - mean}{std}$ trước khi trả về tensor.

### Bài 2 (Easy): Phân chia Train/Validation Dataset
Sử dụng hàm `torch.utils.data.random_split(health_dataset, [0.8, 0.2])` để chia dataset thành 2 tập: `train_set` (80%) và `val_set` (20%). Khởi tạo 2 DataLoader tương ứng (lưu ý: `train_loader` có `shuffle=True`, còn `val_loader` có `shuffle=False`).

### Bài 3 (Medium): Xử lý Batch không đều với `drop_last=True`
Chạy lại DataLoader với tham số `drop_last=True`. Quan sát xem tổng số batch trong 1 epoch thay đổi như thế nào và giải thích khi nào trong thi đấu Olympic ta nên bật `drop_last=True` (gợi ý: khi một số tầng BatchNorm yêu cầu batch size lớn hơn 1).

### Bài 4 (Medium): Kiểm tra tốc độ nạp dữ liệu với `num_workers`
Tìm hiểu tham số `num_workers` trong DataLoader. Thử nghiệm đặt `num_workers=0` (mặc định chạy trên tiến trình chính) và `num_workers=2` (đa tiến trình). Nêu lưu ý quan trọng khi dùng `num_workers > 0` trên hệ điều hành Windows (yêu cầu khối lệnh `if __name__ == '__main__':`).

### Bài 5 (Challenge): Viết Custom Collate Function
Trong nhiều bài toán xử lý văn bản hoặc đồ thị, các mẫu trong cùng một batch có thể có độ dài khác nhau. Hãy tìm hiểu tham số `collate_fn` của DataLoader và viết một hàm collate đơn giản tự động đệm thêm số 0 (padding) cho các vector có kích thước không đồng nhất.
"""
    voai_connection_content = """
Trong các bài thi Deep Learning của Olympic AI, nghẽn cổ chai lớn nhất thường **không nằm ở GPU mà nằm ở CPU nạp dữ liệu (Data Loading Bottleneck)**! Nếu bạn viết hàm `__getitem__` quá nặng (ví dụ: mở file đĩa liên tục, tiền xử lý chuỗi phức tạp bằng vòng lặp Python bên trong getitem), GPU sẽ phải ngồi chơi xơi nước 80% thời gian chờ CPU chuyển dữ liệu sang. Thiết kế `Dataset` gọn gàng, tính toán trước các thao tác nặng trong `__init__`, và dùng `DataLoader` hợp lý sẽ giúp bạn rút ngắn thời gian huấn luyện từ 1 giờ xuống còn 10 phút!
"""
    debugging_content = """
**Tình huống thực tế:** Một thí sinh viết `__getitem__` trả về nhãn mục tiêu dạng số nguyên nhưng quên chuyển sang Tensor, hoặc chuyển sai kiểu dữ liệu thành `torch.float64` (Double) thay vì `torch.float32`:

```python
# CODE CÓ LỖI KIỂU DỮ LIỆU TENSOR KHI NẠP:
# PyTorch mac dinh cac lop Linear yeu cau dau vao la Float32!
# Neu tra ve Float64 (DoubleTensor), mo hinh se văng loi:
# RuntimeError: Expected object of scalar type Float but got scalar type Double for argument #4 'mat1'
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao PyTorch chuẩn hóa toàn bộ tính toán mạng nơ-ron ở kiểu dữ liệu `torch.float32` (Single Precision) thay vì `float64`.
2. Sửa lỗi triệt để trong `__getitem__` bằng cách luôn tường minh `dtype=torch.float32`.
"""
    knowledge_check_content = """
1. Khi kế thừa lớp `torch.utils.data.Dataset`, ba phương thức nào là BẮT BUỘC phải cài đặt?
   - A. `__start__`, `__run__`, `__stop__`
   - B. `__init__`, `__len__`, `__getitem__`
   - C. `fit`, `predict`, `score`
   - D. `forward`, `backward`, `step`

2. Tham số `shuffle=True` trong DataLoader có ý nghĩa gì và nên được áp dụng cho tập dữ liệu nào?
   - A. Xáo trộn thứ tự các cột đặc trưng; áp dụng cho cả Train và Test.
   - B. Xáo trộn ngẫu nhiên thứ tự các hàng mẫu thử ở đầu mỗi epoch; nên áp dụng cho tập Train để tránh thiên lệch thứ tự học.
   - C. Xóa các hàng ngẫu nhiên để giảm dung lượng RAM.
   - D. Chỉ áp dụng cho tập Test để tăng độ khó.

3. Phương thức `__len__(self)` trong Custom Dataset được gọi khi người dùng thực hiện câu lệnh nào?
   - A. `dataset[0]`
   - B. `len(dataset)`
   - C. `dataset.size()`
   - D. `dataset.shape`

4. Hiện tượng `CUDA out of memory` (OOM) trong quá trình huấn luyện mạng nơ-ron thường được khắc phục nhanh nhất bằng cách điều chỉnh tham số nào?
   - A. Tăng số lượng epochs.
   - B. Giảm giá trị `batch_size` trong DataLoader (ví dụ giảm từ 128 xuống 64 hoặc 32).
   - C. Tăng Learning Rate lên gấp 10 lần.
   - D. Tắt mạng Internet.

5. Tham số `drop_last=True` trong DataLoader thực hiện nhiệm vụ gì?
   - A. Xóa mẫu dữ liệu cuối cùng của tập dữ liệu vĩnh viễn.
   - B. Bỏ qua mini-batch cuối cùng nếu số lượng phần tử của nó nhỏ hơn `batch_size` quy định.
   - C. Xóa tầng cuối cùng của mạng nơ-ron.
   - D. Bỏ qua epoch cuối cùng.
"""
    session_summary_content = """
* **What I learned:**
  - Nguyên nhân và cách phòng chống lỗi tràn bộ nhớ (OOM) bằng xử lý theo Batch.
  - Cấu trúc 3 phương thức chuẩn mực của `torch.utils.data.Dataset`.
  - Cơ chế hoạt động của `DataLoader`: Batching, Shuffling, Multi-processing.
* **What I can now do:**
  - Tự tay đóng gói bất kỳ bảng dữ liệu nào thành PyTorch Dataset chuyên nghiệp.
  - Thiết lập luồng nạp dữ liệu Mini-batch chuẩn hóa sẵn sàng cấp liệu cho mạng nơ-ron.
* **What I still need to practice:**
  - Thiết kế kiến trúc mạng nơ-ron hoàn chỉnh bằng **`torch.nn.Module`** ở Buổi 3.
"""
    next_session_content = """
Trong **Buổi M3_S03**, chúng ta sẽ chính thức xây dựng bộ não nơ-ron: **Thiết Kế Kiến Trúc Mạng Nơ-ron với `torch.nn.Module`** — Tìm hiểu cấu trúc Perceptron đa tầng (MLP), các hàm kích hoạt (ReLU, Sigmoid), và cách quản lý tham số mạng!
"""
    trace = {
        "items": [
            "Hiểu Dataset & DataLoader trong PyTorch",
            "Tùy biến Custom Dataset (__init__, __len__, __getitem__)",
            "Quản lý batch_size, shuffle & nạp dữ liệu an toàn bộ nhớ"
        ],
        "deliverable": "Mã nguồn Data Pipeline tùy biến đóng gói dữ liệu bảng thành Mini-batches"
    }
    nb = create_notebook(
        module_id="M3",
        module_name="XÂY MODEL AI SÂU HƠN BẰNG PYTORCH",
        session_id="S02",
        topic="PyTorch Data Pipeline: Tùy Biến Dataset & DataLoader",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook xây dựng Custom Dataset và DataLoader hoàn chỉnh",
        competition_relevance="Quản lý bộ nhớ tối ưu, loại bỏ triệt để nghẽn cổ chai nạp dữ liệu khi thi đấu.",
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
    save_notebook(nb, "course/M3_PYTORCH/M3_S02.ipynb")


def build_m3_s03():
    objectives = [
        "Hiểu kiến trúc Perceptron Đa tầng (Multi-Layer Perceptron - MLP) và nguyên lý xâu chuỗi các tầng tuyến tính.",
        "Kế thừa và làm chủ lớp trừu tượng `torch.nn.Module` để đóng gói kiến trúc mạng nơ-ron.",
        "Hiểu bản chất toán học và sự cần thiết của các Hàm kích hoạt phi tuyến (Non-linear Activations): ReLU, Sigmoid, LeakyReLU.",
        "Kiểm tra, đếm số lượng tham số học được (`parameters()`) và truy vết kích thước Tensor qua từng tầng mạng."
    ]
    prerequisites = [
        "Hoàn thành Buổi M3_S01 và M3_S02.",
        "Hiểu phép nhân ma trận giữa tensor đầu vào và ma trận trọng số tầng ẩn."
    ]
    why_content = """
Nếu chúng ta chỉ xếp chồng nhiều tầng tuyến tính liên tiếp nhau: $\hat{y} = W_2 (W_1 X + b_1) + b_2$, thì theo định lý đại số tuyến tính, tích của hai ma trận tuyến tính $W_2 W_1$ vẫn chỉ là MỘT ma trận tuyến tính duy nhất! Điều đó có nghĩa là dù bạn có xếp 100 tầng tuyến tính, mạng nơ-ron của bạn vẫn chỉ tương đương với một mô hình Hồi quy Tuyến tính đơn giản, hoàn toàn bất lực trước dữ liệu phi tuyến!

Để tạo ra "trí thông minh nhân tạo", chúng ta bắt buộc phải chèn các **Hàm kích hoạt phi tuyến (Non-linear Activation Functions)** vào giữa các tầng. Lớp `torch.nn.Module` là công cụ quyền năng nhất của PyTorch giúp bạn thiết kế, tùy biến và kiểm soát mọi kiến trúc mạng nơ-ron từ đơn giản đến những mô hình vĩ đại nhất thế giới.
"""
    concept_content = r"""
### 1. Kiến trúc Perceptron Đa tầng (MLP)
Một mạng MLP chuẩn mực gồm 3 thành phần:
1. **Input Layer (Tầng đầu vào):** Nhận vector đặc trưng có $D_{in}$ chiều.
2. **Hidden Layers (Các tầng ẩn):** Nơi mạng nơ-ron học cách biến đổi và trừu tượng hóa đặc trưng. Mỗi tầng thực hiện: $h = \sigma(W x + b)$.
3. **Output Layer (Tầng đầu ra):** Đưa ra kết quả dự đoán với số chiều $D_{out}$ (1 nơ-ron cho hồi quy/nhị phân, $C$ nơ-ron cho phân loại đa lớp).

### 2. Tại sao hàm ReLU (Rectified Linear Unit) lại thống trị Deep Learning?
$$\text{ReLU}(z) = \max(0, z)$$
* **Giải quyết triệt tiêu đạo hàm (Vanishing Gradient):** Đạo hàm của ReLU luôn bằng 1 khi $z > 0$, giúp dòng gradient có thể lan truyền ngược qua hàng chục tầng mạng sâu mà không bị suy giảm về 0.
* **Tốc độ tính toán siêu nhanh:** Chỉ là phép so sánh lớn hơn 0 trên phần cứng CPU/GPU.

### 3. Cấu trúc chuẩn của một lớp `nn.Module`
* Trong `__init__(self)`: Khai báo các tầng mạng (`nn.Linear`, `nn.ReLU`, `nn.Dropout`).
* Trong `forward(self, x)`: Định nghĩa luồng dữ liệu di chuyển từ đầu vào xuyên qua các tầng để tạo ra đầu ra.
"""
    visual_intuition_content = """
```
Cấu trúc Mạng nơ-ron Perceptron Đa tầng (MLP):
  x1 ──○
        \
  x2 ──○ ──> [ Tầng ẩn 1: nn.Linear(3, 8) + ReLU ] ──> [ Tầng ẩn 2: nn.Linear(8, 4) + ReLU ] ──> [ Output: nn.Linear(4, 1) ] ──> y_hat
        /
  x3 ──○
```
"""
    code_cells = [
        (
            """import torch
import torch.nn as nn

# 1. Dinh nghia kien truc mang Perceptron Da tang (MLP) ke thua tu nn.Module
class MedicalClassifierMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=1):
        super(MedicalClassifierMLP, self).__init__()
        
        # Tang an 1 (Hidden Layer 1)
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.relu1 = nn.ReLU()
        
        # Tang an 2 (Hidden Layer 2)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim // 2)
        self.relu2 = nn.ReLU()
        
        # Tang dau ra (Output Layer - Logits)
        self.out = nn.Linear(in_features=hidden_dim // 2, out_features=output_dim)
        
    def forward(self, x):
        # Dinh nghia luong chay (Forward Pass)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        logits = self.out(x)
        return logits

# Khoi tao mo hinh
model = MedicalClassifierMLP(input_dim=4, hidden_dim=16, output_dim=1)
print("===== KIEN TRUC MANG NƠ-RON (MODEL ARCHITECTURE) =====")
print(model)
""",
            """Khối code thiết kế một mạng MLP hoàn chỉnh bằng `nn.Module`:
- Kế thừa chuẩn `super().__init__()`.
- Tách bạch 2 tầng ẩn có kích thước thu hẹp dần (4 -> 16 -> 8 -> 1) kết hợp phi tuyến `nn.ReLU()`.
- Tầng cuối cùng trả về `logits` thô (chưa qua Sigmoid) — chuẩn thiết kế tối ưu số học của PyTorch."""
        ),
        (
            """# 2. Kiem tra Forward Pass, Debug kich thuoc Tensor va Dem so tham so
# Gia lap mot Mini-Batch gom 8 benh nhan, moi nguoi co 4 dac trung
dummy_batch = torch.randn(8, 4)

# Chay Forward Pass
with torch.no_grad():
    output_logits = model(dummy_batch)

print("===== KIEM THU FORWARD PASS =====")
print(f"Kich thuoc dau vao (Input Batch Shape)  : {dummy_batch.shape}")
print(f"Kich thuoc dau ra (Output Logits Shape): {output_logits.shape}")
print("\\nGia tri 3 logits dau tien:\\n", output_logits[:3])

# Dem tong so luong tham so co the huan luyen (Trainable Parameters)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\\nTong so tham so can huan luyen cua mang: {total_params} parameters")
for name, param in model.named_parameters():
    print(f" - {name:15s}: shape = {str(list(param.shape)):15s} ({param.numel()} params)")
""",
            """Khối code kiểm tra tính đúng đắn của kiến trúc mạng:
- Đưa một batch giả lập $8 \times 4$ qua mạng để xác minh kích thước đầu ra đúng chuẩn $(8, 1)$.
- Sử dụng hàm tổng hợp `numel()` duyệt qua `named_parameters()` để kiểm kê chi tiết số lượng trọng số (Weights) và độ lệch (Biases) của từng tầng."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Xây dựng mạng bằng `nn.Sequential`
Tìm hiểu module đóng gói nhanh `nn.Sequential`. Viết lại mạng nơ-ron trên chỉ bằng một khối `nn.Sequential` duy nhất và kiểm tra xem đầu ra có giống hệt không.

### Bài 2 (Easy): Đổi sang bài toán Phân loại 3 lớp (Multi-class)
Điều chỉnh mô hình để giải quyết bài toán phân loại loài hoa Iris có 3 lớp (`output_dim=3`). Nêu rõ kích thước đầu ra thay đổi như thế nào khi đưa batch 8 mẫu vào mạng.

### Bài 3 (Medium): So sánh hàm kích hoạt ReLU vs LeakyReLU vs GELU
Thay thế `nn.ReLU()` bằng `nn.LeakyReLU(negative_slope=0.01)` và `nn.GELU()`. Giải thích tại sao LeakyReLU giúp giải quyết hiện tượng "Chết nơ-ron ReLU" (Dying ReLU Problem).

### Bài 4 (Medium): Khởi tạo trọng số tùy biến (Custom Weight Initialization)
Mặc định PyTorch sử dụng khởi tạo Kaiming/He hoặc Xavier. Hãy viết một hàm sử dụng `torch.nn.init.kaiming_normal_` để khởi tạo lại trọng số cho tất cả các tầng `nn.Linear` trong mô hình.

### Bài 5 (Challenge): Tự viết Custom Layer trừu tượng
Tạo một lớp kế thừa từ `nn.Module` tên là `LinearWithSkipConnection` kết hợp một tầng `nn.Linear` với một đường tắt thẳng (Residual/Skip connection) cộng trực tiếp đầu vào vào đầu ra: $y = \text{Linear}(x) + x$ (nền tảng của kiến trúc ResNet).
"""
    voai_connection_content = """
Trong phòng thi Olympic AI, lỗi phổ biến nhất khiến thí sinh bế tắc hàng giờ là **Lỗi Lệch Kích Thước Tensor (Dimension Mismatch Error)**:
Ví dụ: Tầng Linear 1 có `out_features=32` nhưng tầng Linear 2 lại khai báo `in_features=16`, khiến code lập tức sập ngay ở forward pass đầu tiên. Thói quen luôn chạy thử một `dummy_batch` qua mạng để in shape của từng tầng là kỹ năng tự bảo vệ không thể thiếu của một đấu thủ chuyên nghiệp.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh thêm hàm `nn.Sigmoid()` ở tầng cuối cùng của mạng nhưng lại sử dụng hàm mất mát `nn.BCEWithLogitsLoss()` trong quá trình huấn luyện:

```python
# CODE CÓ LỖI NHÂN ĐÔI HÀM SIGMOID:
# nn.BCEWithLogitsLoss() da tich hop san ham Sigmoid ben trong de on dinh so hoc (Numerical Stability)!
# Neu ban tu goi Sigmoid o tang cuoi, ham Sigmoid se bi ap dung 2 LAN LIEN TIEP,
# khien xac suat bi bop meo va gradient gan nhu bien mat!
```

**Nhiệm vụ của bạn:**
1. Giải thích sự khác biệt giữa `nn.BCELoss` (yêu cầu đầu vào đã qua Sigmoid) và `nn.BCEWithLogitsLoss` (yêu cầu đầu vào là Logits thô).
2. Nêu lý do tại sao các chuyên gia luôn khuyến nghị sử dụng `BCEWithLogitsLoss` kết hợp với tầng cuối không có activation.
"""
    knowledge_check_content = """
1. Điều gì sẽ xảy ra nếu ta xây dựng một mạng nơ-ron sâu gồm 10 tầng `nn.Linear` liên tiếp nhưng hoàn toàn KHÔNG sử dụng bất kỳ hàm kích hoạt phi tuyến nào?
   - A. Mạng sẽ đạt độ chính xác 100%.
   - B. Toàn bộ mạng 10 tầng sẽ suy biến toán học tương đương với đúng MỘT tầng tuyến tính duy nhất, mất khả năng học các mối quan hệ phi tuyến.
   - C. PyTorch sẽ tự động ngắt kết nối.
   - D. Mô hình sẽ chạy chậm hơn 100 lần.

2. Trong phương thức `forward(self, x)` của một lớp kế thừa `nn.Module`, tham số `x` đại diện cho điều gì?
   - A. Nhãn mục tiêu của bài toán.
   - B. Tensor dữ liệu đầu vào của Mini-batch hiện tại.
   - C. Giá trị đạo hàm của hàm mất mát.
   - D. Số lượng epochs đã hoàn thành.

3. Ưu điểm nổi bật nhất của hàm kích hoạt ReLU ($\max(0, z)$) so với Sigmoid trong các mạng nơ-ron sâu là gì?
   - A. ReLU có giá trị âm lớn.
   - B. ReLU không bị bão hòa ở miền dương (đạo hàm luôn bằng 1 khi $z > 0$), giúp triệt tiêu hiện tượng biến mất gradient (Vanishing Gradient).
   - C. ReLU chỉ chạy được trên CPU.
   - D. ReLU làm giảm kích thước dữ liệu.

4. Tổng số lượng tham số (Weights + Biases) của một tầng `nn.Linear(in_features=10, out_features=5)` là bao nhiêu?
   - A. $10 \times 5 = 50$
   - B. $10 \times 5 + 5 = 55$
   - C. $10 + 5 = 15$
   - D. $10 \times 5 + 10 = 60$

5. Khái niệm "Logits" trong học sâu dùng để chỉ giá trị nào?
   - A. Giá trị hàm mất mát sau khi đạo hàm.
   - B. Giá trị thô chưa chuẩn hóa được xuất ra từ tầng tuyến tính cuối cùng của mạng trước khi đi qua hàm kích hoạt xác suất (như Sigmoid hay Softmax).
   - C. Số lượng nơ-ron bị tắt.
   - D. Giá trị tỷ lệ phần trăm chính xác.
"""
    session_summary_content = """
* **What I learned:**
  - Kiến trúc Perceptron Đa tầng (MLP) và tầm quan trọng sống còn của Hàm kích hoạt phi tuyến.
  - Cách thiết kế mạng nơ-ron chuyên nghiệp bằng cách kế thừa `torch.nn.Module`.
  - Cơ chế đếm và kiểm soát tham số của từng tầng (`parameters()`, `named_parameters()`).
* **What I can now do:**
  - Tự tay lập trình và tùy biến bất kỳ kiến trúc mạng MLP nào cho bài toán hồi quy hoặc phân loại.
  - Debug kích thước Tensor qua từng tầng mạng bằng kỹ thuật `dummy_batch`.
* **What I still need to practice:**
  - Kết nối Dataset, DataLoader, Model và Optimizer vào một **Training Loop chuẩn mực** ở Buổi 4.
"""
    next_session_content = """
Trong **Buổi M3_S04**, chúng ta sẽ ghép nối tất cả các mảnh ghép lại: **Tự Viết Training Loop Chuẩn Mực Bằng PyTorch** — Làm chủ 5 bước thần thánh của một vòng lặp huấn luyện, phân biệt `model.train()` vs `model.eval()`, và theo dõi độ mất mát trên từng batch!
"""
    trace = {
        "items": [
            "Kiến trúc MLP & Non-linear activations (ReLU, Sigmoid)",
            "Kế thừa và thiết kế mạng nơ-ron bằng torch.nn.Module",
            "Forward pass, theo dõi shape tensor & quản lý parameters"
        ],
        "deliverable": "Mã nguồn định nghĩa kiến trúc mạng nơ-ron MLP hoàn chỉnh bằng nn.Module"
    }
    nb = create_notebook(
        module_id="M3",
        module_name="XÂY MODEL AI SÂU HƠN BẰNG PYTORCH",
        session_id="S03",
        topic="Thiết Kế Kiến Trúc Mạng Nơ-ron với torch.nn.Module",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook thiết kế kiến trúc mạng nơ-ron MLP chuẩn mực",
        competition_relevance="Khả năng xây dựng kiến trúc mạng nơ-ron tùy biến cho mọi bài toán Olympic AI.",
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
    save_notebook(nb, "course/M3_PYTORCH/M3_S03.ipynb")


def build_m3_s04():
    objectives = [
        "Nắm vững như in 5 bước cốt tử của một vòng lặp huấn luyện PyTorch (Training Loop).",
        "Hiểu sự khác biệt sống còn giữa hai trạng thái mô hình: `model.train()` và `model.eval()`.",
        "Sử dụng đúng hàm mất mát (`nn.BCEWithLogitsLoss`) và thuật toán tối ưu hóa thích nghi (Adam Optimizer).",
        "Xây dựng vòng lặp kiểm chứng (Validation Loop) không tính gradient (`torch.no_grad()`) và tích lũy metric đánh giá chính xác qua các batch."
    ]
    prerequisites = [
        "Hoàn thành các buổi M3_S01, M3_S02 và M3_S03.",
        "Nắm rõ khái niệm Epoch, Mini-Batch và Learning Rate."
    ]
    why_content = """
Trong nhiều khóa học cấp tốc, học viên thường chỉ được dạy gọi các thư viện bọc sẵn (high-level wrappers) như Keras hay Trainer abstractions. Điều đó khiến bạn hoàn toàn bất lực khi bước vào phòng thi Olympic AI: Bạn không biết chèn thêm code đo lường metric tùy biến ở đâu, không biết cách clip gradient khi nổ dốc, và không hiểu tại sao mô hình của mình dự đoán sai trên tập validation.

Một kỹ sư AI đẳng cấp quốc tế **bắt buộc phải tự tay viết được một Training Loop chuẩn mực từ đầu đến cuối bằng PyTorch thuần**. Tự viết training loop trao cho bạn quyền kiểm soát 100% từng dòng dữ liệu, từng bước cập nhật trọng số và từng con số mất mát của mô hình.
"""
    concept_content = r"""
### 1. 5 bước thần thánh của một Training Step trong PyTorch
Trong mỗi Mini-Batch của tập Train, bạn bắt buộc phải thực hiện tuần tự đúng 5 bước:
1. `optimizer.zero_grad()`: Xóa sạch gradient cũ còn sót lại từ batch trước.
2. `outputs = model(inputs)`: Lan truyền xuôi (Forward pass) để lấy dự đoán.
3. `loss = criterion(outputs, targets)`: Tính toán hàm mất mát.
4. `loss.backward()`: Lan truyền ngược (Backward pass) tự động tính gradient theo Chain Rule.
5. `optimizer.step()`: Cập nhật trọng số theo hướng tối ưu dốc.

### 2. `model.train()` vs `model.eval()`
* `model.train()`: Đưa mô hình vào chế độ huấn luyện. Bật tính năng Dropout (ngẫu nhiên tắt nơ-ron) và cập nhật thống kê động (running mean/var) của BatchNorm.
* `model.eval()`: Đưa mô hình vào chế độ đánh giá/kiểm thử. Tắt hoàn toàn Dropout (sử dụng 100% trọng số) và đóng băng BatchNorm để bảo đảm tính ổn định tuyệt đối của kết quả dự đoán.

### 3. Vòng lặp Kiểm chứng (Validation Loop)
* Luôn bọc toàn bộ validation trong khối `with torch.no_grad():` để:
  - Tiết kiệm bộ nhớ RAM/VRAM (không tốn chỗ lưu đồ thị tính toán).
  - Tăng tốc độ tính toán lên gấp 2-3 lần.
"""
    visual_intuition_content = """
```
Cấu trúc Chu trình Training Loop hoàn chỉnh qua từng Epoch:
┌─────────────────────────────────────────────────────────────────────────────┐
│ FOR EPOCH in range(1, NUM_EPOCHS + 1):                                      │
│   ├── [PHASE 1: TRAINING] -> model.train()                                  │
│   │     For batch_X, batch_y in train_loader:                               │
│   │       1. zero_grad() -> 2. forward() -> 3. loss() -> 4. backward() -> 5. step()
│   │                                                                         │
│   └── [PHASE 2: VALIDATION] -> model.eval() with torch.no_grad():           │
│         For val_X, val_y in val_loader:                                     │
│           forward() -> compute val_loss -> accumulate val_accuracy          │
│                                                                             │
│   └── In báo cáo tiến độ: Epoch N | Train Loss: ... | Val Accuracy: ...     │
└─────────────────────────────────────────────────────────────────────────────┘
```
"""
    code_cells = [
        (
            """import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# 1. Tao tap du lieu mau de huan luyen (400 mau, 5 dac trung, nhan nhi phan 0/1)
torch.manual_seed(42)
np.random.seed(42)

N = 400
X = torch.randn(N, 5)
# Tao nhan muc tieu co quy luat toan hoc + nhieu
true_weights = torch.tensor([[1.5], [-2.0], [0.8], [-1.2], [0.5]])
logits_true = X @ true_weights - 0.3
probs = torch.sigmoid(logits_true)
y = (probs > 0.5).float()

# Chia 80% Train (320 mau) va 20% Val (80 mau)
train_dataset = TensorDataset(X[:320], y[:320])
val_dataset = TensorDataset(X[320:], y[320:])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 2. Xay dung kien truc mang nơ-ron
model = nn.Sequential(
    nn.Linear(5, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1) # Output Logits
)

# Khai bao Ham mat mat va Thuat toan toi uu Adam
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Da chuan bi xong Du lieu, Mo hinh, Loss va Optimizer!")
""",
            """Khối code chuẩn bị bài toán huấn luyện hoàn chỉnh:
- Sử dụng `TensorDataset` và `DataLoader` để quản lý phân chia Train/Validation theo Mini-batch.
- Mạng nơ-ron MLP xuất ra logits.
- `criterion = nn.BCEWithLogitsLoss()`: Tích hợp hàm mất mát Binary Cross-Entropy tối ưu số học.
- `optimizer = optim.Adam(..., lr=0.01)`: Thuật toán tối ưu hóa thích nghi hàng đầu hiện nay."""
        ),
        (
            """# 3. Training & Validation Loop hoan chinh tu so 0
num_epochs = 20

print("===== BAT DAU HUAN LUYEN (TRAINING & VALIDATION LOOP) =====")
for epoch in range(1, num_epochs + 1):
    # --- PHASE 1: TRAINING ---
    model.train() # Bat che do huan luyen
    running_train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()               # 1. Xoa gradient cu
        outputs = model(batch_X)            # 2. Forward pass
        loss = criterion(outputs, batch_y)  # 3. Tinh loss
        loss.backward()                     # 4. Backward pass
        optimizer.step()                    # 5. Cap nhat trong so
        
        running_train_loss += loss.item() * batch_X.size(0)
        
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    
    # --- PHASE 2: VALIDATION ---
    model.eval() # Bat che do danh gia
    running_val_loss = 0.0
    correct_preds = 0
    total_val_samples = 0
    
    with torch.no_grad(): # Khong tinh gradient de tiet kiem bo nho
        for val_X, val_y in val_loader:
            val_outputs = model(val_X)
            val_loss = criterion(val_outputs, val_y)
            running_val_loss += val_loss.item() * val_X.size(0)
            
            # Tinh do chinh xac (Accuracy): bien doi Logits -> Xac suat bang Sigmoid
            predicted_labels = (torch.sigmoid(val_outputs) >= 0.5).float()
            correct_preds += (predicted_labels == val_y).sum().item()
            total_val_samples += val_y.size(0)
            
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_val_acc = (correct_preds / total_val_samples) * 100
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch [{epoch:2d}/{num_epochs:2d}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.1f}%")

print("\\n-> Huan luyen thanh cong! Vong lap chay muot ma khong loi.")
""",
            """Khối code minh họa một Training & Validation Loop hoàn chỉnh đạt chuẩn thi đấu:
- Tách bạch rõ ràng 2 pha `model.train()` và `model.eval()`.
- Cài đặt đầy đủ 5 bước cập nhật trọng số trong pha huấn luyện.
- Tính toán chính xác độ mất mát trung bình theo số lượng mẫu thực tế (`* batch_X.size(0)`).
- Tính chỉ số độ chính xác (Validation Accuracy) bằng cách chuyển logits qua Sigmoid và so khớp với nhãn thực tế."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Theo dõi lịch sử huấn luyện (History Tracking)
Tạo hai danh sách rỗng: `train_losses = []` và `val_losses = []`. Trong mỗi epoch, lưu giá trị mất mát vào danh sách tương ứng. Sau khi huấn luyện kết thúc, dùng Matplotlib vẽ đồ thị đường cong học tập (Learning Curves) so sánh Train Loss vs Val Loss.

### Bài 2 (Easy): Thử nghiệm với thuật toán SGD
Thay thế `optim.Adam` bằng `optim.SGD(model.parameters(), lr=0.05, momentum=0.9)`. So sánh tốc độ hội tụ của SGD có Momentum so với Adam.

### Bài 3 (Medium): Tích lũy Metric F1-Score trên tập Validation
Thay vì chỉ tính Accuracy, hãy import `f1_score` từ Scikit-Learn. Thu thập toàn bộ dự đoán `all_preds` và nhãn thực tế `all_targets` trên tập Validation, sau đó tính Macro-F1 ở cuối mỗi epoch.

### Bài 4 (Medium): Kỹ thuật Cắt tỉa Gradient (Gradient Clipping)
Trong các mạng nơ-ron sâu, gradient có thể bùng nổ quá lớn (Exploding Gradients). Hãy tìm hiểu hàm `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` và chèn nó vào giữa bước `loss.backward()` và `optimizer.step()`.

### Bài 5 (Challenge): Viết Lớp Trainer tổng quát
Đóng gói toàn bộ logic của Training và Validation Loop thành một class `PyTorchTrainer` có phương thức:
`fit(train_loader, val_loader, epochs=10)`
giúp tái sử dụng linh hoạt cho bất kỳ mô hình và bài toán nào trong tương lai.
"""
    voai_connection_content = """
Trong phòng thi Olympic AI, nhiều thí sinh gặp hiện tượng kỳ lạ: Điểm số lúc kiểm thử thì rất cao nhưng khi nộp bài cho hệ thống chấm tự động thì bị lỗi `Time Limit Exceeded` hoặc kết quả nhảy loạn xạ. Nguyên nhân cốt lõi là thí sinh **quên gọi `model.eval()` và `with torch.no_grad():` trong hàm nộp bài**! Khi quên hai lệnh này, Dropout vẫn tiếp tục hoạt động làm kết quả mỗi lần chạy một khác, đồng thời PyTorch tiếp tục lưu đồ thị tính toán làm tràn bộ nhớ và kéo sập máy chủ chấm thi.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh tính toán độ mất mát trung bình của epoch bằng cách cộng dồn trực tiếp `loss.item()` rồi chia cho số lượng batch:

```python
# CODE CÓ LỖI TÍNH SAI TRỌNG SỐ BATCH CUỐI:
# running_loss += loss.item()
# avg_loss = running_loss / len(train_loader)
# Neu batch cuoi cung co it mau hon (vi du chi co 8 mau thay vi 32 mau), 
# viec chia binh quan tren so batch se lam meo mo gia tri mat mat thuc te!
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao công thức chuẩn mực phải là nhân `loss.item()` với kích thước mẫu của batch hiện tại (`batch_X.size(0)`), sau đó mới chia cho tổng số mẫu `len(train_loader.dataset)`.
2. Sửa lại code để bảo đảm độ mất mát luôn được tính trung bình chuẩn xác trên từng mẫu dữ liệu.
"""
    knowledge_check_content = """
1. Thứ tự đúng đắn của 5 thao tác trong một bước huấn luyện (Training Step) của PyTorch là gì?
   - A. `backward()` -> `step()` -> `zero_grad()` -> `forward()` -> `loss()`.
   - B. `zero_grad()` -> `forward()` -> `loss()` -> `backward()` -> `step()`.
   - C. `forward()` -> `step()` -> `loss()` -> `backward()` -> `zero_grad()`.
   - D. `step()` -> `zero_grad()` -> `forward()` -> `loss()` -> `backward()`.

2. Lệnh `model.eval()` có tác dụng gì trong quá trình kiểm thử mô hình?
   - A. Xóa bỏ hoàn toàn mô hình khỏi bộ nhớ.
   - B. Chuyển mô hình sang chế độ đánh giá: Tắt cơ chế Dropout và đóng băng thống kê BatchNorm để cho ra dự đoán ổn định, có tính tất định.
   - C. Tự động tính toán điểm F1-score.
   - D. Tăng số lượng tham số của mạng.

3. Tại sao chúng ta bắt buộc phải gọi `loss.backward()` TRƯỚC KHI gọi `optimizer.step()`?
   - A. Vì `optimizer.step()` cần đọc các giá trị đạo hàm đã được tính và lưu trữ trong thuộc tính `.grad` của các tham số để biết hướng cập nhật.
   - B. Để xóa sạch dữ liệu của batch hiện tại.
   - C. Vì nếu gọi ngược lại sẽ gây lỗi cú pháp Python.
   - D. Vì hàm backward làm giảm kích thước ma trận.

4. Đoạn code `loss.item()` khác gì so với biến `loss` thông thường?
   - A. `loss.item()` trả về một số thực Python thông thường và ngắt kết nối khỏi đồ thị tính toán, giúp giải phóng bộ nhớ.
   - B. `loss.item()` là một tensor đa chiều.
   - C. `loss.item()` chạy trên card GPU.
   - D. Không có sự khác biệt nào.

5. Việc quên gọi lệnh `optimizer.zero_grad()` ở đầu mỗi batch sẽ dẫn đến hậu quả gì?
   - A. Mô hình sẽ bị dừng chạy ngay lập tức.
   - B. Gradient của batch hiện tại sẽ bị cộng dồn với gradient của tất cả các batch trước đó, khiến bước cập nhật trọng số trở nên khổng lồ và sai lệch hướng tối ưu.
   - C. Tập dữ liệu bị xóa mất nhãn.
   - D. Tốc độ nạp dữ liệu bị chậm lại.
"""
    session_summary_content = """
* **What I learned:**
  - Nằm lòng 5 bước thần thánh của một chu trình huấn luyện PyTorch.
  - Phân định rạch ròi vai trò của `model.train()` và `model.eval()`.
  - Cách thiết lập vòng lặp kiểm chứng (Validation Loop) an toàn với `torch.no_grad()`.
  - Theo dõi và tính toán độ mất mát cũng như độ chính xác chuẩn xác qua từng Epoch.
* **What I can now do:**
  - Tự tay viết trọn vẹn một Training Loop từ đầu đến cuối mà không cần dựa dẫm vào các thư viện bọc ngoài.
  - Tự tin huấn luyện mạng nơ-ron sâu trên bất kỳ tập dữ liệu nào.
* **What I still need to practice:**
  - Kiểm soát hiện tượng quá khớp (Overfitting), lưu vết mô hình (Checkpointing) và xuất file nộp bài ở Buổi 5.
"""
    next_session_content = """
Buổi tiếp theo (**M3_S05**) là buổi tổng kết đỉnh cao của Module 3: **Chống Overfitting, Learning Curves, Checkpoint & Save/Load Model**! Bạn sẽ hoàn thiện notebook PyTorch có training loop, xuất file Checkpoint `.pt` và chính thức nhận Chứng chỉ **Certificate of Deep Learning Practice**!
"""
    trace = {
        "items": [
            "Tự viết Training loop hoàn chỉnh (Forward, Loss, Backward, Optimizer)",
            "Chế độ model.train() vs model.eval() & no_grad validation",
            "Đo lường loss & metric tích lũy qua từng Mini-batch"
        ],
        "deliverable": "Mã nguồn Training Loop & Validation Loop chuẩn mực bằng PyTorch thuần"
    }
    nb = create_notebook(
        module_id="M3",
        module_name="XÂY MODEL AI SÂU HƠN BẰNG PYTORCH",
        session_id="S04",
        topic="Tự Viết Training Loop Chuẩn Mực Bằng PyTorch",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook huấn luyện mạng nơ-ron với Training Loop và Validation Loop hoàn chỉnh",
        competition_relevance="Làm chủ 100% luồng thực thi huấn luyện mô hình sâu trong phòng thi Olympic AI.",
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
    save_notebook(nb, "course/M3_PYTORCH/M3_S04.ipynb")


def build_m3_s05():
    objectives = [
        "Chẩn đoán chính xác hiện tượng Quá khớp (Overfitting) thông qua Đồ thị đường cong học tập (Train vs Validation Loss Curves).",
        "Áp dụng thành thạo hai kỹ thuật điều quy (Regularization) kinh điển: Suy giảm trọng số (Weight Decay / L2) và Dropout (`nn.Dropout`).",
        "Làm chủ kỹ thuật Dừng sớm (Early Stopping) để tự động bảo tồn trạng thái mô hình tốt nhất.",
        "Thực hành Lưu trữ (Save) và Tái nạp (Load) Model Checkpoint (`state_dict`) đạt chuẩn công nghiệp và hoàn thành sản phẩm Module 3."
    ]
    prerequisites = [
        "Hoàn thành Buổi M3_S04 (Đã viết thành thạo Training & Validation Loop).",
        "Hiểu khái niệm Overfitting từ Module M2."
    ]
    why_content = """
Trong các kỳ thi AI kéo dài nhiều giờ, tài nguyên tính toán luôn có hạn. Nếu bạn để mô hình chạy 200 epochs mà không kiểm soát, có thể sau epoch thứ 20 mô hình đã bắt đầu "học vẹt" (Overfitting), khiến hiệu năng trên tập kiểm tra thực tế tụt dốc không phanh. Chưa kể, nếu giữa chừng xảy ra sự cố sập nguồn, mất kết nối Colab hoặc hết thời gian thi, toàn bộ trọng số học được sẽ biến mất vĩnh viễn!

Làm chủ kỹ thuật **Điều quy hóa (Regularization)**, đọc **Learning Curves**, và cơ chế **Lưu vết Checkpoint (Save/Load State Dict)** chính là áo giáp bảo vệ thành quả của bạn: Nó đảm bảo bạn luôn giữ lại được phiên bản mô hình thông minh nhất và sẵn sàng phục hồi trạng thái chỉ trong 1 giây để nộp bài thi!
"""
    concept_content = r"""
### 1. Đọc vị Đường cong học tập (Learning Curves)
* **Underfitting (Chưa khớp):** Cả Train Loss và Val Loss đều cao và không chịu giảm. Nguyên nhân: Mô hình quá yếu hoặc chưa học đủ lâu.
* **Good Fit (Hội tụ tối ưu):** Cả Train Loss và Val Loss cùng giảm đều và tiệm cận một giá trị thấp ổn định.
* **Overfitting (Quá khớp):** Train Loss tiếp tục giảm sâu, nhưng **Val Loss bắt đầu quay đầu tăng ngược trở lại**! Đây là thời điểm vàng cần dừng huấn luyện ngay lập tức.

### 2. Các kỹ thuật chống Overfitting trong PyTorch
1. **Weight Decay (L2 Regularization):** Phạt các trọng số có độ lớn quá cao, ép mô hình chọn các nghiệm êm dịu hơn:
   `optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)`
2. **Dropout (`nn.Dropout(p=0.3)`):** Trong mỗi lượt huấn luyện, ngẫu nhiên "tắt" (gán bằng 0) một tỷ lệ $p$ các nơ-ron, buộc mạng không thể phụ thuộc vào một cá nhân nơ-ron nào mà phải học các đặc trưng tập thể mạnh mẽ.

### 3. Cơ chế Lưu và Nạp Model Checkpoint chuẩn (`state_dict`)
Trong PyTorch, chuẩn mực vàng là **chỉ lưu bảng tham số `state_dict`** (chứa tensor trọng số và bias), không lưu toàn bộ đối tượng class để tránh lỗi không tương thích môi trường:
* **Lưu Checkpoint:**
  ```python
  torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'val_loss': best_val_loss,
  }, 'best_checkpoint.pt')
  ```
* **Nạp lại mô hình:**
  ```python
  model = MyModel()
  checkpoint = torch.load('best_checkpoint.pt')
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval() # Bắt buộc chuyển sang eval sau khi load!
  ```
"""
    visual_intuition_content = """
```
Trực quan hóa Overfitting & Early Stopping:
  Loss
   ▲
   │        \                     / ──> Validation Loss (Bắt đầu tăng: Overfitting!)
   │         \     VÙNG TỐI ƯU   /
   │          \         │       /
   │           \        ▼      /
   │            \───────★─────'  <─── BEST CHECKPOINT (Lưu tại đây!)
   │             \
   │              \───────────────────> Train Loss (Tiếp tục giảm sâu: Học vẹt)
   └───────────────────────────────────► Epochs
```
"""
    code_cells = [
        (
            """import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# 1. Thiet ke mang MLP co Regularization manh me: Dropout
class RegularizedMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, dropout_rate=0.3):
        super(RegularizedMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), # Ngau nhien tat 30% no-ron khi train de chong Overfitting
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# Khoi tao mo hinh
model = RegularizedMLP(input_dim=10, hidden_dim=32, dropout_rate=0.3)
print("===== MO HINH REGULARIZED MLP CO DROPOUT =====")
print(model)
""",
            """Khối code tích hợp kỹ thuật điều quy Dropout trực tiếp vào kiến trúc:
- `nn.Dropout(p=0.3)` được chèn xen kẽ giữa các tầng ẩn.
- Giúp phá vỡ sự đồng thích nghi (Co-adaptation) của các nơ-ron, giảm thiểu triệt để nguy cơ quá khớp."""
        ),
        (
            """# 2. Huan luyen, Luu Checkpoint tai diem Val Loss tot nhat va Tai nap lai
torch.manual_seed(42)
X_dummy = torch.randn(200, 10)
y_dummy = torch.randint(0, 2, (200, 1)).float()

criterion = nn.BCEWithLogitsLoss()
# Tich hop L2 Regularization (weight_decay=1e-4) trong Adam
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

best_val_loss = float('inf')
checkpoint_path = 'best_model_checkpoint.pt'

# Gia lap qua trinh huan luyen va luu checkpoint
epochs = 15
train_history = []
val_history = []

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(X_dummy[:150])
    loss = criterion(out, y_dummy[:150])
    loss.backward()
    optimizer.step()
    
    # Validation step
    model.eval()
    with torch.no_grad():
        val_out = model(X_dummy[150:])
        v_loss = criterion(val_out, y_dummy[150:]).item()
        
    train_history.append(loss.item())
    val_history.append(v_loss)
    
    # Kiem tra va Luu Checkpoint neu dat Val Loss ky luc moi
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_val_loss
        }, checkpoint_path)

print(f"-> Da luu Checkpoint tot nhat tai Val Loss = {best_val_loss:.4f} vao file: {checkpoint_path}")

# 3. Kiem thu Tai nap lai (Load Checkpoint) de kiem tra tinh toan ven
new_model = RegularizedMLP(input_dim=10, hidden_dim=32)
loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
new_model.eval() # Chuyen sang che do danh gia

print("-> Tai nap Checkpoint thanh cong! San sang suy dien (Inference Ready).")
print(f"Thong tin Checkpoint luu o Epoch: {loaded_checkpoint['epoch']}")
""",
            """Khối code minh họa chu trình chuẩn mực về Checkpointing trong Deep Learning:
- Theo dõi `val_loss` qua từng epoch và tự động ghi đè file `best_model_checkpoint.pt` khi đạt mốc kỷ lục mới.
- Lưu trữ đầy đủ siêu dữ liệu: `model_state_dict`, `optimizer_state_dict`, số epoch.
- Tái nạp mô hình vào một đối tượng mới hoàn toàn bằng `load_state_dict` và chuyển ngay sang `eval()`."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Vẽ biểu đồ Learning Curves
Sử dụng dữ liệu `train_history` và `val_history` ở trên, dùng Matplotlib vẽ đồ thị đường cong học tập với 2 đường: Train Loss (màu xanh) và Val Loss (màu đỏ). Đánh dấu điểm Epoch có Val Loss nhỏ nhất bằng một ngôi sao trên biểu đồ.

### Bài 2 (Easy): Thử nghiệm thay đổi tỷ lệ Dropout
Thay đổi `dropout_rate` thành $0.7$ (quá cao) và $0.05$ (quá thấp). Quan sát sự chênh lệch giữa tốc độ học của tập Train và tập Validation.

### Bài 3 (Medium): Xây dựng Lớp EarlyStopping tùy biến
Viết một class `EarlyStopping(patience=5, min_delta=0.001)` có phương thức theo dõi `step(val_loss)`. Nếu sau 5 epoch liên tiếp mà `val_loss` không giảm thêm ít nhất `0.001`, tự động kích hoạt cờ dừng `should_stop = True` để ngắt vòng lặp huấn luyện.

### Bài 4 (Medium): Nạp Checkpoint để tiếp tục huấn luyện (Resume Training)
Giả sử máy tính bị mất điện ở epoch thứ 15. Hãy viết đoạn code nạp lại cả `model_state_dict` VÀ `optimizer_state_dict` từ checkpoint đã lưu, sau đó tiếp tục huấn luyện từ epoch 16 đến epoch 30 một cách liên tục không đứt đoạn.

### Bài 5 (Challenge): Xuất dự đoán ra file `submission.csv` từ Checkpoint
Viết một hàm `generate_submission(model, test_loader, output_file='test_predictions.csv')` nạp trọng số từ file checkpoint tốt nhất, chạy dự đoán trên tập dữ liệu kiểm tra, và xuất ra file CSV gồm 2 cột: `ID` và `Probability_Prediction` chuẩn quy cách phòng thi.
"""
    voai_connection_content = """
Trong thi đấu Olympic AI, quy chế thi thường yêu cầu: *"Thí sinh phải nộp cả file Notebook VÀ file trọng số mô hình đã huấn luyện (.pt hoặc .pth)"*. Nếu notebook của bạn huấn luyện mất 3 tiếng nhưng bạn không lưu Checkpoint, khi ban giám khảo chạy lại để chấm điểm và gặp sự cố timeout, bài thi của bạn sẽ bị đánh rớt. Luôn lưu lại `best_checkpoint.pt` sau mỗi epoch là thói quen sống còn của một tuyển thủ chuyên nghiệp.
"""
    debugging_content = """
**Tình huống thực tế:** Một bạn học viên nạp lại mô hình từ file checkpoint nhưng quên gọi lệnh `model.eval()`, khiến kết quả dự đoán trên cùng một mẫu thử mỗi lần chạy lại cho ra một con số khác nhau:

```python
# CODE CÓ LỖI BẤT ỔN ĐỊNH DO QUÊN MODEL.EVAL():
# Khi quen model.eval(), tang nn.Dropout van tiep tuc ngau nhien tat 30% no-ron
# moi lan du doan, lam ket qua submission bi nhieu va giam diem dang tiec!
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao Dropout chỉ được phép hoạt động trong lúc huấn luyện và bắt buộc phải tắt khi đánh giá.
2. Viết lại đoạn code nạp checkpoint bảo đảm luôn có lệnh `model.eval()` ngay lập tức.
"""
    knowledge_check_content = """
1. Khi quan sát đồ thị Learning Curves, dấu hiệu rõ ràng nhất cho thấy mô hình đang bị Quá khớp (Overfitting) là gì?
   - A. Cả Train Loss và Val Loss đều không thay đổi qua các epoch.
   - B. Train Loss tiếp tục giảm sâu, nhưng Validation Loss bắt đầu tăng ngược trở lại.
   - C. Train Loss tăng cao hơn Validation Loss.
   - D. Tốc độ quạt tản nhiệt của máy tính kêu to hơn.

2. Tầng `nn.Dropout(p=0.4)` trong PyTorch thực hiện thao tác gì trong quá trình huấn luyện?
   - A. Xóa bỏ vĩnh viễn 40% số nơ-ron của tầng đó.
   - B. Trong mỗi Mini-batch, ngẫu nhiên đặt giá trị đầu ra của 40% số nơ-ron thành số 0 để tránh việc các nơ-ron quá phụ thuộc vào nhau.
   - C. Giảm tốc độ học xuống 40%.
   - D. Giảm kích thước batch size đi 40%.

3. Phương thức chuẩn mực và an toàn nhất được khuyến nghị bởi đội ngũ phát triển PyTorch để lưu trạng thái mô hình là gì?
   - A. `torch.save(model, 'model.pt')` (Lưu toàn bộ đối tượng class).
   - B. `torch.save(model.state_dict(), 'model_weights.pt')` (Chỉ lưu từ điển trọng số và tham số).
   - C. Xuất ra file Excel.
   - D. Lưu code Python ra file `.txt`.

4. Cơ chế "Early Stopping" (Dừng sớm) trong huấn luyện mạng nơ-ron có mục đích chính là gì?
   - A. Để máy tính đỡ tốn điện.
   - B. Tự động ngắt vòng lặp huấn luyện khi mô hình không còn cải thiện hiệu năng trên tập Validation sau một số epoch quy định (Patience), tránh lãng phí thời gian và ngăn chặn Overfitting.
   - C. Tự động xóa các dữ liệu ngoại lai.
   - D. Tăng độ sâu của mạng nơ-ron.

5. Sản phẩm đầu ra chính thức mà học viên bắt buộc phải hoàn thành khi kết thúc Module M3 là gì?
   - A. Một bài thuyết trình PowerPoint về lịch sử AI.
   - B. Một Notebook PyTorch hoàn chỉnh có Training Loop tự viết và file Model Checkpoint (.pt) đã được lưu thành công.
   - C. Một website thương mại điện tử.
   - D. Một bài viết blog dài 2000 từ.
"""
    session_summary_content = """
* **What I learned:**
  - Kỹ năng chẩn đoán Overfitting qua đường cong học tập Train/Val Loss.
  - Cách ứng dụng Regularization: Weight Decay (L2) và Dropout (`nn.Dropout`).
  - Cơ chế Early Stopping và kỹ thuật lưu trữ/tái nạp Checkpoint chuẩn mực (`state_dict`).
* **What I can now do:**
  - Tự tin kiểm soát quá trình huấn luyện của mạng nơ-ron sâu, không để mô hình bị học vẹt.
  - Lưu trữ và phục hồi trạng thái mô hình an toàn, sẵn sàng xuất file nộp bài cho ban tổ chức.
* **What I still need to practice:**
  - Áp dụng toàn bộ kiến thức PyTorch để bước vào **Module M4: AI nhìn và hiểu dữ liệu (Computer Vision & NLP)**!
"""
    next_session_content = """
Xin chúc mừng bạn đã chính thức tốt nghiệp Module M3 và nhận Chứng chỉ **Certificate of Deep Learning Practice**!
Trong buổi tiếp theo (**M4_S01**), chúng ta sẽ bước vào Module thực chiến lớn nhất: **Module M4: AI NHÌN VÀ HIỂU DỮ LIỆU** — Bắt đầu chặng đường đầu tiên: Thị giác máy tính (Computer Vision) với chủ đề Biểu diễn ảnh số, Tensors hình ảnh và các kênh màu Grayscale/RGB!
"""
    module_assessment = """
---

## 🏆 MODULE 3 ASSESSMENT & CERTIFICATION

### Phần A: Đánh giá Lý thuyết & Tư duy Deep Learning (Theory & DL Architecture)
- Nắm vững cơ chế Autograd, Computational Graph và 5 bước của Training Step.
- Phân biệt sâu sắc vai trò của Activation phi tuyến (ReLU), `model.train()` vs `model.eval()`.
- Giải thích bản chất của Dropout, Weight Decay và chiến thuật Checkpointing an toàn.

### Phần B: Sản phẩm Thực hành Bắt buộc (Practical Deliverable)
- **Notebook PyTorch có Training Loop tự viết:** Huấn luyện thành công mạng nơ-ron MLP trên dữ liệu thực tế, có tách bạch pha Train và pha Validation không dùng thư viện bọc ngoài.
- **Model Checkpoint (.pt):** Xuất bản thành công file lưu trữ trọng số `state_dict` tốt nhất kèm code kiểm thử tái nạp (load) không lỗi.

### Tiêu chí Đạt chuẩn (Rubric):
| Tiêu chí | Trọng số | Yêu cầu đạt |
| :--- | :--- | :--- |
| **Data Pipeline (Dataset & Loader)** | 25% | Xây dựng Custom Dataset và DataLoader batching chuẩn |
| **Kiến trúc mạng (nn.Module)** | 25% | Thiết kế mạng MLP có Activation và Dropout hợp lý |
| **Training & Validation Loop** | 30% | Viết đủ 5 bước cập nhật, theo dõi Loss và Accuracy |
| **Checkpointing & Regularization**| 20% | Lưu và nạp thành công `best_model.pt`, có Early Stopping |

**Chứng chỉ hoàn thành:**
🏅 **Certificate of Deep Learning Practice** — Đủ điều kiện tiến vào Module M4 (Computer Vision & NLP Applied AI).
"""
    trace = {
        "items": [
            "Chẩn đoán Overfitting qua Learning Curves",
            "Kỹ thuật Regularization: Weight decay & Dropout",
            "Tự viết Training loop & Checkpoint (Save/Load state_dict)",
            "Nhận chứng chỉ Certificate of Deep Learning Practice"
        ],
        "deliverable": "Notebook PyTorch có training loop + Checkpoint (.pt) hoàn chỉnh"
    }
    nb = create_notebook(
        module_id="M3",
        module_name="XÂY MODEL AI SÂU HƠN BẰNG PYTORCH",
        session_id="S05",
        topic="Chống Overfitting, Learning Curves & Model Checkpoint",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook PyTorch có training loop + Checkpoint (.pt)",
        competition_relevance="Bảo tồn mô hình tối ưu, ngăn ngừa rớt hạng do học vẹt trong phòng thi.",
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
    save_notebook(nb, "course/M3_PYTORCH/M3_S05.ipynb")

if __name__ == "__main__":
    build_m3_s01()
    build_m3_s02()
    build_m3_s03()
    build_m3_s04()
    build_m3_s05()
    print("M3 build complete!")
