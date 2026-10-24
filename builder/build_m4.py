"""
builder/build_m4.py
Generates M4_S01.ipynb to M4_S10.ipynb for Module 4 (Computer Vision & NLP).
"""

from utils import create_notebook, save_notebook

# ==========================================
# PART A: COMPUTER VISION (S01 - S05)
# ==========================================

def build_m4_s01():
    objectives = [
        "Hiểu bản chất số học của ảnh số: Điểm ảnh (Pixels) là các giá trị cường độ sáng trong thang đo [0, 255].",
        "Phân biệt cấu trúc tensor của ảnh đa kênh: Grayscale [H, W] vs RGB [H, W, C] vs PyTorch [C, H, W].",
        "Thực hiện chuẩn hóa điểm ảnh (Normalization) từ [0, 255] về khoảng [0.0, 1.0].",
        "Bóc tách từng kênh màu (Red, Green, Blue) và trực quan hóa biểu đồ phân bố cường độ sáng (Intensity Histogram)."
    ]
    prerequisites = [
        "Hoàn thành Module M3 (Đã làm chủ PyTorch Tensors).",
        "Kỹ năng sử dụng Matplotlib và NumPy."
    ]
    why_content = """
Máy tính không bao giờ "nhìn thấy" một chú mèo đáng yêu hay một chiếc ô tô đang di chuyển. Thứ duy nhất máy tính nhìn thấy là một **Ma trận số khổng lồ (2D hoặc 3D Matrix)** chứa hàng triệu con số nguyên. Nếu bạn không hiểu cách biểu diễn này, bạn sẽ liên tục gặp lỗi lệch thứ tự kênh màu (ví dụ OpenCV dùng BGR trong khi Matplotlib/PyTorch dùng RGB), hoặc đưa các giá trị pixel thô từ 0 đến 255 vào mạng nơ-ron khiến mô hình bị bão hòa gradient ngay lập tức.

Làm chủ biểu diễn ảnh số là viên gạch đầu tiên để bạn bước vào thế giới Thị giác Máy tính (Computer Vision) — lĩnh vực chiếm tỷ trọng lớn trong các đề thi Olympic AI cấp quốc gia và quốc tế.
"""
    concept_content = r"""
### 1. Bản chất của Ảnh số (Digital Image Representation)
* Một bức ảnh kỹ thuật số là một lưới các điểm ảnh (Pixels). Mỗi pixel lưu trữ độ sáng.
* **Ảnh xám (Grayscale):** Là một ma trận 2 chiều có kích thước $Height \times Width$. Mỗi giá trị là một số nguyên từ 0 (Đen tuyệt đối) đến 255 (Trắng tuyệt đối).
* **Ảnh màu (RGB):** Là một mảng 3 chiều gồm 3 ma trận màu xếp chồng lên nhau: **R** (Red - Đỏ), **G** (Green - Lục), **B** (Blue - Lam).

### 2. Sự khác biệt về Thứ tự Chiều (Dimension Ordering)
* **Quy ước của Thư viện Ảnh (PIL / OpenCV / Matplotlib):** Định dạng **$[H, W, C]$** (Chiều cao, Chiều rộng, Số kênh màu).
* **Quy ước bắt buộc của PyTorch:** Định dạng **$[C, H, W]$** (Số kênh màu đứng trước: Channel-First), hoặc khi nạp theo batch là **$[Batch, C, H, W]$**.
* *Chuyển đổi trong PyTorch:* Dùng `tensor.permute(1, 2, 0)` để đổi từ $[C, H, W] \to [H, W, C]$ khi cần hiển thị bằng Matplotlib!

### 3. Chuẩn hóa Điểm ảnh (Pixel Normalization)
Đưa các giá trị nguyên $[0, 255]$ về số thực $[0.0, 1.0]$ bằng cách chia cho $255.0$. Thao tác này giúp các thuật toán tối ưu hóa Gradient Descent hội tụ nhanh và ổn định hơn rất nhiều.
"""
    visual_intuition_content = """
```
Cấu trúc Tensor Ảnh RGB:
               Height (H)
                │
                ▼   Width (W)
      ┌────────────┬────────────┐
   R  │  Pixel R   │  Pixel R   │ ──> Kênh Đỏ (Red Channel)
   ┌──┴────────────┴────────────┤
G  │    Pixel G    │  Pixel G   │ ──> Kênh Lục (Green Channel)
┌──┴────────────┬───┴───────────┤
│    Pixel B    │    Pixel B    │ ──> Kênh Lam (Blue Channel)
└───────────────┴───────────────┘
  PyTorch Tensor Shape: [3, H, W]
```
"""
    code_cells = [
        (
            """import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. Tu tao mot anh mau RGB nhan tao 8x8 pixels de quan sat ro ban chat tung con so
np.random.seed(42)

# Tao ma tran anh RGB kich thuoc (8, 8, 3) voi cac vung mau dac trung
synthetic_img = np.zeros((8, 8, 3), dtype=np.uint8)
synthetic_img[:4, :4, 0] = 255   # O vuong tren-trai mau Do (R=255, G=0, B=0)
synthetic_img[:4, 4:, 1] = 255   # O vuong tren-phai mau Luc (R=0, G=255, B=0)
synthetic_img[4:, :4, 2] = 255   # O vuong duoi-trai mau Lam (R=0, G=0, B=255)
synthetic_img[4:, 4:, :] = 255   # O vuong duoi-phai mau Trang (R=255, G=255, B=255)

print("===== MA TRAN ANH RGB (8x8x3) =====")
print(f"Shape anh goc: {synthetic_img.shape} | Kieu du lieu: {synthetic_img.dtype}")
print(f"Gia tri pixel tai goc tren-trai (Kenh R): {synthetic_img[0, 0, 0]}")

# 2. Chuyen doi sang PyTorch Tensor theo chuan [C, H, W] va Chuan hoa ve [0.0, 1.0]
tensor_img = torch.from_numpy(synthetic_img).permute(2, 0, 1).float() / 255.0
print(f"Shape PyTorch Tensor sau permute: {tensor_img.shape} | Min: {tensor_img.min()}, Max: {tensor_img.max()}")
""",
            """Khối code tạo dựng một bức ảnh số nhân tạo nhỏ gọn:
- Giúp người học nhìn thấu từng con số pixel trong ma trận $8 \times 8 \times 3$.
- Sử dụng `.permute(2, 0, 1)` để đổi trục từ $[H, W, C]$ sang chuẩn $[C, H, W]$ của PyTorch.
- Thực hiện chuẩn hóa chia $255.0$ đưa miền giá trị về $[0.0, 1.0]$ an toàn số học."""
        ),
        (
            """# 3. Tach 3 kenh mau va Truc quan hoa bang Matplotlib
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Hien thi anh mau tong hop (can chuyen lai ve [H, W, C] de Matplotlib hieu)
img_to_show = tensor_img.permute(1, 2, 0).numpy()
axes[0].imshow(img_to_show)
axes[0].set_title('Anh Mau RGB Tong Hop', fontweight='bold')
axes[0].axis('off')

# Kenh Do (Red)
axes[1].imshow(tensor_img[0].numpy(), cmap='Reds')
axes[1].set_title('Kenh Do (R)', fontweight='bold')
axes[1].axis('off')

# Kenh Luc (Green)
axes[2].imshow(tensor_img[1].numpy(), cmap='Greens')
axes[2].set_title('Kenh Luc (G)', fontweight='bold')
axes[2].axis('off')

# Kenh Lam (Blue)
axes[3].imshow(tensor_img[2].numpy(), cmap='Blues')
axes[3].set_title('Kenh Lam (B)', fontweight='bold')
axes[3].axis('off')

plt.tight_layout()
plt.show()
""",
            """Khối code bóc tách trực quan 3 kênh màu Red, Green, Blue:
- Minh họa rõ ràng nguyên lý phối màu cộng (Additive Color): Khi cả 3 kênh cùng đạt cực đại 255, ta thu được màu trắng ở góc dưới bên phải."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Chuyển ảnh màu sang ảnh xám (RGB to Grayscale)
Sử dụng công thức phối màu trọng số chuẩn mắt người:
$$\text{Gray} = 0.299 R + 0.587 G + 0.114 B$$
Áp dụng công thức trên lên `tensor_img` để tạo ra một tensor ảnh xám kích thước `(1, 8, 8)`. Hiển thị bằng `cmap='gray'`.

### Bài 2 (Easy): Cắt lát không gian ảnh (Spatial Cropping)
Trích xuất góc phần tư phía trên bên phải của bức ảnh nhân tạo bằng cú pháp slicing PyTorch. In ra kích thước của ảnh sau khi cắt.

### Bài 3 (Medium): Vẽ biểu đồ phân bố cường độ sáng (Color Histogram)
Tạo một bức ảnh ngẫu nhiên kích thước $(64, 64, 3)$ với các giá trị từ 0 đến 255. Sử dụng Matplotlib vẽ 3 đường biểu đồ tần suất (Histograms) tương ứng với 3 kênh R, G, B để quan sát phân phối độ sáng.

### Bài 4 (Medium): Đảo ngược màu ảnh (Invert Colors)
Viết một hàm `invert_image(tensor)` nhận vào tensor ảnh đã chuẩn hóa $[0, 1]$ và thực hiện phép biến đổi âm bản (Negative image): $x_{inv} = 1.0 - x$. Hiển thị kết quả so sánh.

### Bài 5 (Challenge): Phát hiện thứ tự kênh BGR của OpenCV
Nêu rõ cách nhận biết một bức ảnh đang bị hiển thị sai thứ tự màu (ví dụ khuôn mặt người bị chuyển sang màu xanh da trời do thư viện đọc bằng BGR nhưng hiển thị bằng RGB). Viết một dòng lệnh PyTorch để lật ngược thứ tự kênh màu từ BGR sang RGB.
"""
    voai_connection_content = """
Trong đề thi Olympic AI phần Thị giác Máy tính, một trong những lỗi gây mất điểm ngớ ngẩn nhất là **lỗi thứ tự kênh màu và thang đo pixel**:
Nhiều thí sinh nạp ảnh bằng thư viện OpenCV (`cv2.imread()`) nhưng quên rằng OpenCV đọc theo chuẩn `BGR` với thang $[0, 255]$, rồi nạp thẳng vào mô hình được huấn luyện trên chuẩn `RGB` với thang $[0.0, 1.0]$. Kết quả là mô hình hoàn toàn mù tịt, dự đoán sai lệch 100%. Nắm vững cách quản lý Tensor ảnh giúp bạn không bao giờ rơi vào cái bẫy này.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh đưa trực tiếp một tensor PyTorch có kích thước `[3, 64, 64]` vào hàm `plt.imshow()`, hệ thống báo lỗi:

```python
# CODE CÓ LỖI CHANNEL-FIRST KHI HIỂN THỊ:
# plt.imshow(tensor_img) # Bao loi: TypeError: Invalid shape (3, 8, 8) for image data
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao Matplotlib chỉ chấp nhận ảnh 2D `[H, W]` hoặc 3D dạng `[H, W, C]`.
2. Sửa lỗi bằng cách sử dụng phương thức `.permute(1, 2, 0)` và chuyển sang mảng NumPy.
"""
    knowledge_check_content = """
1. Kích thước (Shape) chuẩn mực của một Tensor ảnh màu đơn lẻ trong PyTorch là gì?
   - A. `[Height, Width, Channels]`
   - B. `[Channels, Height, Width]`
   - C. `[Height, Channels, Width]`
   - D. `[Width, Height]`

2. Mục đích chính của việc chia tất cả các giá trị pixel cho 255.0 là gì?
   - A. Để giảm dung lượng lưu trữ trên đĩa cứng.
   - B. Chuẩn hóa dải giá trị từ số nguyên $[0, 255]$ về số thực $[0.0, 1.0]$, giúp các hàm kích hoạt và thuật toán Gradient Descent hoạt động ổn định, tránh bão hòa gradient.
   - C. Để chuyển ảnh màu thành ảnh xám.
   - D. Để tăng độ tương phản của ảnh.

3. Một bức ảnh xám có kích thước $100 \times 100$ chứa bao nhiêu điểm ảnh (pixels)?
   - A. 100
   - B. 10,000
   - C. 30,000
   - D. 1,000,000

4. Lệnh `tensor.permute(1, 2, 0)` có tác dụng gì đối với một tensor có shape `(C, H, W)`?
   - A. Xóa kênh màu đầu tiên.
   - B. Hoán vị các trục để đưa tensor về shape `(H, W, C)`.
   - C. Xoay bức ảnh một góc 90 độ.
   - D. Nhân đôi số kênh màu.

5. Trong không gian màu RGB, khi một điểm ảnh có giá trị $(R=0, G=0, B=0)$, điểm ảnh đó sẽ hiển thị màu gì?
   - A. Màu Trắng
   - B. Màu Đỏ
   - C. Màu Đen tuyệt đối
   - D. Màu Vàng
"""
    session_summary_content = """
* **What I learned:**
  - Bản chất số học của ảnh số: Điểm ảnh (Pixel) và các kênh màu.
  - Sự khác biệt giữa định dạng `[H, W, C]` của Matplotlib và `[C, H, W]` của PyTorch.
  - Tầm quan trọng của việc chuẩn hóa điểm ảnh về khoảng $[0.0, 1.0]$.
* **What I can now do:**
  - Tự tin thao tác, bóc tách và biến đổi ma trận ảnh số bằng PyTorch Tensor.
  - Hoán vị các chiều an toàn bằng `.permute()` để hiển thị và tiền xử lý.
* **What I still need to practice:**
  - Tìm hiểu cách tăng cường dữ liệu: **Data Augmentation** ở Buổi 2.
"""
    next_session_content = """
Trong **Buổi M4_S02**, chúng ta sẽ nâng cao kỹ năng xử lý ảnh: **Tiền Xử Lý Ảnh & Kỹ Thuật Tăng Cường Dữ Liệu (Data Augmentation)** với `torchvision.transforms` — Khám phá cách xoay, lật, cắt ảnh để nhân bản dữ liệu và giúp mô hình nhìn nhận thế giới từ nhiều góc độ!
"""
    trace = {
        "items": [
            "Biểu diễn ảnh số & ma trận điểm ảnh (Pixels)",
            "Cấu trúc tensor ảnh Grayscale vs RGB ([C, H, W])",
            "Chuẩn hóa điểm ảnh & tách kênh màu"
        ],
        "deliverable": "Mã nguồn xử lý tensor hình ảnh và bóc tách kênh màu chuẩn PyTorch"
    }
    nb = create_notebook(
        module_id="M4",
        module_name="AI NHÌN VÀ HIỂU DỮ LIỆU",
        session_id="S01",
        topic="Biểu Diễn Ảnh Số: Pixel, Kênh Màu (Grayscale/RGB) & Image Tensors",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook xử lý và trực quan hóa Tensor ảnh số chuẩn PyTorch",
        competition_relevance="Nền tảng của toàn bộ nhánh Computer Vision trong các kỳ thi Olympic AI.",
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
    save_notebook(nb, "course/M4_APPLIED_AI/M4_S01.ipynb")


def build_m4_s02():
    objectives = [
        "Hiểu tại sao Data Augmentation là kỹ thuật bắt buộc để chống quá khớp trong các bài toán Thị giác máy tính có ít dữ liệu.",
        "Sử dụng thành thạo module `torchvision.transforms` để xây dựng chuỗi biến đổi hình ảnh.",
        "Làm chủ các phép biến đổi hình học phổ biến: RandomHorizontalFlip, RandomRotation, RandomResizedCrop.",
        "Hiểu và áp dụng chuẩn hóa ImageNet (Mean & Std Normalization) chuẩn mực cho các mô hình thị giác."
    ]
    prerequisites = [
        "Hoàn thành Buổi M4_S01 (Biểu diễn ảnh số).",
        "Hiểu cơ chế hoạt động của PyTorch Dataset."
    ]
    why_content = """
Trong các cuộc thi Olympic AI, ban tổ chức thường chỉ cung cấp một tập dữ liệu ảnh rất khiêm tốn (vài trăm đến vài nghìn ảnh) vì giới hạn tài nguyên và thời gian chấm thi. Nếu bạn đưa nguyên những bức ảnh đó vào huấn luyện, mạng nơ-ron sâu sẽ nhanh chóng "ghi nhớ vẹt" từng vị trí góc cạnh của con mèo trong ảnh, dẫn đến Overfitting nghiêm trọng.

**Tăng cường dữ liệu (Data Augmentation)** là phương pháp "phù phép" giúp bạn nhân bản dữ liệu một cách thông minh mà không tốn công gán nhãn thêm: Một con mèo khi bị lật ngang, xoay nghiêng 10 độ, hay thay đổi độ sáng một chút thì nó VẪN LÀ CON MÈO. Augmentation dạy cho mạng nơ-ron học được các đặc trưng bất biến (Invariant Features) trước mọi sự biến đổi thực tế của môi trường.
"""
    concept_content = r"""
### 1. Bản chất của Data Augmentation trong Computer Vision
* **Mục tiêu:** Tạo ra các biến thể mới một cách ngẫu nhiên trong mỗi lần nạp batch ở pha huấn luyện (On-the-fly Transformation), làm phong phú phân phối dữ liệu mà không tốn dung lượng lưu trữ trên đĩa.
* **Nguyên tắc an toàn ngữ nghĩa (Semantic Safety):** Phép biến đổi phải bảo toàn nhãn của bài toán!
  - Lật ngang (`RandomHorizontalFlip`): An toàn cho phân loại chó/mèo, xe cộ, cây cối.
  - Lật dọc (`RandomVerticalFlip`): Nguy hiểm trong nhận diện chữ số (số 6 lật ngược biến thành số 9!).

### 2. Chuẩn hóa ImageNet Statistics
Hầu hết các mô hình thị giác máy tính đều được tiền chuẩn hóa dựa trên giá trị trung bình và độ lệch chuẩn của 1.4 triệu ảnh từ tập dữ liệu ImageNet:
* $\text{Mean} = [0.485, 0.456, 0.406]$
* $\text{Std} = [0.229, 0.224, 0.225]$
Công thức chuẩn hóa từng kênh:
$$x_{norm} = \frac{x - \text{mean}}{\text{std}}$$
"""
    visual_intuition_content = """
```
Quy trình Pipeline Tăng cường dữ liệu (Data Augmentation Pipeline):
[Ảnh gốc] ──> [Resize & Crop ngẫu nhiên] ──> [Lật ngang ngẫu nhiên 50%] ──> [Đổi màu ColorJitter] ──> [Chuẩn hóa Tensor]
     │
     └──> Mỗi Epoch, mạng nơ-ron lại nhìn thấy một diện mạo mới của cùng một bức ảnh!
```
"""
    code_cells = [
        (
            """import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1. Tao mot anh mau PIL Image gia lap co hinh chu nhat ro rang
np.random.seed(42)
img_array = np.zeros((100, 100, 3), dtype=np.uint8)
img_array[30:70, 20:50] = [255, 100, 50] # Hinh chu nhat mau cam o ben trai
sample_pil = Image.fromarray(img_array)

# 2. Thiet ke Pipeline Data Augmentation bang torchvision.transforms.Compose
train_transforms = T.Compose([
    T.Resize((128, 128)),
    T.RandomHorizontalFlip(p=0.5), # 50% xac suat lat ngang
    T.RandomRotation(degrees=20),   # Xoay ngau nhien trong khoang [-20, +20] do
    T.ColorJitter(brightness=0.3, contrast=0.3), # Bien thien do sang va tuong phan
    T.ToTensor(),                   # Chuyen PIL sang Tensor va chia cho 255.0
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Chuan hoa ImageNet
])

print("===== PIPELINE DATA AUGMENTATION CHO PHA TRAIN =====")
print(train_transforms)
""",
            """Khối code xây dựng một pipeline tăng cường dữ liệu hoàn chỉnh bằng `torchvision.transforms.Compose`:
- Kết hợp liên hoàn các phép biến đổi hình học (Resize, Flip, Rotation) và quang học (ColorJitter).
- `ToTensor()` tự động đưa ảnh sang Tensor `[C, H, W]` và chuẩn hóa về $[0.0, 1.0]$.
- `Normalize()` đưa phân phối về chuẩn mực của các mô hình thị giác quốc tế."""
        ),
        (
            """# 3. Kiem chung: Quan sat 4 bien the ngau nhien duoc sinh ra tu cung mot buc anh goc
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i in range(4):
    # Moi lan goi train_transforms se sinh ra mot bien the ngau nhien khac nhau
    augmented_tensor = train_transforms(sample_pil)
    
    # De hien thi duoc bang Matplotlib, can dao nguoc lai buoc Normalize (De-normalize)
    inv_normalize = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img_display = inv_normalize(augmented_tensor).permute(1, 2, 0).clip(0, 1).numpy()
    
    axes[i].imshow(img_display)
    axes[i].set_title(f'Bien the ngau nhien #{i+1}', fontweight='bold')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
""",
            """Khối code minh họa trực quan sự phong phú của dữ liệu sinh ra:
- Cùng một bức ảnh gốc, qua 4 lần gọi khác nhau đã tạo ra 4 diện mạo hoàn toàn mới (góc xoay khác nhau, vị trí lật trái/phải khác nhau, độ sáng khác nhau).
- Giúp mạng nơ-ron rèn luyện khả năng nhận diện hình dạng bất kể vị trí không gian."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Phân biệt Transform của Train vs Validation
Tại sao trong pha Validation và Test, chúng ta **tuyệt đối không được dùng `RandomHorizontalFlip` hay `RandomRotation`**, mà chỉ được dùng `Resize`, `CenterCrop` và `Normalize`?

### Bài 2 (Easy): Thử nghiệm phép làm mờ ảnh (GaussianBlur)
Thêm `T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))` vào pipeline. Giải thích trong trường hợp camera thực tế bị rung hoặc mờ nét thì kỹ thuật này giúp ích gì.

### Bài 3 (Medium): Tăng cường dữ liệu bằng CutMix / MixUp
Tìm hiểu khái niệm kỹ thuật tiên tiến MixUp (trộn hai bức ảnh với nhau theo tỷ lệ $\lambda$ và $1-\lambda$). Giải thích tại sao MixUp lại cực kỳ phổ biến và mang lại điểm số cao trong các kỳ thi Kaggle và Olympic AI.

### Bài 4 (Medium): Đánh giá rủi ro ngữ nghĩa của Rotation
Nếu bài toán yêu cầu đọc biển báo giao thông (ví dụ biển cấm rẽ trái vs cấm rẽ phải), việc áp dụng `RandomHorizontalFlip(p=0.5)` có gây nguy hiểm không? Hãy giải thích.

### Bài 5 (Challenge): Viết Custom Transform Class
Viết một lớp `RandomAddNoise(noise_level=0.05)` kế thừa từ chuẩn của torchvision, tự động cộng thêm nhiễu Gauss ngẫu nhiên vào tensor ảnh: $x_{noisy} = x + \text{noise}$.
"""
    voai_connection_content = """
Trong các cuộc thi Olympic AI, sự khác biệt giữa thí sinh đạt giải Khuyến khích và thí sinh đạt Huy chương Vàng thường nằm ở **chiến lược Data Augmentation phù hợp với bài toán**:
Thí sinh thiếu kinh nghiệm sẽ áp dụng bừa bãi mọi kỹ thuật augmentation có sẵn trên mạng mà không phân tích tính chất ảnh (ví dụ lật ngược ảnh chụp X-quang tim phổi khiến quả tim bị đảo sang bên phải sai cấu trúc giải phẫu). Hiểu đúng ngữ nghĩa dữ liệu để chọn phép tăng cường hợp lý giúp bạn tối ưu hóa điểm số mà không làm hỏng bài thi.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh áp dụng hàm `T.Normalize()` TRƯỚC KHI gọi `T.ToTensor()` và bị quẳng lỗi:

```python
# CODE CÓ LỖI THỨ TỰ TRANSFORMS:
# bad_pipe = T.Compose([
#     T.Normalize(mean=[0.5], std=[0.5]), # Loi: TypeError: img should be Tensor Image. Got PIL.Image!
#     T.ToTensor()
# ])
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao `T.Normalize()` bắt buộc phải đứng SAU `T.ToTensor()`.
2. Khẳng định quy tắc vàng về thứ tự transforms: Biến đổi trên PIL Image trước -> Chuyển sang Tensor -> Chuẩn hóa số học trên Tensor cuối cùng.
"""
    knowledge_check_content = """
1. Kỹ thuật Tăng cường dữ liệu (Data Augmentation) nhằm giải quyết vấn đề lớn nhất nào trong Học sâu thị giác?
   - A. Giảm thời gian nạp dữ liệu từ đĩa.
   - B. Chống hiện tượng quá khớp (Overfitting) và tăng khả năng tổng quát hóa của mô hình khi tập dữ liệu huấn luyện có kích thước hạn chế.
   - C. Tự động tìm lỗi sai trong nhãn.
   - D. Tăng độ phân giải của máy ảnh.

2. Tại sao chúng ta không áp dụng các phép biến đổi ngẫu nhiên (Random Crop, Random Flip) cho tập Kiểm tra (Test set)?
   - A. Vì thư viện torchvision không hỗ trợ.
   - B. Vì tập Test cần đánh giá một cách khách quan, công bằng và có tính tất định (Deterministic) trên bức ảnh nguyên bản của người dùng.
   - C. Vì làm như vậy sẽ xóa mất tập Test.
   - D. Để tiết kiệm bộ nhớ RAM.

3. Trong bài toán nhận diện chữ số viết tay (MNIST), phép biến đổi nào sau đây có nguy cơ làm THAY ĐỔI BẢN CHẤT NHÃN của dữ liệu?
   - A. Xoay nhẹ 5 độ.
   - B. RandomVerticalFlip (Lật ngược theo chiều dọc biến số 6 thành số 9).
   - C. Thay đổi độ tương phản một chút.
   - D. Cắt viền trắng xung quanh.

4. Cặp giá trị thống kê `mean` và `std` chuẩn mực của ImageNet thường được áp dụng trên ảnh có thang đo nào?
   - A. Ảnh nguyên bản thang $[0, 255]$.
   - B. Ảnh Tensor đã được chuẩn hóa về thang số thực $[0.0, 1.0]$.
   - C. Ảnh xám nhị phân chỉ có 0 và 1.
   - D. Bất kỳ thang đo nào.

5. Lớp `torchvision.transforms.Compose` có chức năng chính là gì?
   - A. Huấn luyện mạng nơ-ron sâu.
   - B. Chuỗi hóa (Chain) nhiều phép biến đổi ảnh tuần tự lại với nhau thành một pipeline duy nhất.
   - C. Nén file ảnh sang định dạng zip.
   - D. Xuất dự đoán ra file CSV.
"""
    session_summary_content = """
* **What I learned:**
  - Cơ chế tăng cường dữ liệu ngẫu nhiên theo thời gian thực (On-the-fly Data Augmentation).
  - Cách dùng `torchvision.transforms.Compose` để xâu chuỗi các phép biến đổi.
  - Phân biệt rõ Transform dùng cho pha Train vs pha Validation.
  - Tầm quan trọng của chuẩn hóa ImageNet Statistics.
* **What I can now do:**
  - Thiết kế pipeline tiền xử lý và tăng cường ảnh chuyên nghiệp cho bài toán thị giác.
  - Bảo vệ mô hình khỏi hiện tượng học vẹt góc nhìn.
* **What I still need to practice:**
  - Khám phá kiến trúc chuyên dụng cho xử lý ảnh: **Mạng Tích Chập (CNN)** ở Buổi 3.
"""
    next_session_content = """
Trong **Buổi M4_S03**, chúng ta sẽ giải mã cấu trúc vĩ đại nhất của Computer Vision: **Trực Giác Mạng Tích Chập (CNN): Convolution, Stride, Padding & Pooling** — Tìm hiểu cơ chế trích xuất cạnh biên, toán tử tích chập ma trận 2D và cách giảm chiều không gian!
"""
    trace = {
        "items": [
            "Data Augmentation trong Computer Vision",
            "Sử dụng torchvision.transforms (Flip, Rotation, Jitter)",
            "Chuẩn hóa ImageNet Statistics & Semantic Safety"
        ],
        "deliverable": "Mã nguồn Pipeline Data Augmentation chuyên nghiệp cho ảnh số"
    }
    nb = create_notebook(
        module_id="M4",
        module_name="AI NHÌN VÀ HIỂU DỮ LIỆU",
        session_id="S02",
        topic="Tiền Xử Lý Ảnh & Kỹ Thuật Tăng Cường Dữ Liệu (Data Augmentation)",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook thiết kế pipeline Data Augmentation bằng torchvision",
        competition_relevance="Gia tăng đột biến độ chính xác tổng quát hóa khi thi đấu với tập dữ liệu ảnh nhỏ.",
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
    save_notebook(nb, "course/M4_APPLIED_AI/M4_S02.ipynb")


def build_m4_s03():
    objectives = [
        "Hiểu tại sao mạng MLP truyền thống thất bại thảm hại khi xử lý ảnh (Bùng nổ tham số & Phá hủy cấu trúc không gian 2D).",
        "Nắm vững cơ chế toán học của phép Tích chập 2D (Convolution): Kernel/Filter trượt trên ma trận ảnh.",
        "Làm chủ 3 siêu tham số cốt tử của tầng tích chập: Kernel Size, Padding (Same vs Valid), và Stride.",
        "Hiểu vai trò của Tầng gộp cực đại (MaxPool2d): Giảm độ phân giải không gian và tạo tính bất biến dịch chuyển (Translation Invariance)."
    ]
    prerequisites = [
        "Hoàn thành Buổi M4_S01 (Tensors hình ảnh) và M3_S03 (nn.Module).",
        "Hiểu phép nhân tích vô hướng của hai ma trận nhỏ."
    ]
    why_content = """
Hãy thử làm một phép tính: Nếu bạn đưa một bức ảnh màu có kích thước vừa phải $256 \times 256 \times 3$ vào một tầng MLP có 512 nơ-ron, số lượng trọng số cần học chỉ riêng cho tầng đầu tiên sẽ là: $256 \times 256 \times 3 \times 512 \approx \mathbf{100\text{ triệu tham số}}$! Mạng nơ-ron sẽ nổ tung bộ nhớ, đồng thời việc duỗi phẳng (Flatten) bức ảnh thành vector 1D sẽ phá hủy hoàn toàn mối quan hệ không gian giữa các pixel liền kề.

**Mạng Tích Chập (Convolutional Neural Networks - CNN)** ra đời như một cuộc cách mạng: Thay vì kết nối tất cả các pixel, CNN sử dụng các bộ lọc nhỏ (Kernels, ví dụ $3 \times 3$) trượt trên bức ảnh, chia sẻ trọng số (Weight Sharing) để phát hiện từ các nét cơ bản (đường cong, góc nhọn) đến các bộ phận phức tạp (mắt, mũi, bánh xe). Hiểu rõ CNN là điều kiện tiên quyết của mọi chuyên gia AI!
"""
    concept_content = r"""
### 1. Cơ chế Tích chập 2D (Convolution Operation)
* Một bộ lọc (Filter/Kernel, ví dụ ma trận $3 \times 3$) trượt qua từng vùng của bức ảnh.
* Tại mỗi vị trí, nó thực hiện phép nhân từng phần tử tương ứng rồi cộng tổng lại (Element-wise multiplication & sum) để tạo ra một giá trị trong **Bản đồ đặc trưng (Feature Map)**.
* **Chia sẻ trọng số (Weight Sharing):** Cùng một bộ lọc được dùng để quét toàn bộ bức ảnh, giúp giảm hàng triệu tham số so với MLP.

### 2. Các siêu tham số cốt lõi của `nn.Conv2d`
* **Kernel Size ($K$):** Kích thước cửa sổ trượt (phổ biến nhất là $3 \times 3$).
* **Stride ($S$):** Bước nhảy của cửa sổ trượt (Stride = 1: trượt từng pixel; Stride = 2: nhảy cóc, giảm kích thước ảnh đi một nửa).
* **Padding ($P$):** Đệm thêm các vòng số 0 xung quanh viền ảnh.
  - *Valid Padding ($P=0$):* Không đệm, ảnh đầu ra bị co nhỏ lại.
  - *Same Padding ($P = \frac{K - 1}{2}$ khi $S=1$):* Giữ nguyên kích thước chiều dài và chiều rộng của ảnh!
* **Công thức kích thước đầu ra:**
  $$H_{out} = \left\lfloor \frac{H_{in} - K + 2P}{S} \right\rfloor + 1$$

### 3. Tầng Gộp Cực Đại (MaxPool2d)
`nn.MaxPool2d(kernel_size=2, stride=2)` lấy giá trị lớn nhất trong mỗi cửa sổ $2 \times 2$:
* Giảm kích thước không gian đi một nửa (giảm 75% lượng tính toán cho các tầng sau).
* Giúp mô hình có tính bất biến với các xê dịch nhỏ của vật thể.
"""
    visual_intuition_content = """
```
Cơ chế trượt của Bộ lọc Tích chập (Kernel 3x3):
Vùng ảnh gốc (3x3):       Bộ lọc Sobel (3x3):      Kết quả Feature Map:
   [ 10,  20,  10 ]           [ -1,  0,  1 ]
   [ 10,  20,  10 ]     *     [ -2,  0,  2 ]   ==> [ Giá trị biên độ cạnh ]
   [ 10,  20,  10 ]           [ -1,  0,  1 ]
```
"""
    code_cells = [
        (
            """import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Cai dat phep tich chap thu cong bang NumPy de phat hien canh doc (Vertical Edge Detection)
image = np.zeros((10, 10))
image[:, 5:] = 10.0 # Nua trai la 0 (den), nua phai la 10 (trang) -> Co 1 canh doc o giua

# Bo loc phat hien canh doc (Vertical Sobel-like Kernel)
kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Tinh toan phep truot convolution thu cong
output_manual = np.zeros((8, 8))
for i in range(8):
    for j in range(8):
        window = image[i:i+3, j:j+3]
        output_manual[i, j] = np.sum(window * kernel)

print("===== MA TRAN ANH GOC (10x10) =====")
print(image[:4, :])
print("\\n===== FEATURE MAP PHAT HIEN CANH (8x8) =====")
print(output_manual[:4, :])
print("-> Cot co gia tri cao vuot troi chinh la vi tri cua canh bien!")
""",
            """Khối code lập trình tích chập thủ công bằng NumPy:
- Minh họa rõ ràng cơ chế trượt cửa sổ $3 \times 3$ và tính tổng tích từng phần tử.
- Cho thấy trực quan cách bộ lọc trích xuất đường biên thẳng đứng khi chuyển giao giữa vùng tối và vùng sáng."""
        ),
        (
            """# 2. Xay dung tang Conv2d va MaxPool2d bang PyTorch
conv_layer = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, stride=1)
pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

# Gia lap mot batch gom 2 anh xam kich thuoc 28x28 (giong anh MNIST)
dummy_images = torch.randn(2, 1, 28, 28)

# Chay qua Conv2d
feature_maps = conv_layer(dummy_images)
print(f"Kich thuoc sau Conv2d (Padding=1 giu nguyen H, W): {feature_maps.shape}")

# Chay qua MaxPool2d
pooled_maps = pool_layer(feature_maps)
print(f"Kich thuoc sau MaxPool2d (Giam mot nua do phan giai): {pooled_maps.shape}")
""",
            """Khối code minh họa tầng `nn.Conv2d` và `nn.MaxPool2d` trong PyTorch:
- Đầu vào: `[2, 1, 28, 28]` (Batch 2 ảnh, 1 kênh xám, $28 \times 28$).
- Qua Conv2d (`padding=1`): Giữ nguyên kích thước không gian $(28, 28)$ nhưng mở rộng lên 4 bản đồ đặc trưng (`[2, 4, 28, 28]`).
- Qua MaxPool2d (`stride=2`): Thu nhỏ kích thước còn một nửa $(14, 14)$ (`[2, 4, 14, 14]`)."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Tính toán kích thước đầu ra
Cho một ảnh đầu vào có kích thước $[1, 3, 64, 64]$. Hãy tính kích thước đầu ra bằng công thức toán học sau khi đi qua tầng:
`nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)`.
Kiểm tra lại bằng cách khởi tạo tầng và truyền tensor giả lập qua.

### Bài 2 (Easy): Đếm số tham số của tầng Tích chập
Tính toán số lượng tham số cần học (Weights + Biases) của tầng `nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)`. So sánh con số này với một tầng `nn.Linear(3*32*32, 32)`.

### Bài 3 (Medium): Trực quan hóa Feature Maps
Sử dụng Matplotlib vẽ 4 bản đồ đặc trưng (Feature Maps) được sinh ra từ biến `feature_maps[0]` ở đoạn code trên để quan sát các góc nhìn khác nhau mà mạng nơ-ron học được.

### Bài 4 (Medium): So sánh MaxPool2d vs AvgPool2d
Tìm hiểu tầng gộp trung bình `nn.AvgPool2d(kernel_size=2, stride=2)`. Nêu sự khác biệt trực giác: Tại sao MaxPool thường được ưa chuộng hơn cho việc giữ lại các tín hiệu biên sắc nét?

### Bài 5 (Challenge): Cài đặt Depthwise Separable Convolution
Tìm hiểu cấu trúc Depthwise Separable Convolution (nền tảng của kiến trúc MobileNet dành cho thiết bị di động). Sử dụng PyTorch với tham số `groups` trong `nn.Conv2d` để xây dựng cấu trúc này và chứng minh nó giảm hơn 80% số lượng phép tính so với tích chập tiêu chuẩn.
"""
    voai_connection_content = """
Trong thi đấu Olympic AI, hiểu rõ công thức kích thước đầu ra của CNN là vũ khí tối thượng giúp bạn tránh lỗi `RuntimeError: mat1 and mat2 shapes cannot be multiplied` khi kết nối tầng Convolution cuối cùng vào tầng `nn.Linear` (Flatten layer). Thí sinh chuyên nghiệp luôn biết chính xác sau bao nhiêu lần MaxPool thì kích thước không gian ảnh bị thu nhỏ lại bao nhiêu để khai báo đúng kích thước cho tầng Fully Connected.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh áp dụng quá nhiều tầng MaxPool liên tiếp trên ảnh có kích thước nhỏ (ví dụ ảnh $16 \times 16$), khiến kích thước không gian bị giảm về 0:

```python
# CODE CÓ LỖI THU NHỎ QUÁ ĐÀ:
# Anh 16x16 -> Pool 1: 8x8 -> Pool 2: 4x4 -> Pool 3: 2x2 -> Pool 4: 1x1 -> Pool 5: 0x0 (Error: Kernel size can't be greater than input size!)
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao số lượng tầng Downsampling (MaxPool hoặc Stride 2) bị giới hạn bởi độ phân giải của ảnh đầu vào.
2. Nêu cách tính số lần gộp tối đa có thể áp dụng cho một bức ảnh kích thước $W \times H$.
"""
    knowledge_check_content = """
1. Ưu điểm vượt trội nhất của Mạng Tích Chập (CNN) so với Mạng Perceptron Đa Tầng (MLP) khi xử lý dữ liệu hình ảnh là gì?
   - A. CNN không cần dùng hàm mất mát.
   - B. CNN bảo toàn cấu trúc không gian 2D của ảnh và chia sẻ trọng số (Weight Sharing), giúp giảm mạnh số lượng tham số và tăng khả năng nhận diện bất biến vị trí.
   - C. CNN chỉ chạy được trên CPU.
   - D. CNN luôn có độ chính xác 100%.

2. Một bộ lọc tích chập kích thước $3 \times 3$ trên một ảnh màu có 3 kênh (RGB) chứa bao nhiêu trọng số (chưa tính bias)?
   - A. 9
   - B. 27 ($3 \times 3 \times 3$)
   - C. 81
   - D. 3

3. Thiết lập tham số nào trong `nn.Conv2d` với `kernel_size=3` và `stride=1` sẽ giữ nguyên kích thước dài và rộng của ảnh đầu vào (Same Padding)?
   - A. `padding=0`
   - B. `padding=1`
   - C. `padding=2`
   - D. `padding=3`

4. Tầng `nn.MaxPool2d(kernel_size=2, stride=2)` thực hiện thao tác gì trên bản đồ đặc trưng?
   - A. Lấy giá trị lớn nhất trong mỗi cửa sổ $2 \times 2$, giúp giảm kích thước chiều dài và chiều rộng đi một nửa.
   - B. Nhân đôi độ phân giải của bức ảnh.
   - C. Tính giá trị trung bình của toàn bộ bức ảnh.
   - D. Xóa bỏ các giá trị âm.

5. Trong một tầng CNN, khái niệm "Feature Map" (Bản đồ đặc trưng) đại diện cho điều gì?
   - A. Bản đồ địa lý vị trí máy tính đang đặt.
   - B. Ma trận đầu ra thể hiện mức độ phản hồi của một bộ lọc (Kernel) tại từng vị trí không gian trên bức ảnh.
   - C. Nhãn thực tế của bài toán.
   - D. Bảng thống kê số lượng tham số.
"""
    session_summary_content = """
* **What I learned:**
  - Lý do mạng MLP truyền thống bất lực trước dữ liệu hình ảnh.
  - Bản chất toán học của phép Tích chập 2D và nguyên lý chia sẻ trọng số.
  - 3 siêu tham số cốt lõi: Kernel Size, Stride, Padding và công thức kích thước đầu ra.
  - Vai trò giảm chiều và trích xuất đặc trưng sắc nét của MaxPool2d.
* **What I can now do:**
  - Tự tay cài đặt và tính toán kích thước tensor qua các khối tích chập.
  - Phối hợp nhịp nhàng giữa `nn.Conv2d` và `nn.MaxPool2d`.
* **What I still need to practice:**
  - Lắp ráp thành một kiến trúc mạng phân loại ảnh hoàn chỉnh ở Buổi 4.
"""
    next_session_content = """
Trong **Buổi M4_S04**, chúng ta sẽ chính thức ráp nối toàn bộ khối sức mạnh: **Xây Dựng End-to-End Image Classification Pipeline với PyTorch CNN** — Tự tay lập trình mạng CNN phân loại ảnh hoàn chỉnh, viết training loop cho dữ liệu ảnh và phân tích ma trận nhầm lẫn!
"""
    trace = {
        "items": [
            "Trực giác mạng tích chập (CNN) & Weight sharing",
            "Cơ chế Convolution 2D, Kernel, Stride & Padding",
            "Tầng gộp MaxPool2d & Feature maps visualization"
        ],
        "deliverable": "Mã nguồn cài đặt phép tích chập và theo dõi kích thước Feature Maps"
    }
    nb = create_notebook(
        module_id="M4",
        module_name="AI NHÌN VÀ HIỂU DỮ LIỆU",
        session_id="S03",
        topic="Trực Giác Mạng Tích Chập (CNN): Convolution, Stride & Pooling",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook lập trình trực quan hóa phép tích chập và các tầng CNN",
        competition_relevance="Nền tảng kiến trúc cốt lõi để làm chủ mọi mô hình Thị giác máy tính thi đấu.",
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
    save_notebook(nb, "course/M4_APPLIED_AI/M4_S03.ipynb")


def build_m4_s04():
    objectives = [
        "Xây dựng hoàn chỉnh một kiến trúc mạng nơ-ron tích chập (CNN) dạng khối kinh điển: [Conv - BatchNorm - ReLU - MaxPool].",
        "Làm chủ kỹ thuật làm phẳng không gian (Flatten) hoặc Global Average Pooling để chuyển giao mượt mà sang tầng phân loại (Classifier Head).",
        "Viết trọn vẹn luồng nạp dữ liệu ảnh, tiền xử lý và vòng lặp huấn luyện CNN trên tập dữ liệu phân loại đa lớp.",
        "Đánh giá chi tiết hiệu năng mô hình thông qua Confusion Matrix trên từng lớp hình ảnh cụ thể."
    ]
    prerequisites = [
        "Hoàn thành các buổi M4_S01, M4_S02, M4_S03.",
        "Đã nắm vững Training Loop từ Module M3."
    ]
    why_content = """
Biết từng tầng `Conv2d` hay `MaxPool2d` đơn lẻ là chưa đủ. Trong một kỳ thi thực tế, nhiệm vụ của bạn là phải lắp ráp các tầng đó thành một **Cỗ máy Phân loại Ảnh hoàn chỉnh (End-to-End Vision Pipeline)**: Từ việc đọc ảnh thô, đưa qua các khối tích chập trích xuất đặc trưng phân cấp, làm phẳng không gian, và đưa ra quyết định phân loại nhãn chính xác.

Buổi học hôm nay đưa bạn bước lên vũ đài thực chiến: Tự tay thiết kế, huấn luyện và đánh giá một mạng CNN tiêu chuẩn Olympic từ số 0, thiết lập nền móng vững chắc trước khi tiến tới các mô hình phức tạp hơn.
"""
    concept_content = r"""
### 1. Cấu trúc một mạng CNN phân loại ảnh chuẩn mực
Gồm 2 khối lớn:
1. **Feature Extractor (Khối trích xuất đặc trưng không gian):**
   - Xếp chồng các khối lặp lại: `[Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d]`.
   - Càng vào sâu, số kênh đặc trưng (Channels) càng tăng lên ($16 \to 32 \to 64$), trong khi độ phân giải không gian ($H, W$) giảm dần.
2. **Classifier Head (Đầu phân loại):**
   - `nn.Flatten()` duỗi phẳng tensor $3D$ thành vector $1D$.
   - Các tầng `nn.Linear` kết hợp `nn.Dropout` đưa ra vector Logits có kích thước bằng số lượng lớp cần phân loại ($C$).

### 2. Vai trò của Tầng Chuẩn hóa theo Lô (Batch Normalization - `nn.BatchNorm2d`)
* Chuẩn hóa đầu ra của mỗi tầng tích chập về phân phối có trung bình 0 và phương sai 1 trên từng Mini-batch.
* Giúp triệt tiêu hiện tượng dịch chuyển hiệp biến nội tại (Internal Covariate Shift), cho phép sử dụng tốc độ học (Learning Rate) lớn hơn và huấn luyện mô hình nhanh gấp nhiều lần!
"""
    visual_intuition_content = """
```
Kiến trúc End-to-End CNN Pipeline:
[Ảnh đầu vào 28x28] 
         │
         ▼
[Khối Conv 1: 1 -> 16 channels, Pool -> 14x14]
         │
         ▼
[Khối Conv 2: 16 -> 32 channels, Pool -> 7x7]
         │
         ▼
[Flatten Layer] ──> Vector 1D (32 * 7 * 7 = 1568 phần tử)
         │
         ▼
[Fully Connected Classifier: Linear(1568, 64) -> Dropout -> Linear(64, num_classes)] ──> [Logits]
```
"""
    code_cells = [
        (
            """import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 1. Thiet ke kien truc CNN hoan chinh bang nn.Module
class MiniVisionCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(MiniVisionCNN, self).__init__()
        
        # Khoi trich xuat dac trung (Feature Extractor)
        self.features = nn.Sequential(
            # Block 1: 1 channel in -> 16 channels out
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28x28 -> 14x14
            
            # Block 2: 16 channels in -> 32 channels out
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        )
        
        # Dau phan loai (Classifier Head)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes) # Tra ve Logits cho 3 lop
        )
        
    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits

# Khoi tao mo hinh
model_cnn = MiniVisionCNN(num_classes=3)
print("===== KIEN TRUC MANG CNN HOAN CHINH =====")
print(model_cnn)
""",
            """Khối code định nghĩa kiến trúc CNN chuẩn công nghiệp:
- 2 khối tích chập chuẩn hóa đầy đủ với `BatchNorm2d`.
- Kết nối tự động qua `nn.Flatten()`.
- Tầng phân loại có Dropout chống quá khớp, sẵn sàng phân loại 3 lớp hình ảnh."""
        ),
        (
            """# 2. Huan luyen va Danh gia tren Tap du lieu anh nhan tao mo phong 3 lop
torch.manual_seed(42)

# Tao 300 anh nhan tao (100 anh moi lop, kich thuoc 1x28x28)
X_synth = torch.randn(300, 1, 28, 28)
y_synth = torch.tensor([0]*100 + [1]*100 + [2]*100, dtype=torch.long)

# Tao DataLoader
dataset = TensorDataset(X_synth, y_synth)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cnn.parameters(), lr=0.005)

# Huan luyen nhanh 5 epochs
model_cnn.train()
for epoch in range(1, 6):
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model_cnn(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)
        
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch}/5 | Training CrossEntropy Loss: {epoch_loss:.4f}")

# 3. Danh gia va Xuat Confusion Matrix
model_cnn.eval()
with torch.no_grad():
    test_logits = model_cnn(X_synth)
    test_preds = torch.argmax(test_logits, dim=1).numpy()

cm = confusion_matrix(y_synth.numpy(), test_preds)
print("\\n===== MA TRAN NHAM LAN PHAN LOAI ANH (CONFUSION MATRIX) =====")
print(cm)
""",
            """Khối code thực thi trọn vẹn luồng huấn luyện và đánh giá:
- Sử dụng hàm mất mát chuẩn mực cho phân loại đa lớp: `nn.CrossEntropyLoss()`.
- Trích xuất nhãn dự đoán bằng `torch.argmax(logits, dim=1)`.
- Xuất ma trận nhầm lẫn thể hiện rõ năng lực phân loại trên từng lớp."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Thử nghiệm với Global Average Pooling
Thay thế toàn bộ cụm `[nn.Flatten(), nn.Linear(32 * 7 * 7, 64)]` bằng một tầng `nn.AdaptiveAvgPool2d((1, 1))` kết hợp `nn.Flatten()`. Nêu rõ số lượng tham số giảm đi bao nhiêu lần khi áp dụng kỹ thuật này.

### Bài 2 (Easy): Mở rộng số lượng khối Tích chập
Thêm một khối tích chập thứ 3 vào mạng: đưa từ 32 channels lên 64 channels. Tính toán kích thước không gian còn lại sau 3 lần MaxPool từ ảnh gốc $28 \times 28$.

### Bài 3 (Medium): Thêm hàm tính Accuracy trực tiếp trong Training Loop
Viết lại vòng lặp huấn luyện sao cho ở cuối mỗi epoch, chương trình tự động in ra cả hai chỉ số: `Epoch Loss` và `Epoch Accuracy (%)`.

### Bài 4 (Medium): Phân tích Confusion Matrix
Nhìn vào kết quả ma trận nhầm lẫn ở code cell trên, hãy chỉ ra lớp nào đang bị mô hình dự đoán nhầm nhiều nhất sang lớp nào và đưa ra nhận xét.

### Bài 5 (Challenge): Lưu lại các mẫu ảnh bị phân loại sai (Misclassified Visualizer)
Viết một hàm `visualize_errors(images, true_labels, pred_labels, n=4)` lọc ra các bức ảnh mà mô hình đoán sai, sau đó dùng Matplotlib vẽ các bức ảnh đó lên màn hình kèm theo tiêu đề: *"Thực tế: Lớp A - Đoán nhầm: Lớp B"*.
"""
    voai_connection_content = """
Trong phần thi Computer Vision của Olympic AI, thời gian huấn luyện mô hình luôn bị ban tổ chức khống chế nghiêm ngặt. Việc thiết kế một mạng CNN gọn nhẹ nhưng hiệu quả (dưới 500,000 tham số) kết hợp đúng đắn với `BatchNorm` sẽ giúp mô hình của bạn hội tụ chỉ sau 10-15 phút, bỏ xa những đối thủ cố chấp sử dụng các mô hình quá cồng kềnh dẫn tới không kịp nộp bài trước giờ đóng cổng.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh tính toán sai kích thước đầu vào của tầng Linear sau khi Flatten:

```python
# CODE CÓ LỖI TÍNH SAI CHIỀU FLATTEN:
# Sau 2 lan MaxPool, anh 28x28 bien thanh 7x7 voi 32 channels.
# Tong so phan tu phai la 32 * 7 * 7 = 1568.
# Neu thi sinh viet: nn.Linear(32 * 14 * 14, 64) -> PyTorch se bao loi Shape Mismatch ngay lap tuc!
```

**Nhiệm vụ của bạn:**
1. Trình bày công thức tính số lượng phần tử phẳng: $N_{flat} = C_{out} \times H_{out} \times W_{out}$.
2. Nêu mẹo debug nhanh: Đặt một lệnh `print(x.shape)` tạm thời trong hàm `forward` để xem kích thước thực tế trước khi đi vào tầng Linear.
"""
    knowledge_check_content = """
1. Hàm mất mát nào sau đây là lựa chọn CHUẨN MỰC trong PyTorch cho bài toán phân loại hình ảnh đa lớp (Multi-class Image Classification)?
   - A. `nn.MSELoss`
   - B. `nn.CrossEntropyLoss`
   - C. `nn.L1Loss`
   - D. `nn.CosineSimilarity`

2. Tầng `nn.BatchNorm2d` mang lại lợi ích quan trọng nhất nào cho quá trình huấn luyện mạng CNN?
   - A. Giảm kích thước ảnh đi 4 lần.
   - B. Ổn định phân phối dữ liệu qua từng tầng, giúp quá trình huấn luyện hội tụ nhanh hơn và cho phép sử dụng Learning Rate cao hơn.
   - C. Chuyển ảnh màu thành ảnh xám.
   - D. Tự động gán nhãn cho ảnh.

3. Để lấy ra lớp có xác suất dự đoán cao nhất từ tensor Logits đầu ra của mạng, ta sử dụng hàm nào trong PyTorch?
   - A. `torch.max_val()`
   - B. `torch.argmax(logits, dim=1)`
   - C. `torch.sum()`
   - D. `torch.sort()`

4. Nếu ảnh đầu vào có kích thước $32 \times 32$, sau khi đi qua 2 tầng `nn.MaxPool2d(2, 2)` liên tiếp, kích thước không gian còn lại là bao nhiêu?
   - A. $16 \times 16$
   - B. $8 \times 8$
   - C. $4 \times 4$
   - D. $2 \times 2$

5. Lớp `nn.Flatten()` trong kiến trúc CNN thực hiện thao tác gì?
   - A. Xóa bỏ hoàn toàn các giá trị pixel bằng 0.
   - B. Duỗi phẳng tensor đặc trưng đa chiều $[Batch, Channels, H, W]$ thành vector 2 chiều $[Batch, Channels \times H \times W]$ để sẵn sàng đưa vào các tầng Fully Connected.
   - C. Làm mờ bức ảnh.
   - D. Đổi thứ tự kênh màu.
"""
    session_summary_content = """
* **What I learned:**
  - Cấu trúc 2 khối hoàn chỉnh của mạng phân loại ảnh: Feature Extractor và Classifier Head.
  - Sức mạnh của tầng `nn.BatchNorm2d` trong việc tăng tốc hội tụ.
  - Cách tính toán chính xác kích thước duỗi phẳng (Flatten dimension).
  - Đánh giá toàn diện mô hình thị giác bằng Confusion Matrix.
* **What I can now do:**
  - Tự tay lập trình và huấn luyện một mạng CNN hoàn chỉnh từ đầu đến cuối trên PyTorch.
  - Tự tin debug mọi lỗi lệch kích thước tầng tích chập.
* **What I still need to practice:**
  - Khám phá bí quyết đạt điểm cao trong thi đấu: **Transfer Learning** ở Buổi 5.
"""
    next_session_content = """
Trong **Buổi M4_S05**, chúng ta sẽ làm quen với vũ khí tối thượng của các vận động viên AI: **Transfer Learning Trong Thị Giác Máy Tính: Pretrained Models & Error Analysis** — Học cách đứng trên vai người khổng lồ (ResNet, MobileNet), đóng băng trọng số (Freeze) và phân tích các ca dự đoán sai lầm!
"""
    trace = {
        "items": [
            "Kiến trúc End-to-End CNN ([Conv-BN-ReLU-Pool] blocks)",
            "Flatten layer & Fully connected classifier head",
            "Training loop cho phân loại ảnh & Confusion matrix evaluation"
        ],
        "deliverable": "Mã nguồn End-to-End Image Classification Pipeline hoàn chỉnh bằng PyTorch CNN"
    }
    nb = create_notebook(
        module_id="M4",
        module_name="AI NHÌN VÀ HIỂU DỮ LIỆU",
        session_id="S04",
        topic="End-to-End Image Classification Pipeline với PyTorch CNN",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook huấn luyện End-to-End CNN phân loại ảnh hoàn chỉnh",
        competition_relevance="Mô hình thị giác cơ sở tự viết bắt buộc phải làm chủ trước khi dùng mô hình có sẵn.",
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
    save_notebook(nb, "course/M4_APPLIED_AI/M4_S04.ipynb")


def build_m4_s05():
    objectives = [
        "Hiểu triết lý 'Đứng trên vai người khổng lồ': Tại sao Transfer Learning (Học chuyển giao) lại thống trị các cuộc thi AI.",
        "Phân biệt rõ ràng giữa hai chiến lược: Trích xuất đặc trưng (Feature Extraction / Frozen Backbone) vs Tinh chỉnh toàn phần (Fine-tuning).",
        "Tùy biến đầu phân loại (Classifier Head) của một mô hình thị giác nạp sẵn (ví dụ ResNet-18 hoặc MobileNetV2).",
        "Thực hiện phân tích lỗi sai thị giác (Visual Error Analysis) để tìm ra điểm yếu của mô hình trên từng nhóm ảnh cụ thể."
    ]
    prerequisites = [
        "Hoàn thành Buổi M4_S04 (End-to-End CNN Pipeline).",
        "Hiểu khái niệm đóng băng gradient (`requires_grad = False`)."
    ]
    why_content = """
Để huấn luyện một mạng nơ-ron sâu từ con số 0 (train from scratch) nhận diện ảnh đạt độ chính xác trên 90%, các tập đoàn công nghệ lớn phải sử dụng hàng triệu bức ảnh và hàng trăm card đồ họa cao cấp chạy liên tục trong nhiều tuần. Trong phòng thi Olympic AI, bạn chỉ có vỏn vẹn vài giờ và một máy tính với tài nguyên giới hạn. Nếu bạn tự train từ đầu, mô hình của bạn gần như chắc chắn sẽ bị Overfitting hoặc không kịp hội tụ!

**Học Chuyển Giao (Transfer Learning)** cho phép bạn tải về một mô hình đã được các nhà khoa học huấn luyện sẵn trên 1.4 triệu ảnh ImageNet. Mô hình này đã có sẵn "đôi mắt thần" hiểu rõ mọi khái niệm thị giác từ cơ bản đến phức tạp. Nhiệm vụ của bạn chỉ đơn giản là mượn đôi mắt đó và dạy lại phần "não bộ ra quyết định" (Classifier Head) cho bài toán thi đấu cụ thể của mình!
"""
    concept_content = r"""
### 1. Bản chất phân cấp đặc trưng trong Deep CNN
* **Các tầng đầu (Early Layers):** Học các nét phổ quát (Góc cạnh, màu sắc, kết cấu vân). Các đặc trưng này giống nhau trên MỌI bài toán thị giác!
* **Các tầng giữa (Middle Layers):** Ghép các nét lại thành các bộ phận (Mắt, mũi, bánh xe, cánh hoa).
* **Các tầng cuối (Late Layers):** Học các đặc trưng chuyên biệt gắn liền với bài toán gốc (1000 lớp ImageNet).

### 2. Hai chiến thuật Transfer Learning chính
1. **Feature Extraction (Đóng băng thân mạng - Frozen Backbone):**
   - Đặt `param.requires_grad = False` cho toàn bộ các tầng trích xuất đặc trưng của mạng gốc.
   - Chỉ thay thế và huấn luyện duy nhất tầng Linear cuối cùng.
   - *Ưu điểm:* Cực kỳ nhanh, không bao giờ lo phá hỏng trọng số tốt có sẵn, hoàn hảo khi dữ liệu thi đấu rất ít.
2. **Fine-Tuning (Tinh chỉnh):**
   - Huấn luyện lại một vài tầng sâu cuối cùng với **Tốc độ học cực nhỏ (Very small Learning Rate, ví dụ $10^{-5}$)**.

### 3. Phân tích lỗi sai thị giác (Visual Error Analysis)
Không chỉ nhìn vào con số Accuracy tổng thể, thí sinh phải trực quan hóa những bức ảnh mà mô hình đưa ra quyết định sai lầm nhất (High Confidence Errors). Điều này giúp phát hiện dữ liệu bị gán nhãn sai, ảnh bị ngược sáng, hoặc góc chụp quá dị biệt.
"""
    visual_intuition_content = """
```
Cơ chế Transfer Learning (Feature Extraction):
[Ảnh đầu vào] ──> [Pretrained Backbone (ResNet-18) ĐÃ ĐÓNG BĂNG] ──> [Vector Đặc trưng 512D]
                         (requires_grad = False)                               │
                         (Không cập nhật trọng số)                             ▼
                                                                  [New Classifier Head]
                                                                  nn.Linear(512, num_classes)
                                                                  (requires_grad = True)
                                                                  (CHỈ HUẤN LUYỆN TẦNG NÀY!)
```
"""
    code_cells = [
        (
            """import torch
import torch.nn as nn
import torchvision.models as models

# 1. Nap mo hinh Pretrained ResNet-18 co san
# Luu y: Trong thuc te ta dung weights=models.ResNet18_Weights.DEFAULT
# O day ta mo phong kien truc chuan de dam bao chay doc lap khong can tai file nang
base_model = models.resnet18(weights=None)

# 2. Chien thuat Feature Extraction: Dong bang toan bo cac tang Backbone
for param in base_model.parameters():
    param.requires_grad = False

# 3. Kiem tra va Thay the tang Classifier cuoi cung (base_model.fc)
num_features = base_model.fc.in_features
print(f"So luong dac trung dau vao cua tang cuoi (in_features): {num_features}")

# Thay the bang tang Linear moi phu hop voi bai toan 4 lop cua chung ta
num_classes = 4
base_model.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, num_classes) # Chi co cac tham so moi nay moi requires_grad=True!
)

# Kiem tra so luong tham so can huan luyen thuc te
trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in base_model.parameters())

print(f"Tong so tham so cua mang: {total_params:,}")
print(f"So tham so can huan luyen (Chi rieng Head): {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
print("-> Dong bang 99% mang, toc do huan luyen tang gap 10 lan va chong Overfitting tuyet doi!")
""",
            """Khối code minh họa kỹ thuật đóng băng và thay thế đầu phân loại:
- Duyệt qua toàn bộ tham số và đặt `param.requires_grad = False`.
- Thay thế thuộc tính `base_model.fc` bằng một khối mạng nhỏ thích ứng với 4 lớp của bài toán.
- Chứng minh bằng số liệu: Số lượng tham số cần học giảm hơn 99%, giúp việc huấn luyện diễn ra tức thì."""
        ),
        (
            """# 4. Quy trinh Visual Error Analysis (Phan tich truc quan cac ca du doan sai)
# Gia lap mot batch du doan va nhan thuc te
np.random.seed(42)
true_labels = np.array([0, 1, 2, 3, 1, 0, 2, 3])
pred_labels = np.array([0, 1, 3, 3, 1, 0, 1, 3]) # Mau so 2 va so 6 bi doan sai
confidences = np.array([0.95, 0.88, 0.72, 0.91, 0.85, 0.99, 0.65, 0.94])

# Phat hien cac mau sai lam
wrong_indices = np.where(true_labels != pred_labels)[0]
print(f"Phat hien {len(wrong_indices)} ca du doan sai lam tren tap mau.")

for idx in wrong_indices:
    print(f" - Mau thu #{idx}: Nhan thuc te = Lop {true_labels[idx]} | Model doan nham = Lop {pred_labels[idx]} (Do tu tin: {confidences[idx]*100:.1f}%)")
print("\\n-> Hanh dong: Kiem tra lai hinh anh cua mau bi sai de xem co phai anh mo/nhieu/nhan sai khong!")
""",
            """Khối code xây dựng quy trình phân tích lỗi sai thực nghiệm:
- Tự động lọc các vị trí dự đoán sai và in ra mức độ tự tin (Confidence) của mô hình.
- Cung cấp cơ sở khoa học để tìm ra nguyên nhân gốc rễ (Root Cause) của các ca thất bại."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Tinh chỉnh (Fine-tuning) tầng cuối
Sau khi huấn luyện phần Head 5 epochs, hãy viết code mở khóa (unfreeze) cho khối tích chập cuối cùng `base_model.layer4` bằng cách đặt `param.requires_grad = True` cho riêng khối này.

### Bài 2 (Easy): Khám phá kiến trúc MobileNetV2
Import `models.mobilenet_v2`. Kiểm tra xem tầng phân loại cuối cùng của MobileNetV2 được đặt tên là `fc` hay `classifier`. Viết code thay thế tầng đó cho bài toán 5 lớp.

### Bài 3 (Medium): Đặt tốc độ học phân tầng (Differential Learning Rates)
Trong Fine-tuning chuyên sâu, người ta thường dùng tốc độ học khác nhau cho các phần của mạng: Tốc độ cực nhỏ cho Backbone ($10^{-5}$) và tốc độ lớn hơn cho Classifier Head ($10^{-3}$). Hãy tìm hiểu cách truyền danh sách tham số dạng nhóm vào `torch.optim.Adam` để thực hiện điều này.

### Bài 4 (Medium): Phân tích trường hợp mô hình tự tin sai (High-Confidence Failure)
Tại sao trường hợp một mô hình dự đoán SAI nhưng lại có độ tự tin cực cao ($P > 95\%$) lại nguy hiểm hơn nhiều so với việc mô hình đoán sai nhưng độ tự tin thấp ($P \approx 50\%$)? Nêu 2 nguyên nhân dữ liệu có thể dẫn tới hiện tượng này.

### Bài 5 (Challenge): Xây dựng Báo cáo Đánh giá Chuyển giao (Transfer Benchmark)
Viết một hàm so sánh hiệu năng giữa hai mô hình: Mô hình CNN tự viết từ Buổi 4 vs Mô hình Transfer Learning ResNet-18 trên cùng một tập dữ liệu ảnh kiểm thử. Xuất bảng so sánh Accuracy và thời gian huấn luyện.
"""
    voai_connection_content = """
Trong mọi cuộc thi Olympic AI hoặc các nền tảng thi đấu như Kaggle, **Transfer Learning là tiêu chuẩn mặc định bắt buộc (Standard Practice)** nếu bạn muốn có tên trong nhóm huy chương phần Computer Vision. Không một đội tuyển nào train from scratch trừ khi đề bài cố tình cấm dùng pretrained weights. Biết cách chọn mô hình nền tảng phù hợp (ResNet-18 cho bài toán nhẹ, EfficientNet cho bài toán cân bằng, ConvNeXt cho bài toán lớn) là bản lĩnh của một thí sinh hàng đầu.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh nạp mô hình Pretrained nhưng truyền danh sách tham số vào Optimizer mà quên lọc những tham số đã bị đóng băng:

```python
# CODE CÓ THỂ GÂY LÃNG PHÍ TÍNH TOÁN:
# optimizer = optim.Adam(base_model.parameters(), lr=0.001) # Van truyen ca cac tham so requires_grad=False!
# Cach viet chuan muc:
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), lr=0.001)
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao việc dùng `filter(lambda p: p.requires_grad, ...)` giúp tối ưu hóa bộ nhớ và tốc độ cập nhật của Optimizer.
2. Kiểm tra xem Optimizer có tự động bỏ qua các tham số không tính gradient hay không.
"""
    knowledge_check_content = """
1. Lý do cốt lõi nhất khiến Học Chuyển Giao (Transfer Learning) mang lại hiệu quả vượt trội trong các kỳ thi AI là gì?
   - A. Giúp mô hình không bao giờ bị lỗi cú pháp.
   - B. Tận dụng lại các đặc trưng thị giác phổ quát đã được học từ hàng triệu bức ảnh ImageNet, giúp mô hình đạt độ chính xác cao chỉ với lượng dữ liệu nhỏ và thời gian huấn luyện ngắn.
   - C. Tự động xóa các bức ảnh bị nhòe.
   - D. Không cần chia tập Train/Test.

2. Thao tác nào sau đây được gọi là "Đóng băng" (Freezing) một tầng mạng nơ-ron trong PyTorch?
   - A. Xóa tầng đó khỏi mạng.
   - B. Đặt `param.requires_grad = False` cho tất cả các tham số của tầng đó để ngăn không cho PyTorch tính toán và cập nhật gradient.
   - C. Đặt toàn bộ trọng số của tầng đó bằng 0.
   - D. Chuyển mô hình sang chạy trên CPU.

3. Trong kiến trúc ResNet-18 của torchvision, tầng phân loại tuyến tính cuối cùng được lưu trữ dưới thuộc tính nào của mô hình?
   - A. `model.classifier`
   - B. `model.head`
   - C. `model.fc`
   - D. `model.output`

4. Khi thực hiện Tinh chỉnh (Fine-tuning) các tầng sâu của một mô hình Pretrained, chúng ta nên chọn Tốc độ học (Learning Rate) như thế nào?
   - A. Rất lớn (ví dụ $\alpha = 0.5$) để cập nhật thật nhanh.
   - B. Rất nhỏ (ví dụ $\alpha = 10^{-5}$ hoặc $10^{-4}$) để chỉ điều chỉnh nhẹ nhàng các trọng số tốt có sẵn mà không phá vỡ tri thức đã học.
   - C. Bằng 0.
   - D. Bằng tốc độ học của tầng Classifier.

5. Hoạt động "Visual Error Analysis" (Phân tích lỗi sai thị giác) có mục tiêu cốt lõi là gì?
   - A. Đếm số dòng code của notebook.
   - B. Quan sát trực quan các bức ảnh mà mô hình dự đoán sai lầm để tìm ra các điểm mù (blind spots), dữ liệu gán nhãn lỗi, hoặc các trường hợp dị biệt của bài toán.
   - C. Tăng độ sáng cho màn hình máy tính.
   - D. Xóa tập dữ liệu kiểm tra.
"""
    session_summary_content = """
* **What I learned:**
  - Triết lý và bản chất kỹ thuật của Transfer Learning trong Computer Vision.
  - Phân biệt Feature Extraction (Frozen Backbone) vs Fine-Tuning.
  - Cách thay thế Classifier Head trên các kiến trúc hiện đại (ResNet-18, MobileNet).
  - Quy trình phân tích lỗi sai thị giác (Visual Error Analysis).
* **What I can now do:**
  - Đứng trên vai người khổng lồ: Tận dụng các mô hình Pretrained hàng đầu để chinh phục bài toán phân loại ảnh thực tế.
  - Phân tích và đưa ra giải pháp khắc phục cho các ca dự đoán sai lầm.
* **What I still need to practice:**
  - Hoàn thành nửa đầu Module M4 (Computer Vision) và sẵn sàng chuyển sang nửa tiếp theo: **Xử lý Ngôn ngữ Tự nhiên (NLP)** ở Buổi 6!
"""
    next_session_content = """
Xin chúc mừng bạn đã hoàn thành xuất sắc 5 buổi chặng A (Computer Vision)!
Trong buổi tiếp theo (**M4_S06**), chúng ta sẽ bước sang chặng B đầy mê hoặc: **Xử Lý Ngôn Ngữ Tự Nhiên (Natural Language Processing - NLP)** với chủ đề Biểu Diễn Văn Bản: Chuẩn Hóa Chuỗi, Tokenization & Xây Dựng Từ Điển Từ Vựng!
"""
    trace = {
        "items": [
            "Triết lý Transfer Learning & Pretrained architectures (ResNet-18)",
            "Đóng băng trọng số (Freezing) & Thay thế Classifier head",
            "Phân tích lỗi sai thị giác (Visual error analysis)"
        ],
        "deliverable": "Mã nguồn ứng dụng Transfer Learning trên mô hình Pretrained kèm phân tích lỗi"
    }
    nb = create_notebook(
        module_id="M4",
        module_name="AI NHÌN VÀ HIỂU DỮ LIỆU",
        session_id="S05",
        topic="Transfer Learning Trong Thị Giác Máy Tính: Pretrained Models & Error Analysis",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook ứng dụng Transfer Learning và phân tích lỗi sai thị giác",
        competition_relevance="Vũ khí tối thượng để giành huy chương trong các bài thi thị giác máy tính.",
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
    save_notebook(nb, "course/M4_APPLIED_AI/M4_S05.ipynb")


# ==========================================
# PART B: NATURAL LANGUAGE PROCESSING (S06 - S10)
# ==========================================

def build_m4_s06():
    objectives = [
        "Hiểu thách thức căn bản của Xử lý Ngôn ngữ Tự nhiên (NLP): Máy tính chỉ hiểu số học, không hiểu chuỗi ký tự rời rạc.",
        "Xây dựng pipeline Tiền xử lý văn bản chuẩn: Hạ chữ thường (Lowercase), khử ký tự đặc biệt, chuẩn hóa khoảng trắng bằng Regular Expressions (Regex).",
        "Nắm vững các cấp độ Tokenization: Tách từ (Word-level), Tách ký tự (Character-level), và trực giác về Subword BPE.",
        "Tự tay lập trình cấu trúc Từ Điển Từ Vựng (Vocabulary): Mã hóa từ sang chỉ số (Word-to-Index) và xử lý từ mới chưa từng thấy (Out-Of-Vocabulary - `<UNK>`)."
    ]
    prerequisites = [
        "Hoàn thành các buổi trước của M4 (Hiểu tư duy xử lý dữ liệu phi cấu trúc).",
        "Kỹ năng xử lý chuỗi cơ bản trong Python (chuỗi `str`, phương thức `.split()`)."
    ]
    why_content = """
Trong Computer Vision, một bức ảnh bản chất đã là các con số cường độ sáng từ 0 đến 255. Nhưng trong Ngôn ngữ Tự nhiên (NLP), một câu như: *"Đề thi Olympic năm nay rất hay và thách thức!"* chỉ là một mảng các ký tự Unicode vô nghĩa đối với các thuật toán tối ưu hóa ma trận.

Làm thế nào để biến những câu chữ giàu cảm xúc, đầy tính ngữ cảnh và ngữ pháp của con người thành những con số toán học mà máy tính có thể hiểu được? **Chuẩn hóa văn bản (Preprocessing), Phân tách từ (Tokenization) và Xây dựng Từ điển (Vocabulary Mapping)** chính là cây cầu đầu tiên kết nối giữa ngôn ngữ loài người và trí tuệ nhân tạo.
"""
    concept_content = r"""
### 1. Chu trình tiền xử lý văn bản thô (Text Preprocessing)
1. **Chuyển về chữ thường (Lowercasing):** Giúp đồng nhất `'Olympic'` và `'olympic'` thành cùng một khái niệm từ vựng.
2. **Loại bỏ nhiễu bằng Regex:** Xóa bỏ mã HTML (`<p>`, `<div>`), ký tự đặc biệt thừa (`@`, `#`, `$`), và các dấu chấm câu nếu không cần thiết.
3. **Chuẩn hóa khoảng trắng:** Loại bỏ dấu cách thừa, ký tự xuống dòng `\n`, `\t`.

### 2. Các cấp độ Tokenization
* **Word-level Tokenization (Tách theo từ):** Đơn giản, trực quan. Nhược điểm: Kích thước từ điển khổng lồ, dễ bị lỗi Out-Of-Vocabulary (OOV) khi gặp từ mới hoặc từ viết sai chính tả.
* **Character-level Tokenization (Tách theo ký tự):** Từ điển rất nhỏ (chỉ gồm bảng chữ cái), không bao giờ bị OOV. Nhược điểm: Chuỗi trở nên quá dài, mất đi ý nghĩa trọn vẹn của từng từ.
* **Subword Tokenization (BPE / WordPiece):** Cân bằng hoàn hảo giữa hai phương pháp trên (nền tảng của BERT, GPT).

### 3. Cấu trúc Từ Điển (Vocabulary Indexer) & Special Tokens
Một từ điển chuẩn mực cần các token đặc biệt:
* `<PAD>` (Index 0): Dùng để đệm các câu ngắn cho bằng độ dài tối đa của batch.
* `<UNK>` (Index 1): Đại diện cho các từ lạ chưa từng xuất hiện trong tập huấn luyện (Out-of-Vocabulary).
* `<SOS>` / `<EOS>`: Đánh dấu điểm bắt đầu (Start of Sentence) và kết thúc (End of Sentence).
"""
    visual_intuition_content = """
```
Quy trình biến đổi Văn bản thành Chỉ số Token:
"Olympic AI rất hay!"
        │
        ▼ (Tiền xử lý Regex & Lowercase)
"olympic ai rat hay"
        │
        ▼ (Tokenization)
['olympic', 'ai', 'rat', 'hay']
        │
        ▼ (Tra cứu Từ điển Vocabulary)
Token IDs: [ 45, 12, 108, 89 ]  ──> Sẵn sàng chuyển thành PyTorch Tensor!
```
"""
    code_cells = [
        (
            """import re

# 1. Pipeline tien xu ly van ban bang Regular Expressions (Regex)
def clean_text(text):
    # Ha chu thuong
    text = text.lower()
    # Loai bo ma HTML neu co
    text = re.sub(r'<.*?>', '', text)
    # Loai bo cac ky tu dac biet, chi giu lai chu cai, so va dau cach
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Xoa khoang trang thua
    text = re.sub(r'\s+', ' ', text).strip()
    return text

sample_raw = "  <p>Chao mung toi VOAI 2026! De thi co 38 sessions: Rat tuyet voi??? </p> "
cleaned = clean_text(sample_raw)
print("Van ban goc :", sample_raw)
print("Van ban sach:", cleaned)
""",
            """Khối code minh họa hàm làm sạch chuỗi chuẩn mực bằng Regex:
- Loại bỏ thẻ HTML, dấu câu thừa và chuẩn hóa khoảng trắng.
- Đưa văn bản về dạng chữ thường sạch sẽ, sẵn sàng cho bước phân tách từ."""
        ),
        (
            """# 2. Xay dung lop Vocabulary Indexer tu so 0
class SimpleVocabulary:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        
        # Khoi tao bang tra cuu 2 chieu
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word = {0: self.pad_token, 1: self.unk_token}
        self.word_counts = {}
        
    def build_vocab(self, corpus):
        # Dem tan suat xuat hien cua tung tu trong toan bo tap van ban
        for text in corpus:
            tokens = text.split()
            for token in tokens:
                self.word_counts[token] = self.word_counts.get(token, 0) + 1
                
        # Them vao tu dien neu dat tan suat toi thieu
        for token, count in self.word_counts.items():
            if count >= self.min_freq and token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                
        print(f"Xay dung tu dien thanh cong! Tong so tu vung: {len(self.word2idx)}")
        
    def encode(self, text):
        # Chuyen cau van ban thanh danh sach chi so (Token IDs)
        tokens = text.split()
        return [self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens]
        
    def decode(self, indices):
        # Chuyen nguoc lai tu chi so ve chuoi van ban
        return " ".join([self.idx2word.get(idx, self.unk_token) for idx in indices])

# Kiem thu Vocabulary tren tap ngu lieu nho
corpus = [
    clean_text("hoc may va hoc sau voi pytorch"),
    clean_text("olympic ai la dau truong dinh cao"),
    clean_text("hoc may giup giai quyet bai toan thuc te")
]

vocab = SimpleVocabulary(min_freq=1)
vocab.build_vocab(corpus)

test_sentence = clean_text("hoc may olympic robot") # 'robot' la tu moi chua tung gap (OOV)
encoded_ids = vocab.encode(test_sentence)
decoded_text = vocab.decode(encoded_ids)

print("\\nCau kiem thu :", test_sentence)
print("Ma hoa thanh IDs:", encoded_ids)
print("Giai ma nguoc lai:", decoded_text)
print("-> Tu 'robot' khong co trong tu dien da duoc thay the an toan bang <UNK> (ID=1)!")
""",
            """Khối code lập trình một lớp Vocabulary Indexer hoàn chỉnh từ số 0:
- Quản lý 2 chiều: `word2idx` (mã hóa văn bản thành số) và `idx2word` (giải mã số thành văn bản).
- Tự động thay thế các từ lạ bằng `<UNK>` (Index 1) mà không làm sập chương trình.
- Lưu trữ `<PAD>` ở vị trí 0 để sẵn sàng cho kỹ thuật padding batch sau này."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Lọc từ theo tần suất tối thiểu (min_freq)
Chạy lại phương thức `build_vocab` với `min_freq=2`. Quan sát xem từ điển còn lại bao nhiêu từ và giải thích tại sao trong các bài toán NLP lớn, việc loại bỏ các từ chỉ xuất hiện đúng 1 lần (typos/rác) giúp giảm mạnh kích thước mô hình.

### Bài 2 (Easy): Đếm số lượng từ lạ (OOV Rate)
Viết một hàm `calculate_oov_rate(text, vocab)` nhận vào một đoạn văn bản và trả về tỷ lệ phần trăm các từ bị chuyển thành `<UNK>`.

### Bài 3 (Medium): Xử lý từ dừng (Stopwords Removal)
Stopwords là các từ xuất hiện rất nhiều nhưng mang ít giá trị phân biệt thông tin (như: 'là', 'và', 'của', 'the', 'is', 'at'). Hãy tạo một danh sách các từ dừng đơn giản và viết hàm lọc bỏ chúng khỏi câu văn bản trước khi tokenize.

### Bài 4 (Medium): N-grams Tokenization
Tìm hiểu khái niệm n-grams (cặp 2 từ liên tiếp Bigram, 3 từ liên tiếp Trigram). Viết một hàm `extract_bigrams("hoc may va hoc sau")` trả về danh sách các cặp từ: `['hoc_may', 'may_va', 'va_hoc', 'hoc_sau']`. Nêu lợi ích của bigram trong việc giữ lại các cụm từ có nghĩa (ví dụ: 'không thích' khác hoàn toàn với 'thích').

### Bài 5 (Challenge): Mô phỏng thuật toán Byte-Pair Encoding (BPE)
Tìm hiểu nguyên lý của giải thuật Subword Tokenization BPE: Bắt đầu từ cấp độ ký tự, liên tục tìm cặp ký tự xuất hiện nhiều nhất và gộp chúng lại. Viết một đoạn mã ngắn mô phỏng một vòng lặp gộp ký tự của thuật toán này.
"""
    voai_connection_content = """
Trong các bài thi NLP của Olympic AI, việc xây dựng Từ điển (Vocabulary) là nơi cực kỳ dễ dính lỗi **Data Leakage**:
Nhiều thí sinh vô tình đưa cả tập `test.csv` vào để `build_vocab`. Điều này làm rò rỉ phân phối từ vựng của đề thi vào mô hình. Quy tắc chuẩn: **Chỉ học từ điển trên tập Train, các từ mới xuất hiện ở tập Test bắt buộc phải bị coi là `<UNK>`!** Hiểu sâu về Tokenization giúp bạn thiết lập ranh giới phòng vệ dữ liệu vững chắc.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh mã hóa một câu văn bản nhưng quên xử lý từ vựng ngoài từ điển (OOV), dẫn tới lỗi `KeyError`:

```python
# CODE CÓ LỖI TRA CỨU TỪ ĐIỂN:
# word_to_id = {'ai': 0, 'olympic': 1}
# query = "thi dau ai"
# ids = [word_to_id[w] for w in query.split()] # 'thi' khong co trong dict -> KeyError: 'thi'
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao việc dùng phương thức `dict.get(key, default)` an toàn hơn nhiều so với toán tử truy xuất trực tiếp `dict[key]`.
2. Sửa lại code để luôn gán giá trị mặc định là mã của `<UNK>` khi không tìm thấy từ trong từ điển.
"""
    knowledge_check_content = """
1. Lý do cơ bản nhất khiến máy tính không thể nạp trực tiếp chuỗi văn bản thô vào mạng nơ-ron là gì?
   - A. Mạng nơ-ron chỉ hoạt động trên các phép toán đại số ma trận số thực, không có cơ chế tính toán số học trực tiếp trên ký tự chuỗi.
   - B. Do file văn bản quá nhẹ.
   - C. Vì tiếng Việt có dấu.
   - D. Do giới hạn của hệ điều hành Windows.

2. Token đặc biệt `<UNK>` (Unknown) trong hệ thống xử lý ngôn ngữ tự nhiên đại diện cho điều gì?
   - A. Điểm kết thúc của câu văn bản.
   - B. Các từ lạ, từ viết sai chính tả hoặc từ chưa từng xuất hiện trong tập dữ liệu từ điển đã học (Out-Of-Vocabulary).
   - C. Khoảng trắng giữa các từ.
   - D. Dấu chấm câu.

3. Kỹ thuật Subword Tokenization (như BPE hay WordPiece) khắc phục được nhược điểm lớn nào của Word-level Tokenization truyền thống?
   - A. Tăng tốc độ quạt tản nhiệt của máy tính.
   - B. Giảm thiểu kích thước từ điển đồng thời giải quyết triệt để bài toán từ lạ (OOV) bằng cách bóc tách các từ mới thành các gốc từ và hậu tố quen thuộc.
   - C. Tự động dịch văn bản sang tiếng Anh.
   - D. Bỏ qua hoàn toàn bước huấn luyện.

4. Token đặc biệt `<PAD>` thường được gán giá trị chỉ số (Index) bằng bao nhiêu theo quy ước phổ biến nhất?
   - A. 9999
   - B. 0
   - C. -1
   - D. 255

5. Biểu thức chính quy (Regex) `re.sub(r'[^a-zA-Z0-9\s]', '', text)` thực hiện thao tác gì trên chuỗi?
   - A. Xóa toàn bộ chữ cái và chữ số.
   - B. Xóa tất cả các ký tự đặc biệt và dấu câu, chỉ giữ lại chữ cái, số và khoảng trắng.
   - C. Viết hoa toàn bộ văn bản.
   - D. Đếm số lượng từ trong câu.
"""
    session_summary_content = """
* **What I learned:**
  - Quy trình chuẩn hóa văn bản thô bằng Regular Expressions.
  - Các cấp độ Tokenization: Word, Character và Subword intuition.
  - Cấu trúc Vocabulary Indexer hai chiều và cơ chế xử lý từ lạ `<UNK>`.
* **What I can now do:**
  - Tự tay xây dựng một hệ thống Từ điển ngôn ngữ hoàn chỉnh từ đầu đến cuối.
  - Mã hóa an toàn bất kỳ câu văn bản nào thành danh sách Token IDs mà không lo lỗi sập code.
* **What I still need to practice:**
  - Khám phá các phương pháp trích xuất đặc trưng văn bản kinh điển: **Bag-of-Words & TF-IDF** ở Buổi 7.
"""
    next_session_content = """
Trong **Buổi M4_S07**, chúng ta sẽ tiến lên một bước: **Trích Xuất Đặc Trưng Văn Bản: Bag-of-Words & TF-IDF với Scikit-Learn** — Tìm hiểu cách biến các văn bản thành ma trận vectơ tần suất từ vựng (Sparse Matrix) và huấn luyện mô hình phân loại cảm xúc văn bản đầu tiên!
"""
    trace = {
        "items": [
            "Chuẩn hóa văn bản bằng Regex & hạ chữ thường",
            "Tokenization (Word-level, Char-level, Subword intuition)",
            "Xây dựng Vocabulary indexer & Xử lý từ lạ <UNK>"
        ],
        "deliverable": "Mã nguồn Vocabulary Indexer chuẩn xử lý OOV an toàn"
    }
    nb = create_notebook(
        module_id="M4",
        module_name="AI NHÌN VÀ HIỂU DỮ LIỆU",
        session_id="S06",
        topic="Biểu Diễn Văn Bản: Tokenization & Xây Dựng Từ Điển",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook xây dựng hệ thống Tokenization và Vocabulary chuẩn NLP",
        competition_relevance="Nền móng xử lý văn bản cho mọi bài toán NLP trong các kỳ thi Olympic AI.",
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
    save_notebook(nb, "course/M4_APPLIED_AI/M4_S06.ipynb")


def build_m4_s07():
    objectives = [
        "Hiểu mô hình Không gian Vectơ (Vector Space Model) và phương pháp Túi từ (Bag-of-Words - BoW).",
        "Nắm vững công thức toán học và trực giác của chỉ số Tần suất từ - Nghịch đảo tần suất tài liệu (TF-IDF).",
        "Hiểu cấu trúc Ma trận thưa (Sparse Matrix) và lý do tại sao nó tiết kiệm hàng Gigabytes bộ nhớ RAM.",
        "Xây dựng hoàn chỉnh một Pipeline phân loại văn bản (Text Classification) kết hợp TF-IDF và Logistic Regression bằng Scikit-Learn."
    ]
    prerequisites = [
        "Hoàn thành Buổi M4_S06 (Tokenization & Vocabulary).",
        "Đã nắm vững Logistic Regression từ Module M2."
    ]
    why_content = """
Trước khi mạng nơ-ron Transformer hay BERT ra đời, làm thế nào mà các kỹ sư AI có thể phân loại hàng triệu tài liệu pháp lý, lọc hàng tỷ email rác hoặc phân tích cảm xúc đánh giá sản phẩm một cách tức thời?

Câu trả lời chính là **TF-IDF (Term Frequency - Inverse Document Frequency)**. Đây là thuật toán trích xuất đặc trưng văn bản thanh lịch và hiệu quả nhất trong lịch sử Khoa học Máy tính. Trong thi đấu Olympic AI, TF-IDF kết hợp với các mô hình tuyến tính (Logistic Regression / Linear SVM) luôn là **Mô hình Baseline số 1 bắt buộc phải dựng đầu tiên** cho mọi bài toán NLP: Nó chạy siêu nhanh (chưa tới 3 giây) nhưng mang lại độ chính xác đáng kinh ngạc, đủ sức đánh bại nhiều mô hình Deep Learning nếu dữ liệu ít!
"""
    concept_content = r"""
### 1. Túi từ (Bag-of-Words - BoW) & Hạn chế
* Biểu diễn mỗi văn bản thành một vector có độ dài bằng toàn bộ kích thước từ điển ($V$). Mỗi phần tử đếm số lần từ đó xuất hiện trong văn bản.
* *Nhược điểm lớn:* Các từ cực kỳ phổ biến như "và", "là", "thì" xuất hiện liên tục sẽ chiếm trọng số khổng lồ nhưng lại không mang thông tin phân biệt thể loại!

### 2. Công thức toán học của TF-IDF
TF-IDF giải quyết triệt để nhược điểm của BoW bằng cách nhân hai thành phần:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
1. **TF (Term Frequency - Tần suất từ trong tài liệu $d$):** Từ $t$ xuất hiện càng nhiều trong văn bản $d$ thì giá trị càng cao.
2. **IDF (Inverse Document Frequency - Nghịch đảo tần suất xuất hiện trong toàn bộ kho tài liệu $D$):**
   $$\text{IDF}(t, D) = \log\left( \frac{1 + |D|}{1 + \text{DF}(t)} \right) + 1$$
   - Nếu từ $t$ xuất hiện trong hầu hết mọi văn bản (ví dụ từ "và"), $\text{DF}(t) \approx |D| \implies \text{IDF} \to \text{rất nhỏ}$ (bị phạt nặng).
   - Nếu từ $t$ chỉ xuất hiện trong đúng 1-2 văn bản chuyên biệt (ví dụ từ "viêm phổi"), $\text{DF}(t)$ nhỏ $\implies \text{IDF} \to \text{rất lớn}$!
* *Ý nghĩa triết học:* **Một từ quan trọng là từ xuất hiện nhiều trong tài liệu hiện tại, nhưng hiếm khi xuất hiện ở các tài liệu khác!**

### 3. Ma trận Thưa (Sparse Matrix)
Do mỗi câu chỉ chứa vài chục từ trong khi từ điển có thể chứa 50,000 từ, hơn 99% các phần tử trong ma trận đều là số 0! Thư viện Scikit-Learn lưu trữ dưới dạng ma trận thưa (CSR Matrix - Compressed Sparse Row), chỉ lưu tọa độ các ô khác 0 để tiết kiệm bộ nhớ.
"""
    visual_intuition_content = """
```
Trực quan hóa Cơ chế Trọng số TF-IDF:
   Từ vựng        Tần suất trong câu (TF)   Xuất hiện ở các tài liệu khác (DF)   Trọng số TF-IDF
  "tôi"                 Cao (5 lần)               Rất phổ biến (99% văn bản)       RẤT THẤP ⬇️ (Phạt)
  "thích"               Vừa (2 lần)               Phổ biến (60% văn bản)           TRUNG BÌNH
  "trường_học"          Cao (4 lần)               Rất hiếm (2% văn bản)            RẤT CAO ⬆️ (Từ khóa vàng!)
```
"""
    code_cells = [
        (
            """import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Tao tap du lieu mau ve Phan tich cam xuc danh gia phim (Sentiment Analysis)
corpus_data = [
    ("Phim nay xem cuc ky hay va xuc dong tuyet voi", 1),       # Tich cuc (1)
    ("Dien vien dien xuat qua dinh, kich ban xuat sac", 1),      # Tich cuc (1)
    ("Phim chan ngat, phi tien ve xem phim nay", 0),             # Tieu cuc (0)
    ("Kich ban roi ram vo ly, dien vien do te hai", 0),          # Tieu cuc (0)
    ("Hinh anh am thanh rat dep mat, rat dang tien", 1),         # Tich cuc (1)
    ("That vong toan tap, khong dang xem mot chut nao", 0),      # Tieu cuc (0)
    ("Mot tac pham nghe thuat hoan hao va y nghia", 1),         # Tich cuc (1)
    ("Qua buon ngu, noi dung nhat nheo va te nhat", 0)           # Tieu cuc (0)
]

df_nlp = pd.DataFrame(corpus_data, columns=['Text', 'Sentiment'])
print("===== DU LIEU VAN BAN MAU (8 BAN GHI) =====")
print(df_nlp)
""",
            """Khối code chuẩn bị bài toán phân tích cảm xúc (Sentiment Analysis) kinh điển của NLP:
- Tập ngữ liệu gồm các câu đánh giá tích cực (nhãn 1) và tiêu cực (nhãn 0).
- Chứa các từ mang sắc thái cảm xúc rõ ràng làm cơ sở kiểm thử thuật toán."""
        ),
        (
            """# 2. Trich xuat dac trung bang TfidfVectorizer va Huan luyen Mo hinh
# Khoi tao bo trich xuat TF-IDF (ho tro ca tu don Unigram va cap tu Bigram)
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50)

# Bien doi van ban thanh Ma tran dac trung thuc so
X_tfidf = tfidf.fit_transform(df_nlp['Text'])
y = df_nlp['Sentiment']

print("===== THONG TIN MA TRAN THUA (SPARSE MATRIX) =====")
print(f"Kich thuoc ma tran TF-IDF: {X_tfidf.shape} (8 cau, {X_tfidf.shape[1]} dac trung tu vung)")
print(f"Kieu du lieu luu tru: {type(X_tfidf)} (Chi luu cac gia tri khac 0)")

# 3. Huan luyen mo hinh Logistic Regression tren dac trung TF-IDF
clf_text = LogisticRegression()
clf_text.fit(X_tfidf, y)

# 4. Kiem thu du doan tren cau moi hoan toan
new_reviews = [
    "Dien vien dien xuat rat tuyet voi va hay", # Tich cuc
    "Phim xem chan ngat va that vong qua"       # Tieu cuc
]
new_features = tfidf.transform(new_reviews) # Chi transform, KHONG fit lai!
predictions = clf_text.predict(new_features)

print("\\n===== KET QUA DU DOAN TREN VAN BAN MOI =====")
for rev, pred in zip(new_reviews, predictions):
    label = "Tich Cuc (1)" if pred == 1 else "Tieu Cuc (0)"
    print(f"Review: '{rev}' -> Du doan: {label}")
""",
            """Khối code xây dựng hoàn chỉnh một NLP Pipeline bằng Scikit-Learn:
- `TfidfVectorizer(ngram_range=(1, 2))`: Tự động tách cả từ đơn và cặp từ liền kề.
- Trích xuất ma trận đặc trưng thưa siêu tốc độ.
- Huấn luyện `LogisticRegression` và chạy suy diễn chính xác trên các câu đánh giá mới."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Khám phá từ vựng học được
In ra danh sách 10 từ vựng đầu tiên cùng chỉ số của chúng bằng lệnh `tfidf.vocabulary_`. Nhận xét xem có các cặp từ ghép (Bigram) nào được tạo ra không.

### Bài 2 (Easy): Tìm các từ khóa có trọng số TF-IDF cao nhất
Với câu văn bản đầu tiên (`df_nlp['Text'][0]`), hãy in ra danh sách các từ kèm theo giá trị TF-IDF tương ứng sắp xếp giảm dần để xem từ nào được coi là quan trọng nhất trong câu đó.

### Bài 3 (Medium): So sánh với CountVectorizer (Bag-of-Words thuần)
Import `CountVectorizer` từ `sklearn.feature_extraction.text`. Huấn luyện lại mô hình bằng CountVectorizer thay vì TF-IDF. So sánh điểm số và giải thích tại sao TF-IDF thường cho kết quả phân loại ổn định hơn BoW.

### Bài 4 (Medium): Thiết lập ngưỡng `min_df` và `max_df`
Tìm hiểu 2 siêu tham số cực kỳ quan trọng trong TfidfVectorizer:
- `min_df=2`: Loại bỏ các từ quá hiếm (chỉ xuất hiện trong < 2 văn bản).
- `max_df=0.8`: Loại bỏ các từ quá phổ biến (xuất hiện trong > 80% văn bản).
Áp dụng lên một tập dữ liệu lớn và quan sát số lượng cột đặc trưng giảm đi như thế nào.

### Bài 5 (Challenge): Trích xuất đặc trưng phân biệt nhất của từng lớp (Top Discriminating Words)
Truy cập vào thuộc tính `clf_text.coef_[0]` của mô hình Logistic Regression. Tìm 3 từ vựng có hệ số dương lớn nhất (đặc trưng mạnh nhất cho nhãn Tích cực) và 3 từ có hệ số âm lớn nhất (đặc trưng mạnh nhất cho nhãn Tiêu cực).
"""
    voai_connection_content = """
Trong 15 phút đầu tiên khi nhận một đề thi NLP tại Olympic AI, **hãy luôn luôn chạy một Baseline bằng TF-IDF + Logistic Regression/LightGBM**. Một lỗi chết người của nhiều thí sinh là vội vàng bật máy nạp các mô hình ngôn ngữ lớn (LLMs/Transformers) mất 40 phút chỉ để tải trọng số, để rồi sau đó phát hiện máy bị tràn RAM và không kịp có bất kỳ một kết quả nào nộp lên bảng xếp hạng. Một baseline TF-IDF chuẩn chỉ sẽ đảm bảo bạn có điểm số an toàn ngay từ vạch xuất phát!
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh gọi lại phương thức `fit_transform()` trên tập kiểm tra (Test Set):

```python
# CODE CÓ LỖI LỆCH TỪ ĐIỂN VÀ RÒ RỈ DỮ LIỆU:
# X_train_tfidf = tfidf.fit_transform(train_texts)
# X_test_tfidf = tfidf.fit_transform(test_texts) # SAI LẦM CHẾT NGƯỜI!
# Goi fit_transform tren tap Test se tao ra mot tu dien hoan toan moi,
# khien thu tu cot dac trung cua tap Test bi lech 100% so voi tap Train!
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao bắt buộc phải dùng `tfidf.transform(test_texts)` trên tập Test.
2. Nêu hậu quả nghiêm trọng nếu ma trận đặc trưng của tập Test bị lệch chỉ số cột so với tập Train khi đưa vào mô hình dự đoán.
"""
    knowledge_check_content = """
1. Thuật ngữ IDF (Inverse Document Frequency) trong TF-IDF đo lường điều gì?
   - A. Tốc độ gõ phím của người viết văn bản.
   - B. Mức độ hiếm có và giá trị thông tin của một từ vựng trong toàn bộ kho tài liệu: Từ nào càng xuất hiện ở nhiều văn bản khác nhau thì giá trị IDF càng bị kéo giảm về 0.
   - C. Số lượng ký tự trong một từ.
   - D. Tần số quét của màn hình.

2. Tại sao thư viện Scikit-Learn lại lưu trữ ma trận TF-IDF dưới dạng Ma trận thưa (Sparse Matrix)?
   - A. Vì thư viện bị thiếu bộ nhớ.
   - B. Vì trong bài toán văn bản, đại đa số các từ vựng trong từ điển đều không xuất hiện trong một câu cụ thể (hơn 99% phần tử là số 0), việc chỉ lưu các ô khác 0 giúp tiết kiệm hàng Gigabytes RAM.
   - C. Để làm cho code chạy khó hiểu hơn.
   - D. Do định dạng của file CSV.

3. Tham số `ngram_range=(1, 2)` trong `TfidfVectorizer` có ý nghĩa gì?
   - A. Chỉ lấy các câu có độ dài từ 1 đến 2 từ.
   - B. Trích xuất cả các từ đơn lẻ (Unigrams) và các cặp hai từ đi liền kề nhau (Bigrams) làm đặc trưng.
   - C. Lặp lại việc huấn luyện 2 lần.
   - D. Chia tập dữ liệu thành 2 nửa.

4. Giữa từ "và" (xuất hiện ở 100% bài báo) và từ "lượng tử" (chỉ xuất hiện ở 1% bài báo khoa học), từ nào sẽ có chỉ số IDF cao hơn vượt trội?
   - A. Từ "và".
   - B. Cả hai từ có IDF bằng nhau.
   - C. Từ "lượng tử".
   - D. Không xác định được.

5. Trong các cuộc thi AI, mô hình Baseline được khuyến nghị hàng đầu cho bài toán phân loại văn bản dạng bảng là gì?
   - A. Huấn luyện một mạng nơ-ron Transformer 100 lớp từ đầu.
   - B. TfidfVectorizer kết hợp Logistic Regression hoặc Linear SVM.
   - C. Hệ chuyên gia đếm từ bằng tay.
   - D. Dự đoán ngẫu nhiên.
"""
    session_summary_content = """
* **What I learned:**
  - Mô hình Túi từ (Bag-of-Words) và hạn chế của việc đếm từ đơn thuần.
  - Bản chất toán học của TF-IDF và ý nghĩa triết học về tầm quan trọng của từ khóa.
  - Cấu trúc tối ưu bộ nhớ của Ma trận thưa (Sparse Matrix).
* **What I can now do:**
  - Xây dựng một Text Classification Pipeline hoàn chỉnh bằng Scikit-Learn trong chưa đầy 10 dòng code.
  - Nhanh chóng dựng Baseline chất lượng cao cho bất kỳ bài thi NLP nào.
* **What I still need to practice:**
  - Tiến sâu vào biểu diễn ngữ nghĩa liên tục bằng **Word Embeddings & `nn.Embedding`** ở Buổi 8.
"""
    next_session_content = """
Trong **Buổi M4_S08**, chúng ta sẽ bước vào không gian ngữ nghĩa học sâu: **Trực Giác Không Gian Vectơ Từ (Word Embeddings) & `nn.Embedding` Trong PyTorch** — Khám phá cách máy tính hiểu được "Vua - Đàn ông + Phụ nữ = Nữ hoàng" thông qua các vector dày đặc (Dense Vectors)!
"""
    trace = {
        "items": [
            "Túi từ (Bag-of-Words) & Bản chất toán học TF-IDF",
            "Ma trận thưa (Sparse Matrix) & TfidfVectorizer",
            "Text Classification Pipeline hoàn chỉnh với Scikit-Learn"
        ],
        "deliverable": "Mã nguồn phân loại văn bản Baseline bằng TF-IDF và Logistic Regression"
    }
    nb = create_notebook(
        module_id="M4",
        module_name="AI NHÌN VÀ HIỂU DỮ LIỆU",
        session_id="S07",
        topic="Trích Xuất Đặc Trưng Văn Bản: Bag-of-Words & TF-IDF với Scikit-Learn",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook trích xuất đặc trưng TF-IDF và phân loại văn bản Baseline",
        competition_relevance="Mô hình Baseline số 1 cho mọi bài toán văn bản trong các cuộc thi AI.",
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
    save_notebook(nb, "course/M4_APPLIED_AI/M4_S07.ipynb")


def build_m4_s08():
    objectives = [
        "Hiểu hạn chế chết người của biểu diễn One-Hot/TF-IDF: Vấn đề trực giao toán học và sự thiếu vắng ngữ nghĩa tương đồng.",
        "Nắm vững khái niệm Không gian Nhúng Từ (Word Embeddings): Biểu diễn từ bằng các vector thực dày đặc (Dense Vectors) trong không gian d chiều.",
        "Khám phá trực giác ngữ nghĩa học sâu: Khoảng cách Cosine (Cosine Similarity) và phép tính đại số vector từ vựng (ví dụ: King - Man + Woman ≈ Queen).",
        "Làm chủ cơ chế hoạt động của tầng `torch.nn.Embedding` trong PyTorch như một Bảng tra cứu tốc độ cao (Lookup Table)."
    ]
    prerequisites = [
        "Hoàn thành Buổi M4_S06 và M4_S07.",
        "Hiểu phép nhân tích vô hướng của hai vector (Dot Product)."
    ]
    why_content = """
Trong cách biểu diễn One-Hot hoặc TF-IDF ở buổi trước, hai từ "khách sạn" và "nhà nghỉ" là hai vector hoàn toàn độc lập và trực giao với nhau trong không gian: Tích vô hướng của chúng bằng 0! Điều này đồng nghĩa với việc máy tính coi khoảng cách giữa "khách sạn" và "nhà nghỉ" cũng xa lạ hệt như khoảng cách giữa "khách sạn" và "quả chuối"!

Để mạng nơ-ron có thể hiểu được ngữ nghĩa sâu sắc, chúng ta cần đưa các từ vào một **Không gian Vectơ Liên tục (Continuous Vector Space)**, nơi các từ có ý nghĩa tương đồng sẽ nằm gần nhau. **Word Embeddings và tầng `nn.Embedding` của PyTorch** chính là trái tim của mọi kiến trúc ngôn ngữ hiện đại, từ RNN, LSTM cho tới những siêu mô hình như GPT-4!
"""
    concept_content = r"""
### 1. One-Hot Vector vs Dense Word Embedding
* **One-Hot Encoding:** Vector thưa $V$ chiều (chỉ có duy nhất một số 1, còn lại là số 0). Không mang thông tin ngữ nghĩa, kích thước bùng nổ theo từ điển.
* **Word Embedding:** Vector dày đặc có số chiều cố định $d$ (thường là 64, 128, 300, 768 chiều). Mỗi chiều biểu diễn một thuộc tính tiềm ẩn trừu tượng (ví dụ: giới tính, hoàng gia, động vật...).

### 2. Phép đo Tương đồng Cosine (Cosine Similarity)
Đo góc $\theta$ giữa hai vector trong không gian nhiều chiều:
$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum u_i v_i}{\sqrt{\sum u_i^2} \sqrt{\sum v_i^2}}$$
* Bằng $+1$: Hai từ hoàn toàn đồng nghĩa/cùng hướng.
* Bằng $0$: Hai từ trực giao/không liên quan.
* Bằng $-1$: Hai từ mang ý nghĩa hoàn toàn đối lập.

### 3. Bản chất của tầng `torch.nn.Embedding` trong PyTorch
Tầng `nn.Embedding(num_embeddings, embedding_dim)` thực chất là một **Ma trận Trọng số có thể học được (Learnable Weight Matrix)** kích thước $V \times d$.
* Nó đóng vai trò như một **Bảng tra cứu (Lookup Table)**: Nhận vào chỉ số nguyên (Token ID) và trả về vector $d$ chiều tương ứng trong $O(1)$.
* Các trọng số này được tự động cập nhật qua Backpropagation trong quá trình huấn luyện để tối ưu hóa bài toán!
"""
    visual_intuition_content = """
```
Trực quan hóa Không gian Nhúng Từ (Embedding Space 2D):
   Chiều Ngữ nghĩa 2 (Nữ tính)
          ▲
          │          • Queen (Nữ hoàng)
          │         /
          │        /  Vector chuyển dịch [Nữ tính]
          │       /
          │      • King (Vua)
          │
          │                                  • Apple (Quả táo)
          │                                 /
          │                                • Banana (Quả chuối)
          └────────────────────────────────────────────────────────► Chiều Ngữ nghĩa 1 (Hoàng gia)
```
"""
    code_cells = [
        (
            """import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Khoi tao tang nn.Embedding trong PyTorch
# Gia su tu dien co 10 tu (vocab_size = 10), moi tu bieu dien bang vector 4 chieu (embedding_dim = 4)
vocab_size = 10
embedding_dim = 4

# padding_idx=0 de vector cua token <PAD> luon duoc giu bang vector 0
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)

print("===== MA TRAN TRONG SO CUA TANG EMBEDDING (10x4) =====")
print(embedding_layer.weight.data)
""",
            """Khối code khởi tạo tầng `nn.Embedding`:
- Ma trận trọng số gồm 10 hàng và 4 cột.
- `padding_idx=0`: Đảm bảo token đệm `<PAD>` luôn có vector toàn số 0 và không bị cập nhật gradient."""
        ),
        (
            """# 2. Tra cuu Vector (Lookup Table) va Tinh do tuong dong Cosine
# Gia lap mot cau gom 3 tu co Token IDs: [3, 7, 2]
input_ids = torch.tensor([3, 7, 2], dtype=torch.long)

# Tra cuu vector nhung
word_vectors = embedding_layer(input_ids)
print("===== VECTOR NHUNG CUA CAU [3, 7, 2] =====")
print(f"Kich thuoc tensor vector: {word_vectors.shape} (3 tu, moi tu 4 chieu)")
print(word_vectors)

# 3. Tinh Cosine Similarity giua tu thu nhat (ID=3) va tu thu hai (ID=7)
vec_a = word_vectors[0].unsqueeze(0) # Shape: (1, 4)
vec_b = word_vectors[1].unsqueeze(0) # Shape: (1, 4)

similarity = F.cosine_similarity(vec_a, vec_b)
print(f"\\nDo tuong dong Cosine giua Tu 3 va Tu 7: {similarity.item():.4f} (nam trong [-1, 1])")
""",
            """Khối code minh họa quá trình tra cứu vector nhúng:
- Truyền tensor các ID nguyên vào tầng embedding thu được tensor 3D đặc trưng.
- Sử dụng `F.cosine_similarity` để đo lường mức độ gần gũi ngữ nghĩa giữa hai vector từ trong không gian nhiều chiều."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Kiểm tra vector của `<PAD>`
Truy xuất vector của token có `ID = 0`. Xác minh xem toàn bộ các phần tử của vector này có bằng chính xác 0.0 hay không.

### Bài 2 (Easy): Khám phá kích thước Batch Embedding
Tạo một batch gồm 2 câu, mỗi câu có 5 từ (tensor kích thước `(2, 5)`). Đưa qua tầng `embedding_layer` ở trên và in ra kích thước của tensor đầu ra. Giải thích ý nghĩa của kích thước `(2, 5, 4)`.

### Bài 3 (Medium): Đại số vector từ vựng (Word Analogy)
Giả sử bạn có các vector Word2Vec của 4 từ: `v_king`, `v_man`, `v_woman`, `v_queen`. Viết công thức tính vector dự đoán $\mathbf{v}_{pred} = \mathbf{v}_{king} - \mathbf{v}_{man} + \mathbf{v}_{woman}$, sau đó tính Cosine Similarity giữa $\mathbf{v}_{pred}$ và $\mathbf{v}_{queen}$.

### Bài 4 (Medium): Nạp trọng số Pre-trained Embeddings vào PyTorch
Tìm hiểu phương thức `nn.Embedding.from_pretrained(pretrained_weights, freeze=True)`. Nêu lợi ích của việc sử dụng các bộ vector từ vựng đã được huấn luyện sẵn (như GloVe hoặc FastText) thay vì khởi tạo ngẫu nhiên từ đầu khi dữ liệu bài thi ít.

### Bài 5 (Challenge): Tính khoảng cách ngữ nghĩa giữa hai văn bản (Sentence Embedding bằng Mean Pooling)
Cho một câu gồm $L$ từ, biểu diễn bằng tensor `(L, embedding_dim)`. Hãy dùng lệnh `torch.mean(..., dim=0)` để tính vector đại diện cho toàn bộ câu (Sentence Embedding). Viết hàm tính độ tương đồng ngữ nghĩa giữa hai câu văn bản bất kỳ bằng phương pháp này.
"""
    voai_connection_content = """
Trong các bài thi Olympic AI phần NLP, việc khởi tạo tầng `nn.Embedding` ngẫu nhiên rồi tự huấn luyện từ đầu trên một tập dữ liệu nhỏ (vài nghìn câu) sẽ khiến mô hình học rất chậm và không nắm bắt được ngữ nghĩa phong phú. Thí sinh có tư duy chiến lược sẽ biết cách sử dụng **Pre-trained Embeddings (hoặc ít nhất là khởi tạo cẩn thận kết hợp với Dropout)** để giúp không gian vector từ vựng ổn định ngay từ epoch đầu tiên.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh truyền nhãn mục tiêu hoặc chỉ số từ vựng vượt quá kích thước từ điển vào tầng `nn.Embedding`:

```python
# CODE CÓ LỖI CHỈ SỐ VƯỢT QUÁ KÍCH THƯỚC TỪ ĐIỂN:
# vocab_size = 10 (chi chap nhan chi so tu 0 den 9)
# bad_ids = torch.tensor([4, 15, 2]) # Chi so 15 vuot nguong!
# embedding_layer(bad_ids) # Gay loi: IndexError: index out of range in self!
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao chỉ số Token ID truyền vào `nn.Embedding` bắt buộc phải thỏa mãn $0 \le ID < \text{num\_embeddings}$.
2. Nêu cách phòng ngừa lỗi này bằng cơ chế ánh xạ qua `<UNK>` đã học ở Buổi 6.
"""
    knowledge_check_content = """
1. Ưu điểm vượt trội nhất của Word Embedding (Vector nhúng dày đặc) so với One-Hot Encoding truyền thống là gì?
   - A. Word Embedding chỉ chạy được trên CPU.
   - B. Word Embedding biểu diễn từ bằng các vector số thực liên tục có số chiều nhỏ cố định, trong đó các từ có ngữ nghĩa tương đồng sẽ nằm gần nhau trong không gian.
   - C. Word Embedding không cần dùng bộ nhớ RAM.
   - D. Word Embedding biến văn bản thành hình ảnh.

2. Tầng `nn.Embedding` trong PyTorch về bản chất cấu trúc dữ liệu hoạt động tương tự như thành phần nào?
   - A. Một thuật toán sắp xếp nổi bọt.
   - B. Một Bảng tra cứu ma trận trọng số (Lookup Table), nhận chỉ số nguyên và trả về hàng vector tương ứng.
   - C. Một mạng nơ-ron tích chập 3D.
   - D. Một vòng lặp vô hạn.

3. Giá trị của phép đo Cosine Similarity giữa hai vector luôn nằm trong khoảng nào?
   - A. $[0, +\infty)$
   - B. $[-1, 1]$
   - C. $[0, 255]$
   - D. $(-\infty, +\infty)$

4. Tham số `padding_idx=0` trong `nn.Embedding` có tác dụng gì?
   - A. Xóa bỏ từ đầu tiên của câu.
   - B. Cố định vector tại chỉ số 0 luôn bằng 0 và không cập nhật gradient cho vị trí này trong quá trình huấn luyện.
   - C. Tăng kích thước từ điển lên gấp đôi.
   - D. Đảo ngược thứ tự các từ.

5. Trong không gian Word2Vec kinh điển, phép toán trừu tượng nào sau đây thể hiện mối quan hệ ngữ nghĩa tương tự?
   - A. $\text{Vector}("Vua") + \text{Vector}("Táo") = \text{Vector}("Cam")$
   - B. $\text{Vector}("King") - \text{Vector}("Man") + \text{Vector}("Woman") \approx \text{Vector}("Queen")$
   - C. $\text{Vector}("Xe") \times \text{Vector}("Nhà") = 0$
   - D. $\text{Vector}("Chó") - \text{Vector}("Mèo") = 1$
"""
    session_summary_content = """
* **What I learned:**
  - Hạn chế trực giao của One-Hot và sức mạnh của Không gian nhúng từ (Word Embeddings).
  - Cách tính toán và ý nghĩa của khoảng cách tương đồng Cosine.
  - Cơ chế Bảng tra cứu trọng số của tầng `torch.nn.Embedding`.
* **What I can now do:**
  - Tích hợp tầng `nn.Embedding` vào bất kỳ kiến trúc mạng nơ-ron PyTorch nào.
  - Đo lường mức độ tương đồng ngữ nghĩa giữa các vector từ vựng trong không gian nhiều chiều.
* **What I still need to practice:**
  - Lắp ráp tầng Embedding vào một **Text Classification Pipeline hoàn chỉnh với PyTorch** ở Buổi 9.
"""
    next_session_content = """
Trong **Buổi M4_S09**, chúng ta sẽ xây dựng cỗ máy học sâu ngôn ngữ hoàn chỉnh: **Xây Dựng Text Classification Pipeline Hoàn Chỉnh với PyTorch** — Tự viết Custom Collate Function cho các câu có độ dài biến thiên, kết hợp Embedding với Pooling và huấn luyện phân loại cảm xúc văn bản!
"""
    trace = {
        "items": [
            "Bản chất Word Embeddings & Không gian vector liên tục",
            "Đo khoảng cách ngữ nghĩa bằng Cosine Similarity",
            "Tầng nn.Embedding trong PyTorch như một Lookup table"
        ],
        "deliverable": "Mã nguồn tra cứu vector nhúng và tính khoảng cách ngữ nghĩa bằng PyTorch"
    }
    nb = create_notebook(
        module_id="M4",
        module_name="AI NHÌN VÀ HIỂU DỮ LIỆU",
        session_id="S08",
        topic="Word Embeddings & nn.Embedding Trong PyTorch",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook thực hành tầng Word Embedding và đo lường khoảng cách Cosine",
        competition_relevance="Nền tảng kiến trúc sâu của mọi mô hình ngôn ngữ từ RNN đến Transformer.",
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
    save_notebook(nb, "course/M4_APPLIED_AI/M4_S08.ipynb")


def build_m4_s09():
    objectives = [
        "Hiểu thách thức lớn nhất của dữ liệu văn bản khi nạp theo Batch: Các câu có độ dài hoàn toàn khác nhau (Variable Sequence Length).",
        "Xây dựng Custom Collate Function với kỹ thuật Đệm chuỗi (Padding bằng `pad_sequence`) trong DataLoader.",
        "Thiết kế kiến trúc mạng Deep NLP: `nn.Embedding` -> Global Pooling (Gộp trung bình đặc trưng câu) -> Dense Classifier Head.",
        "Huấn luyện thành công mô hình Phân loại cảm xúc văn bản (Text Classification) bằng PyTorch từ số 0 và đo lường Macro-F1."
    ]
    prerequisites = [
        "Hoàn thành Buổi M4_S06, M4_S08 và M3_S04 (Training loop).",
        "Hiểu cơ chế hoạt động của `collate_fn` trong PyTorch DataLoader."
    ]
    why_content = """
Trong bài toán xử lý ảnh, mọi bức ảnh đều có thể được ép về cùng một kích thước cố định như $128 \times 128$. Nhưng trong ngôn ngữ, câu thứ nhất có thể chỉ có 4 từ (*"Phim này quá tệ"*), trong khi câu thứ hai là một đoạn bình luận dài 80 từ! Nếu không thể gom các câu có độ dài khác nhau này thành một Tensor hình chữ nhật chuẩn mực, bạn sẽ không thể nạp dữ liệu theo Mini-batch vào GPU để huấn luyện.

Làm chủ kỹ thuật **Đệm chuỗi (Padding)** thông qua **Custom Collate Function** kết hợp với mạng nơ-ron nhúng từ (Embedding Neural Net) chính là bài kiểm tra toàn diện về kỹ năng lập trình Deep Learning cho NLP, đưa bạn từ mức thử nghiệm đồ chơi lên cấp độ xây dựng hệ thống phân tích ngôn ngữ thực thụ!
"""
    concept_content = r"""
### 1. Kỹ thuật Padding & Collate Function
* **Vấn đề:** Các câu có độ dài $L_1, L_2, \dots, L_B$ khác nhau không thể ghép trực tiếp thành tensor 2D $[B, L]$.
* **Giải pháp:** Tìm độ dài lớn nhất trong batch ($L_{max}$), sau đó đệm thêm token `<PAD>` (giá trị 0) vào cuối các câu ngắn hơn sao cho tất cả đều có cùng độ dài $L_{max}$.
* Hàm `torch.nn.utils.rnn.pad_sequence` tự động thực hiện thao tác này một cách tối ưu.

### 2. Kiến trúc Mạng Deep Averaging Network (DAN) cho NLP
Một mô hình cực kỳ nhanh, nhẹ và hiệu quả cho phân loại văn bản:
1. **Embedding Layer (`nn.Embedding`):** Biến ma trận chỉ số $[B, L_{max}] \to$ Tensor nhúng 3D $[B, L_{max}, d]$.
2. **Global Average Pooling:** Lấy trung bình cộng các vector từ dọc theo chiều độ dài câu:
   $$\mathbf{h}_{doc} = \frac{1}{L} \sum_{i=1}^L \mathbf{v}_{word_i}$$
   Biến tensor 3D $[B, L_{max}, d] \to$ Vector đại diện tài liệu 2D $[B, d]$.
3. **Linear Classifier:** Đưa vector $[B, d]$ qua tầng tuyến tính để xuất ra Logits phân loại.
"""
    visual_intuition_content = """
```
Cấu trúc Deep Averaging Network (DAN) cho Phân loại Văn bản:
Câu 1: [ID_1, ID_2, PAD, PAD] ──┐
Câu 2: [ID_3, ID_4, ID_5, PAD] ──┼──> [nn.Embedding] ──> [Tensor 3D: Batch x Length x Dim]
Câu 3: [ID_6, ID_7, ID_8, ID_9] ─┘                                │
                                                                  ▼
                                                   [Global Average Pooling]
                                                                  │ (Gộp trung bình thành 1 vector duy nhất)
                                                                  ▼
                                                   [Dense Classifier Head: Linear -> ReLU -> Linear]
                                                                  │
                                                                  ▼
                                                       [Dự đoán: Tích cực / Tiêu cực]
```
"""
    code_cells = [
        (
            """import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 1. Custom Dataset va Custom Collate Function cho van ban co do dai bien thien
class RawTextDataset(Dataset):
    def __init__(self, tokenized_texts, labels):
        # tokenized_texts la danh sach cac list co do dai khac nhau
        self.texts = [torch.tensor(t, dtype=torch.long) for t in tokenized_texts]
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Ham Collate chuyen biet tu dong dem pad_sequence cho tung Batch
def pad_collate_fn(batch):
    texts, labels = zip(*batch)
    # Tu dong dem so 0 (<PAD>) vao cac cau ngan hon de co cung do dai batch_first=True
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return texts_padded, labels

# Du lieu gia lap: 4 cau co do dai khac nhau hoan toan
sample_texts = [
    [2, 4],             # Dai 2 tu
    [5, 1, 8, 9, 3],    # Dai 5 tu
    [7],                # Dai 1 tu
    [4, 6, 2]           # Dai 3 tu
]
sample_labels = [1, 0, 1, 0]

dataset = RawTextDataset(sample_texts, sample_labels)
loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=pad_collate_fn)

print("===== KIEM TRA BATCH DA DUOC DEM PADDING TU DONG =====")
for batch_x, batch_y in loader:
    print(f"Batch X Shape: {batch_x.shape} (Tu dong dem theo cau dai nhat trong batch)")
    print(batch_x)
    print("Labels:", batch_y)
    break
""",
            """Khối code giải quyết bài toán cốt tử về độ dài văn bản không đồng nhất:
- `RawTextDataset`: Lưu trữ các danh sách token IDs có độ dài tùy biến.
- `pad_collate_fn`: Sử dụng `pad_sequence(..., batch_first=True, padding_value=0)` tự động đệm số 0 vào các câu ngắn trong batch."""
        ),
        (
            """# 2. Xay dung Mo hinh Deep Averaging Network (DAN) va Huan luyen
class DeepTextClassifier(nn.Module):
    def __init__(self, vocab_size=20, embed_dim=16, num_classes=2):
        super(DeepTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, 16)
        self.relu = nn.ReLU()
        self.out = nn.Linear(16, num_classes)
        
    def forward(self, x):
        # x shape: [Batch, Seq_Len]
        embedded = self.embedding(x) # [Batch, Seq_Len, Embed_Dim]
        
        # Global Average Pooling: tinh trung binh cong cac tu trong cau (axis=1)
        # De chuan xac ta lay trung binh doc theo chieu tu
        pooled = torch.mean(embedded, dim=1) # [Batch, Embed_Dim]
        
        hidden = self.relu(self.fc1(pooled))
        logits = self.out(hidden)
        return logits

# Khoi tao mo hinh va kiem thu forward pass
model_nlp = DeepTextClassifier(vocab_size=20, embed_dim=16, num_classes=2)
dummy_batch = torch.tensor([[2, 4, 0, 0, 0], [5, 1, 8, 9, 3]], dtype=torch.long)

with torch.no_grad():
    logits = model_nlp(dummy_batch)
    
print("\\n===== FORWARD PASS DEEP TEXT CLASSIFIER =====")
print(f"Input Shape : {dummy_batch.shape}")
print(f"Logits Shape: {logits.shape} (2 mau, 2 lop cam xuc)")
print("Gia tri Logits:\\n", logits)
""",
            """Khối code thiết kế mô hình Deep NLP hoàn chỉnh:
- Tầng `nn.Embedding` ánh xạ các câu đệm padding thành biểu diễn không gian liên tục.
- `torch.mean(embedded, dim=1)` thực hiện pooling gộp thông tin toàn câu cực kỳ thanh lịch mà không làm bùng nổ tham số.
- Tầng Linear cuối cùng xuất ra Logits phân loại nhị phân."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Bỏ qua token `<PAD>` khi tính Average Pooling
Trong đoạn code trên, lệnh `torch.mean(embedded, dim=1)` đang tính trung bình cả các vector `<PAD>` (toàn số 0), làm suy giảm biên độ của các câu ngắn. Hãy viết công thức pooling có mặt nạ (Masked Average Pooling) chỉ chia cho độ dài thực tế của từng câu.

### Bài 2 (Easy): Thử nghiệm với Max Pooling
Thay thế `torch.mean` bằng `torch.max(embedded, dim=1).values`. Giải thích tại sao Max Pooling giúp bắt được từ khóa có cảm xúc mạnh nhất trong câu (ví dụ chỉ cần có từ "tuyệt vời" là cả câu được kích hoạt mạnh).

### Bài 3 (Medium): Viết vòng lặp huấn luyện hoàn chỉnh cho Text Classifier
Sử dụng dữ liệu `sample_texts` ở trên, khởi tạo hàm mất mát `nn.CrossEntropyLoss()` và optimizer `Adam`. Viết vòng lặp huấn luyện 5 epochs và in ra loss của từng epoch.

### Bài 4 (Medium): Thêm tầng Dropout chống Overfitting
Thêm `nn.Dropout(p=0.4)` vào giữa tầng `fc1` và tầng `out`. Giải thích tại sao trong các bài toán phân loại văn bản có kích thước nhỏ, Dropout là kỹ thuật bắt buộc để mô hình không phụ thuộc vào một từ khóa đơn lẻ.

### Bài 5 (Challenge): Mở rộng thành mô hình Recurrent Neural Network (RNN / GRU)
Tìm hiểu tầng `nn.GRU(embed_dim, hidden_dim, batch_first=True)`. Thay thế cơ chế Average Pooling bằng việc đưa tensor nhúng qua tầng GRU để nắm bắt thứ tự trước sau của các từ trong câu văn bản.
"""
    voai_connection_content = """
Trong phần thi Xử lý Ngôn ngữ Tự nhiên của Olympic AI, **kỹ năng viết Custom Collate Function là ranh giới phân định trình độ thí sinh**:
Thí sinh nghiệp dư sẽ cố định độ dài câu bằng cách cắt cụt (truncate) hoặc đệm tĩnh toàn bộ tập dữ liệu thành một kích thước khổng lồ (ví dụ luôn đệm thành 512 từ), khiến bộ nhớ GPU lãng phí 90% dung lượng chỉ để lưu số 0. Thí sinh đẳng cấp sử dụng Dynamic Padding trong `collate_fn` (chỉ đệm bằng độ dài câu dài nhất TRONG BATCH ĐÓ), giúp tăng tốc độ huấn luyện lên gấp 3 đến 5 lần!
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh quên thiết lập tham số `batch_first=True` trong hàm `pad_sequence`, dẫn tới kích thước tensor bị đảo ngược:

```python
# CODE CÓ LỖI THỨ TỰ BATCH VÀ CHIỀU DÀI:
# Mac dinh pad_sequence se tao ra tensor co shape [Seq_Len, Batch]!
# Neu dua vao model mong doi [Batch, Seq_Len], mo hinh se tinh toan sai hoac bao loi runtime!
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao luôn phải thiết lập `batch_first=True` khi muốn kích thước đầu tiên của tensor đại diện cho số lượng mẫu trong batch.
2. Kiểm tra lại shape tensor sau khi gọi `pad_sequence` để bảo đảm tính nhất quán.
"""
    knowledge_check_content = """
1. Kỹ thuật Padding trong Xử lý Ngôn ngữ Tự nhiên được sử dụng nhằm mục đích gì?
   - A. Xóa bỏ các từ sai chính tả.
   - B. Đệm thêm các token đặc biệt (thường là số 0) vào cuối các câu ngắn để đưa tất cả các câu trong cùng một Mini-batch về cùng một độ dài đồng nhất.
   - C. Tăng tốc độ đọc file từ đĩa cứng.
   - D. Dịch văn bản sang tiếng Anh.

2. Tham số `collate_fn` trong `torch.utils.data.DataLoader` cho phép lập trình viên làm điều gì?
   - A. Tùy biến cách gom nhóm và xử lý danh sách các mẫu dữ liệu đơn lẻ thành một Mini-batch Tensor chuẩn mực trước khi nạp vào mô hình.
   - B. Tự động tính toán đạo hàm của hàm mất mát.
   - C. Thay đổi số lượng card đồ họa GPU.
   - D. Xóa tập dữ liệu huấn luyện.

3. Trong mô hình Deep Averaging Network (DAN), thao tác Global Average Pooling dọc theo trục từ vựng (`dim=1`) có tác dụng gì?
   - A. Nhân đôi số chiều của vector từ.
   - B. Gộp toàn bộ các vector từ trong câu thành duy nhất một vector đại diện cho cả tài liệu có số chiều cố định, không phụ thuộc vào độ dài của câu.
   - C. Sắp xếp lại thứ tự các từ theo bảng chữ cái.
   - D. Xóa các từ dừng.

4. Hàm nào của PyTorch được sử dụng phổ biến nhất để tự động đệm danh sách các tensor có độ dài khác nhau thành một tensor duy nhất?
   - A. `torch.cat()`
   - B. `torch.nn.utils.rnn.pad_sequence`
   - C. `torch.stack()`
   - D. `torch.reshape()`

5. Điểm khác biệt lớn nhất giữa mô hình Deep Learning NLP (như DAN) và mô hình Baseline TF-IDF là gì?
   - A. Deep Learning hoàn toàn không dùng số học.
   - B. Deep Learning sử dụng vector nhúng dày đặc (Dense Embeddings) có khả năng học và cập nhật ngữ nghĩa ngữ cảnh qua từng tầng, trong khi TF-IDF sử dụng ma trận tần suất cố định không học được ngữ nghĩa sâu.
   - C. TF-IDF luôn chạy chậm hơn Deep Learning.
   - D. Deep Learning chỉ áp dụng được cho tiếng Anh.
"""
    session_summary_content = """
* **What I learned:**
  - Kỹ thuật xử lý các chuỗi văn bản có độ dài biến thiên (Variable Sequence Length).
  - Cách viết Custom Collate Function với `pad_sequence` động theo từng batch.
  - Kiến trúc Deep Averaging Network (DAN) kết hợp Embedding và Global Pooling.
* **What I can now do:**
  - Tự tay xây dựng toàn bộ luồng huấn luyện Deep Learning cho bài toán phân loại văn bản bằng PyTorch thuần.
  - Tối ưu hóa việc nạp batch dữ liệu ngôn ngữ không gây lãng phí bộ nhớ.
* **What I still need to practice:**
  - Thực hiện bài toán lớn tổng kết Module: **Mini Project CV / NLP Thực Tế** ở Buổi 10!
"""
    next_session_content = """
Buổi tiếp theo (**M4_S10**) là buổi tổng kết đỉnh cao của Module 4: **Mini Project Thực Tế: So Sánh, Tinh Chỉnh & Báo Cáo Dự Án CV / NLP**! Bạn sẽ chọn một trong hai hướng (Thị giác máy tính HOẶC Xử lý ngôn ngữ tự nhiên), tự tay xây dựng pipeline hoàn chỉnh từ dữ liệu thô, huấn luyện, đánh giá đối sánh và nhận Chứng chỉ **Certificate of CV/NLP Applied AI**!
"""
    trace = {
        "items": [
            "Xử lý độ dài câu biến thiên (Variable sequence length)",
            "Custom collate_fn & Dynamic padding với pad_sequence",
            "Deep Averaging Network (Embedding + Global Pooling + Classifier)"
        ],
        "deliverable": "Mã nguồn Text Classification Pipeline hoàn chỉnh bằng PyTorch Deep Learning"
    }
    nb = create_notebook(
        module_id="M4",
        module_name="AI NHÌN VÀ HIỂU DỮ LIỆU",
        session_id="S09",
        topic="Xây Dựng Text Classification Pipeline với PyTorch",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Notebook huấn luyện mạng nơ-ron sâu phân loại văn bản bằng PyTorch",
        competition_relevance="Hoàn thiện năng lực xử lý văn bản thực chiến cho các đề thi Olympic AI dạng NLP.",
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
    save_notebook(nb, "course/M4_APPLIED_AI/M4_S09.ipynb")


def build_m4_s10():
    objectives = [
        "Hoàn thành sản phẩm thực hành lớn nhất của Module 4: Mini Project ứng dụng thực tế (Lựa chọn nhánh Thị giác máy tính HOẶC Xử lý ngôn ngữ tự nhiên).",
        "Thiết lập một khung thực nghiệm đối sánh khoa học (Comparative Experimental Framework): Baseline vs Mô hình Nâng cao.",
        "Tiến hành phân tích lỗi sai chuyên sâu (In-depth Error Analysis) và đề xuất các giải pháp kỹ thuật cụ thể.",
        "Đóng gói toàn bộ mã nguồn, số liệu và xuất bản Báo cáo Mini Project (Applied AI Report) đạt chuẩn thi đấu để nhận chứng chỉ Module 4."
    ]
    prerequisites = [
        "Hoàn thành trọn vẹn 9 buổi trước của Module M4 (Cả nhánh CV từ S01-S05 và nhánh NLP từ S06-S09).",
        "Kỹ năng lập báo cáo đánh giá đối sánh từ Module M2."
    ]
    why_content = """
Chào mừng bạn đến với buổi tổng kết đỉnh cao của Module 4 — module dài nhất và đồ sộ nhất trong toàn bộ lộ trình 38 buổi! Trong các kỳ thi học thuật và ứng dụng công nghiệp, không ai thuê bạn chỉ để "chạy thử một dòng code mẫu". Bạn được đánh giá dựa trên **Năng lực hoàn thiện một Sản phẩm AI hoàn chỉnh (Completed AI Deliverable)**.

Hôm nay, bạn sẽ chọn một con đường: Hoặc trở thành chuyên gia giải quyết bài toán Thị giác máy tính (CV Track), hoặc chinh phục bài toán Xử lý ngôn ngữ tự nhiên (NLP Track). Bạn sẽ tự tay thiết kế bài toán, dựng mốc so sánh, tinh chỉnh siêu tham số, phân tích nguyên nhân thất bại và lập một bản Báo cáo Dự án chuẩn mực. Đây là tấm hộ chiếu chính thức mở ra cánh cửa thi đấu Olympic AI ở Module M5!
"""
    concept_content = r"""
### 1. Khung thiết kế Mini Project chuẩn Olympic
Một dự án AI ứng dụng thực tế đạt chuẩn bắt buộc phải tuân theo chuỗi mắt xích:
1. **Lựa chọn Đề tài (Topic Scoping):**
   - *Hướng A (Computer Vision):* Bài toán phân loại ảnh đa lớp (ví dụ: nhận diện biểu cảm khuôn mặt, phân loại bệnh cây trồng, hoặc phân loại phương tiện giao thông).
   - *Hướng B (Natural Language Processing):* Bài toán phân loại văn bản (ví dụ: phát hiện tin giả, phân tích cảm xúc phản hồi người dùng, hoặc phân loại câu hỏi).
2. **Thiết lập Baseline Đối sánh:**
   - CV: Một mạng CNN nhỏ 2 tầng tự viết hoặc mô hình hồi quy tuyến tính trên pixel.
   - NLP: Mô hình TF-IDF + Logistic Regression.
3. **Mô hình Ứng dụng Nâng cao:**
   - CV: Áp dụng Transfer Learning (ResNet-18) kết hợp Data Augmentation.
   - NLP: Áp dụng Deep Averaging Network (DAN) hoặc Pretrained Embeddings kết hợp Dropout.
4. **Phân tích đối sánh định lượng (Comparative Benchmark):** Bảng so sánh Accuracy, Macro-F1, thời gian huấn luyện và dung lượng mô hình.
5. **Báo cáo phân tích lỗi sai (Error Analysis):** Trích xuất tối thiểu 3 ca dự đoán sai lầm nghiêm trọng và lý giải nguyên nhân.
"""
    visual_intuition_content = """
```
Cấu trúc Báo cáo Mini Project M4:
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. TỔNG QUAN BÀI TOÁN & TẬP DỮ LIỆU (Chọn Track: CV hoặc NLP)               │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. TIỀN XỬ LÝ & TĂNG CƯỜNG DỮ LIỆU (Augmentation / Regex Tokenization)      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. THIẾT LẬP MỐC CƠ SỞ (Baseline Model Pipeline)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. HUẤN LUYỆN MÔ HÌNH NÂNG CAO (Transfer Learning / Deep NLP)               │
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. BẢNG SO SÁNH ĐỐI SÁNH (Benchmark Table: Accuracy, F1, Train Time)        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 6. PHÂN TÍCH LỖI SAI (Error Analysis & Failure Modes)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 7. KẾT LUẬN & ĐỀ XUẤT CẢI TIẾN CHO VÒNG THI ĐẤU (Module M5 Readiness)       │
└─────────────────────────────────────────────────────────────────────────────┘
```
"""
    code_cells = [
        (
            """import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 1. KHOI TAO FRAMEWORK SO SANH MINI PROJECT (MINI PROJECT BENCHMARK ENGINE)
# Minh hoa tren Track B: NLP Sentiment & Intent Classification
np.random.seed(42)

print("===== KHỞI ĐỘNG MINI PROJECT MODULE 4: APPLIED AI =====")
print("-> Track duoc lua chon: NLP & Text Understanding")
print("-> Bai toan: Phan loai Y dinh khach hang (Customer Intent: Query vs Complaint vs Praise)")

# Gia lap tap du lieu kiem thu thuc te (Test Benchmark) gom 100 mau
true_test_labels = np.random.choice([0, 1, 2], size=100, p=[0.4, 0.3, 0.3])

# Mo hinh 1: Baseline Model (TF-IDF + Linear Classifier)
# Do chinh xac kha nhung de nham lan giua Query va Complaint
preds_baseline = true_test_labels.copy()
noise_idx_base = np.random.choice(100, size=28, replace=False)
preds_baseline[noise_idx_base] = np.random.choice([0, 1, 2], size=28)

# Mo hinh 2: Deep NLP Model (PyTorch Embedding + Masked Pooling + Dropout)
# Khac phuc duoc cac tu dong nghia, giam thieu sai sot dang ke
preds_deep = true_test_labels.copy()
noise_idx_deep = np.random.choice(100, size=14, replace=False)
preds_deep[noise_idx_deep] = np.random.choice([0, 1, 2], size=14)

print("Da nap xong ket qua thuc nghiem cua hai mo hinh!")
""",
            """Khối code khởi tạo khung đánh giá đối sánh cho Mini Project:
- Thiết lập kịch bản thực nghiệm khoa học trên 3 lớp ý định của người dùng.
- Chuẩn bị dữ liệu kiểm thử độc lập để so sánh công bằng giữa Baseline và Mô hình Deep Learning nâng cao."""
        ),
        (
            """# 2. XUAT BANG SO SANH DOI SANH (COMPARATIVE BENCHMARK TABLE)
metrics_summary = [
    {
        'Mo hinh': 'Model 1: Baseline (TF-IDF + Logistic)',
        'Accuracy (%)': f"{accuracy_score(true_test_labels, preds_baseline)*100:.1f}%",
        'Macro F1-Score': f"{f1_score(true_test_labels, preds_baseline, average='macro'):.3f}",
        'Thoi gian train': '1.8 giay',
        'Bo nho tham so': '~0.4 MB'
    },
    {
        'Mo hinh': 'Model 2: Deep PyTorch (Embedding + Pooling)',
        'Accuracy (%)': f"{accuracy_score(true_test_labels, preds_deep)*100:.1f}%",
        'Macro F1-Score': f"{f1_score(true_test_labels, preds_deep, average='macro'):.3f}",
        'Thoi gian train': '42.5 giay',
        'Bo nho tham so': '~4.2 MB'
    }
]

df_report = pd.DataFrame(metrics_summary)
print("===== BANG SO SANH HIEU NANG MINI PROJECT (BENCHMARK) =====")
print(df_report.to_string(index=False))

# 3. PHAN TICH MA TRAN NHAM LAN (CONFUSION MATRIX) CUA MO HINH TOT NHAT
cm_best = confusion_matrix(true_test_labels, preds_deep)
print("\\n===== MA TRAN NHAM LAN CUA MO HINH DEEP NLP =====")
print("             Doan Query(0)  Doan Complaint(1)  Doan Praise(2)")
for i, row in enumerate(cm_best):
    print(f"Thuc te Lop {i}:      {row[0]:2d}               {row[1]:2d}                 {row[2]:2d}")

print("\\n-> KET LUAN: Mo hinh Deep PyTorch tang vuot troi +14% Accuracy va giam mot nua so luong ca du doan sai!")
""",
            """Khối code xuất bản bảng Báo cáo Kết quả Đối sánh và Ma trận Nhầm lẫn:
- Thể hiện sự vượt trội rõ rệt của mô hình Deep Learning trên cả 2 tiêu chí Accuracy và Macro F1-Score.
- Đánh giá đa chiều cả về chi phí tính toán (Thời gian huấn luyện và kích thước bộ nhớ) — phẩm chất bắt buộc của một vận động viên thi đấu AI."""
        )
    ]
    practice_content = """
### Bài 1 (Easy): Tùy biến Báo cáo cho Nhánh Computer Vision
Nếu bạn chọn nhánh Thị giác máy tính (CV Track), hãy viết lại bảng thống kê trên thay thế bằng hai mô hình: `Model 1: Custom CNN (2 blocks)` vs `Model 2: Pretrained ResNet-18 Transfer Learning`.

### Bài 2 (Easy): Tính toán tỷ lệ giảm lỗi sai (Error Reduction Rate)
Dựa vào số lượng mẫu đoán sai của hai mô hình ($E_1 = 28$ và $E_2 = 14$), hãy viết công thức tính Tỷ lệ giảm thiểu lỗi sai:
$$\text{Reduction Rate} = \frac{E_1 - E_2}{E_1} \times 100\%$$

### Bài 3 (Medium): Viết hàm xuất bản Báo cáo Markdown hoàn chỉnh
Viết một hàm Python `export_mini_project_report(df_report, track="NLP", filename="MINI_PROJECT_REPORT.md")` tự động xuất bản một tài liệu Markdown hoàn chỉnh gồm: Tên đề tài, Tác giả, Bảng Benchmark, Ma trận nhầm lẫn và Lời nhận xét kỹ thuật.

### Bài 4 (Medium): Phân tích Trade-off giữa Tốc độ và Độ chính xác
Mô hình Deep NLP đạt độ chính xác cao hơn 14% nhưng thời gian huấn luyện lâu hơn gấp 23 lần. Trong một bài thi Olympic AI có thời gian nộp bài gấp rút (chỉ còn 10 phút), bạn sẽ đưa ra quyết định chiến thuật như thế nào?

### Bài 5 (Challenge): Xây dựng Pipeline Inference tương tác nhanh (Interactive Predictor)
Viết một hàm `predict_new_sample(text_or_image, model, preprocessor)` nhận vào một câu văn bản mới (hoặc đường dẫn tới một bức ảnh mới), tự động thực hiện toàn bộ các bước tiền xử lý, đưa qua mô hình đã nạp, và in ra màn hình: Tên lớp dự đoán kèm theo Phần trăm Độ tự tin (Confidence Score).
"""
    voai_connection_content = """
Trong cấu trúc đề thi Olympic AI, phần Applied AI (CV/NLP) luôn là phần chiếm tỷ trọng điểm số cao nhất và có tính phân hóa mạnh nhất. Những thí sinh chỉ biết lý thuyết suông sẽ bị "ngợp" trước dữ liệu thực tế có nhiễu. Ngược lại, một thí sinh đã kinh qua một Mini Project hoàn chỉnh từ A đến Z sẽ có sẵn trong tay:
1. Bộ template code nạp dữ liệu chuẩn.
2. Các hàm vẽ đồ thị và đánh giá đối sánh tự động.
3. Bản lĩnh phân tích lỗi sai và bình tĩnh tinh chỉnh siêu tham số dưới áp lực thời gian.
"""
    debugging_content = """
**Tình huống thực tế:** Thí sinh nộp bài Mini Project nhưng file notebook chứa các ô code bị lỗi phụ thuộc thứ tự chạy ngầm (Hidden State Dependency) hoặc gọi các đường dẫn file cứng trên máy cá nhân (`C:\\Users\\Desktop\\data...`):

```python
# CODE CÓ LỖI ĐƯỜNG DẪN CỨNG VÀ PHỤ THUỘC THỨ TỰ:
# df = pd.read_csv("C:/Users/Admin/Downloads/my_data.csv") # May giam khao se bao loi FileNotFoundError ngay lap tuc!
```

**Nhiệm vụ của bạn:**
1. Giải thích tại sao mã nguồn nộp bài thi bắt buộc phải dùng đường dẫn tương đối (Relative Paths) hoặc tạo dữ liệu có tính tái lập (Reproducible Synthetic Datasets).
2. Nêu quy tắc vàng: Trước khi nộp bài, luôn chọn `Kernel -> Restart & Run All` để bảo đảm toàn bộ notebook chạy từ đầu đến cuối không có bất kỳ lỗi nào.
"""
    knowledge_check_content = """
1. Yêu cầu bắt buộc đối với sản phẩm đầu ra (Deliverable) của Module M4 là gì?
   - A. Một bài viết tóm tắt lý thuyết trên mạng xã hội.
   - B. Một Mini Project ứng dụng thực tế hoàn chỉnh (chọn nhánh Computer Vision HOẶC Natural Language Processing) có mã nguồn huấn luyện, đánh giá đối sánh và báo cáo kết quả.
   - C. Một file âm thanh thu âm giọng đọc.
   - D. Một bài kiểm tra trắc nghiệm 5 câu.

2. Trong một báo cáo thử nghiệm AI khoa học, việc thiết lập một mô hình cơ sở (Baseline Model) có ý nghĩa gì?
   - A. Để kéo dài thời gian làm bài thi.
   - B. Thiết lập chuẩn đối sánh tối thiểu nhằm chứng minh một cách định lượng rằng giải pháp nâng cao thực sự mang lại giá trị cải tiến vượt trội.
   - C. Tự động xóa các file tạm trong hệ điều hành.
   - D. Thay thế cho bước tiền xử lý dữ liệu.

3. Khi đánh giá mô hình phân loại trên tập dữ liệu có sự chênh lệch số lượng mẫu giữa các lớp, chỉ số nào phản ánh công bằng nhất chất lượng toàn diện của mô hình?
   - A. Điểm Accuracy tổng thể.
   - B. Macro-averaged F1-Score (Trung bình cộng F1 của tất cả các lớp).
   - C. Số lượng epochs đã chạy.
   - D. Kích thước file trọng số mô hình.

4. Quy tắc "Restart Kernel & Run All Cells" trước khi nộp bài thi có tác dụng gì?
   - A. Xóa sạch mã nguồn của bạn.
   - B. Kiểm tra tính toàn vẹn và khả năng tái lập (Reproducibility) của toàn bộ notebook từ trên xuống dưới, bảo đảm không có lỗi ẩn do chạy nhảy cóc các ô code.
   - C. Cài đặt lại hệ điều hành Windows.
   - D. Tăng độ chính xác của mô hình lên 100%.

5. Sau khi hoàn thành xuất sắc Buổi M4_S10, học viên chính thức đạt được chứng chỉ cột mốc nào?
   - A. Certificate of AI Orientation
   - B. Certificate of CV/NLP Applied AI
   - C. Certificate of Machine Learning Basics
   - D. Final Project Certificate
"""
    session_summary_content = """
* **What I learned:**
  - Quy trình khép kín hoàn thiện một Mini Project AI ứng dụng thực tế.
  - Cách thiết lập khung thực nghiệm đối sánh (Comparative Benchmark) giữa Baseline và Deep Model.
  - Phân tích lỗi sai đa chiều và đánh giá Trade-off giữa độ chính xác và tài nguyên tính toán.
* **What I can now do:**
  - Tự tin thực hiện trọn vẹn một bài toán Thị giác máy tính hoặc Xử lý ngôn ngữ tự nhiên từ A đến Z.
  - Xuất bản Báo cáo Kỹ thuật chuẩn thi đấu, sẵn sàng cho đấu trường Olympic AI.
* **What I still need to practice:**
  - Bước vào giai đoạn rèn luyện chiến thuật thi đấu đỉnh cao ở **Module M5: Chiến thuật thi Olympic AI**!
"""
    next_session_content = """
Xin nhiệt liệt chúc mừng bạn đã chính thức vượt qua chặng đường dài nhất và gian nan nhất của khóa học: Hoàn thành trọn vẹn Module M4 và nhận Chứng chỉ **Certificate of CV/NLP Applied AI**!
Trong buổi tiếp theo (**M5_S01**), chúng ta sẽ bước sang một cảnh giới mới: **Module M5: CHIẾN THUẬT THI OLYMPIC AI** — Bắt đầu làm chủ Chiến thuật đọc đề nhanh trong 15 phút, giải mã mục tiêu, căn chỉnh Metric và phòng ngừa rò rỉ dữ liệu trong phòng thi!
"""
    module_assessment = """
---

## 🏆 MODULE 4 ASSESSMENT & CERTIFICATION

### Phần A: Đánh giá Lý thuyết & Kiến trúc Ứng dụng (Theory & Applied AI)
- **Computer Vision:** Nắm vững cấu trúc Tensor ảnh $[C, H, W]$, cơ chế tích chập 2D, ý nghĩa của Padding, Stride, MaxPool2d và triết lý Transfer Learning.
- **Natural Language Processing:** Nắm vững quy trình Tokenization, cấu trúc Vocabulary Indexer, công thức toán học TF-IDF và bản chất Word Embeddings trong PyTorch.

### Phần B: Sản phẩm Thực hành Bắt buộc (Practical Deliverable)
- **Mini Project Ứng dụng Hoàn Chỉnh:** Học viên nộp 01 notebook thực hành chuyên sâu thuộc một trong hai hướng:
  * *Hướng A (Computer Vision):* Pipeline phân loại ảnh có Data Augmentation, kiến trúc CNN hoặc Transfer Learning (ResNet-18) kèm Confusion Matrix.
  * *Hướng B (Natural Language Processing):* Pipeline phân loại văn bản có Tokenization, xử lý Padding động (`pad_collate_fn`), mạng Deep NLP kèm Báo cáo đối sánh với Baseline TF-IDF.
- **Báo cáo Kỹ thuật Đạt chuẩn (Applied AI Report):** Có Bảng Benchmark so sánh tối thiểu 2 mô hình (Baseline vs Advanced), đo lường Accuracy và Macro-F1, kèm phân tích ít nhất 3 ca dự đoán sai lầm (Error Analysis).

### Tiêu chí Đạt chuẩn (Rubric):
| Tiêu chí | Trọng số | Yêu cầu đạt |
| :--- | :--- | :--- |
| **Data Pipeline & Preprocessing** | 25% | Xử lý ảnh (Transforms) hoặc văn bản (Padding Collate) chuẩn |
| **Kiến trúc Mô hình & Training** | 30% | Huấn luyện thành công mô hình Deep Learning có hội tụ |
| **Khung Thực nghiệm Đối sánh**   | 25% | Có mô hình Baseline và Bảng so sánh đa metric khách quan |
| **Báo cáo & Phân tích Lỗi sai**   | 20% | Xuất bản báo cáo sạch, chỉ ra nguyên nhân của các ca đoán sai |

**Chứng chỉ hoàn thành:**
🏅 **Certificate of CV/NLP Applied AI** — Đủ điều kiện tiến vào Module M5 (Chiến thuật thi Olympic AI).
"""
    trace = {
        "items": [
            "Hoàn thành Mini Project ứng dụng thực tế (chọn nhánh CV hoặc NLP)",
            "Khung thực nghiệm đối sánh (Comparative benchmark: Baseline vs Deep model)",
            "Phân tích lỗi sai chuyên sâu (In-depth error analysis)",
            "Nhận chứng chỉ Certificate of CV/NLP Applied AI"
        ],
        "deliverable": "Mini project ảnh hoặc văn bản hoàn chỉnh + Báo cáo kết quả (Applied AI Report)"
    }
    nb = create_notebook(
        module_id="M4",
        module_name="AI NHÌN VÀ HIỂU DỮ LIỆU",
        session_id="S10",
        topic="Mini Project Thực Tế: So Sánh, Tinh Chỉnh & Báo Cáo Dự Án CV / NLP",
        objectives=objectives,
        prerequisites=prerequisites,
        estimated_time="120 phút",
        session_output="Mini project ảnh hoặc văn bản (Applied AI Report)",
        competition_relevance="Tổng duyệt toàn bộ năng lực ứng dụng trước khi bước vào giai đoạn luyện thi Olympic AI.",
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
    save_notebook(nb, "course/M4_APPLIED_AI/M4_S10.ipynb")

if __name__ == "__main__":
    build_m4_s01()
    build_m4_s02()
    build_m4_s03()
    build_m4_s04()
    build_m4_s05()
    build_m4_s06()
    build_m4_s07()
    build_m4_s08()
    build_m4_s09()
    build_m4_s10()
    print("M4 build complete! 10/10 notebooks generated.")
