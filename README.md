# 🫀 Heart Attack Risk Prediction

> **Dự án Khai phá dữ liệu:** Dự báo nguy cơ sốc tim bằng các thuật toán Machine Learning trên dữ liệu BRFSS.

## 📋 Mục tiêu

Xây dựng và so sánh nhiều mô hình phân loại để dự đoán nguy cơ heart attack:
- ✅ Tiền xử lý dữ liệu & Feature Selection
- ✅ Huấn luyện 5 mô hình: SVM, Naive Bayes, Random Forest, KNN, ANN
- ✅ Đánh giá hiệu suất (ưu tiên **Recall** - tránh bỏ sót ca bệnh)
- ✅ Chọn mô hình tối ưu

---

## 📁 Cấu trúc thư mục

```
KhaiPhaDuLieu/
├── CODE/                       # Jupyter notebooks
│   ├── feature_selection.ipynb # Chọn features quan trọng
│   ├── SVM.ipynb              # Support Vector Machine
│   ├── NaiveBayes.ipynb       # Naive Bayes
│   ├── RandomForest.ipynb     # Random Forest
│   ├── KNN.ipynb              # K-Nearest Neighbors
│   └── ANN.ipynb              # Artificial Neural Network
│
├── DATA/                       # Dữ liệu (không commit file lớn)
│   ├── BRFSS.csv              # Dataset gốc (tải từ Drive)
│   └── selected_columns.csv   # Features đã chọn
│
├── MODELS/                     # Models đã train (không commit)
│   ├── svm.pkl
│   ├── nb.pkl
│   ├── rf.pkl
│   ├── knn.pkl
│   └── ann.pkl
│
├── REPORT/                     # Báo cáo & slides
│   ├── report.docx
│   └── slides.pptx
│
├── .gitignore
├── requirements.txt            # Danh sách thư viện
└── README.md                   # File này
```

---

## ⚙️ Thiết lập môi trường

### 🐧 Linux / macOS

```bash
# 1. Tạo môi trường ảo
python3 -m venv venv

# 2. Kích hoạt môi trường
source venv/bin/activate

# 3. Cài đặt thư viện
pip install -r requirements.txt

# 4. Kiểm tra
python --version
pip list
```

### 🪟 Windows

```cmd
# 1. Tạo môi trường ảo
python -m venv venv

# 2. Kích hoạt môi trường
venv\Scripts\activate

# 3. Cài đặt thư viện
pip install -r requirements.txt

# 4. Kiểm tra
python --version
pip list
```

**Lưu ý Windows:** Nếu gặp lỗi PowerShell không cho chạy script, chạy lệnh này (mở PowerShell as Admin):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Nếu bạn dùng `.venv` (tôi đã tạo `.venv` trong project này):**

```powershell
# Kích hoạt trong PowerShell
.\.venv\Scripts\Activate.ps1

# Hoặc, nếu không muốn kích hoạt, chạy trực tiếp Python từ venv:
.\\.venv\\Scripts\\python -m pip install -r requirements.txt
.\\.venv\\Scripts\\python -m notebook
```

---

## 🚀 Cách chạy Jupyter Notebook

### Cách 1: Jupyter Notebook (khuyến nghị)

**Linux/macOS:**
```bash
source venv/bin/activate    # Kích hoạt venv trước
jupyter notebook            # Mở trình duyệt tự động
```

**Windows:**
```cmd
venv\Scripts\activate       # Kích hoạt venv trước
jupyter notebook            # Mở trình duyệt tự động
```

### Cách 2: JupyterLab

```bash
# Cài JupyterLab (nếu chưa có)
pip install jupyterlab

# Chạy
jupyter lab
```

### Cách 3: VS Code

1. Mở VS Code
2. Cài extension: **Jupyter** (Microsoft)
3. Mở file `.ipynb` và chọn kernel: `venv (Python 3.x)`

---

## 📊 Hướng dẫn chạy Pipeline

### Bước 1: Tải dữ liệu

Tải file `BRFSS.csv` từ Drive và đặt vào thư mục `DATA/`:

📦 **Link tải dữ liệu:** `[Điền link Google Drive/OneDrive ở đây]`

```bash
# Cấu trúc sau khi tải
DATA/
├── BRFSS.csv              # ← File này cần tải từ Drive
└── selected_columns.csv
```

### Bước 2: Chạy Feature Selection

```bash
# Mở notebook
jupyter notebook CODE/feature_selection.ipynb

# Hoặc chạy tất cả cells: Kernel → Restart & Run All
```

**Output:** File `DATA/selected_columns.csv` chứa danh sách features đã chọn

### Bước 3: Huấn luyện các mô hình

Chạy lần lượt các notebook trong `CODE/`:

1. ✅ `SVM.ipynb`
2. ✅ `NaiveBayes.ipynb`
3. ✅ `RandomForest.ipynb`
4. ✅ `KNN.ipynb`
5. ✅ `ANN.ipynb`

**Output:** Các file model `.pkl` trong thư mục `MODELS/`

### Bước 4: Đánh giá & so sánh

Xem kết quả trong từng notebook hoặc tạo notebook tổng hợp để so sánh:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve

---

---

## 📦 Dữ liệu & Mô hình

### ⚠️ Không commit vào Git:
- ❌ `DATA/BRFSS.csv` (file quá lớn)
- ❌ `MODELS/*.pkl` (mô hình đã train)
- ❌ `venv/` (thư viện Python)

### 📥 Link tải:

| Tên file | Mô tả | Link |
|----------|-------|------|
| `BRFSS.csv` | Dataset gốc (~50MB) | [Điền link Drive] |
| `*.pkl` | Models đã train | [Điền link Drive] |

**Cách tải nhanh (Linux/macOS):**
```bash
# Dùng gdown để tải từ Google Drive
pip install gdown
gdown "LINK_GOOGLE_DRIVE" -O DATA/BRFSS.csv
```

---

## 🎯 Tiêu chí đánh giá mô hình

### Ưu tiên chỉ số:
1. **Recall** (Sensitivity) - Tránh bỏ sót người bệnh
2. **F1-Score** - Cân bằng Precision & Recall
3. **Precision** - Độ chính xác dự đoán
4. **Accuracy** - Tỷ lệ dự đoán đúng tổng thể

### Công thức:
```
Recall = TP / (TP + FN)        ← Tỷ lệ phát hiện đúng người bệnh
Precision = TP / (TP + FP)     ← Tỷ lệ dự đoán bệnh chính xác
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## 👥 Phân công công việc

| Thành viên | Vai trò | Nhiệm vụ |
|------------|---------|----------|
| **Member A** | Data Engineer | Preprocessing, Feature Selection |
| **Member B** | ML Engineer | Train & Tune models (SVM, NB, RF, KNN, ANN) |
| **Member C** | Analyst/Reporter | Evaluation, Visualization, Report, Slides |

---

## 🛠️ Ghi chú kỹ thuật

### 1. Tái lập kết quả (Reproducibility)
Luôn dùng `random_state=42` khi split data:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 2. Xử lý Class Imbalance
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### 3. Loại bỏ Output Notebook (tránh file quá lớn)

**Linux/macOS:**
```bash
pip install nbstripout
nbstripout --install          # Tự động xóa output khi commit
```

**Windows:**
```cmd
pip install nbstripout
nbstripout --install
```

### 4. Kiểm tra môi trường

```bash
# Xem các thư viện đã cài
pip list

# Xem đường dẫn Python đang dùng
which python    # Linux/macOS
where python    # Windows
```

---

## 🐛 Xử lý lỗi thường gặp

### Lỗi: `ModuleNotFoundError: No module named 'sklearn'`
```bash
# Chưa kích hoạt venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows

# Cài lại thư viện
pip install -r requirements.txt
```

### Lỗi: `Kernel died` trong Jupyter
```bash
# Cài lại ipykernel
pip install --upgrade ipykernel
python -m ipykernel install --user
```

### Lỗi: Không mở được Jupyter trên Windows
```cmd
# Chạy với Python module
python -m notebook
```

---

## 📚 Tài liệu tham khảo

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Imbalanced-learn](https://imbalanced-learn.org/)
- [BRFSS Dataset Info](https://www.cdc.gov/brfss/)

---

## 📧 Liên hệ

- **Leader:** [Tên] - [Email]
- **Member A:** [Tên] - [Email]
- **Member B:** [Tên] - [Email]
- **Member C:** [Tên] - [Email]

---

**🎓 Trường:** [Tên trường]  
**📖 Môn học:** Khai phá dữ liệu / Data Mining  
**👨‍🏫 Giảng viên:** [Tên GV]  
**📅 Học kỳ:** [HK1/2024-2025]

