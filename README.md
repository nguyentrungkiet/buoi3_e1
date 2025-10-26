# Nhận Dạng Chữ Số MNIST với CNN

## 🎯 Mô tả dự án

Đây là dự án nhận dạng chữ số viết tay từ dataset MNIST sử dụng Convolutional Neural Network (CNN). Dự án được phát triển trong khuôn khổ **Olympic AI - Buổi 3 - Bài tập E1** với mục tiêu đạt được **accuracy cao nhất có thể** trên dataset MNIST.

### 🏆 **Kết quả đạt được: 99.40% Validation Accuracy**

## 🎯 Mục tiêu

- **Mục tiêu chính**: Làm quen với CNN cơ bản và đạt accuracy cao
- **Dataset**: Digit Recognizer (MNIST) - 70,000 ảnh chữ số viết tay 28x28 pixel
- **Metric đánh giá**: Accuracy 
- **Kiến trúc**: CNN với các layer Conv–ReLU–Pool, BatchNorm, data augmentation
- **Target**: Đạt >99% accuracy trên validation set ✅

## 🧠 CNN là gì và tại sao phù hợp với MNIST?

### Convolutional Neural Network (CNN)
CNN là một loại neural network đặc biệt hiệu quả cho các tác vụ xử lý ảnh:

1. **Convolutional Layer**: Phát hiện các đặc tr�ng cục bộ (edges, patterns)
2. **Pooling Layer**: Giảm kích thước và tăng tính bất biến
3. **Fully Connected Layer**: Phân loại dựa trên features đã trích xuất

### Tại sao CNN phù hợp với MNIST?
- **Spatial Information**: CNN giữ được thông tin không gian của pixel
- **Parameter Sharing**: Giảm số parameters so với Dense layer
- **Translation Invariance**: Nhận dạng được chữ số ở các vị trí khác nhau
- **Feature Hierarchy**: Học được các pattern từ đơn giản đến phức tạp

## Đặc điểm kỹ thuật

### Kiến trúc mô hình

Mô hình CNN được thiết kế với:

1. **Data Augmentation Layer**:
   - RandomRotation (10%)
   - RandomZoom (10%) 
   - RandomTranslation (10%)

2. **3 Convolutional Blocks**:
   - **Block 1**: 2x Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)
   - **Block 2**: 2x Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
   - **Block 3**: Conv2D(128) + BatchNorm + Dropout(0.25)

3. **Dense Layers**:
   - Flatten
   - Dense(512) + BatchNorm + Dropout(0.5)
   - Dense(256) + Dropout(0.5) 
   - Dense(10, softmax) - output layer

## 📊 Phân tích kết quả chi tiết

### 🎯 **Tại sao đạt được 99.40% accuracy?**

1. **Kiến trúc CNN phù hợp**:
   - 3 Conv blocks với số filters tăng dần (32→64→128)
   - MaxPooling giúp tạo translation invariance
   - BatchNorm giúp training ổn định

2. **Data Augmentation hiệu quả**:
   - RandomRotation: Nhận dạng chữ số bị xoay nhẹ
   - RandomZoom: Thích ứng với kích thước khác nhau  
   - RandomTranslation: Chữ số không cần ở chính giữa

3. **Regularization tốt**:
   - Dropout ngăn overfitting
   - Early Stopping dừng đúng lúc
   - Learning rate scheduling fine-tune cuối

4. **Preprocessing chuẩn**:
   - Normalization [0,255] → [0,1]
   - One-hot encoding cho multi-class
   - Train/validation split stratified

### 📈 **Biểu đồ kết quả**

Project tự động tạo ra 3 biểu đồ quan trọng:

1. **`training_history.png`**: 
   - Training vs Validation Accuracy/Loss theo epochs
   - Thấy được overfitting hay underfitting
   - Xem hiệu quả của learning rate scheduling

2. **`confusion_matrix.png`**:
   - Ma trận 10x10 cho digits 0-9
   - Xem model hay nhầm lẫn giữa số nào
   - Đường chéo chính → predictions chính xác

3. **`predictions_sample.png`**:
   - 10 ảnh test đầu tiên với predictions
   - Kiểm tra trực quan model có hoạt động đúng

### 🔍 **Phân tích Confusion Matrix**

Từ kết quả thực tế, model rất ít nhầm lẫn:
- **Digit 1**: 99% precision/recall (it nhầm với 7)
- **Digit 8**: 99% precision/recall (it nhầm với 0, 6)  
- **Digit 9**: 99% precision/recall (it nhầm với 4, 7)

→ **Kết luận**: Model học được đặc trưng rất tốt!

## 📦 Cài đặt và chạy

### 📋 **Yêu cầu hệ thống**
- **Python**: 3.8+ (Recommended: 3.9-3.11)
- **RAM**: Tối thiểu 4GB (Recommended: 8GB+)
- **Storage**: ~500MB cho code + data + results
- **GPU**: Không bắt buộc nhưng khuyến nghị (giảm thời gian từ 15 phút → 3 phút)

### 1. 🚀 **Cài đặt nhanh (Recommended)**

```bash
# Bước 1: Clone repository
git clone https://github.com/nguyentrungkiet/buoi3_e1.git
cd buoi3_e1

# Bước 2: Tạo virtual environment (Recommended)
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:  
source venv/bin/activate

# Bước 3: Cài đặt dependencies
pip install -r requirements.txt

# Bước 4: Chạy training
python resolve.py
```

### 2. 🐍 **Cài đặt từng bước (Chi tiết)**

```bash
# Kiểm tra Python version
python --version  # Phải >= 3.8

# Tạo thư mục project
mkdir mnist_cnn_project
cd mnist_cnn_project

# Clone code
git clone https://github.com/nguyentrungkiet/buoi3_e1.git .

# Kiểm tra cấu trúc
ls -la  # Phải thấy resolve.py, requirements.txt, dataset/

# Cài đặt từng package (nếu pip install -r không work)
pip install tensorflow>=2.8.0
pip install pandas>=1.3.0  
pip install numpy>=1.21.0
pip install matplotlib>=3.4.0
pip install seaborn>=0.11.0
pip install scikit-learn>=1.0.0

# Kiểm tra installation
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

### 3. 📁 **Kiểm tra dữ liệu**

```bash
# Vào thư mục dataset
cd dataset

# Kiểm tra files có đủ không
ls -la
# Phải thấy:
# train.csv (73MB - 42,000 samples)
# test.csv (48MB - 28,000 samples)  
# sample_submission.csv (nhỏ - format mẫu)

# Kiểm tra dữ liệu train
head -n 2 train.csv
# Output mong đợi:
# label,pixel0,pixel1,pixel2,...,pixel783
# 1,0,0,0,0,0,0,0,0,0,...
```

### 4. ▶️ **Chạy training**

```bash
# Về thư mục gốc
cd ..

# Chạy training (mất ~15 phút CPU, ~3 phút GPU)
python resolve.py

# Output mong đợi:
# ==================================================
# NHẬN DẠNG CHỮ SỐ MNIST VỚI CNN  
# ==================================================
# Đang tải dữ liệu...
# Kích thước dữ liệu:
# X_train: (33600, 28, 28, 1)
# X_val: (8400, 28, 28, 1)
# X_test: (28000, 28, 28, 1)
# ...
# Validation Accuracy: 0.9940
```

### 5. 📊 **Kiểm tra kết quả**

Sau khi chạy xong, sẽ có các files mới:

```bash
ls -la *.png *.csv
# training_history.png     - Biểu đồ training process
# confusion_matrix.png     - Ma trận nhầm lẫn  
# predictions_sample.png   - Ví dụ predictions
# submission.csv           - File kết quả cuối (28,000 dự đoán)
```

### 6. 🔧 **Troubleshooting Installation**

#### Lỗi thường gặp:

**1. TensorFlow installation failed:**
```bash
# Thử cài TensorFlow CPU version
pip install tensorflow-cpu>=2.8.0

# Hoặc dùng conda
conda install tensorflow-gpu  # Nếu có GPU
conda install tensorflow      # CPU only
```

**2. Memory Error:**
```bash
# Giảm batch size trong resolve.py
# Tìm dòng: batch_size=128
# Thay thành: batch_size=64 hoặc batch_size=32
```

**3. Permission Error (Windows):**
```bash
# Chạy PowerShell as Administrator
# Hoặc thay đường dẫn không có space:
# Thay vì: "G:\My Drive\..." 
# Dùng: "C:\projects\buoi3_e1"
```

**4. CUDA Error (GPU):**
```bash
# Kiểm tra GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Nếu không có GPU, TensorFlow tự chuyển sang CPU
# Performance sẽ chậm hơn nhưng vẫn chạy được
```

## Cấu trúc dự án

```
buoi3_e1/
├── dataset/                 # Thư mục chứa dữ liệu
│   ├── train.csv           # Dữ liệu training (42,000 samples)
│   ├── test.csv            # Dữ liệu test (28,000 samples)
│   └── sample_submission.csv
├── resolve.py              # File code chính
├── requirements.txt        # Dependencies
├── README.md              # File hướng dẫn này
└── outputs/               # Thư mục kết quả (tự động tạo)
    ├── submission.csv     # File submission
    ├── training_history.png
    ├── confusion_matrix.png
    └── predictions_sample.png
```

## Kết quả thực tế đạt được

### 🎉 **Hiệu suất xuất sắc**
- **Validation Accuracy**: **99.40%** 
- **Training Accuracy**: **98.90%**
- **Training Time**: 41 epochs (với Early Stopping)
- **Total Training Time**: ~15 phút trên CPU

### 📊 **Classification Report chi tiết**
```
              precision    recall  f1-score   support
           0       1.00      1.00      1.00       827
           1       1.00      0.99      0.99       937
           2       0.99      1.00      0.99       835
           3       1.00      1.00      1.00       870
           4       0.99      0.99      0.99       814
           5       0.99      1.00      1.00       759
           6       1.00      0.99      1.00       827
           7       0.99      0.99      0.99       880
           8       0.99      1.00      0.99       813
           9       1.00      0.99      0.99       838

    accuracy                           0.99      8400
   macro avg       0.99      0.99      0.99      8400
weighted avg       0.99      0.99      0.99      8400
```

### 📈 **Quá trình training**
- **Epoch 1**: 57.95% accuracy → 41: 99.40% accuracy
- **Learning Rate Reduction**: 0.001 → 0.0002 → 0.0001 (tự động)
- **Early Stopping**: Dừng khi không cải thiện trong 10 epochs
- **Best Performance**: Epoch 31 với 99.40% val_accuracy

## Kết quả mong đợi

- **Training Accuracy**: ~99%+ ✅
- **Validation Accuracy**: ~98%+ ✅ (Đạt 99.40%)
- **Test Performance**: Dự kiến đạt accuracy cao trên leaderboard

## Giải thích code chi tiết

### 🏗️ **Cấu trúc Class DigitRecognizer**

```python
class DigitRecognizer:
    def __init__(self):
        self.model = None      # Lưu trữ CNN model
        self.history = None    # Lưu training history
```

#### 1. `load_data()` - Tải và xử lý dữ liệu
```python
def load_data(self):
    # Bước 1: Đọc CSV files
    train_df = pd.read_csv('dataset/train.csv')  # 42,000 samples
    test_df = pd.read_csv('dataset/test.csv')    # 28,000 samples
    
    # Bước 2: Tách features và labels  
    X = train_df.drop('label', axis=1).values    # 784 pixel values
    y = train_df['label'].values                 # Digit labels (0-9)
    
    # Bước 3: Reshape thành ảnh 28x28
    X = X.reshape(-1, 28, 28, 1)                # (N, 28, 28, 1)
    
    # Bước 4: Normalize về [0,1]
    X = X.astype('float32') / 255.0              # Từ [0,255] → [0,1]
    
    # Bước 5: One-hot encoding cho labels
    y = keras.utils.to_categorical(y, 10)        # [3] → [0,0,0,1,0,0,0,0,0,0]
    
    # Bước 6: Chia train/validation (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
```

**Giải thích quan trọng:**
- **Reshape**: MNIST gốc là vector 784 chiều → CNN cần input 2D (28x28)
- **Normalization**: Giúp model học nhanh hơn và ổn định hơn
- **One-hot encoding**: CNN output 10 neurons → cần label dạng [0,0,1,0,...]

#### 2. `build_model()` - Xây dựng kiến trúc CNN

```python
def build_model(self):
    model = keras.Sequential([
        # Data Augmentation (ngẫu nhiên hóa ảnh)
        layers.RandomRotation(0.1),        # Xoay ±10%
        layers.RandomZoom(0.1),            # Zoom ±10%
        layers.RandomTranslation(0.1, 0.1), # Dịch chuyển ±10%
        
        # Convolutional Block 1
        layers.Conv2D(32, (3,3), activation='relu'),  # 32 filters 3x3
        layers.BatchNormalization(),                   # Chuẩn hóa batch
        layers.Conv2D(32, (3,3), activation='relu'),  # Thêm 1 Conv layer
        layers.MaxPooling2D((2,2)),                   # Giảm size xuống 1/2
        layers.Dropout(0.25),                         # Ngẫu nhiên tắt 25% neurons
        
        # Convolutional Block 2 (tương tự nhưng 64 filters)
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Convolutional Block 3 (128 filters)
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Chuyển từ 2D sang 1D
        layers.Flatten(),
        
        # Dense Layers cho classification
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'), 
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 classes (0-9)
    ])
```

**Giải thích từng thành phần:**

1. **Data Augmentation**: Tạo thêm dữ liệu bằng cách biến đổi ảnh
2. **Conv2D**: Phát hiện patterns (edges, shapes) với sliding window
3. **BatchNormalization**: Chuẩn hóa để training ổn định hơn
4. **MaxPooling2D**: Giảm kích thước, giữ lại thông tin quan trọng nhất
5. **Dropout**: Ngẫu nhiên "tắt" neurons để tránh overfitting
6. **Dense**: Fully connected layers cho classification cuối cùng

#### 3. `train_model()` - Huấn luyện với Callbacks

```python
def train_model(self, X_train, X_val, y_train, y_val, epochs=50):
    callbacks = [
        # Dừng sớm nếu không cải thiện
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,                    # Chờ 10 epochs
            restore_best_weights=True       # Lấy lại weights tốt nhất
        ),
        
        # Giảm learning rate khi loss plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,                     # Giảm 5 lần (×0.2)
            patience=5,                     # Chờ 5 epochs
            min_lr=0.0001                   # Không giảm dưới 0.0001
        )
    ]
```

**Callbacks giải thích:**
- **EarlyStopping**: Tự động dừng khi model không cải thiện → tiết kiệm thời gian
- **ReduceLROnPlateau**: Giảm learning rate khi "mắc kẹt" → giúp model học tinh hơn

#### 4. `evaluate_model()` - Đánh giá chi tiết

```python
def evaluate_model(self, X_val, y_val):
    # Dự đoán
    y_pred = self.model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)  # [0.1,0.9,0.0,...] → 1
    y_true_classes = np.argmax(y_val, axis=1)   # One-hot → class index
    
    # Tính accuracy
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    
    # Classification report (precision, recall, f1-score)
    print(classification_report(y_true_classes, y_pred_classes))
    
    # Confusion Matrix - xem model nhầm lẫn ở đâu
    cm = confusion_matrix(y_true_classes, y_pred_classes)
```

#### 5. Flow tổng thể của chương trình

```python
def main():
    # 1. Khởi tạo
    recognizer = DigitRecognizer()
    
    # 2. Load và preprocess data
    X_train, X_val, y_train, y_val, X_test = recognizer.load_data()
    
    # 3. Build CNN model  
    model = recognizer.build_model()
    
    # 4. Train với validation
    history = recognizer.train_model(X_train, X_val, y_train, y_val)
    
    # 5. Evaluate và visualize
    accuracy = recognizer.evaluate_model(X_val, y_val)
    recognizer.plot_training_history()
    
    # 6. Predict trên test set
    submission = recognizer.predict_and_save(X_test)
```

## Hyperparameters

| Parameter | Value | Giải thích |
|-----------|-------|------------|
| Learning Rate | Adam default (0.001) | Tự động điều chỉnh |
| Batch Size | 128 | Cân bằng tốc độ và memory |
| Epochs | 50 | Với early stopping |
| Dropout | 0.25-0.5 | Tăng dần qua các layer |
| Data Aug Rate | 0.1 | Augmentation nhẹ |

## Mở rộng và cải tiến

### Đã implement:
- [x] CNN cơ bản với Conv-ReLU-Pool
- [x] BatchNormalization  
- [x] Data augmentation nhẹ
- [x] Dropout tuning
- [x] Learning rate scheduling

### Có thể thêm:
- [ ] **Cutout**: Random mask patches
- [ ] **Mixup**: Trộn ảnh và labels
- [ ] **TTA (Test Time Augmentation)**: Augment khi predict
- [ ] **Ensemble**: Kết hợp nhiều models
- [ ] **Advanced architectures**: ResNet, DenseNet

## Tác giả

- **Tên**: Nguyễn Trung Kiệt
- **Email**: [your-email@example.com]
- **GitHub**: [nguyentrungkiet](https://github.com/nguyentrungkiet)

## Phiên bản

- **v1.0**: Implementation cơ bản với CNN baseline
- **Framework**: TensorFlow/Keras 2.8+
- **Python**: 3.8+

## License

Dự án này được phát triển cho mục đích học tập trong Olympic AI.

---

### Lưu ý quan trọng

1. **Thời gian training**: Khoảng 10-30 phút tùy GPU
2. **Memory requirement**: Tối thiểu 4GB RAM
3. **GPU support**: Khuyến khích sử dụng GPU để tăng tốc
4. **Results**: Kết quả có thể khác nhau do random initialization

### 🔧 **Troubleshooting thực tế**

#### ❌ **Lỗi thường gặp và cách fix:**

**1. `Out of Memory` Error:**
```python
# SOLUTION: Giảm batch_size trong resolve.py
# Tìm dòng 152:
batch_size=128,

# Thay thành:  
batch_size=64,    # Hoặc 32 nếu vẫn lỗi
```

**2. `Slow Training` (>30 phút):**
```bash
# CHECK: GPU có được sử dụng không?
python -c "import tensorflow as tf; print(f'GPU Available: {tf.config.list_physical_devices(\"GPU\")}')"

# SOLUTION: Nếu không có GPU
# - Giảm epochs từ 50 → 20
# - Giảm model size (ít Conv layers)
```

**3. `Poor Accuracy` (<95%):**
```python
# POSSIBLE CAUSES & SOLUTIONS:
# 1. Data không đúng format → Kiểm tra load_data()
# 2. Learning rate quá cao → Thử lr=0.0001  
# 3. Overfitting → Tăng dropout lên 0.5
# 4. Underfitting → Tăng model capacity hoặc epochs
```

**4. `Module not found` Error:**
```bash
# SOLUTION: Reinstall dependencies
pip uninstall tensorflow pandas numpy matplotlib
pip install -r requirements.txt

# Hoặc dùng conda:
conda install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

#### 💡 **Tips để cải thiện kết quả:**

**1. Hyperparameter Tuning:**
```python
# Thử các giá trị khác nhau:
- learning_rate: [0.001, 0.0005, 0.0001]
- dropout_rate: [0.2, 0.3, 0.4, 0.5]  
- batch_size: [32, 64, 128, 256]
- data_augmentation: Tăng/giảm rotation, zoom
```

**2. Model Architecture:**
```python
# Thêm layers:
layers.Conv2D(256, (3,3), activation='relu'),  # Block 4
layers.GlobalAveragePooling2D(),               # Thay Flatten

# Hoặc thử Transfer Learning:
base_model = tf.keras.applications.MobileNetV2(...)
```

**3. Advanced Techniques:**
```python
# Cutout (Random Erasing)
def cutout(image, mask_size=8):
    h, w = image.shape[:2]
    y = np.random.randint(h)
    x = np.random.randint(w)
    image[y:y+mask_size, x:x+mask_size] = 0

# Mixup  
def mixup(x1, x2, y1, y2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y
```

#### 📈 **Monitoring Training:**

```python
# Thêm vào train_model():
# Thêm TensorBoard callback
keras.callbacks.TensorBoard(
    log_dir='logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

# Chạy TensorBoard để theo dõi:
# tensorboard --logdir=logs
```

#### 🎯 **Benchmarks để so sánh:**

| Method | Validation Accuracy | Training Time |
|--------|-------------------|---------------|
| **Our CNN** | **99.40%** | **15 min (CPU)** |
| Simple MLP | ~97.5% | 5 min |
| Basic CNN | ~98.5% | 10 min |
| ResNet-50 | ~99.6% | 30 min |
| Ensemble (5 models) | ~99.7% | 75 min |

→ **Kết luận**: Model của chúng ta đạt balance tốt giữa accuracy và training time!

Chúc bạn thành công với dự án! 🚀