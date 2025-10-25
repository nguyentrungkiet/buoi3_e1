# Nhận Dạng Chữ Số MNIST với CNN

## Mô tả dự án

Đây là dự án nhận dạng chữ số viết tay từ dataset MNIST sử dụng Convolutional Neural Network (CNN). Dự án được phát triển trong khuôn khổ Olympic AI - Buổi 3 - Bài tập E1.

## Mục tiêu

- **Mục tiêu chính**: Làm quen với CNN cơ bản
- **Dataset**: Digit Recognizer (MNIST) 
- **Metric đánh giá**: Accuracy
- **Kiến trúc**: CNN với các layer Conv–ReLU–Pool, BatchNorm, data augmentation

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

### Kỹ thuật nâng cao được áp dụng

- ✅ **BatchNormalization**: Cải thiện tốc độ hội tụ và ổn định training
- ✅ **Data Augmentation**: Tăng cường dữ liệu để tránh overfitting
- ✅ **Dropout**: Regularization với tỷ lệ 0.25-0.5
- ✅ **Early Stopping**: Dừng training khi validation accuracy không cải thiện
- ✅ **Learning Rate Scheduling**: Giảm learning rate khi loss không giảm

## Cài đặt và chạy

### 1. Cài đặt môi trường

```bash
# Clone repository
git clone https://github.com/nguyentrungkiet/buoi3_e1.git
cd buoi3_e1

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Đảm bảo các file dữ liệu được đặt trong thư mục `dataset/`:
```
dataset/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### 3. Chạy training

```bash
python resolve.py
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

## Kết quả mong đợi

- **Training Accuracy**: ~99%+
- **Validation Accuracy**: ~98%+
- **Test Performance**: Dự kiến đạt accuracy cao trên leaderboard

## Giải thích code

### Class DigitRecognizer

Đây là class chính chứa toàn bộ logic:

#### 1. `load_data()` 
- Tải và tiền xử lý dữ liệu
- Reshape ảnh về 28x28x1
- Normalize pixel values về [0,1]
- Chia train/validation set

#### 2. `build_model()`
- Xây dựng kiến trúc CNN
- Áp dụng data augmentation
- Compile model với Adam optimizer

#### 3. `train_model()`
- Huấn luyện mô hình với callbacks
- Early stopping và learning rate reduction
- Validation trên 20% dữ liệu train

#### 4. `evaluate_model()`
- Đánh giá performance
- Tạo confusion matrix
- Classification report chi tiết

#### 5. `predict_and_save()`
- Dự đoán trên test set
- Lưu kết quả theo format yêu cầu

### Tính năng nổi bật

1. **Reproducibility**: Set random seeds cho kết quả nhất quán
2. **Visualization**: Tự động tạo các biểu đồ phân tích
3. **Modular Design**: Code được tổ chức thành class dễ hiểu
4. **Comprehensive Evaluation**: Đánh giá đa chiều với nhiều metrics

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

### Troubleshooting

**Lỗi thường gặp:**

1. **Out of Memory**: Giảm batch_size xuống 64 hoặc 32
2. **Slow training**: Kiểm tra GPU setup
3. **Poor accuracy**: Tăng epochs hoặc điều chỉnh learning rate

**Tips để cải thiện:**

1. Thử nghiệm với different dropout rates
2. Adjust data augmentation parameters  
3. Experiment với model architecture
4. Use different optimizers (SGD, RMSprop)

Chúc bạn thành công với dự án! 🚀