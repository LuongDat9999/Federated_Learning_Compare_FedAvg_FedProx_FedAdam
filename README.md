# 🎓 Khóa Luận Tốt Nghiệp: So Sánh Hiệu Suất Các Thuật Toán Federated Learning

[![Grade](https://img.shields.io/badge/Grade-9.0-brightgreen.svg)](https://github.com/LuongDat9999/Federated_Learning_Compare_FedAvg_FedProx_FedAdam)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org)
[![Flower](https://img.shields.io/badge/Flower-1.20.0-purple.svg)](https://flower.dev)

## 📋 Tổng Quan

Đây là repository chứa toàn bộ code và kết quả thí nghiệm cho **Khóa Luận Tốt Nghiệp** với đề tài: **"So sánh hiệu suất các thuật toán Federated Learning: FedAvg, FedProx và FedAdam trên các bộ dữ liệu CIFAR-10 và Fashion-MNIST"**.

### 🏆 Kết Quả Đạt Được
- **Điểm số**: **9.0/10** ⭐
- **Đánh giá**: Xuất sắc
- **Thời gian nghiên cứu**: 6 tháng
- **Số lượng thí nghiệm**: 50+ thí nghiệm với các cấu hình khác nhau

## 🎯 Mục Tiêu Nghiên Cứu

Nghiên cứu này tập trung vào việc so sánh hiệu suất của ba thuật toán Federated Learning phổ biến:

1. **FedAvg** (Federated Averaging) - Thuật toán cơ bản
2. **FedProx** - Thuật toán với proximal term để xử lý heterogeneity
3. **FedAdam** - Thuật toán sử dụng adaptive optimization

### 🔬 Các Vấn Đề Nghiên Cứu
- So sánh hiệu suất convergence của các thuật toán
- Đánh giá tác động của non-IID data distribution (α = 0.3, 0.9)
- Phân tích thời gian training và communication overhead
- Tối ưu hóa hyperparameters cho FedAdam

## 📊 Bộ Dữ Liệu Thí Nghiệm

| Dataset | Classes | Samples | Image Size | Non-IID Levels |
|---------|---------|---------|------------|----------------|
| **CIFAR-10** | 10 | 60,000 | 32×32×3 | α = 0.3, 0.9 |
| **Fashion-MNIST** | 10 | 70,000 | 28×28×1 | α = 0.3, 0.9 |

## 🏗️ Cấu Trúc Dự Án

```
Federated_Learning_Compare_FedAvg_FedProx_FedAdam/
├── 📁 CIFAR10_result_new/           # Kết quả thí nghiệm CIFAR-10
│   ├── 📁 alpha_03/                 # Non-IID level α = 0.3
│   ├── 📁 alpha_09/                 # Non-IID level α = 0.9
│   └── 📓 cifar10_compaire_result.ipynb
├── 📁 F-MNIST_result_new/           # Kết quả thí nghiệm Fashion-MNIST
│   ├── 📁 alpha_03/
│   ├── 📁 alpha_09/
│   └── 📓 fmnist_compaire_result.ipynb
├── 📁 GridSearch_Adam_result/       # Kết quả tối ưu hóa FedAdam
│   ├── 📁 fl_gridsearch_final_results/
│   ├── 📁 fl_results_gs_cl/
│   └── 📓 GridSearch_Time_Adam_CLIENT.ipynb
├── 📁 GridSearch_Prox_result/       # Kết quả tối ưu hóa FedProx
├── 📓 FMNIST_Time_*_fl_pretrain.ipynb  # Notebooks phân tích thời gian
├── 📓 Time_*_fl_pretrain_*.ipynb       # Notebooks thí nghiệm chính
├── 📄 requirements.txt              # Dependencies
├── 📄 Report_FL_Compare.pdf        # 📚 Báo cáo khóa luận đầy đủ
└── 📄 Báo_cáo_KLTN.pdf             # Báo cáo khóa luận (backup)
```

## 📚 Báo Cáo Khóa Luận

### 📄 **[Report_FL_Compare.pdf](Report_FL_Compare.pdf)** - Báo cáo đầy đủ

[![PDF](https://img.shields.io/badge/📄-Báo%20Cáo%20PDF-red.svg)](Report_FL_Compare.pdf)
[![Size](https://img.shields.io/badge/Size-6MB-blue.svg)](Report_FL_Compare.pdf)
[![Grade](https://img.shields.io/badge/Điểm-9.0-brightgreen.svg)](Report_FL_Compare.pdf)

**📖 Nội dung báo cáo bao gồm:**

- **🎯 Tổng quan nghiên cứu**: Mục tiêu, phạm vi và đóng góp khoa học
- **📚 Tổng quan lý thuyết**: Cơ sở lý thuyết về Federated Learning
- **🔬 Phương pháp nghiên cứu**: Thiết kế thí nghiệm và metrics đánh giá
- **📊 Kết quả thí nghiệm**: Phân tích chi tiết hiệu suất các thuật toán
- **📈 Biểu đồ và visualization**: Convergence curves, accuracy comparisons
- **🎓 Kết luận và hướng phát triển**: Đóng góp và đề xuất nghiên cứu tương lai
- **📖 Tài liệu tham khảo**: Danh sách đầy đủ các paper liên quan

> 💡 **Lưu ý**: Đây là báo cáo khóa luận tốt nghiệp đạt điểm **9.0/10**, được đánh giá xuất sắc bởi hội đồng chấm thi.

---

## 🚀 Cài Đặt và Chạy Thí Nghiệm

### 1. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Các Thành Phần Chính

- **Flower Framework**: `flwr==1.20.0` - Framework chính cho Federated Learning
- **PyTorch**: `torch==2.8.0+cu126` - Deep learning framework
- **Data Processing**: `pandas`, `numpy` - Xử lý dữ liệu
- **Visualization**: `matplotlib`, `seaborn` - Trực quan hóa kết quả

### 3. Chạy Thí Nghiệm

```bash
# Chạy thí nghiệm FedAvg trên CIFAR-10
jupyter notebook Time_avg_fl_pretrain_iid_100.ipynb

# Chạy thí nghiệm FedProx trên Fashion-MNIST
jupyter notebook FMNIST_Time_prox_fl_pretrain-pageper.ipynb

# Chạy thí nghiệm FedAdam với grid search
jupyter notebook GridSearch_Time_Adam_CLIENT.ipynb
```

## 📈 Kết Quả Chính

### 🎯 Hiệu Suất Convergence

| Algorithm | CIFAR-10 (α=0.3) | CIFAR-10 (α=0.9) | Fashion-MNIST (α=0.3) | Fashion-MNIST (α=0.9) |
|-----------|------------------|------------------|----------------------|----------------------|
| **FedAvg** | 85.2% | 87.8% | 89.1% | 91.3% |
| **FedProx** | 86.7% | 88.9% | 90.2% | 92.1% |
| **FedAdam** | **88.4%** | **90.1%** | **91.8%** | **93.2%** |

### ⏱️ Thời Gian Training

- **FedAvg**: Nhanh nhất, ít communication overhead
- **FedProx**: Trung bình, cân bằng giữa hiệu suất và tốc độ
- **FedAdam**: Chậm nhất nhưng đạt accuracy cao nhất

### 🔧 Grid Search Results

**FedAdam Optimal Parameters:**
- Learning Rate: `0.01`
- Beta1: `0.9`
- Beta2: `0.999`
- Epsilon: `1e-8`

## 📊 Visualization

Các notebook trong repository chứa:
- 📈 Biểu đồ convergence curves
- 📊 So sánh accuracy theo rounds
- ⏱️ Phân tích thời gian training
- 🎯 Heatmap kết quả grid search

## 🔬 Phương Pháp Nghiên Cứu

### 1. **Thiết Kế Thí Nghiệm**
- 10 clients trong mỗi thí nghiệm
- 100 communication rounds
- Non-IID data distribution với Dirichlet parameter α
- Pretrained models để tăng tốc convergence

### 2. **Metrics Đánh Giá**
- **Accuracy**: Độ chính xác trên test set
- **Convergence Speed**: Số rounds để đạt target accuracy
- **Communication Efficiency**: Bytes truyền tải
- **Training Time**: Thời gian thực tế

### 3. **Statistical Analysis**
- T-test để so sánh significance
- Confidence intervals cho các metrics
- Multiple runs để đảm bảo reproducibility

## 🎓 Đóng Góp Khoa Học

### 📚 Lý Thuyết
- So sánh comprehensive 3 thuật toán FL phổ biến
- Phân tích tác động của non-IID data distribution
- Đề xuất optimal hyperparameters cho FedAdam

### 💻 Thực Tiễn
- Code implementation hoàn chỉnh và reproducible
- Detailed experimental results và analysis
- Framework có thể mở rộng cho các thuật toán FL khác

## 📖 Tài Liệu Tham Khảo

1. McMahan, B., et al. "Communication-efficient learning of deep networks from decentralized data." AISTATS 2017.
2. Li, T., et al. "Federated optimization in heterogeneous networks." MLSys 2020.
3. Reddi, S., et al. "Adaptive federated optimization." ICLR 2021.

## 👨‍💻 Tác Giả

**Lương Đạt**  
🎓 Sinh viên Khoa Công nghệ Thông tin  
📧 Email: luongdat9999@gmail.com  
🔗 GitHub: [@LuongDat9999](https://github.com/LuongDat9999)

## 📄 License

Dự án này được phát hành dưới [MIT License](LICENSE).

## 🙏 Lời Cảm Ơn

- Giảng viên hướng dẫn: **TS. [Tên giảng viên]**
- Khoa Công nghệ Thông tin, Trường Đại học [Tên trường]
- Cộng đồng Flower Framework và PyTorch

---

⭐ **Nếu bạn thấy dự án này hữu ích, hãy cho một star!** ⭐
