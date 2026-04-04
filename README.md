# Ứng dụng Phân loại bình luận Tiếng Việt: Tiêu cực - Tích cực - Bình thường

Ứng dụng này sử dụng mô hình **PhoBERT + SVM** để phân loại bình luận tiếng Việt trên mạng xã hội (TikTok, Facebook) thành **3 nhãn**:  

- **Tiêu cực**  
- **Tích cực**  
- **Bình thường**  

**Mục tiêu:** Giúp lọc bình luận xấu, hỗ trợ quản lý cộng đồng mạng, và phân tích cảm xúc người dùng.

---

## Tính năng ứng dụng

1. **Giới thiệu & Khám phá dữ liệu (EDA)**
   - Xem dữ liệu mẫu từ `dataset.csv`
   - Phân phối nhãn bình luận
   - Biểu đồ phân tích đặc trưng và ma trận tương quan

2. **Dự đoán bình luận**
   - Nhập bình luận tiếng Việt hoặc tải file CSV
   - Dự đoán loại bình luận (Tiêu cực, Tích cực, Bình thường)
   - Hiển thị độ tin cậy dự đoán
   - Tải kết quả dự đoán ra CSV

3. **Đánh giá & Hiệu năng mô hình**
   - Báo cáo phân loại (Accuracy, F1-score)
   - Ma trận nhầm lẫn
   - Nhận xét về sai số và hướng cải thiện

---

## Cấu trúc thư mục dự án

Sinh viên: Nguyễn Phước Bảo Thắng
MSSV: 21T1020109
Email: 21T1020109@husc.edu.vn
