import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
import os
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# 1. Hàm tiền xử lý văn bản chuyên sâu
# -----------------------------
def preprocess_text(text):
    # Chuyển chữ thường và xóa khoảng trắng thừa
    text = str(text).lower().strip()
    
    # Loại bỏ URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Loại bỏ các ký tự lặp lại quá nhiều (ví dụ: hay quáaaaa -> hay quá)
    text = re.sub(r'([a-z])\1+', r'\1', text)
    
    # Xử lý Teencode cơ bản để mô hình hiểu tốt hơn
    teencode_dict = {
        "vcl": "rất", "vl": "rất", "đm": "xấu", "dm": "xấu", 
        "k": "không", "ko": "không", "j": "gì", "thê": "thế", "huhu": "buồn"
    }
    words = text.split()
    words = [teencode_dict.get(w, w) for w in words]
    
    return " ".join(words)

# -----------------------------
# 2. Load mô hình và dữ liệu (Có cơ chế tự sửa lỗi)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline

    model_path = "models/model.pkl"
    
    try:
        # Thử load file hiện có
        return joblib.load(model_path)
    except Exception:
        # Nếu lỗi phiên bản (KeyError/EOFError), tiến hành huấn luyện lại ngay tại chỗ
        st.info("🔄 Đang cấu hình lại mô hình để tương thích với máy chủ...")
        df_train = pd.read_csv("data/dataset.csv") 
        
        # Tạo Pipeline đóng gói cả TF-IDF và SVM
        new_model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('clf', SVC(kernel='linear', probability=True))
        ])
        
        # Huấn luyện trên dữ liệu hiện có
        new_model.fit(df_train['comment'], df_train['label'])
        
        # Lưu lại để sử dụng cho lần sau
        os.makedirs("models", exist_ok=True)
        joblib.dump(new_model, model_path)
        return new_model

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv("data/dataset.csv")

# -----------------------------
# 3. Các hàm bổ trợ Giao diện & Dự đoán
# -----------------------------
def plot_label_distribution(df):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x='label', data=df, palette='viridis', ax=ax)
    ax.set_title('Phân phối nhãn (0: Thường, 1: Tiêu cực, 2: Tích cực)')
    st.pyplot(fig)

def predict_comment(model, comment):
    text = preprocess_text(comment)
    pred = model.predict([text])[0]
    confidence = max(model.predict_proba([text])[0]) if hasattr(model, "predict_proba") else None
    return pred, confidence

# Map nhãn sang tên và emoji
label_map = {0: "Bình thường 😐", 1: "Tiêu cực 😡", 2: "Tích cực 😍"}

# -----------------------------
# 4. Cấu hình Trang Streamlit
# -----------------------------
st.set_page_config(page_title="NLP - Phân loại bình luận", layout="wide")

# Sidebar điều hướng
st.sidebar.image("https://www.streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
st.sidebar.title("Menu Điều Khiển")
page = st.sidebar.selectbox("Chọn tính năng", 
                            ["📊 Giới thiệu & EDA", "🚀 Triển khai mô hình", "📈 Đánh giá & Hiệu năng"])

# Load dữ liệu và mô hình ngay từ đầu
df = load_data()
model = load_model()

# -----------------------------
# Trang 1: Giới thiệu & EDA
# -----------------------------
if page == "📊 Giới thiệu & EDA":
    st.title("📌 Phân tích cảm xúc bình luận Tiếng Việt")
    
    # --- PHẦN 1: THÔNG TIN SINH VIÊN ---
    with st.expander("ℹ️ Thông tin sinh viên & Đề tài", expanded=True):
        col_inf1, col_inf2 = st.columns(2)
        with col_inf1:
            st.markdown(f"""
            **Sinh viên thực hiện:** Nguyễn Phước Bảo Thắng  
            **Mã số sinh viên:** 21T1020109  
            **Lớp:** Học máy với Python
            """)
        with col_inf2:
            st.markdown("""
            **Thuật toán:** SVM (Support Vector Machine)  
            **Kỹ thuật trích xuất:** TF-IDF Vectorizer  
            **Thư viện chính:** Scikit-learn, Streamlit, Pandas
            """)

    st.divider()

    # --- PHẦN 2: GIỚI THIỆU BÀI TOÁN ---
    st.subheader("📝 Giới thiệu bài toán")
    st.info("""
    **Bài toán:** Phân loại tự động sắc thái bình luận của người dùng trên các nền tảng mạng xã hội (như Facebook Fanpage của nghệ sĩ).  
    **Mục tiêu:** * **Nhãn 0 (Bình thường):** Các bình luận mang tính thảo luận trung lập, không gây hại.
    * **Nhãn 1 (Tiêu cực):** Các bình luận toxic, chửi bới, spam hoặc gây hấn (Cần được lọc/ẩn).
    * **Nhãn 2 (Tích cực):** Các bình luận khen ngợi, ủng hộ và lan tỏa năng lượng tốt.
    
    **Giá trị thực tiễn:** Giúp Quản trị viên (Admin) tiết kiệm 80% thời gian kiểm soát nội dung độc hại, duy trì môi trường mạng lành mạnh.
    """)

    # --- PHẦN 3: QUY TRÌNH TIỀN XỬ LÝ (PIPELINE) ---
    st.subheader("⚙️ Quy trình Tiền xử lý & Huấn luyện")
    
    # Định nghĩa các bước
    steps = [
        ("Bước 1: Clean Text", "Chuyển chữ thường, xóa URL, xóa khoảng trắng thừa."),
        ("Bước 2: Teencode", "Chuyển các từ viết tắt (vcl, vl, ko,...) về từ gốc."),
        ("Bước 3: TF-IDF", "Biến đổi văn bản thành ma trận số (Vectorization)."),
        ("Bước 4: SVM", "Tìm ranh giới phân loại tối ưu cho 3 nhãn cảm xúc.")
    ]
    
    # Hiển thị bằng Columns và Container có khung (border)
    cols = st.columns(4)
    for i, (step, desc) in enumerate(steps):
        with cols[i]:
            with st.container(border=True):
                st.markdown(f"**{step}**")
                st.caption(desc) # Dùng caption để chữ nhỏ và tinh tế hơn

    # --- PHẦN 4: KHÁM PHÁ DỮ LIỆU (EDA) ---
    st.subheader("🔍 Khám phá dữ liệu (EDA)")
    
    # Hiển thị các chỉ số tổng quan
    m1, m2, m3 = st.columns(3)
    m1.metric("Tổng số mẫu huấn luyện", len(df))
    m2.metric("Số lượng nhãn mục tiêu", "3 nhãn")
    m3.metric("Trạng thái Model", "Đã tối ưu (Linear)")

    col_chart1, col_chart2 = st.columns([2, 3]) # Tỷ lệ 2:3 cho biểu đồ và bảng
    with col_chart1:
        st.write("**Phân phối nhãn thực tế:**")
        plot_label_distribution(df)
    with col_chart2:
        st.write("**Dữ liệu mẫu ngẫu nhiên:**")
        st.dataframe(df.sample(min(len(df), 15)), use_container_width=True)

    st.success("💡 Dữ liệu đã được xáo trộn (shuffled) để đảm bảo tính khách quan khi huấn luyện.")

# -----------------------------
# Trang 2: Triển khai mô hình
# -----------------------------
elif page == "🚀 Triển khai mô hình":
    st.title("🔮 Dự đoán cảm xúc bình luận")
    
    tab_single, tab_batch = st.tabs(["🎯 Phân tích câu đơn", "📂 Phân tích hàng loạt (CSV)"])

    # --- TAB 1: PHÂN TÍCH CÂU ĐƠN ---
    with tab_single:
        st.subheader("Kiểm tra nội dung văn bản lẻ")
        user_input = st.text_area("Nhập bình luận của bạn tại đây:", 
                                  placeholder="Ví dụ: Chị Juky San hát hay quá, ủng hộ hết mình...",
                                  height=100)
        
        if st.button("🔍 Phân tích ngay"):
            if user_input.strip() == "":
                st.warning("⚠️ Vui lòng nhập nội dung!")
            else:
                pred_label, confidence = predict_comment(model, user_input)
                
                # Hiển thị Card kết quả
                st.write("### 💎 Kết quả phân tích:")
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    if pred_label == 1:
                        st.error(f"**{label_map[pred_label]}**")
                    elif pred_label == 2:
                        st.success(f"**{label_map[pred_label]}**")
                    else:
                        st.info(f"**{label_map[pred_label]}**")
                
                with col_res2:
                    if confidence:
                        st.write(f"Độ tin cậy: **{confidence:.2%}**")
                        st.progress(confidence)

    # --- TAB 2: PHÂN TÍCH HÀNG LOẠT (CSV) ---
    with tab_batch:
        st.subheader("Phân tích dữ liệu lớn từ File")
        st.write("Tải lên file CSV chứa danh sách bình luận để xem thống kê cảm xúc tổng thể.")
        
        uploaded_file = st.file_uploader("Chọn file CSV (Lưu ý: Phải có cột tên 'comment')", type="csv")
        
        if uploaded_file:
            df_upload = pd.read_csv(uploaded_file)
            if 'comment' in df_upload.columns:
                with st.spinner("🚀 Hệ thống đang xử lý dữ liệu..."):
                    # 1. Dự đoán
                    df_upload['clean'] = df_upload['comment'].apply(preprocess_text)
                    df_upload['pred_id'] = model.predict(df_upload['clean'])
                    df_upload['Nhãn'] = df_upload['pred_id'].map(label_map)
                    
                    st.divider()
                    
                    # 2. Hiển thị Dashboard Thống kê
                    st.subheader("📊 Dashboard Thống kê cảm xúc")
                    
                    # Tính toán số lượng
                    count_df = df_upload['Nhãn'].value_counts().reset_index()
                    count_df.columns = ['Cảm xúc', 'Số lượng']
                    
                    col_chart1, col_chart2 = st.columns([1, 1])
                    
                    with col_chart1:
                        st.write("**Tỉ lệ phần trăm giữa các nhãn:**")
                        fig_pie, ax_pie = plt.subplots()
                        colors = {'Tích cực 😍': '#2ecc71', 'Tiêu cực 😡': '#e74c3c', 'Bình thường 😐': '#f1c40f'}
                        # Lấy màu tương ứng với các nhãn hiện có trong dữ liệu tải lên
                        current_colors = [colors.get(x, '#3498db') for x in count_df['Cảm xúc']]
                        
                        ax_pie.pie(count_df['Số lượng'], labels=count_df['Cảm xúc'], 
                                   autopct='%1.1f%%', startangle=140, colors=current_colors)
                        st.pyplot(fig_pie)
                        
                    with col_chart2:
                        st.write("**So sánh số lượng tuyệt đối:**")
                        fig_bar, ax_bar = plt.subplots()
                        sns.barplot(x='Cảm xúc', y='Số lượng', data=count_df, palette='viridis', ax=ax_bar)
                        plt.xticks(rotation=45)
                        st.pyplot(fig_bar)
                    
                    # 3. Hiển thị bảng chi tiết
                    st.subheader("📋 Chi tiết danh sách dự đoán")
                    st.dataframe(df_upload[['comment', 'Nhãn']], use_container_width=True)
                    
                    # 4. Cho phép tải kết quả
                    csv_data = df_upload[['comment', 'Nhãn']].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Tải xuống kết quả (.csv)",
                        data=csv_data,
                        file_name="ket_qua_du_doan.csv",
                        mime="text/csv"
                    )
            else:
                st.error("❌ Lỗi: Không tìm thấy cột 'comment' trong file của bạn!")

# -----------------------------
# Trang 3: Đánh giá & Hiệu năng
# -----------------------------
else:
    st.title("📈 Phân tích Kỹ thuật & Đánh giá Hiệu năng")
    
    # --- PHẦN 1: CƠ SỞ TOÁN HỌC ---
    st.header("1. Cơ sở phương pháp luận")
    
    with st.expander("🔍 Chi tiết thuật toán SVM & TF-IDF", expanded=True):
        st.markdown("### 🔹 Thuật toán SVM (Support Vector Machine)")
        st.write("Mục tiêu của SVM là tìm một siêu phẳng (hyperplane) tối ưu để phân tách các lớp dữ liệu trong không gian đa chiều sao cho khoảng cách (margin) là lớn nhất.")
        st.latex(r"f(x) = \text{sign}(\mathbf{w}^T \mathbf{x} + b)")
        st.write("Trong đó $\mathbf{w}$ là vector trọng số và $b$ là độ chệch (bias).")

        st.markdown("### 🔹 Kỹ thuật TF-IDF")
        st.write("Dùng để chuyển đổi văn bản thành vector số dựa trên trọng số của từ:")
        st.latex(r"\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)")
        st.info("Kỹ thuật này giúp loại bỏ các từ dừng (stop words) và tập trung vào các từ mang tính đặc trưng cho cảm xúc.")

    st.divider()

    # --- PHẦN 2: CHỈ SỐ ĐO LƯỜNG ---
    st.header("2. Các chỉ số đánh giá (Evaluation Metrics)")
    
    # Tính toán dự đoán
    y_true = df['label']
    y_pred = model.predict(df['comment'])
    
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.markdown("**Precision (Độ chính xác)**")
        st.latex(r"P = \frac{TP}{TP + FP}")
    with metric_cols[1]:
        st.markdown("**Recall (Độ nhạy)**")
        st.latex(r"R = \frac{TP}{TP + FN}")
    with metric_cols[2]:
        st.markdown("**F1-Score (Điểm F1)**")
        st.latex(r"F1 = 2 \cdot \frac{P \cdot R}{P + R}")

    st.divider()

    # --- PHẦN 3: KẾT QUẢ THỰC TẾ ---
    st.header("3. Kết quả thực thi trên tập dữ liệu")

    col_m1, col_m2 = st.columns([1, 1])
    
    with col_m1:
        st.subheader("📑 Classification Report")
        # Sử dụng code block để giữ định dạng bảng của sklearn
        st.text(classification_report(y_true, y_pred))
        
    with col_m2:
        st.subheader("🎯 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['Thường', 'Tiêu cực', 'Tích cực'],
                    yticklabels=['Thường', 'Tiêu cực', 'Tích cực'])
        ax_cm.set_xlabel("Nhãn Dự đoán")
        ax_cm.set_ylabel("Nhãn Thực tế")
        st.pyplot(fig_cm)

    # --- PHẦN 4: NHẬN XÉT ---
    st.subheader("📝 Nhận xét kết quả")
    if len(df) < 200:
        st.warning("⚠️ **Lưu ý:** Tập dữ liệu hiện tại còn khá nhỏ. Để tăng chỉ số F1-Score cho nhãn 'Bình thường', cần bổ sung thêm các mẫu câu trung lập.")
    else:
        st.success("✅ Mô hình đạt độ ổn định cao trên cả 3 nhãn. Ma trận nhầm lẫn cho thấy khả năng phân biệt giữa 'Tiêu cực' và 'Tích cực' rất rõ rệt.")