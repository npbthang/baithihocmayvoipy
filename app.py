import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Hàm tiền xử lý văn bản
# -----------------------------
def preprocess_text(text):
    text = str(text).lower().strip()
    # TODO: loại bỏ URL, emoji, teencode nếu cần
    return text

# -----------------------------
# Load mô hình và dữ liệu
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    import os
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline

    model_path = "models/model.pkl"
    
    # Thử load file hiện có
    try:
        return joblib.load(model_path)
    except Exception:
        # Nếu lỗi (như KeyError), tiến hành huấn luyện lại ngay tại chỗ
        st.info("Đang cấu hình lại mô hình để tương thích với máy chủ, vui lòng đợi giây lát...")
        
        # Đọc dữ liệu từ file dataset.csv bạn đã có trên GitHub
        df_train = pd.read_csv("data/dataset.csv") 
        
        # Tạo Pipeline chuẩn như bạn đã làm
        new_model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('clf', SVC(kernel='linear', probability=True))
        ])
        
        # Huấn luyện (Giả sử cột văn bản là 'comment' và nhãn là 'label')
        new_model.fit(df_train['comment'], df_train['label'])
        
        # Lưu đè lại file pkl để lần sau không phải train lại
        os.makedirs("models", exist_ok=True)
        joblib.dump(new_model, model_path)
        
        return new_model

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv("data/dataset.csv")

# -----------------------------
# Biểu đồ và EDA
# -----------------------------
def plot_label_distribution(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x='label', data=df)
    plt.title('Phân phối nhãn bình luận')
    st.pyplot(plt)

def plot_correlation(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Ma trận tương quan')
        st.pyplot(plt)
    else:
        st.write("Không có đặc trưng số để hiển thị ma trận tương quan.")

# -----------------------------
# Hàm dự đoán bình luận
# -----------------------------
def predict_comment(model, comment):
    text = preprocess_text(comment)
    pred = model.predict([text])[0]
    confidence = max(model.predict_proba([text])[0]) if hasattr(model, "predict_proba") else None
    return pred, confidence

def predict_bulk(model, df):
    df = df.copy()
    df['comment_clean'] = df['comment'].apply(preprocess_text)
    df['pred_label'] = model.predict(df['comment_clean'])
    if hasattr(model, "predict_proba"):
        df['confidence'] = df['comment_clean'].apply(lambda x: max(model.predict_proba([x])[0]))
    else:
        df['confidence'] = None
    return df

# Map nhãn sang tên dễ hiểu
label_map = {
    0: "Bình thường",
    1: "Tiêu cực",
    2: "Tích cực"
}

# -----------------------------
# Cấu hình Streamlit
# -----------------------------
st.set_page_config(page_title="Phân loại bình luận tiếng Việt", layout="wide")
st.title("Ứng dụng Phân loại bình luận Tiêu cực - Tích cực - Bình thường")

# Thanh điều hướng 3 trang
page = st.sidebar.radio("Chọn trang", 
                        ("Giới thiệu & EDA", "Triển khai mô hình", "Đánh giá & Hiệu năng"))

# Load dữ liệu và mô hình
data_load_state = st.text("Đang tải dữ liệu và mô hình...")
df = load_data()
model = load_model()
data_load_state.text("Tải xong!")

# -----------------------------
# Trang 1: Giới thiệu & EDA
# -----------------------------
if page == "Giới thiệu & EDA":
    st.header("Giới thiệu bài toán & Khám phá dữ liệu")
    st.markdown("""
    - **Đề tài:** Phân loại bình luận tiếng Việt thành Bình thường – Tiêu cực – Tích cực
    - **Sinh viên:** Nguyễn Phước Bảo Thắng , MSSV: 21T1020109
    - **Giá trị thực tiễn:** Hỗ trợ quản lý bình luận trên mạng xã hội, tự động ẩn bình luận xấu, bảo vệ cộng đồng mạng.
    """)

    st.subheader("Dữ liệu mẫu")
    st.dataframe(df.sample(10))

    st.subheader("Phân phối nhãn bình luận")
    plot_label_distribution(df)

    st.subheader("Phân tích đặc trưng")
    plot_correlation(df)

    st.markdown("""
    **Nhận xét:**  
    - Dữ liệu có phân phối nhãn như trên.  
    - Cần tiền xử lý kỹ để mô hình đạt hiệu quả cao.
    """)

# -----------------------------
# Trang 2: Triển khai mô hình
# -----------------------------
elif page == "Triển khai mô hình":
    st.header("Dự đoán bình luận")

    mode = st.radio("Chọn chế độ dự đoán", ["Một bình luận", "Nhiều bình luận từ CSV"])

    if mode == "Một bình luận":
        user_input = st.text_area("Nhập bình luận tiếng Việt:", height=150)
        if st.button("Dự đoán"):
            if user_input.strip() == "":
                st.warning("Vui lòng nhập bình luận trước khi dự đoán!")
            else:
                with st.spinner("Đang phân loại..."):
                    pred_label, confidence = predict_comment(model, user_input)
                    pred_text = label_map.get(pred_label, "Không xác định")
                    st.success(f"Kết quả dự đoán: **{pred_text}**")
                    if confidence is not None:
                        st.info(f"Độ tin cậy dự đoán: {confidence:.2%}")

    else:  # Nhiều bình luận từ CSV
        uploaded_file = st.file_uploader("Tải file CSV chứa cột 'comment'", type="csv")
        if uploaded_file:
            df_upload = pd.read_csv(uploaded_file)
            if 'comment' not in df_upload.columns:
                st.error("File CSV phải có cột 'comment'.")
            else:
                with st.spinner("Đang dự đoán hàng loạt..."):
                    df_result = predict_bulk(model, df_upload)
                    df_result['pred_label_name'] = df_result['pred_label'].map(label_map)
                    st.success("Dự đoán hoàn tất!")
                    st.dataframe(df_result[['comment', 'pred_label_name', 'confidence']])

                    # Cho phép tải file kết quả
                    csv = df_result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Tải kết quả dự đoán",
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv'
                    )

# -----------------------------
# Trang 3: Đánh giá & Hiệu năng
# -----------------------------
else:
    st.header("Đánh giá hiệu năng mô hình")
    y_true = df['label']
    y_pred = model.predict(df['comment'].apply(preprocess_text))

    st.subheader("Báo cáo phân loại")
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    st.text(classification_report(y_true, y_pred))

    st.subheader("Ma trận nhầm lẫn")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    st.pyplot(fig)

    st.subheader("Phân tích sai số")
    st.write("""
    - Mô hình có thể nhầm lẫn giữa bình luận trung tính và tích cực do ngôn ngữ tương đồng.
    - Bình luận chứa teencode hoặc emoji phức tạp có thể gây sai lệch dự đoán.
    - Hướng cải thiện: mở rộng tập dữ liệu, cải tiến tiền xử lý, tinh chỉnh mô hình.
    """)