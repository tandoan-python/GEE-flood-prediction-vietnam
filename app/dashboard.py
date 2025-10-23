import streamlit as st
import requests
import json

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Dự báo Ngập lụt", layout="centered")
st.title("Ứng dụng Dự báo Nguy cơ Ngập lụt")
st.markdown("Nhập các thông số địa hình và chuỗi mưa dự báo để nhận được xác suất ngập lụt.")

# --- GIAO DIỆN NHẬP LIỆU ---
with st.sidebar:
    st.header("Thông số đầu vào")
    st.subheader("Đặc trưng Địa hình (Tĩnh)")
    elevation = st.number_input("Độ cao (mét)", -50.0, 3000.0, 15.0, 1.0)
    slope = st.number_input("Độ dốc (độ)", 0.0, 90.0, 1.5, 0.1, format="%.1f")
    twi = st.number_input("Chỉ số TWI", 0.0, 30.0, 9.2, 0.1, format="%.1f")
    lulc = st.selectbox("Loại sử dụng đất", options=[1, 2, 3, 4, 5], format_func=lambda x: {10: 'Cây cối', 20: 'Cây bụi', 30: 'Đồng cỏ', 40: 'Đất trồng trọt', 50: 'Khu dân cư'}.get(x, 'Khác'), index=3)

    st.subheader("Dự báo Mưa (7 ngày tới)")
    # --- SỬA LỖI ---
    rain_seq = []
    for i in range(7):
        day_rain = st.number_input(f"Lượng mưa ngày {i+1} (mm)", 0.0, 500.0, 10.0, 0.1, key=f"rain_{i}", format="%.1f")
        rain_seq.append(day_rain)

# --- NÚT DỰ BÁO VÀ XỬ LÝ ---
if st.button("Dự báo Nguy cơ Ngập lụt", use_container_width=True, type="primary"):
    request_data = {
        "elevation": elevation, "slope": slope, "twi": twi,
        "lulc": lulc, "rainfall_sequence": rain_seq
    }
    API_URL = "http://127.0.0.1:8000/predict"
    
    with st.spinner("Đang gửi yêu cầu đến mô hình..."):
        try:
            response = requests.post(API_URL, data=json.dumps(request_data), timeout=10)
            if response.status_code == 200:
                result = response.json()
                prob = result.get("prediction_probability", 0)
                st.metric(label="Xác suất Ngập lụt", value=f"{prob * 100:.2f}%")
                if prob > 0.75:
                    st.error("Cảnh báo: Nguy cơ ngập lụt RẤT CAO!")
                elif prob > 0.5:
                    st.warning("Cảnh báo: Nguy cơ ngập lụt CAO.")
                else:
                    st.success("Thông báo: Nguy cơ ngập lụt thấp.")
                st.progress(prob)
            else:
                st.error(f"Lỗi từ API: {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Không thể kết nối đến API. Bạn đã khởi chạy server FastAPI (main.py) chưa?")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi không xác định: {e}")