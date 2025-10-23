import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
import os
import joblib

# --- CÁC THAM SỐ ---
DATA_PATH = os.path.join('..', 'data', 'flood_data_vn.csv')
MODEL_SAVE_PATH = os.path.join('..', 'models', 'flood_model.keras')
SCALER_SAVE_PATH = os.path.join('..', 'models', 'scaler.joblib')

RAINFALL_LOOKBACK_DAYS = 7  # Phải khớp với script prepare_data.py
STATIC_FEATURES = ["elevation", "slope", "twi", "lulc"]
TARGET_VARIABLE = "flood_label"

# SỬA LỖI: Tự động tạo danh sách cột mưa
rain_cols = [f"rain_{i}" for i in range(RAINFALL_LOOKBACK_DAYS)]

# --- 1. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU ---
print("Đang tải dữ liệu...")
if not os.path.exists(DATA_PATH):
    print(f"LỖI: Không tìm thấy tệp dữ liệu tại {DATA_PATH}")
    print(
        "Vui lòng chạy Giai đoạn 3 (prepare_data.py) và di chuyển tệp CSV vào thư mục 'data'."
    )
    exit()

df = pd.read_csv(DATA_PATH)
df.replace(-9999, np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Kích thước dữ liệu sau khi làm sạch: {df.shape}")

# SỬA LỖI: Kiểm tra kích thước dữ liệu
if df.shape[0] < 50:  # SỬA LỖI: df.shape[0] thay vì df.shape
    print(
        f"LỖI: Dữ liệu quá nhỏ ({df.shape[0]} hàng) để huấn luyện. Cần ít nhất 50 hàng."
    )
    exit()

# SỬA LỖI: Tách X và y chính xác
if TARGET_VARIABLE not in df.columns:
    print(f"LỖI: Không tìm thấy cột '{TARGET_VARIABLE}' trong CSV.")
    exit()

# SỬA LỖI: Phân tách X và y đúng cách
X_static_df = df[STATIC_FEATURES]
X_ts_df = df[rain_cols]
y_series = df[TARGET_VARIABLE]

# SỬA LỖI: Kiểm tra lỗi 'The least populated class in y has only 1 member'
class_counts = y_series.value_counts()
print(f"Phân phối lớp:\n{class_counts}")
if class_counts.min() < 2:
    print("LỖI: Lớp ít phổ biến nhất có ít hơn 2 mẫu. Không thể thực hiện 'stratify'.")
    print(
        "Vui lòng kiểm tra lại dữ liệu đầu vào từ GEE (Giai đoạn 3) để đảm bảo có đủ mẫu cho cả lớp '0' và '1'."
    )
    exit()

X_static_train, X_static_test, X_ts_train, X_ts_test, y_train, y_test = (
    train_test_split(
        X_static_df,
        X_ts_df,
        y_series,
        test_size=0.2,
        random_state=42,
        stratify=y_series,  # SỬA LỖI: stratify trên y_series
    )
)

print(
    f"Kích thước tập huấn luyện: {len(y_train)}, Kích thước tập kiểm tra: {len(y_test)}"
)

# Chuẩn hóa dữ liệu tĩnh
scaler = StandardScaler()
X_static_train_scaled = scaler.fit_transform(X_static_train)
X_static_test_scaled = scaler.transform(X_static_test)

# Reshape dữ liệu chuỗi thời gian cho LSTM (samples, timesteps, features)
X_ts_train_reshaped = X_ts_train.values.reshape(-1, RAINFALL_LOOKBACK_DAYS, 1)
X_ts_test_reshaped = X_ts_test.values.reshape(-1, RAINFALL_LOOKBACK_DAYS, 1)

# --- 2. XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU ---
print("Tính toán trọng số lớp...")
try:
    weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights_dict = dict(enumerate(weights))
    print(f"Trọng số lớp: {class_weights_dict}")
except ValueError as e:
    print(f"Lỗi khi tính trọng số lớp: {e}. Sử dụng trọng số mặc định.")
    class_weights_dict = None

# --- 3. XÂY DỰNG MÔ HÌNH ---
print("Xây dựng kiến trúc mô hình...")

# Đầu vào cho chuỗi thời gian (mưa)
ts_input = Input(shape=(RAINFALL_LOOKBACK_DAYS, 1), name="ts_input")
lstm_out = LSTM(32, activation="relu")(ts_input)

# Đầu vào cho dữ liệu tĩnh (địa hình)
static_input = Input(shape=(len(STATIC_FEATURES),), name="static_input")
dense_out = Dense(16, activation="relu")(static_input)

# Kết hợp
concatenated = Concatenate()([lstm_out, dense_out])
x = Dense(16, activation="relu")(concatenated)
output = Dense(1, activation="sigmoid", name="output")(x)

model = Model(inputs=[ts_input, static_input], outputs=output)

# SỬA LỖI: Thêm metrics
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Recall(name="recall")],
)

model.summary()

# --- 4. HUẤN LUYỆN MÔ HÌNH ---
print("Bắt đầu huấn luyện...")
history = model.fit(
    [X_ts_train_reshaped, X_static_train_scaled],
    y_train,
    epochs=30,  # Tăng epochs để học tốt hơn
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weights_dict,
    verbose=1,
)

# --- 5. ĐÁNH GIÁ MÔ HÌNH ---
print("\nĐánh giá mô hình trên tập kiểm tra...")
loss, accuracy, recall = model.evaluate(
    [X_ts_test_reshaped, X_static_test_scaled], y_test, verbose=0
)
print(
    f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}, Test Recall: {recall:.4f}"
)

# --- 6. LƯU MÔ HÌNH VÀ SCALER ---
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
print(f"Lưu mô hình vào: {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)

print(f"Lưu scaler vào: {SCALER_SAVE_PATH}")
joblib.dump(scaler, SCALER_SAVE_PATH)

print("Hoàn thành Giai đoạn 4!")
