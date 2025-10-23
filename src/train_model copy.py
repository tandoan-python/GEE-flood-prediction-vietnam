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
RAINFALL_LOOKBACK_DAYS = 7
STATIC_FEATURES = ['elevation', 'slope', 'twi', 'lulc']
TARGET_VARIABLE = 'flood_label'

# --- 1. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU ---
print("Đang tải dữ liệu...")
df = pd.read_csv(DATA_PATH)
df.replace(-9999, np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Kích thước dữ liệu sau khi làm sạch: {df.shape}")

rain_cols = ['rain_0', 'rain_1', 'rain_2', 'rain_3', 'rain_4', 'rain_5', 'rain_6']
X_static = df
X_ts = df[rain_cols]
y = df

X_static_train, X_static_test, X_ts_train, X_ts_test, y_train, y_test = train_test_split(
    X_static, X_ts, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_static_train_scaled = scaler.fit_transform(X_static_train)
X_static_test_scaled = scaler.transform(X_static_test)

X_ts_train_reshaped = X_ts_train.values.reshape(-1, RAINFALL_LOOKBACK_DAYS, 1)
X_ts_test_reshaped = X_ts_test.values.reshape(-1, RAINFALL_LOOKBACK_DAYS, 1)

# --- 2. XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU ---
print("Tính toán trọng số lớp...")
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(weights))
print(f"Trọng số lớp: {class_weights_dict}")

# --- 3. XÂY DỰNG MÔ HÌNH ---
print("Xây dựng kiến trúc mô hình...")
ts_input = Input(shape=(RAINFALL_LOOKBACK_DAYS, 1), name='ts_input')
lstm_out = LSTM(32, activation='relu')(ts_input)
static_input = Input(shape=(len(STATIC_FEATURES),), name='static_input')
dense_out = Dense(16, activation='relu')(static_input)
concatenated = Concatenate()([lstm_out, dense_out])
x = Dense(16, activation='relu')(concatenated)
output = Dense(1, activation='sigmoid', name='output')(x)
model = Model(inputs=[ts_input, static_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'recall'])
model.summary()

# --- 4. HUẤN LUYỆN MÔ HÌNH ---
print("Bắt đầu huấn luyện...")
history = model.fit(
    [X_ts_train_reshaped, X_static_train_scaled], y_train,
    epochs=30, batch_size=64, validation_split=0.2,
    class_weight=class_weights_dict, verbose=1
)

# --- 5. ĐÁNH GIÁ MÔ HÌNH ---
print("\nĐánh giá mô hình trên tập kiểm tra...")
loss, accuracy, recall = model.evaluate([X_ts_test_reshaped, X_static_test_scaled], y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}, Test Recall: {recall:.4f}")

# --- 6. LƯU MÔ HÌNH VÀ SCALER (CẢI TIẾN) ---
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
print(f"Lưu mô hình vào: {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)
print(f"Lưu scaler vào: {SCALER_SAVE_PATH}")
joblib.dump(scaler, SCALER_SAVE_PATH)
print("Hoàn thành!")