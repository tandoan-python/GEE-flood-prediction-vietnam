from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import tensorflow as tf
import numpy as np
import os
import joblib
from typing import List

# --- KHỞI TẠO ỨNG DỤNG VÀ TẢI MÔ HÌNH, SCALER ---
app = FastAPI(title="Flood Prediction API")

MODEL_PATH = os.path.join('..', 'models', 'flood_model.keras')
SCALER_PATH = os.path.join('..', 'models', 'scaler.joblib')
model = None
scaler = None

@app.on_event("startup")
def load_model_and_scaler():
    global model, scaler
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("Tải mô hình và scaler thành công!")
        else:
            print("Lỗi: Không tìm thấy file mô hình hoặc scaler.")
    except Exception as e:
        print(f"Lỗi khi tải mô hình hoặc scaler: {e}")

# --- ĐỊNH NGHĨA CẤU TRÚC DỮ LIỆU ĐẦU VÀO ---
class PredictionRequest(BaseModel):
    elevation: float = Field(..., example=15.0)
    slope: float = Field(..., example=1.5)
    twi: float = Field(..., example=9.2)
    lulc: int = Field(..., example=40)
    rainfall_sequence: List[float] = Field(..., example=[5.0, 10.2, 30.1, 25.5, 12.0, 8.7, 2.1])

# --- ĐIỂM CUỐI (ENDPOINT) DỰ BÁO ---
@app.post("/predict")
async def predict(request: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Mô hình hoặc scaler chưa sẵn sàng")

    # 1. Chuẩn bị dữ liệu đầu vào
    static_data = np.array([[request.elevation, request.slope, request.twi, request.lulc]])
    static_data_scaled = scaler.transform(static_data)

    if len(request.rainfall_sequence)!= 7:
        raise HTTPException(status_code=400, detail="Chuỗi mưa phải có đúng 7 giá trị.")
    
    ts_data = np.array(request.rainfall_sequence).reshape(1, 7, 1)

    # 2. Thực hiện dự báo
    try:
        prediction_prob = model.predict([ts_data, static_data_scaled])
        prob_value = float(prediction_prob)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự báo: {e}")

    # 3. Trả về kết quả
    return {
        "prediction_probability": prob_value,
        "prediction_percentage": f"{prob_value * 100:.2f}%"
    }

@app.get("/")
def read_root():
    return {"message": "Chào mừng đến với API Dự báo Ngập lụt!"}