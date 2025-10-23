import ee
import os
import google.auth

# --- CẤU HÌNH ---
# 1. Xác thực Google Cloud (cho project)
# Đảm bảo bạn đã chạy 'gcloud auth login' và 'gcloud init'
try:
    credentials, project_id = google.auth.default()
    ee.Initialize(credentials=credentials, project='gee-project-tandoan')
    print(f"Đã xác thực GEE thành công với dự án: gee-project-tandoan")
except Exception as e:
    print(f"Lỗi khi xác thực GEE. Hãy đảm bảo bạn đã chạy 'gcloud auth login'. Lỗi: {e}")
    # Nếu chạy trên Colab hoặc môi trường khác, hãy thử ee.Authenticate()
    # ee.Authenticate()
    # ee.Initialize(project='gee-project-tandoan')

# --- ĐỊNH NGHĨA KHU VỰC VÀ THAM SỐ ---
# Khu vực nghiên cứu (Toàn Việt Nam)
region_vn = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'Viet Nam')).geometry()
RAINFALL_LOOKBACK_DAYS = 7  # Số ngày nhìn lại (N)
SAMPLE_POINTS = 5000       # Tổng số điểm lấy mẫu (cân bằng)
EXPORT_FILE_NAME = 'flood_data_vn'
# Tọa độ các điểm ngập lụt lịch sử (VÍ DỤ - Cần thay thế bằng dữ liệu thực tế của bạn)
# Đây là bước quan trọng, bạn cần có dữ liệu điểm (lat, lon) và nhãn (1=ngập, 0=không ngập)
# Dưới đây là các điểm VÍ DỤ, bạn PHẢI thay thế bằng dữ liệu thật
historic_flood_points = [
    ee.Feature(ee.Geometry.Point(105.8, 21.0), {'flood_label': 1}), # Hà Nội
    ee.Feature(ee.Geometry.Point(106.7, 10.8), {'flood_label': 1}), # TPHCM
    ee.Feature(ee.Geometry.Point(108.2, 16.0), {'flood_label': 1})  # Đà Nẵng
]
# Điểm không ngập (ví dụ ở vùng cao)
non_flood_points = [
    ee.Feature(ee.Geometry.Point(105.4, 21.3), {'flood_label': 0}), # Vùng núi
    ee.Feature(ee.Geometry.Point(107.0, 11.5), {'flood_label': 0}), # Cao nguyên
    ee.Feature(ee.Geometry.Point(108.0, 14.5), {'flood_label': 0})  # Tây Nguyên
]
# Nếu bạn có file SHP/CSV các điểm, hãy tải lên GEE Assets và load bằng ee.FeatureCollection('projects/your-project/assets/your_asset_name')
# Ví dụ tạm thời: tạo điểm ngẫu nhiên
# LƯU Ý: Dữ liệu ngẫu nhiên sẽ cho mô hình chất lượng KÉM. Cần dữ liệu thật.
# points = ee.FeatureCollection(historic_flood_points + non_flood_points)
points_flood = ee.FeatureCollection.randomPoints(region_vn, int(SAMPLE_POINTS / 2), 123).map(lambda p: p.set('flood_label', 1))
points_non_flood = ee.FeatureCollection.randomPoints(region_vn, int(SAMPLE_POINTS / 2), 456).map(lambda p: p.set('flood_label', 0))
points = points_flood.merge(points_non_flood)

# --- NGUỒN DỮ LIỆU GEE ---
# 1. Dữ liệu địa hình (Static)
dem = ee.Image("USGS/SRTMGL1_003").clip(region_vn)
slope = ee.Terrain.slope(dem)
twi = dem.expression(
    'log( (flow_accumulation + 1) * pixel_area / tan(slope) )', {
        'flow_accumulation': ee.Image("WWF/HydroSHEDS/15ACC"),
        'slope': ee.Terrain.slope(dem).multiply(ee.Number(3.14159).divide(180)), # độ sang radian
        'pixel_area': ee.Image.pixelArea()
    }
).rename('twi')

# 2. Dữ liệu sử dụng đất (Static)
lulc = ee.ImageCollection("ESA/WorldCover/v100").first().clip(region_vn).select('Map').rename('lulc')

# 3. Dữ liệu mưa (Dynamic)
rainfall = ee.ImageCollection("NASA/GPM_L3/IMERG_V06").filterDate('2010-01-01', '2024-01-01') \
    .select('precipitationCal')

# Gộp các đặc trưng tĩnh
static_features = dem.rename('elevation').addBands(slope).addBands(twi).addBands(lulc)

# --- HÀM TRÍCH XUẤT DỮ LIỆU CHUỖI THỜI GIAN (MƯA) ---
def extract_rainfall(feature):
    # Lấy ngày ngẫu nhiên cho điểm (để huấn luyện)
    # Tạm thời dùng ngày cố định, lý tưởng là bạn có ngày lịch sử cho từng điểm
    event_date = ee.Date('2021-10-15') # VÍ DỤ
    start_date = event_date.advance(-RAINFALL_LOOKBACK_DAYS, 'day')
    
    # Lấy chuỗi thời gian mưa
    rainfall_series = rainfall.filterDate(start_date, event_date)
    
    # Hàm để lặp qua từng ngày và gán tên thuộc tính (rain_0, rain_1, ...)
    def get_daily_rain(n):
        date = start_date.advance(n, 'day')
        daily_total = rainfall_series.filterDate(date, date.advance(1, 'day')).sum()
        # Đặt tên thuộc tính là 'rain_N'
        prop_name = ee.String('rain_').cat(ee.Number(n).int().format())
        # Lấy giá trị tại điểm và gán vào feature
        value = daily_total.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=feature.geometry(),
            scale=10000 # Scale của GPM
        ).get('precipitationCal')
        return feature.set(prop_name, ee.Algorithms.If(value, value, -9999)) # Gán -9999 nếu null

    # SỬA LỖI: ee.List([]) là khởi tạo đúng, ee.List() bị lỗi
    day_indices = ee.List.sequence(0, RAINFALL_LOOKBACK_DAYS - 1)
    
    # Áp dụng hàm get_daily_rain cho từng ngày
    # Sử dụng iterate để xây dựng feature một cách tuần tự
    def accumulate(day_index, intermediate_feature):
        return get_daily_rain(ee.Number(day_index))(ee.Feature(intermediate_feature))

    # Bắt đầu với feature gốc
    feature_with_rain = ee.Feature(day_indices.iterate(accumulate, feature))
    
    return feature_with_rain

# --- TRÍCH XUẤT ĐẶC TRƯNG TĨNH ---
def extract_static(feature):
    return feature.addBands(static_features.sample(feature.geometry(), 30).first())

# Áp dụng các hàm
sampled_points = points.map(extract_static)
training_data = sampled_points.map(extract_rainfall)

# --- XUẤT DỮ LIỆU ---
# Các thuộc tính cần xuất
# SỬA LỖI: Tạo danh sách tên cột mưa động
rain_prop_names = ee.List([ee.String('rain_').cat(ee.Number(i).int().format()) for i in range(RAINFALL_LOOKBACK_DAYS)])
static_prop_names = ee.List(['elevation', 'slope', 'twi', 'lulc'])
label_prop_name = ee.List(['flood_label'])
all_properties = static_prop_names.cat(rain_prop_names).cat(label_prop_name)

print(f"Bắt đầu tác vụ xuất '{EXPORT_FILE_NAME}'...")
print("Vui lòng kiểm tra tab 'Tasks' trong GEE Code Editor để 'Run'.")

task = ee.batch.Export.table.toDrive(
    collection=training_data,
    description=EXPORT_FILE_NAME,
    folder='GEE_Exports',
    fileNamePrefix=EXPORT_FILE_NAME,
    fileFormat='CSV',
    selectors=all_properties
)
task.start()
