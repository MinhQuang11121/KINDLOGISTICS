import pandas as pd
import numpy as np

num_steps = 12    # 12 bước thời gian (3 tiếng quá khứ)
num_nodes = 2891  # 2.891 ngã tư Đà Nẵng
total_rows = num_steps * num_nodes # 34.692 dòng

# Giả lập dữ liệu số cho các cột
np.random.seed(42) # Cố định dữ liệu cho ổn định
traffic_data = np.random.randint(5, 30, size=total_rows) # Lưu lượng đơn từ 5 đến 30
feat2_data = np.zeros(total_rows)                        # Mức độ ngập = 0
feat3_data = np.full(total_rows, 5)                      # Tốc độ gió = 5 m/s
temp_data = np.full(total_rows, 28)                      # Nhiệt độ = 28 độ C
rain_data = np.zeros(total_rows)                         # Lượng mưa = 0
feat6_data = np.zeros(total_rows)                        # Ngày thường = 0

# Tạo DataFrame đúng tên cột mô hình yêu cầu
df_chuan = pd.DataFrame({
    'traffic': traffic_data,
    'feat2': feat2_data,
    'feat3': feat3_data,
    'temp': temp_data,
    'rain': rain_data,
    'feat6': feat6_data
})

# Xuất ra file CSV
df_chuan.to_csv('data_demo_chuan.csv', index=False)
print(f"✅ Đã tạo file thành công với {len(df_chuan)} dòng!")