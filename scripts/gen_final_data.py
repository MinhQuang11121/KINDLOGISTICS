import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm # Để hiện thanh tiến trình cho chuyên nghiệp

def generate_danang_master_data():
    num_nodes = 2851
    num_days = 365
    steps_per_day = 96  # 15 phút/mốc
    total_steps = num_days * steps_per_day
    start_date = datetime(2025, 1, 1)

    # Khởi tạo ma trận: [Thời gian, Node, 5 Kênh]
    # Kênh: 0:Flow, 1:Rain, 2:Temp, 3:Holiday, 4:Sale
    data = np.zeros((total_steps, num_nodes, 5), dtype=np.float32)

    print(f"🚀 Bắt đầu kiến tạo dữ liệu Đà Nẵng 2025 cho {num_nodes} điểm giao hàng...")

    for d in tqdm(range(num_days)):
        curr_date = start_date + timedelta(days=d)
        month = curr_date.month
        is_weekend = curr_date.weekday() >= 5
        
        # 1. Xác định Ngày Lễ & Ngày Sale (Logic Expert)
        is_holiday = 1 if curr_date.strftime('%d-%m') in ['01-01', '30-04', '01-05', '02-09', '24-12'] else 0
        # Tết Nguyên Đán 2025 (Ất Tỵ) rơi vào cuối tháng 1
        if 25 <= curr_date.day <= 31 and month == 1: is_holiday = 1 
        
        is_sale = 1 if curr_date.day == curr_date.month else 0 # 10/10, 11/11, 12/12...

        # 2. Giả lập Thời tiết Đà Nẵng (Logic Mùa vụ)
        # Mùa bão (Tháng 9-12), Nắng gắt (Tháng 5-8)
        has_storm = (9 <= month <= 11) and (np.random.rand() > 0.85)
        base_rain = 30 if has_storm else (10 if 9 <= month <= 12 else 1)
        daily_rain = np.random.gamma(base_rain, 0.5) if np.random.rand() > 0.7 else 0
        
        base_temp = 25 + 8 * np.sin(np.pi * (month - 1) / 6) # Đồ thị hình sin cho nhiệt độ năm
        daily_temp = base_temp + np.random.normal(0, 2)

        for t in range(steps_per_day):
            idx = d * steps_per_day + t
            hour = (t * 15) / 60
            
            # 3. Tạo nhịp sinh học Flow (2 đỉnh cao điểm Việt Nam)
            # Đỉnh trưa (11:30-13:00) & Đỉnh tối (17:30-19:30)
            peak_lunch = np.exp(-((hour - 12)**2) / (2 * 1.0**2))
            peak_dinner = np.exp(-((hour - 18.5)**2) / (2 * 1.2**2))
            base_flow = 0.2 + 0.8 * peak_lunch + 0.9 * peak_dinner
            
            # 4. Áp dụng các "Trải nghiệm Shipper" (Impact Factors)
            rain_impact = 1.4 if daily_rain > 15 else (1.1 if daily_rain > 0 else 1.0)
            sale_impact = 1.6 if (is_sale and 20 <= hour <= 23) else (1.2 if is_sale else 1.0)
            temp_impact = 1.3 if (daily_temp > 37 and 11 <= hour <= 14) else 1.0
            holiday_impact = 0.5 if (is_holiday and hour < 10) else (1.4 if is_holiday else 1.0)

            # 5. Phân bổ lưu lượng cho từng Node (Có độ nhiễu thực tế)
            # Node quan trọng (trung tâm Hải Châu) sẽ có hệ số nhân cao hơn
            node_importance = np.random.rayleigh(1.0, num_nodes) 
            
            final_flow = base_flow * rain_impact * sale_impact * temp_impact * holiday_impact * node_importance * 20
            
            # Gán dữ liệu vào 5 kênh
            data[idx, :, 0] = final_flow + np.random.poisson(2, num_nodes) # Kênh Flow + Nhiễu Poisson
            data[idx, :, 1] = daily_rain
            data[idx, :, 2] = daily_temp
            data[idx, :, 3] = is_holiday
            data[idx, :, 4] = is_sale

    # Lưu file
    np.save('danang_1year_multimodal.npy', data)
    print(f"\n✅ Thành công! Đã xuất file: danang_1year_multimodal.npy")
    print(f"📊 Kích thước dữ liệu: {data.shape} (Thời gian x Node x Kênh)")

if __name__ == "__main__":
    generate_danang_master_data()