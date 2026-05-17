import os

txt_path = r'D:\CDIOFN\data/cd\road_distance_direction.txt'

print("--- 5 DÒNG ĐẦU TIÊN CỦA FILE 1.1GB ---")
try:
    with open(txt_path, 'r', encoding='utf-8') as f:
        for i in range(5):
            # In nguyên bản để xem dấu phẩy, khoảng trắng xếp thế nào
            print(repr(f.readline().strip()))
except Exception as e:
    print(f"❌ Lỗi: {e}")