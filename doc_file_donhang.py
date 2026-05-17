import pandas as pd

def inspect_didi_data(file_path=r'D:\CDIOFN\data\cd\filtered_lines.csv'):
    print(f"🚀 Đang đọc file: {file_path}...\n")
    
    # BẢN VÁ LỖI: Bỏ qua các dòng dị biệt bị bung dấu phẩy
    try:
        # Dùng on_bad_lines='skip' để lơ đi những dòng hỏng, hoặc dùng engine='python'
        df = pd.read_csv(file_path, on_bad_lines='skip', nrows=1000) 
        
        print("=== THÔNG TIN TỔNG QUAN KHỐI DỮ LIỆU ===")
        df.info()
        
        print("\n=== 5 ĐƠN HÀNG/QUỸ ĐẠO ĐẦU TIÊN ===")
        print(df.head())
        
    except Exception as e:
        print("Vẫn lỗi đọc CSV, ta chuyển sang đọc thô bằng Python cơ bản luôn!")
        # Đọc thô 5 dòng đầu tiên dưới dạng văn bản (text)
        with open(file_path, 'r', encoding='utf-8') as f:
            for i in range(5):
                print(f"Dòng {i+1}: {f.readline().strip()}")

if __name__ == "__main__":
    inspect_didi_data()