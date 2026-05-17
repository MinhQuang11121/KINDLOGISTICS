import matplotlib.pyplot as plt

# Dữ liệu từ kịch bản lúc nãy
time_labels = ["17:00", "17:15", "17:30", "17:45", "18:00"]
actual_demand = [90, 105, 125, 140, 130]      # Nhu cầu thực tế
basic_stgcn = [88, 102, 115, 120, 125]        # AI cũ (Mù thời tiết)
multi_stgcn = [92, 108, 128, 142, 135]        # AI KINDLOGISTICS (Đa đặc trưng)

# Cấu hình biểu đồ
plt.figure(figsize=(10, 6))

# Vẽ 3 đường
plt.plot(time_labels, actual_demand, marker='o', linestyle='-', color='black', linewidth=2.5, label='Thực tế (Ground Truth)')
plt.plot(time_labels, basic_stgcn, marker='s', linestyle='--', color='blue', linewidth=2, label='STGCN Cơ bản (Baseline)')
plt.plot(time_labels, multi_stgcn, marker='^', linestyle='-', color='red', linewidth=2.5, label='KINDLOGISTICS (STGCN Đa đặc trưng)')

# Trang trí
plt.title('So sánh độ nhạy dự báo của các mô hình khi có mưa lớn đột ngột', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Khung giờ dự báo', fontsize=12)
plt.ylabel('Lượng đơn hàng (Lượt)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11, loc='upper left')

# Vẽ một đường dọc gạch nét đứt để đánh dấu lúc 17:30 mưa to
plt.axvline(x=2, color='gray', linestyle=':', linewidth=2, alpha=0.7)
plt.text(2.05, 95, 'Bắt đầu mưa lớn\n(Nhu cầu tăng vọt)', color='gray', fontsize=10, style='italic')

# Lưu ảnh chất lượng cao 300dpi để in không bị vỡ
plt.tight_layout()
plt.savefig('Bieu_do_So_sanh_Kich_ban.png', dpi=300)
print("✅ Đã lưu biểu đồ thành công: Bieu_do_So_sanh_Kich_ban.png")

# Hiển thị lên màn hình
plt.show()