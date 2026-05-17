import osmnx as ox
import matplotlib.pyplot as plt

def get_danang_delivery_core():
    # Tọa độ trung tâm: Cầu Rồng (Đà Nẵng) - Nút thắt giao thông huyết mạch
    center_point = (16.0611, 108.2278)
    
    # Bán kính quét: 6000 mét (6km)
    radius_meters = 6000 
    
    print(f"Đang tải đồ thị mạng lưới bán kính {radius_meters/1000}km từ Cầu Rồng...")
    
    # Dùng hàm graph_from_point thay vì graph_from_place
    G = ox.graph_from_point(
        center_point, 
        dist=radius_meters, 
        network_type='drive'
    )
    
    print(f"Tải thành công! Mạng lưới lõi có {len(G.nodes)} nút và {len(G.edges)} cạnh.")
    
    # Vẽ đồ thị
    fig, ax = ox.plot_graph(
        G, 
        node_size=2, 
        edge_linewidth=0.5, 
        node_color='red', 
        edge_color='#333333', 
        bgcolor='white'
    )
    
    return G

if __name__ == "__main__":
    G = get_danang_delivery_core()