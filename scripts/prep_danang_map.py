import osmnx as ox
import networkx as nx
import numpy as np

# Define the districts in Da Nang
districts = ['Hải Châu', 'Thanh Khê', 'Sơn Trà', 'Ngũ Hành Sơn']

# Download and combine graphs
graphs = []
for district in districts:
    place_name = f"{district}, Da Nang, Vietnam"
    print(f"Downloading graph for {place_name}")
    G = ox.graph_from_place(place_name, network_type='drive')
    graphs.append(G)

# Combine all graphs into one
print("Combining graphs...")
G_combined = nx.compose_all(graphs)

# Convert to undirected graph
G_undirected = nx.to_undirected(G_combined)

# Convert to adjacency matrix
print("Converting to adjacency matrix...")
adj_matrix = nx.to_numpy_array(G_undirected, dtype=int)

# Save the adjacency matrix
np.save('danang_adj.npy', adj_matrix)

# Print the shape
print(f"Adjacency matrix shape: {adj_matrix.shape}")