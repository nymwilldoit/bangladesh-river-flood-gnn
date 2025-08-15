"""
Bangladesh River Flood Forecasting - Graph Utilities
River network graph construction and manipulation functions

Extracted from Phase 1: Steps 4.1 and 4.2
"""

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

def create_river_network_graph(gauge_df):
    """Create a graph representation of river network"""
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes (gauge stations)
    for idx, station in gauge_df.iterrows():
        G.add_node(station['station_id'], 
                   name=station['station_name'],
                   pos=(station['longitude'], station['latitude']),
                   river=station['river_name'],
                   elevation=station['elevation'])
    
    # Add edges based on river connectivity and distance
    stations_by_river = gauge_df.groupby('river_name')
    
    for river_name, river_stations in stations_by_river:
        stations_list = river_stations.sort_values('latitude', ascending=False)
        
        # Connect consecutive stations on same river
        for i in range(len(stations_list) - 1):
            station1 = stations_list.iloc[i]
            station2 = stations_list.iloc[i + 1]
            
            # Calculate distance (simplified)
            dist = np.sqrt((station1['latitude'] - station2['latitude'])**2 + 
                          (station1['longitude'] - station2['longitude'])**2) * 111  # km
            
            G.add_edge(station1['station_id'], station2['station_id'], 
                      weight=dist, river=river_name)
    
    # Add inter-river connections (confluence points)
    # Simplified: connect major river systems
    river_connections = [
        ('Ganges', 'Brahmaputra'),  # Join to form Padma
        ('Ganges', 'Meghna')        # Confluence area
    ]
    
    for river1, river2 in river_connections:
        stations1 = gauge_df[gauge_df['river_name'] == river1]
        stations2 = gauge_df[gauge_df['river_name'] == river2]
        
        # Connect closest stations
        if not stations1.empty and not stations2.empty:
            # Find closest stations between rivers
            min_dist = float('inf')
            closest_pair = None
            
            for _, s1 in stations1.iterrows():
                for _, s2 in stations2.iterrows():
                    dist = np.sqrt((s1['latitude'] - s2['latitude'])**2 + 
                                  (s1['longitude'] - s2['longitude'])**2) * 111
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (s1['station_id'], s2['station_id'])
            
            if closest_pair:
                G.add_edge(closest_pair[0], closest_pair[1], 
                          weight=min_dist, river='confluence')
    
    return G


def networkx_to_pyg(nx_graph, node_features=None):
    """Convert NetworkX graph to PyTorch Geometric format"""
    
    # Get edge indices
    edge_indices = []
    edge_weights = []
    
    # Fixed: Properly unpack the edge tuple
    for node1, node2, data in nx_graph.edges(data=True):
        edge_indices.append([node1-1, node2-1])  # Convert to 0-based indexing
        edge_indices.append([node2-1, node1-1])  # Undirected graph
        weight = data.get('weight', 1.0)
        edge_weights.extend([weight, weight])
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    
    # Node features (if not provided, use node degree as feature)
    if node_features is None:
        node_features = []
        for node in sorted(nx_graph.nodes()):
            degree = nx_graph.degree[node]
            pos = nx_graph.nodes[node]['pos']
            elevation = nx_graph.nodes[node].get('elevation', 0)
            node_features.append([degree, pos[0], pos[1], elevation])
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data


def create_adjacency_matrix(river_graph, num_nodes):
    """Create adjacency matrix from NetworkX graph"""
    
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    for edge in river_graph.edges(data=True):
        node1, node2, data = edge
        # Convert to 0-based indexing
        i, j = node1 - 1, node2 - 1
        weight = 1.0 / (data.get('weight', 1.0) + 1e-6)  # Inverse distance weighting
        adj_matrix[i, j] = weight
        adj_matrix[j, i] = weight  # Symmetric matrix
    
    # Add self-loops
    np.fill_diagonal(adj_matrix, 1.0)
    
    return torch.FloatTensor(adj_matrix)
