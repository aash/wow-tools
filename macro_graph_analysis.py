import cv2
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


# --- 1. Keypoint & Macro-Graph Extraction ---
def get_junction_mapping(raw_junctions, merge_radius):
    junction_map = {}
    merged_junctions = []
    
    if not raw_junctions:
        return junction_map, merged_junctions
        
    J_G = nx.Graph()
    J_G.add_nodes_from(raw_junctions)
    tree = KDTree(raw_junctions)
    pairs = tree.query_pairs(r=merge_radius)
    J_G.add_edges_from([(raw_junctions[i], raw_junctions[j]) for i, j in pairs])
    
    for component in nx.connected_components(J_G):
        pts = np.array(list(component))
        centroid = pts.mean(axis=0)
        closest_node = min(component, key=lambda p: np.linalg.norm(np.array(p) - centroid))
        
        merged_junctions.append(closest_node)
        for node in component:
            junction_map[node] = closest_node
            
    return junction_map, merged_junctions

def extract_macro_graph(img, step=12, max_angle_deg=160, prominence=0.05, merge_radius=10.0):
    
    y_coords, x_coords = np.where(img > 0)
    points = list(zip(x_coords, y_coords))
    
    if not points:
        return img, [], nx.Graph(), [], [], []
    
    tree = KDTree(points)
    pairs = tree.query_pairs(r=1.5) 
    
    G_raw = nx.Graph()
    G_raw.add_nodes_from(points)
    G_raw.add_edges_from([(points[i], points[j]) for i, j in pairs])
    
    macro_G = nx.Graph()
    
    all_deadends = []
    all_merged_junctions = []
    all_breaking_points = []
    
    # Incremental sub-step parameters for breaking points
    max_step = step
    sub_step = 4
    
    for cc in nx.connected_components(G_raw):
        G = G_raw.subgraph(cc).copy()
        
        deadends = [n for n, d in G.degree() if d == 1]
        raw_junctions = [n for n, d in G.degree() if d >= 3]
        junction_map, merged_junctions = get_junction_mapping(raw_junctions, merge_radius)
        
        breaking_points = []
        longest_path = []
        for i in range(len(deadends)):
            for j in range(i + 1, len(deadends)):
                try:
                    path = nx.shortest_path(G, deadends[i], deadends[j])
                    if len(path) > len(longest_path): 
                        longest_path = path
                except nx.NetworkXNoPath: 
                    continue
                    
        angles, path_indices = [], []
        if len(longest_path) > 2 * sub_step:
            for i in range(sub_step, len(longest_path) - sub_step):
                distance_to_edge = min(i, len(longest_path) - 1 - i)
                current_step = min(distance_to_edge, max_step)
                current_step = (current_step // sub_step) * sub_step
                
                p1, p_mid, p2 = np.array(longest_path[i-current_step]), np.array(longest_path[i]), np.array(longest_path[i+current_step])
                v1, v2 = p1 - p_mid, p2 - p_mid
                cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
                angles.append(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
                path_indices.append(longest_path[i])
                
        if angles:
            peaks, _ = find_peaks(-np.array(angles), prominence=prominence)
            max_angle_rad = np.radians(max_angle_deg)
            breaking_points = [path_indices[idx] for idx in peaks if angles[idx] <= max_angle_rad]

        all_deadends.extend(deadends)
        all_merged_junctions.extend(merged_junctions)
        all_breaking_points.extend(breaking_points)

        cc_macro_nodes = set(deadends + merged_junctions + breaking_points)
        macro_G.add_nodes_from(cc_macro_nodes)
        
        stop_nodes = set(deadends + raw_junctions + breaking_points)
        visited_edges = set()
        
        for start_node in stop_nodes:
            for neighbor in G.neighbors(start_node):
                edge = tuple(sorted((start_node, neighbor)))
                if edge in visited_edges: continue
                
                path = [start_node, neighbor]
                visited_edges.add(edge)
                curr = neighbor
                
                while curr not in stop_nodes:
                    next_nodes = [n for n in G.neighbors(curr) if n != path[-2]]
                    valid_next = [n for n in next_nodes if tuple(sorted((curr, n))) not in visited_edges]
                    if not valid_next: break 
                    
                    nxt = valid_next[0]
                    visited_edges.add(tuple(sorted((curr, nxt))))
                    path.append(nxt)
                    curr = nxt
                    
                end_node = curr
                
                if end_node in stop_nodes:
                    M_start = junction_map.get(start_node, start_node)
                    M_end = junction_map.get(end_node, end_node)
                    
                    if M_start != M_end: 
                        pts = np.array(path)
                        path_len = np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1))
                        
                        dist_start = np.linalg.norm(np.array(start_node) - np.array(M_start))
                        dist_end = np.linalg.norm(np.array(end_node) - np.array(M_end))
                        total_length = path_len + dist_start + dist_end
                        
                        # NEW: Store the exact physical pixels making up this edge
                        if macro_G.has_edge(M_start, M_end):
                            if total_length < macro_G[M_start][M_end]['weight']:
                                macro_G[M_start][M_end].update({'weight': total_length, 'pixels': path})
                        else:
                            macro_G.add_edge(M_start, M_end, weight=total_length, pixels=path)

    return img, points, macro_G, all_deadends, all_merged_junctions, all_breaking_points

# --- 2. Post-Processing Pruning ---
def prune_small_branches(img, macro_G, deadends, junctions, prune_threshold=10.0):
    """
    Removes edges connected to dead-ends that fall below the length threshold.
    Updates the image mask, removes the graph edge, and collapses newly invalidated junctions.
    """
    deadend_set = set(deadends)
    junction_set = set(junctions)
    
    # 1. Erase small tails
    for de in list(deadend_set):
        if macro_G.has_node(de) and macro_G.degree(de) == 1:
            neighbor = list(macro_G.neighbors(de))[0]
            
            # Check if this tail connects to a junction
            if neighbor in junction_set:
                edge_data = macro_G.get_edge_data(de, neighbor)
                
                if edge_data['weight'] < prune_threshold:
                    # A. Erase the physical pixels from the image mask
                    for px, py in edge_data['pixels']:
                        img[py, px] = 0
                        
                    # B. Remove the edge and the dead-end node from the macro-graph
                    macro_G.remove_node(de)
                    deadends.remove(de)

    # 2. Collapse junctions mathematically
    # If a junction had 3 branches and we pruned 1, it now has a degree of 2. 
    # It is no longer a junction; it is just a normal point on a continuous line.
    for junc in list(junction_set):
        if macro_G.has_node(junc):
            degree = macro_G.degree(junc)
            
            if degree == 2:
                # Merge the two remaining branches into one continuous edge
                n1, n2 = list(macro_G.neighbors(junc))
                d1 = macro_G.get_edge_data(junc, n1)
                d2 = macro_G.get_edge_data(junc, n2)
                
                new_weight = d1['weight'] + d2['weight']
                new_pixels = d1['pixels'] + d2['pixels']
                
                macro_G.add_edge(n1, n2, weight=new_weight, pixels=new_pixels)
                macro_G.remove_node(junc)
                junctions.remove(junc)
                
            elif degree == 1:
                # If we pruned multiple tails, it might have become a dead-end itself
                junctions.remove(junc)
                deadends.append(junc)
                
            elif degree == 0:
                # Completely isolated (entire structure was erased)
                macro_G.remove_node(junc)
                junctions.remove(junc)
                
    return img, macro_G, deadends, junctions

def find_character_centroid(junctions, deadends, search_radius=60.0, min_features=4):
    """
    Identifies a dense spatial cluster of structural features (junctions + deadends)
    and calculates its centroid.
    """
    # Combine both types of keypoints into a single tracking pool
    features = junctions + deadends
    
    if len(features) < min_features:
        return None, []
        
    # 1. Build a spatial clustering graph for the combined features
    cluster_G = nx.Graph()
    cluster_G.add_nodes_from(features)
    
    # 2. Find all feature pairs that fall within the character-sized radius
    tree = KDTree(features)
    pairs = tree.query_pairs(r=search_radius)
    cluster_G.add_edges_from([(features[i], features[j]) for i, j in pairs])
    
    # 3. Extract the largest connected cluster
    largest_cluster = []
    for component in nx.connected_components(cluster_G):
        if len(component) > len(largest_cluster):
            largest_cluster = list(component)
            
    # 4. Verify it meets the density threshold and calculate the centroid
    if len(largest_cluster) >= min_features:
        pts = np.array(largest_cluster)
        centroid = pts.mean(axis=0)
        return centroid, largest_cluster
        
    return None, []

def macro_graph_analysis(input_image):
    img, points, macro_G, deadends, junctions, breaks = extract_macro_graph(
        input_image, step=12, max_angle_deg=100, prominence=0.05, merge_radius=10.0
    )
    
    # --- PRUNE SMALL BRANCHES (< 12.0 pixels) ---
    img, macro_G, deadends, junctions = prune_small_branches(
        img, macro_G, deadends, junctions, prune_threshold=12.0
    )

    char_centroid, char_junctions = find_character_centroid(
        junctions, deadends, search_radius=60.0, min_features=3
    )    

    
    # Re-extract the remaining active pixels for the plot based on the updated image mask
    # y_coords, x_coords = np.where(img > 0)
    # pruned_points = np.array(list(zip(x_coords, y_coords)))

    # plt.figure(figsize=(12, 8))
    
    # Plot pruned pixels
    # if pruned_points.size > 0:
    #     plt.scatter(pruned_points[:, 0], pruned_points[:, 1], c='lightgray', s=5, label='Active Skeleton Pixels')
    
    # for kp_list, color, marker, size, label in [
    #     (deadends, 'red', 'o', 40, 'Dead-ends'),
    #     (junctions, 'blue', 'o', 60, 'Junctions'),
    #     (breaks, 'green', '*', 120, 'Breaking Points')
    # ]:
    #     if kp_list:
    #         arr = np.array(kp_list)
    #         # Only plot nodes that still exist in the graph (weren't pruned)
    #         active_nodes = [node for node in kp_list if macro_G.has_node(tuple(node))]
    #         if active_nodes:
    #             act_arr = np.array(active_nodes)
    #             plt.scatter(act_arr[:, 0], act_arr[:, 1], c=color, s=size, marker=marker, zorder=5, label=label)
            
    sorted_edges = sorted(macro_G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    top_2_edges = [(u, v) for u, v, d in sorted_edges[:2]]
    return top_2_edges, char_centroid


# --- 3. Main Execution & Visualization ---
if __name__ == "__main__":
    image_name = "tmp/frangi_20260725_071910_mask.png"

    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        raise FileNotFoundError(f"Could not load {image_name}")


    # Extract initial graph and keypoints
    img, points, macro_G, deadends, junctions, breaks = extract_macro_graph(
        img, step=12, max_angle_deg=100, prominence=0.05, merge_radius=10.0
    )
    
    # --- PRUNE SMALL BRANCHES (< 12.0 pixels) ---
    img, macro_G, deadends, junctions = prune_small_branches(
        img, macro_G, deadends, junctions, prune_threshold=12.0
    )

    char_centroid, char_junctions = find_character_centroid(
        junctions, deadends, search_radius=60.0, min_features=3
    )    
    

    if char_centroid is None:
        print("qwe")
    # Re-extract the remaining active pixels for the plot based on the updated image mask
    y_coords, x_coords = np.where(img > 0)
    pruned_points = np.array(list(zip(x_coords, y_coords)))

    plt.figure(figsize=(12, 8))
    
    # Plot pruned pixels
    if pruned_points.size > 0:
        plt.scatter(pruned_points[:, 0], pruned_points[:, 1], c='lightgray', s=5, label='Active Skeleton Pixels')
    
    for kp_list, color, marker, size, label in [
        (deadends, 'red', 'o', 40, 'Dead-ends'),
        (junctions, 'blue', 'o', 60, 'Junctions'),
        (breaks, 'green', '*', 120, 'Breaking Points')
    ]:
        if kp_list:
            arr = np.array(kp_list)
            # Only plot nodes that still exist in the graph (weren't pruned)
            active_nodes = [node for node in kp_list if macro_G.has_node(tuple(node))]
            if active_nodes:
                act_arr = np.array(active_nodes)
                plt.scatter(act_arr[:, 0], act_arr[:, 1], c=color, s=size, marker=marker, zorder=5, label=label)
            
    sorted_edges = sorted(macro_G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    top_2_edges = [(u, v) for u, v, d in sorted_edges[:2]]
    
    def is_longest_edge(u, v):
        return (u, v) in top_2_edges or (v, u) in top_2_edges

    for u, v, data in macro_G.edges(data=True):
        mid_x, mid_y = (u[0] + v[0]) / 2, (u[1] + v[1]) / 2
        length = data['weight']
        
        if is_longest_edge(u, v):
            plt.plot([u[0], v[0]], [u[1], v[1]], '-', color='darkorange', linewidth=3.5, alpha=0.9, zorder=3)
            box_edge, text_color = 'darkorange', 'darkorange'
        else:
            plt.plot([u[0], v[0]], [u[1], v[1]], 'k--', linewidth=1.5, alpha=0.5, zorder=2)
            box_edge, text_color = 'purple', 'purple'
        
        plt.text(mid_x, mid_y, f"{length:.1f}px", 
                 color=text_color, fontsize=10, fontweight='bold', ha='center', va='center', zorder=4,
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor=box_edge, boxstyle='round,pad=0.2'))
    

# --- NEW: Highlight Character Area ---
    if char_centroid is not None:
        # Plot the centroid marker
        plt.scatter(char_centroid[0], char_centroid[1], c='magenta', s=200, marker='X', zorder=10, label='Character Centroid')
        
        # Highlight the junctions belonging to the character
        char_arr = np.array(char_junctions)
        plt.scatter(char_arr[:, 0], char_arr[:, 1], c='magenta', s=100, marker='s', zorder=6, alpha=0.5, label='Character Junctions')
        
        # Optional: Draw a circle indicating the detection radius around the centroid
        circle = plt.Circle((char_centroid[0], char_centroid[1]), 60.0, color='magenta', fill=False, linestyle=':', linewidth=2, alpha=0.7)
        plt.gca().add_patch(circle)

    plt.plot([], [], '-', color='darkorange', linewidth=3.5, label='Top 2 Longest Edges')
    plt.gca().invert_yaxis()
    plt.title("Pruned Topological Graph")
    plt.legend()
    plt.axis('equal')
    plt.show()

