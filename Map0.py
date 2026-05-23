import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from shapely.geometry import LineString
#GIS maps

ox.settings.use_cache = True

ox.settings.cache_folder = 'D:/osmnx_shanghai_cache'


def downsample_waypoints_rdp(waypoints_2d, tolerance=3.0):

    if len(waypoints_2d) <= 2:
        return waypoints_2d

    line = LineString(waypoints_2d)

    simplified_line = line.simplify(tolerance=tolerance, preserve_topology=True)

    filtered_wp = np.array(simplified_line.coords)

    return filtered_wp

def generate_shanghai_logistics_map():
    center_point = (31.2662, 121.3412)

    G = ox.graph_from_point(center_point, dist=2500, network_type='drive')

    G_proj = ox.project_graph(G)

    nodes = list(G_proj.nodes())

    random.seed(42)


    xs = [data['x'] for node, data in G_proj.nodes(data=True)]
    ys = [data['y'] for node, data in G_proj.nodes(data=True)]
    center_x, center_y = np.mean(xs), np.mean(ys)

    start_node = ox.distance.nearest_nodes(G_proj, X=center_x, Y=center_y)
    start_x, start_y = G_proj.nodes[start_node]['x'], G_proj.nodes[start_node]['y']

    min_dist = 1800.0
    quadrants = {'North-East': [], 'North-West': [], 'South-West': [], 'South-East': []}

    for node in G_proj.nodes():
        if node == start_node:
            continue

        nx_x, nx_y = G_proj.nodes[node]['x'], G_proj.nodes[node]['y']

        dist = np.hypot(nx_x - start_x, nx_y - start_y)

        if dist >= min_dist:
            if nx_x >= start_x and nx_y >= start_y:
                quadrants['North-East'].append(node)
            elif nx_x < start_x and nx_y >= start_y:
                quadrants['North-West'].append(node)
            elif nx_x < start_x and nx_y < start_y:
                quadrants['South-West'].append(node)
            elif nx_x >= start_x and nx_y < start_y:
                quadrants['South-East'].append(node)

    target_nodes = []
    for q_name, q_nodes in quadrants.items():
        if not q_nodes:
            all_q_nodes = [n for n in G_proj.nodes() if n != start_node and
                           ((G_proj.nodes[n]['x'] >= start_x and G_proj.nodes[n][
                               'y'] >= start_y) if q_name == 'North-East' else
                            (G_proj.nodes[n]['x'] < start_x and G_proj.nodes[n][
                                'y'] >= start_y) if q_name == 'North-West' else
                            (G_proj.nodes[n]['x'] < start_x and G_proj.nodes[n][
                                'y'] < start_y) if q_name == 'South-West' else
                            (G_proj.nodes[n]['x'] >= start_x and G_proj.nodes[n]['y'] < start_y))]
            all_q_nodes.sort(key=lambda n: np.hypot(G_proj.nodes[n]['x'] - start_x, G_proj.nodes[n]['y'] - start_y),
                             reverse=True)
            target_nodes.append(all_q_nodes[0])
        else:
            target_nodes.append(random.choice(q_nodes))

    all_waypoints_3d = []
    cruise_altitude = 50.0

    route_colors = ['#E63946', '#3A86FF', '#F4A261', '#8338EC']

    fig, ax = plt.subplots(figsize=(12, 12))

    ox.plot_graph(G_proj, ax=ax, show=False, close=False,
                  node_size=0, edge_color='#CCCCCC', edge_linewidth=0.8, edge_alpha=1)

    start_x, start_y = G_proj.nodes[start_node]['x'], G_proj.nodes[start_node]['y']

    global_offset_x = start_x
    global_offset_y = start_y

    for i, target in enumerate(target_nodes):
        c = route_colors[i]
        path_nodes = nx.shortest_path(G_proj, start_node, target, weight='length')

        waypoints_2d = np.array([[G_proj.nodes[n]['x'], G_proj.nodes[n]['y']] for n in path_nodes])
        waypoints_2d[:, 0] -= global_offset_x
        waypoints_2d[:, 1] -= global_offset_y

        waypoints_2d = downsample_waypoints_rdp(waypoints_2d, tolerance=3.0)

        wp_3d = np.zeros((len(waypoints_2d), 3))
        for j, (x, y) in enumerate(waypoints_2d):
            wp_3d[j] = [x, y, cruise_altitude]

        all_waypoints_3d.append(wp_3d)

        path_x = waypoints_2d[:, 0] + global_offset_x
        path_y = waypoints_2d[:, 1] + global_offset_y


        ax.plot(path_x, path_y, c=c, linewidth=4.5, alpha=0.85, zorder=5, label=f'Delivery Route {i + 1}')


        tx, ty = path_x[-1], path_y[-1]
        ax.scatter(tx, ty, c=c, s=400, zorder=10, marker='*', edgecolor='white', linewidth=1.5,
                   label=f'Destination {i + 1}')


        ax.annotate(f'Destination {i + 1}', (tx, ty), textcoords="offset points", xytext=(0, 15),
                    ha='center', fontsize=12, fontweight='bold', color=c, zorder=15,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, lw=1.5, alpha=0.9))


        mid_idx = len(path_x) // 2
        ax.annotate(f'Route {i + 1}', (path_x[mid_idx], path_y[mid_idx]), textcoords="offset points", xytext=(15, 0),
                    ha='left', fontsize=11, fontweight='bold', color='white', zorder=15,
                    bbox=dict(boxstyle="round,pad=0.3", fc=c, ec="none", alpha=0.9))


    ax.scatter(start_x, start_y, c='#000000', s=500, zorder=20, marker='^', edgecolor='white', linewidth=2,
               label='Start Point')
    ax.annotate('Start Point', (start_x, start_y+300), textcoords="offset points", xytext=(0, -25),
                ha='center', fontsize=14, fontweight='bold', color='black', zorder=20,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=2, alpha=0.9))

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)  # 极粗的边框
        spine.set_edgecolor('black')  # 纯黑边界

    ax.annotate('N', xy=(0.06, 0.94), xytext=(0.06, 0.86),
                xycoords='axes fraction', textcoords='axes fraction',
                fontsize=20, fontweight='bold', ha='center', va='center', zorder=30,
                arrowprops=dict(facecolor='black', width=5, headwidth=15, headlength=15))

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    scale_length = 500.0


    start_x = xmax - (xmax - xmin) * 0.05 - scale_length
    start_y = ymin + (ymax - ymin) * 0.05

    ax.plot([start_x, start_x + scale_length], [start_y, start_y], color='black', linewidth=5, zorder=30)
    tick_height = (ymax - ymin) * 0.01
    ax.plot([start_x, start_x], [start_y - tick_height, start_y + tick_height], color='black', linewidth=2, zorder=30)
    ax.plot([start_x + scale_length, start_x + scale_length], [start_y - tick_height, start_y + tick_height],
            color='black', linewidth=2, zorder=30)

    ax.text(start_x + scale_length / 2, start_y + tick_height * 1.5, '500 m',
            color='black', fontsize=14, fontweight='bold', ha='center', va='bottom', zorder=30)

    ax.set_title("UAM Flight Corridors (Shanghai Luyuan Rd / Cao'an Hwy)", fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("UTM X-Coordinate (Meters)", fontsize=14, fontweight='bold')
    ax.set_ylabel("UTM Y-Coordinate (Meters)", fontsize=14, fontweight='bold')


    plt.tight_layout()
    plt.savefig('shanghai_delivery_4_routes_geofence.png', dpi=600, bbox_inches='tight')
    return all_waypoints_3d


if __name__ == "__main__":
    wp_3d = generate_shanghai_logistics_map()