import numpy as np
from scipy.spatial import cKDTree
from ortools.graph.python import min_cost_flow
from joblib import Parallel, delayed
from IPython.display import display, HTML


def _assign_random(gdf_res, gdf_par, max_capacity):
    parking_coords = np.vstack([gdf_par["centroid"].x, gdf_par["centroid"].y]).T
    tree = cKDTree(parking_coords)
    capacity_used = np.zeros(len(gdf_par), dtype=int)
    resident_distances = np.empty(len(gdf_res), dtype=float)
    assigned_lots = np.empty(len(gdf_res), dtype=object)
    random_order = np.random.permutation(len(gdf_res))

    for i in random_order:
        point = gdf_res.geometry[i]
        dists, idxs = tree.query([point.x, point.y], k=len(gdf_par))
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)
        for dist, idx in zip(dists, idxs):
            if capacity_used[idx] < max_capacity:
                capacity_used[idx] += 1
                resident_distances[i] = dist
                assigned_lots[i] = gdf_par.index[idx]
                break
        else:
            resident_distances[i] = np.nan
            assigned_lots[i] = None

    gdf_res["assigned_parking_lot"] = assigned_lots
    gdf_res["distance_to_parking"] = resident_distances
    return gdf_res, resident_distances

def _assign_min_cost_flow(gdf_res, gdf_par, max_capacity):
    n_residents = len(gdf_res)
    n_lots = len(gdf_par)
    lot_coords = np.vstack([gdf_par["centroid"].x, gdf_par["centroid"].y]).T
    res_coords = np.vstack([gdf_res.geometry.x, gdf_res.geometry.y]).T

    def compute_row(i):
        return np.sqrt((res_coords[i, 0] - lot_coords[:, 0])**2 +
                       (res_coords[i, 1] - lot_coords[:, 1])**2)

    cost_matrix = np.array(Parallel(n_jobs=-1)(delayed(compute_row)(i) for i in range(n_residents)))

    start_nodes, end_nodes, capacities, unit_costs = [], [], [], []
    source = n_residents + n_lots
    sink = source + 1

    for i in range(n_residents):
        start_nodes.append(source)
        end_nodes.append(i)
        capacities.append(1)
        unit_costs.append(0)

    for i in range(n_residents):
        for j in range(n_lots):
            start_nodes.append(i)
            end_nodes.append(n_residents + j)
            capacities.append(1)
            # unit_costs.append(int(cost_matrix[i, j]))
            unit_costs.append(int(cost_matrix[i, j] ** 2))  # quadratic penalty if we want to reduce outliers

    for j in range(n_lots):
        start_nodes.append(n_residents + j)
        end_nodes.append(sink)
        capacities.append(max_capacity)
        unit_costs.append(0)

    supplies = [0] * (sink + 1)
    supplies[source] = n_residents
    supplies[sink] = -n_residents

    smcf = min_cost_flow.SimpleMinCostFlow()
    for i in range(len(start_nodes)):
        smcf.add_arcs_with_capacity_and_unit_cost(start_nodes[i], end_nodes[i], capacities[i], unit_costs[i])
    for i in range(len(supplies)):
        smcf.set_node_supply(i, supplies[i])

    status = smcf.solve()
    if status != smcf.OPTIMAL:
        raise RuntimeError("Min-cost flow did not find an optimal solution.")

    assigned_lots = [None] * n_residents
    resident_distances = [None] * n_residents
    for i in range(smcf.num_arcs()):
        if smcf.flow(i) > 0:
            start = start_nodes[i]
            end = end_nodes[i]
            if start < n_residents and end >= n_residents and end < source:
                resident_idx = start
                lot_idx = end - n_residents
                assigned_lots[resident_idx] = gdf_par.index[lot_idx]
                resident_distances[resident_idx] = np.sqrt(unit_costs[i])

    gdf_res["assigned_parking_lot"] = assigned_lots
    gdf_res["distance_to_parking"] = resident_distances
    return gdf_res, resident_distances


def assign_residents_to_parking_lots(gdf_resident_gem, gdf_parking_lots_gem_opt, output_file, mode="random", max_capacity=2500):
    """
    Assign residents to parking lots using either random or optimized method.
    """
    if mode == "random":
        gdf_resident_gem, distances = _assign_random(gdf_resident_gem, gdf_parking_lots_gem_opt, max_capacity)
    elif mode == "min_cost_flow":
        gdf_resident_gem, distances = _assign_min_cost_flow(gdf_resident_gem, gdf_parking_lots_gem_opt, max_capacity)
    else:
        raise ValueError("Invalid mode. Choose 'random' or 'min_cost_flow'.")

    # Back to CRS 4326 and save
    gdf_resident_gem = gdf_resident_gem.to_crs(4326)
    gdf_resident_gem = gdf_resident_gem.astype({
        "gemeentenaam": "string",
        "assigned_parking_lot": "int32",
        "distance_to_parking": "int32"
    })
    gdf_resident_gem.to_file(output_file, driver="GeoJSON")

    return gdf_resident_gem, distances
