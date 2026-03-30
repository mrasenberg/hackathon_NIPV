import numpy as np
from collections import Counter

from .BaseLotSelector import ParkingLotSelector


class EvenSpreadSelector(ParkingLotSelector):
    def select_parking_lots(self):
        selected_indices = [np.random.randint(len(self.lot_coords))]
        for _ in range(self.required_count - 1):
            dist_to_selected = np.min(
                np.linalg.norm(self.lot_coords[:, None] - self.lot_coords[selected_indices], axis=2),
                axis=1,
            )
            dist_to_selected[selected_indices] = -1
            next_idx = np.argmax(dist_to_selected)
            selected_indices.append(next_idx)
        return self.gdf_parking_lots.iloc[selected_indices]


class MinAvgDistanceSelector(ParkingLotSelector):
    def select_parking_lots(self):
        # Coordinates (ensure CRS is metric and consistent upstream)
        resident_coords = np.vstack([self.gdf_residents.geometry.x.values,
                                     self.gdf_residents.geometry.y.values]).T
        lot_coords = self.lot_coords  # shape (n_lots, 2) with same CRS

        # Distance matrix: residents x lots
        dist_matrix = np.linalg.norm(
            resident_coords[:, None, :] - lot_coords[None, :, :], axis=2
        )

        n_res, n_lots = dist_matrix.shape
        k = min(self.required_count, n_lots)

        selected_indices = []
        current_min_dist = np.full(n_res, np.inf)

        for _ in range(k):
            best_idx, best_score = None, np.inf
            for idx in range(n_lots):
                if idx in selected_indices:
                    continue
                new_min_dist = np.minimum(current_min_dist, dist_matrix[:, idx])
                avg_dist = float(np.mean(new_min_dist))
                if avg_dist < best_score:
                    best_score, best_idx = avg_dist, idx
            if best_idx is None:
                break
            selected_indices.append(best_idx)
            current_min_dist = np.minimum(current_min_dist, dist_matrix[:, best_idx])

        return self.gdf_parking_lots.iloc[selected_indices]


class MinMaxDistanceSelector(ParkingLotSelector):
    def select_parking_lots(self):
        # Ensure CRS is metric and consistent upstream.
        resident_coords = np.vstack([
            self.gdf_residents.geometry.x.values,
            self.gdf_residents.geometry.y.values
        ]).T
        lot_coords = self.lot_coords  # shape (n_lots, 2), same CRS

        dist_matrix = np.linalg.norm(
            resident_coords[:, None, :] - lot_coords[None, :, :],
            axis=2
        )
        n_res, n_lots = dist_matrix.shape
        k = min(self.required_count, n_lots)

        selected = []

        # Seed: pick the lot minimizing the worst-case distance over residents
        per_lot_max = dist_matrix.max(axis=0)
        first = int(np.argmin(per_lot_max))
        selected.append(first)

        current_min = dist_matrix[:, first].copy()

        # Add one lot per iteration until we reach k
        while len(selected) < k:
            remaining = [j for j in range(n_lots) if j not in selected]
            if not remaining:
                break

            r_star = int(np.argmax(current_min))
            nearest_idx_in_remaining = int(np.argmin(dist_matrix[r_star, remaining]))
            j_star = int(remaining[nearest_idx_in_remaining])

            current_min = np.minimum(current_min, dist_matrix[:, j_star])
            selected.append(j_star)

        return self.gdf_parking_lots.iloc[selected]
