import numpy as np
from sklearn.cluster import KMeans
from .BaseLotSelector import ParkingLotSelector


class ResidentDensitySelector(ParkingLotSelector):
    def select_parking_lots(self):
        buffer_distance = 500

        # Use .loc for assignment to avoid chained indexing issues
        self.gdf_parking_lots = self.gdf_parking_lots.copy()
        self.gdf_parking_lots.loc[:, "resident_count"] = self.gdf_parking_lots.geometry.apply(
            lambda lot: self.gdf_residents[
                self.gdf_residents.geometry.within(lot.buffer(buffer_distance))
            ].shape[0]
        )

        # Return a copy of the sorted slice
        return (
            self.gdf_parking_lots.sort_values("resident_count", ascending=False)
            .head(self.required_count)
            .copy()
        )


class ClusteringSelector(ParkingLotSelector):
    def select_parking_lots(self):
        # Compute resident coordinates
        resident_coords = np.array([[pt.x, pt.y] for pt in self.gdf_residents.geometry])

        # Cluster residents
        kmeans = KMeans(n_clusters=self.required_count, random_state=42)
        cluster_labels = kmeans.fit_predict(resident_coords)
        cluster_centers = kmeans.cluster_centers_

        # For each cluster, find the closest parking lot
        selected_indices = []
        for center in cluster_centers:
            distances = np.linalg.norm(self.lot_coords - center, axis=1)
            closest_idx = np.argmin(distances)
            selected_indices.append(closest_idx)

        # Remove duplicates
        selected_indices = list(set(selected_indices))

        # If fewer than required, fill with closest remaining lots
        if len(selected_indices) < self.required_count:
            remaining_indices = [i for i in range(len(self.lot_coords)) if i not in selected_indices]

            # Sort remaining lots by distance to the densest cluster center
            densest_cluster_idx = np.argmax(np.bincount(cluster_labels))
            densest_center = cluster_centers[densest_cluster_idx]
            remaining_distances = np.linalg.norm(self.lot_coords[remaining_indices] - densest_center, axis=1)

            extra_indices = [
                remaining_indices[i]
                for i in np.argsort(remaining_distances)[: (self.required_count - len(selected_indices))]
            ]
            selected_indices.extend(extra_indices)

        # Return a copy to avoid warnings downstream
        return self.gdf_parking_lots.iloc[selected_indices].copy()
