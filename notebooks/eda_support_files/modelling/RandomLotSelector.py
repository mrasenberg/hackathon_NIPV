from .BaseLotSelector import ParkingLotSelector


class RandomSelector(ParkingLotSelector):
    def select_parking_lots(self):
        return self.gdf_parking_lots.sample(n=self.required_count, random_state=7).copy()
