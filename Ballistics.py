
from dataclasses import dataclass
import numpy as np


@dataclass
class Vector3d:
    x: float
    y: float
    z: float

@dataclass
class PositionAtTime:
    x: float
    y: float
    z: float
    t: float
    
    @classmethod
    def from_vector(cls, vector: Vector3d, time: float):
        return cls(vector.x, vector.y, vector.z, time)


class Ballistics:
    
    def cm_to_meter(cm: float) -> float:
        return cm / 100.0

    @staticmethod
    def meter_to_cm(meter: float) -> float:
        return meter * 100.0
    
    
    gravity = 9.81  # m/s^2
    
    distance_to_rim =  26.5 #cm
    field_length = meter_to_cm(10)
    red_rim_x_position = distance_to_rim
    blue_rim_x_position = field_length - distance_to_rim
    
    @staticmethod
    
    


    @staticmethod
    def filter_positions_after_rim(positions: list[PositionAtTime], is_blue_rim: bool) -> list[PositionAtTime]:
        if is_blue_rim:
            return [p for p in positions if p.z > Ballistics.blue_rim_x_position]
        else:
            return [p for p in positions if p.z < Ballistics.red_rim_x_position]

    @staticmethod
    def calculate_position_at_time(time: float, positions_list: list[PositionAtTime]) -> Vector3d:
        vx = Ballistics.calculate_vx_regression(positions_list)
        vy = Ballistics.calculate_vy_regression(positions_list)
        initial_position = Ballistics.calculate_initial_position(positions_list)
        x = initial_position.x + vx * time
        y = initial_position.y + vy * time
        z = initial_position.z + (0.5 * Ballistics.gravity * time * time)

        return Vector3d(x, y, z)

    @staticmethod
    def calculate_initial_position(positions: list[PositionAtTime], is_blue_rim: bool) -> Vector3d:
        if not positions or len(positions) < 2:
            return Vector3d(0, 0, 0)

        positions = Ballistics.filter_positions_after_rim(positions, is_blue_rim)
        
        
        t_array = np.array([p.t for p in positions])
        x_array = np.array([p.x for p in positions])
        y_array = np.array([p.y for p in positions])
        z_array = np.array([p.z for p in positions])

        # Solve linear regression for X and Y
        A = np.vstack([t_array, np.ones_like(t_array)]).T
        vx, _ = np.linalg.lstsq(A, x_array, rcond=None)[0]
        vy, _ = np.linalg.lstsq(A, y_array, rcond=None)[0]

        # Initial positions at first timestamp
        x0 = x_array[0] - vx * t_array[0]
        y0 = y_array[0] - vy * t_array[0]

        # Solve Z with gravity correction
        z_prime = z_array + 0.5 * Ballistics.gravity * t_array**2
        vz, _ = np.linalg.lstsq(A, z_prime, rcond=None)[0]
        z0 = z_array[0] - vz * t_array[0]

        return Vector3d(x0, y0, z0)


    @staticmethod
    def calculate_vx_regression(positions):
        t = np.array([p.t for p in positions])
        x = np.array([p.x for p in positions])
        A = np.vstack([t, np.ones_like(t)]).T
        vx = np.linalg.lstsq(A, x, rcond=None)[0]
        return vx
    
    @staticmethod
    def calculate_vy_regression(positions):
        t = np.array([p.t for p in positions])
        y = np.array([p.y for p in positions])
        A = np.vstack([t, np.ones_like(t)]).T
        vy = np.linalg.lstsq(A, y, rcond=None)[0]
        return vy
