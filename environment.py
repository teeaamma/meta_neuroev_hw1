from config import *
import numpy as np

class DroneEnvironment:
    def __init__(self):
        self.actions = [(d, s) for d in DIRECTIONS for s in ACTION_SPEEDS][:ACTION_COUNT]
        self.reset()

    @property
    def state_shape(self):
        return DX_BINS, DY_BINS, BATTERY_BINS, VELOCITY_BINS

    def reset(self):
        self.x = float(np.clip(np.random.normal(START_MEAN[0], START_SIGMA), START_X_CLIP[0], START_X_CLIP[1]))
        self.y = float(np.clip(np.random.normal(START_MEAN[1], START_SIGMA), START_Y_CLIP[0], START_Y_CLIP[1]))
        self.battery = 1.0
        self.velocity = VELOCITIES[0]
        self.steps = 0
        self.wind_timer = 1
        self.wx = float(np.random.uniform(*WIND_X))
        self.wy = float(np.random.uniform(*WIND_Y))
        self.total_energy = 0.0
        self.total_collisions = 0
        self.trajectory = [(self.x, self.y)]
        return self.get_state()

    def disc(self, v, lo, hi, bins):
        i = int((v - lo) / (hi - lo) * bins)
        return max(0, min(bins - 1, i))

    def get_state(self):
        dx = STATION_X - self.x
        dy = STATION_Y - self.y
        dx_lo = STATION_X - X_MAX
        dx_hi = STATION_X
        dy_lo = STATION_Y - Y_MAX
        dy_hi = STATION_Y
        return (
            self.disc(dx, dx_lo, dx_hi, DX_BINS),
            self.disc(dy, dy_lo, dy_hi, DY_BINS),
            self.disc(self.battery, 0.0, 1.0, BATTERY_BINS),
            min(range(VELOCITY_BINS), key=lambda i: abs(VELOCITIES[i] - self.velocity)),
        )

    def in_station(self, x=None, y=None):
        x = self.x if x is None else x
        y = self.y if y is None else y
        return np.hypot(x - STATION_X, y - STATION_Y) < STATION_RADIUS

    def collision(self, x, y):
        for x1, y1, x2, y2 in OBSTACLES:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    def update_wind(self):
        if self.wind_timer % WIND_UPDATE_PERIOD == 0:
            self.wx = float(np.random.uniform(*WIND_X))
            self.wy = float(np.random.uniform(*WIND_Y))

    def step(self, action_idx):
        d, s = self.actions[action_idx]
        self.update_wind()

        nx = self.x + s * np.cos(d) + 0.8 * self.wx
        ny = self.y + s * np.sin(d) + 1.2 * self.wy

        collision = False
        if self.collision(nx, ny):
            collision = True
            self.total_collisions += 1
            nx, ny = self.x, self.y
        else:
            nx = float(np.clip(nx, 0.0, X_MAX))
            ny = float(np.clip(ny, 0.0, Y_MAX))

        self.x, self.y = nx, ny
        self.velocity = s
        in_station = self.in_station()

        energy = 0.020 * s ** 2 + 0.004 * abs(self.wx) + 0.003 * abs(self.wy)
        self.total_energy += energy
        self.battery = min(1.0, self.battery - energy + (CHARGE_POWER if in_station else 0.0))

        self.steps += 1
        self.wind_timer += 1
        self.trajectory.append((self.x, self.y))

        reward = -1.2 - 0.035 * s ** 2 + (0.15 * CHARGE_POWER if in_station else 0.0)

        done = False
        success = False

        if in_station and self.battery > SUCCESS_BATTERY_THRESHOLD and self.steps < SUCCESS_STEP_LIMIT:
            reward = 100.0
            done = True
            success = True
        elif self.battery < FAIL_BATTERY_THRESHOLD:
            reward = -180.0
            done = True
        elif self.steps >= MAX_STEPS:
            done = True

        return self.get_state(), reward, done, {
            "collision": collision,
            "success": success,
            "battery": self.battery,
        }