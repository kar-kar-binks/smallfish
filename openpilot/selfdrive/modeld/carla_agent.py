from __future__ import annotations

import math

import cv2
import numpy as np
import carla

from openpilot.selfdrive.modeld.backend import CustomModelState

WHEELBASE_M = 2.85
MAX_STEER_RAD = math.radians(70)
LEAD_SEARCH_RADIUS_M = 60.0
LEAD_BRAKE_DIST_M = 15.0
SPEED_KP = 0.6
SPEED_KI = 0.1
SPEED_KD = 0.05
CONTROL_DT = 0.05


class CarlaDrivingAgent:
  def __init__(self, world: carla.World, vehicle: carla.Vehicle, cam_w: int, cam_h: int):
    self.world = world
    self.vehicle = vehicle
    self.model = CustomModelState(cam_w, cam_h, usbgpu=False)
    self.latest_rgb: np.ndarray | None = None
    self.camera: carla.Actor | None = None
    self.control = carla.VehicleControl()
    self._integral = 0.0
    self._prev_error = 0.0

  def attach_camera(self, blueprint_library: carla.BlueprintLibrary, transform: carla.Transform) -> None:
    bp = blueprint_library.find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(self.model.cam_w))
    bp.set_attribute("image_size_y", str(self.model.cam_h))
    bp.set_attribute("fov", "110")
    self.camera = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
    self.camera.listen(self._on_image)

  def _on_image(self, image: carla.Image) -> None:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    self.latest_rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)

  def lane_ground_truth(self):
    wp = self.world.get_map().get_waypoint(self.vehicle.get_location())
    return wp.get_left_lane(), wp.get_right_lane()

  def lead_ground_truth(self, radius: float = LEAD_SEARCH_RADIUS_M):
    ego_tf = self.vehicle.get_transform()
    ego_loc = ego_tf.location
    ego_fwd = ego_tf.get_forward_vector()
    nearest, nearest_dist = None, radius
    for actor in self.world.get_actors().filter("vehicle.*"):
      if actor.id == self.vehicle.id:
        continue
      delta = actor.get_location() - ego_loc
      dist = delta.length()
      if dist > nearest_dist:
        continue
      forward_dot = delta.x * ego_fwd.x + delta.y * ego_fwd.y + delta.z * ego_fwd.z
      if forward_dot <= 0:
        continue
      nearest, nearest_dist = actor, dist
    return nearest, nearest_dist

  def step(self, route_lla=None, ego_lla=None, ego_yaw_ned=None) -> carla.VehicleControl:
    if self.latest_rgb is None:
      return self.control

    v_ego = self.vehicle.get_velocity().length()

    if ego_lla is not None and ego_yaw_ned is not None:
      target_point_xy = self.model._target_point(ego_lla, float(ego_yaw_ned), route_lla or [])
    else:
      target_point_xy = np.array([[20.0, 0.0], [40.0, 0.0]], dtype=np.float32)

    plan = self.model._plan_from_frame(self.latest_rgb, v_ego, target_point_xy)
    if plan is None:
      self.control.throttle = 0.0
      self.control.brake = 1.0
      self.control.steer = 0.0
      return self.control

    target_speed = float(plan["v"][2])
    lead_actor, lead_dist = self.lead_ground_truth()
    if lead_actor is not None and lead_dist < LEAD_BRAKE_DIST_M:
      target_speed = min(target_speed, lead_actor.get_velocity().length())

    speed_error = target_speed - v_ego
    self._integral = float(np.clip(self._integral + speed_error * CONTROL_DT, -10.0, 10.0))
    derivative = (speed_error - self._prev_error) / CONTROL_DT
    self._prev_error = speed_error
    throttle_brake = SPEED_KP * speed_error + SPEED_KI * self._integral + SPEED_KD * derivative

    curvature = plan["kappa_v2"] / max(v_ego, 1.0) ** 2
    steer = math.atan(curvature * WHEELBASE_M)
    self.control.steer = float(np.clip(steer / MAX_STEER_RAD, -1.0, 1.0))

    if plan["should_stop"]:
      self.control.throttle = 0.0
      self.control.brake = 1.0
    elif throttle_brake < 0:
      self.control.throttle = 0.0
      self.control.brake = float(np.clip(-throttle_brake, 0.0, 1.0))
    else:
      self.control.throttle = float(np.clip(throttle_brake, 0.0, 1.0))
      self.control.brake = 0.0

    return self.control

  def apply(self) -> None:
    self.vehicle.apply_control(self.control)

  def destroy(self) -> None:
    if self.camera is not None:
      self.camera.stop()
      self.camera.destroy()
      self.camera = None
