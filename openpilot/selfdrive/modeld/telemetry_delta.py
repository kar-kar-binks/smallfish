from __future__ import annotations

import math


def compute_route_delta(state: dict) -> dict:
  current_x = float(state["current_x"])
  current_y = float(state["current_y"])
  heading_rad = float(state["heading_rad"])
  target_x = float(state["target_x"])
  target_y = float(state["target_y"])

  dx = target_x - current_x
  dy = target_y - current_y
  distance = math.hypot(dx, dy)
  target_bearing_rad = math.atan2(dy, dx)
  heading_error_rad = math.atan2(math.sin(target_bearing_rad - heading_rad), math.cos(target_bearing_rad - heading_rad))

  return {
    "distance_m": distance,
    "dx": dx,
    "dy": dy,
    "target_bearing_rad": target_bearing_rad,
    "heading_error_rad": heading_error_rad,
    "heading_error_deg": math.degrees(heading_error_rad),
  }
