from __future__ import annotations

import importlib.util
import math
import os
import sys

import cv2
import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.common.transformations.coordinates import LocalCoord

MODEL_REPO = os.environ.get("MODEL_REPO", "/root/model")
MODEL_CHECKPOINT = os.environ.get(
    "MODEL_CHECKPOINT",
    "/root/model_checkpoints/model/checkpoints/epoch=013.ckpt/pytorch_model.pt")
sys.path.insert(0, MODEL_REPO)
sys.path.insert(0, os.path.join(MODEL_REPO, "team_code"))

DEFAULT_NL_COMMAND = "Drive safely and follow the road."
TARGET_POINT_LOOKAHEAD_M = 20.0
TARGET_POINT_LOOKAHEAD2_M = 40.0


def _nv12_to_rgb(buf, width: int, height: int, stride: int, y_height: int) -> np.ndarray:
  raw = np.frombuffer(buf.data, dtype=np.uint8)
  y_plane = raw[:stride * y_height].reshape(y_height, stride)[:height, :width]
  uv_start = stride * y_height
  uv_plane = raw[uv_start:uv_start + stride * (height // 2)].reshape(height // 2, stride)[:, :width]
  nv12 = np.vstack([y_plane, uv_plane])
  return cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)


class CustomModelState:
  def __init__(self, cam_w: int, cam_h: int, usbgpu: bool):
    self.usbgpu = usbgpu
    self.cam_w, self.cam_h = cam_w, cam_h
    self.vision_input_names = ["img"]
    stride, y_height, uv_height, size = get_nv12_info(cam_w, cam_h)
    self.frame_buf_params = {"img": (stride, y_height, uv_height, size)}
    self._load_model()

  def _load_model(self) -> None:
    import hydra
    import torch
    from omegaconf import OmegaConf
    from pathlib import Path
    from transformers import AutoProcessor
    from simlingo_training.utils.custom_types import DrivingInput
    from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess

    self.torch = torch
    self.DrivingInput = DrivingInput
    self.build_transform = build_transform
    self.dynamic_preprocess = dynamic_preprocess

    if torch.cuda.is_available():
      self.device = torch.device("cuda")
      self.dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
      self.device = torch.device("mps")
      self.dtype = torch.float32
    else:
      self.device = torch.device("cpu")
      self.dtype = torch.float32
    config_path = Path(MODEL_CHECKPOINT).parent.parent.parent / ".hydra" / "config.yaml"
    with open(config_path, "r") as f:
      cfg = OmegaConf.load(f)
    self.cfg = cfg
    cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img

    processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
    self.tokenizer = processor.tokenizer if "tokenizer" in processor.__dict__ else processor
    self.tokenizer.add_special_tokens({"additional_special_tokens": [
        "<WAYPOINTS>", "<WAYPOINTS_DIFF>", "<ORG_WAYPOINTS_DIFF>", "<ORG_WAYPOINTS>",
        "<WAYPOINT_LAST>", "<ROUTE>", "<ROUTE_DIFF>", "<TARGET_POINT>"]})
    self.tokenizer.padding_side = "left"

    cache_dir = f"pretrained/{cfg.model.vision_model.variant.split('/')[1]}"
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(self.dtype)
    self.model = hydra.utils.instantiate(
        cfg.model, cfg_data_module=cfg.data_module, processor=processor,
        cache_dir=cache_dir, _recursive_=False).to(self.device)
    torch.set_default_dtype(default_dtype)
    self.model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=self.device))
    self.model.eval()

    self.transform = build_transform(input_size=448)
    self.nl_command_path = os.environ.get(
        "NL_COMMAND_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "command.txt"))

    cache_dir = os.path.abspath(f"pretrained/{cfg.model.vision_model.variant.split('/')[1]}")
    conv_path = os.path.join(cache_dir, "conversation.py")
    if not os.path.exists(conv_path):
      from huggingface_hub import snapshot_download
      snapshot_download(repo_id=cfg.model.vision_model.variant, local_dir=cache_dir)
    spec = importlib.util.spec_from_file_location("get_conv_template", conv_path)
    conv_module = importlib.util.module_from_spec(spec)
    sys.modules["get_conv_template"] = conv_module
    spec.loader.exec_module(conv_module)
    self.conv_module = conv_module

    from transformers import AutoConfig
    tmp_config = AutoConfig.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
    image_size = tmp_config.force_image_size or tmp_config.vision_config.image_size
    patch_size = tmp_config.vision_config.patch_size
    self.num_image_token = int((image_size // patch_size) ** 2 * (tmp_config.downsample_ratio ** 2))

    from team_code.simlingo_utils import get_camera_intrinsics, get_camera_extrinsics
    self._camera_intrinsics = get_camera_intrinsics(self.cam_w, self.cam_h, 110).unsqueeze(0).to(self.device)
    self._camera_extrinsics = get_camera_extrinsics().unsqueeze(0).to(self.device)

  def _read_nl_command(self) -> str:
    if os.path.exists(self.nl_command_path):
      text = open(self.nl_command_path).read().strip()
      if text:
        return text
    return DEFAULT_NL_COMMAND

  def _target_point(self, ego_lla: np.ndarray, ego_yaw_ned: float, route_lla: list) -> np.ndarray:
    if not route_lla:
      return np.array([[TARGET_POINT_LOOKAHEAD_M, 0.0], [TARGET_POINT_LOOKAHEAD2_M, 0.0]], dtype=np.float32)

    origin = LocalCoord.from_geodetic(ego_lla)
    cy, sy = math.cos(ego_yaw_ned), math.sin(ego_yaw_ned)

    def to_ego_frame(lat, lon):
      ned = origin.geodetic2ned(np.array([lat, lon, ego_lla[2]]))
      north, east = float(ned[0]), float(ned[1])
      forward = north * cy + east * sy
      right = -north * sy + east * cy
      return [forward, -right]

    near = to_ego_frame(*route_lla[0])
    far = to_ego_frame(*route_lla[min(1, len(route_lla) - 1)])
    return np.array([near, far], dtype=np.float32)

  def _preprocess_image(self, rgb: np.ndarray):
    import torch
    from PIL import Image

    _, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    bgr = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = rgb[:int(rgb.shape[0] - (rgb.shape[0] * 4.8) // 16), :, :]

    image = Image.fromarray(rgb)
    tiles = self.dynamic_preprocess(image, image_size=448,
                                    use_thumbnail=self.cfg.model.vision_model.use_global_img, max_num=2)
    pixel_values = torch.stack([self.transform(t) for t in tiles])
    num_patches, C, H, W = pixel_values.shape
    pixel_values = pixel_values.view(1, 1, num_patches, C, H, W)
    image_sizes = torch.tensor([[image.size[1], image.size[0]]])
    return pixel_values, image_sizes

  def _build_prompt(self, speed: float, use_cot: bool, target_point_xy: np.ndarray):
    from simlingo_training.utils.custom_types import LanguageLabel

    prompt_tp = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
    tail = "What should the ego do next?" if use_cot else "Predict the waypoints."
    prompt = f"Current speed: {speed:.1f} m/s. {prompt_tp} {tail}"

    question = f"<image>\n{prompt}"
    conversation = [
      {"role": "user", "content": [{"type": "text", "text": question}, {"type": "image"}]},
      {"role": "assistant", "content": [{"type": "text", "text": "Waypoints:"}]},
    ]
    template = self.conv_module.get_conv_template("internlm2-chat")
    for turn in conversation:
      role = template.roles[1] if turn["role"] == "assistant" else template.roles[0]
      text = turn["content"][0]["text"]
      template.append_message(role, None if turn["role"] == "assistant" else text)
    query = template.get_prompt()
    system_prompt = template.system_template.replace("{system_message}", template.system_message) + template.sep
    query = query.replace(system_prompt, "")

    image_tokens = "<img>" + "<IMG_CONTEXT>" * self.num_image_token * 2 + "</img>"
    query = query.replace("<image>", image_tokens, 1)

    tok = self.tokenizer([query], padding=True, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    phrase_ids = tok["input_ids"].to(self.device)
    phrase_valid = (tok["input_ids"] != self.tokenizer.pad_token_id).to(self.device)
    tp_token_id = self.tokenizer.convert_tokens_to_ids("<TARGET_POINT>")
    return LanguageLabel(
      phrase_ids=phrase_ids, phrase_valid=phrase_valid, phrase_mask=phrase_valid,
      placeholder_values=[{tp_token_id: target_point_xy}], language_string=[query], loss_masking=None,
    )

  def _infer(self, rgb: np.ndarray, v_ego: float, target_point_xy: np.ndarray):
    pixel_values, image_sizes = self._preprocess_image(rgb)
    pixel_values = pixel_values.to(self.device, dtype=self.dtype)

    use_cot = bool(self.cfg.get("use_cot", False))
    language = self._build_prompt(v_ego, use_cot, target_point_xy)

    driving_input = self.DrivingInput(
      camera_images=pixel_values,
      image_sizes=image_sizes,
      camera_intrinsics=self._camera_intrinsics,
      camera_extrinsics=self._camera_extrinsics,
      vehicle_speed=self.torch.tensor([[v_ego]], device=self.device, dtype=self.torch.float32),
      target_point=self.torch.from_numpy(target_point_xy).unsqueeze(0).to(self.device, dtype=self.torch.float32),
      prompt=language,
      prompt_inference=language,
    )
    with self.torch.inference_mode():
      pred_speed_wps, pred_route, _language = self.model(driving_input)
    if pred_route is None:
      return None
    waypoints = pred_route[0].float().cpu().numpy()
    speed_profile = (pred_speed_wps[0, :, 0].float().cpu().numpy()
                     if pred_speed_wps is not None else np.full(len(waypoints), v_ego, dtype=np.float32))
    return waypoints, speed_profile

  def _plan_from_frame(self, rgb: np.ndarray, v_ego: float, target_point_xy: np.ndarray) -> dict | None:
    result = self._infer(rgb, v_ego, target_point_xy)
    if result is None:
      return None
    waypoints, speed_profile = result

    t_src_wp = np.linspace(0.0, 4.0, len(waypoints))
    t_src_speed = np.linspace(0.0, 4.0, len(speed_profile))
    t_dst = np.array(ModelConstants.T_IDXS, dtype=np.float32)

    x = np.interp(t_dst, t_src_wp, waypoints[:, 0])
    y = np.interp(t_dst, t_src_wp, waypoints[:, 1])
    v = np.interp(t_dst, t_src_speed, speed_profile)
    z = np.zeros_like(x)
    a = np.gradient(v, t_dst)

    kappa_v2 = float(2.0 * y[4] / max(t_dst[4], 1e-3) ** 2) if len(y) > 4 else 0.0
    accel = float(a[2]) if len(a) > 2 else 0.0
    should_stop = bool(v[-1] < 0.3 and v_ego < 1.0)

    return {"t": t_dst, "x": x, "y": y, "z": z, "v": v, "a": a,
            "kappa_v2": kappa_v2, "accel": accel, "should_stop": should_stop}

  def run(self, bufs, transforms, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
    stride, y_height, _, _ = self.frame_buf_params["img"]
    rgb = _nv12_to_rgb(bufs["img"], self.cam_w, self.cam_h, stride, y_height)
    v_ego = float(inputs.get("v_ego", 0.0))

    ego_lla = inputs.get("ego_lla")
    ego_yaw_ned = inputs.get("ego_yaw_ned")
    route_lla = inputs.get("route_lla", [])
    if ego_lla is not None and ego_yaw_ned is not None:
      target_point_xy = self._target_point(ego_lla, float(ego_yaw_ned), route_lla)
    else:
      target_point_xy = np.array([[TARGET_POINT_LOOKAHEAD_M, 0.0], [TARGET_POINT_LOOKAHEAD2_M, 0.0]], dtype=np.float32)

    plan_result = self._plan_from_frame(rgb, v_ego, target_point_xy)
    if plan_result is None:
      return None
    x, y, z, v, a = plan_result["x"], plan_result["y"], plan_result["z"], plan_result["v"], plan_result["a"]
    t_dst = plan_result["t"]

    plan = np.zeros((1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH), dtype=np.float32)
    plan[0, :, 0], plan[0, :, 1], plan[0, :, 2] = x, y, z
    plan[0, :, 3] = v
    plan[0, :, 6] = a
    plan_stds = np.ones_like(plan) * 0.5

    kappa_v2 = plan_result["kappa_v2"]
    accel = plan_result["accel"]
    should_stop = plan_result["should_stop"]
    action = np.array([[kappa_v2, accel]], dtype=np.float32)

    lane_lines = np.zeros((1, ModelConstants.NUM_LANE_LINES, ModelConstants.IDX_N, ModelConstants.LANE_LINES_WIDTH), dtype=np.float32)
    lane_lines_prob = np.zeros((1, ModelConstants.NUM_LANE_LINES * 2), dtype=np.float32)
    road_edges = np.zeros((1, ModelConstants.NUM_ROAD_EDGES, ModelConstants.IDX_N, ModelConstants.ROAD_EDGES_WIDTH), dtype=np.float32)
    lead = np.zeros((1, 3, ModelConstants.LEAD_TRAJ_LEN, ModelConstants.LEAD_WIDTH), dtype=np.float32)
    lead_prob = np.zeros((1, 3), dtype=np.float32)
    meta = np.zeros((1, 55), dtype=np.float32)
    desire_state = np.zeros((1, ModelConstants.DESIRE_LEN), dtype=np.float32)
    desire_pred = np.zeros((1, ModelConstants.DESIRE_PRED_LEN, ModelConstants.DESIRE_PRED_WIDTH), dtype=np.float32)
    pose = np.zeros((1, ModelConstants.POSE_WIDTH), dtype=np.float32)
    pose_stds = np.ones((1, ModelConstants.POSE_WIDTH), dtype=np.float32) * 0.5
    wide_from_device_euler = np.zeros((1, ModelConstants.WIDE_FROM_DEVICE_WIDTH), dtype=np.float32)
    wide_from_device_euler_stds = np.ones((1, ModelConstants.WIDE_FROM_DEVICE_WIDTH), dtype=np.float32) * 0.5
    road_transform = np.zeros((1, 3), dtype=np.float32)
    road_transform_stds = np.ones((1, 3), dtype=np.float32) * 0.5

    return {
      "action": action,
      "plan": plan, "plan_stds": plan_stds,
      "lane_lines": lane_lines, "lane_lines_stds": np.ones_like(lane_lines), "lane_lines_prob": lane_lines_prob,
      "road_edges": road_edges, "road_edges_stds": np.ones_like(road_edges),
      "lead": lead, "lead_stds": np.ones_like(lead), "lead_prob": lead_prob,
      "meta": meta,
      "desire_state": desire_state, "desire_pred": desire_pred,
      "pose": pose, "pose_stds": pose_stds,
      "wide_from_device_euler": wide_from_device_euler, "wide_from_device_euler_stds": wide_from_device_euler_stds,
      "road_transform": road_transform, "road_transform_stds": road_transform_stds,
      "hidden_state": np.zeros((ModelConstants.FEATURE_LEN,), dtype=np.float32),
      "_should_stop": should_stop,
    }

  def warmup(self) -> None:
    dummy_rgb = np.zeros((self.cam_h, self.cam_w, 3), dtype=np.uint8)
    dummy_target = np.array([[TARGET_POINT_LOOKAHEAD_M, 0.0], [TARGET_POINT_LOOKAHEAD2_M, 0.0]], dtype=np.float32)
    self._infer(dummy_rgb, 0.0, dummy_target)
