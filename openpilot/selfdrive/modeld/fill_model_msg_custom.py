from __future__ import annotations

import capnp
import numpy as np
from openpilot.cereal import log
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.fill_model_msg import fill_xyzt, PublishState

ConfidenceClass = log.ModelDataV2.ConfidenceClass


def fill_model_msg_custom(msg: capnp._DynamicStructBuilder, net_output_data: dict[str, np.ndarray], action: log.ModelDataV2.Action,
                            publish_state: PublishState, vipc_frame_id: int, vipc_frame_id_extra: int,
                            frame_id: int, frame_drop: float, timestamp_eof: int, model_execution_time: float,
                            valid: bool) -> None:
  frame_age = frame_id - vipc_frame_id if frame_id > vipc_frame_id else 0
  msg.valid = valid

  modelV2 = msg.modelV2
  modelV2.frameId = vipc_frame_id
  modelV2.frameIdExtra = vipc_frame_id_extra
  modelV2.frameAge = frame_age
  modelV2.frameDropPerc = frame_drop * 100
  modelV2.timestampEof = timestamp_eof
  modelV2.modelExecutionTime = model_execution_time

  fill_xyzt(modelV2.position, ModelConstants.T_IDXS, *net_output_data['plan'][0, :, Plan.POSITION].T, *net_output_data['plan_stds'][0, :, Plan.POSITION].T)
  fill_xyzt(modelV2.velocity, ModelConstants.T_IDXS, *net_output_data['plan'][0, :, Plan.VELOCITY].T)
  fill_xyzt(modelV2.acceleration, ModelConstants.T_IDXS, *net_output_data['plan'][0, :, Plan.ACCELERATION].T)
  fill_xyzt(modelV2.orientation, ModelConstants.T_IDXS, *net_output_data['plan'][0, :, Plan.T_FROM_CURRENT_EULER].T)
  fill_xyzt(modelV2.orientationRate, ModelConstants.T_IDXS, *net_output_data['plan'][0, :, Plan.ORIENTATION_RATE].T)

  modelV2.action = action

  LINE_T_IDXS: list[float] = []
  modelV2.init('laneLines', 4)
  for i in range(4):
    fill_xyzt(modelV2.laneLines[i], LINE_T_IDXS, np.array(ModelConstants.X_IDXS),
              net_output_data['lane_lines'][0, i, :, 0], net_output_data['lane_lines'][0, i, :, 1])
  modelV2.laneLineStds = net_output_data['lane_lines_stds'][0, :, 0, 0].tolist()
  modelV2.laneLineProbs = net_output_data['lane_lines_prob'][0, 1::2].tolist()

  modelV2.init('roadEdges', 2)
  for i in range(2):
    fill_xyzt(modelV2.roadEdges[i], LINE_T_IDXS, np.array(ModelConstants.X_IDXS),
              net_output_data['road_edges'][0, i, :, 0], net_output_data['road_edges'][0, i, :, 1])
  modelV2.roadEdgeStds = net_output_data['road_edges_stds'][0, :, 0, 0].tolist()

  modelV2.init('leadsV3', 3)
  for i in range(3):
    lead = modelV2.leadsV3[i]
    lead.t = ModelConstants.LEAD_T_IDXS
    lead.x = net_output_data['lead'][0, i, :, 0].tolist()
    lead.y = net_output_data['lead'][0, i, :, 1].tolist()
    lead.v = net_output_data['lead'][0, i, :, 2].tolist()
    lead.a = net_output_data['lead'][0, i, :, 3].tolist()
    lead.prob = 0.0
    lead.probTime = ModelConstants.LEAD_T_OFFSETS[i]

  meta = modelV2.meta
  meta.desireState = net_output_data['desire_state'][0].reshape(-1).tolist()
  meta.desirePrediction = net_output_data['desire_pred'][0].reshape(-1).tolist()
  meta.engagedProb = 0.0
  meta.init('disengagePredictions')
  dp = meta.disengagePredictions
  dp.t = ModelConstants.META_T_IDXS
  zeros5 = [0.0] * 5
  dp.brakeDisengageProbs = zeros5
  dp.gasDisengageProbs = zeros5
  dp.steerOverrideProbs = zeros5
  dp.brake3MetersPerSecondSquaredProbs = zeros5
  dp.brake4MetersPerSecondSquaredProbs = zeros5
  dp.brake5MetersPerSecondSquaredProbs = zeros5
  dp.gasPressProbs = zeros5
  dp.brakePressProbs = zeros5
  meta.hardBrakePredicted = False
  meta.laneChangeState = log.LaneChangeState.off
  meta.laneChangeDirection = log.LaneChangeDirection.none

  modelV2.confidence = ConfidenceClass.green
