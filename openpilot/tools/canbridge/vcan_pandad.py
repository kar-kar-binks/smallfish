#!/usr/bin/env python3
import argparse
import select
import socket
import struct
import time

from opendbc.car.structs import car
import openpilot.cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.selfdrive.pandad import can_list_to_can_capnp, can_capnp_to_list

CAN_FRAME_FMT = "=IB3x8s"
CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FMT)
CAN_EFF_FLAG = 0x80000000
CAN_ID_MASK = 0x1FFFFFFF


def open_socketcan(iface: str) -> socket.socket:
  s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
  s.bind((iface,))
  s.setblocking(False)
  return s


def recv_frames(sock: socket.socket, bus: int) -> list[tuple[int, bytes, int]]:
  frames = []
  while True:
    try:
      raw = sock.recv(CAN_FRAME_SIZE)
    except BlockingIOError:
      break
    can_id, length, data = struct.unpack(CAN_FRAME_FMT, raw)
    can_id &= CAN_ID_MASK
    frames.append((can_id, data[:length], bus))
  return frames


def send_frame(sock: socket.socket, address: int, dat: bytes) -> None:
  can_id = address
  if can_id > 0x7FF:
    can_id |= CAN_EFF_FLAG
  padded = dat.ljust(8, b"\x00")
  raw = struct.pack(CAN_FRAME_FMT, can_id, len(dat), padded)
  sock.send(raw)


def get_safety_config(params: Params):
  cp_bytes = params.get("CarParams")
  if not cp_bytes:
    return "silent", 0
  with car.CarParams.from_bytes(cp_bytes) as cp:
    if len(cp.safetyConfigs) == 0:
      return "silent", 0
    sc = cp.safetyConfigs[-1]
    return sc.safetyModel, sc.safetyParam


def main():
  parser = argparse.ArgumentParser(description="Drop-in pandad replacement backed by real SocketCAN vcan interfaces.")
  parser.add_argument("--buses", type=int, default=3, help="number of CAN buses / vcan interfaces")
  parser.add_argument("--iface-prefix", default="vcan", help="SocketCAN interface prefix, e.g. vcan -> vcan0, vcan1, ...")
  args = parser.parse_args()

  socks = [open_socketcan(f"{args.iface_prefix}{i}") for i in range(args.buses)]

  params = Params()
  pm = messaging.PubMaster(["can", "pandaStates"])
  sendcan_sock = messaging.sub_sock("sendcan", conflate=False)

  rk = Ratekeeper(100, print_delay_threshold=None)
  idx = 0
  while True:
    readable, _, _ = select.select(socks, [], [], 0)
    can_msgs = []
    for s in readable:
      bus = socks.index(s)
      can_msgs.extend(recv_frames(s, bus))
    if can_msgs:
      pm.send("can", can_list_to_can_capnp(can_msgs))

    for raw in messaging.drain_sock_raw(sendcan_sock):
      for _, frames in can_capnp_to_list([raw], msgtype="sendcan"):
        for address, dat, bus in frames:
          if 0 <= bus < len(socks):
            send_frame(socks[bus], address, dat)

    if idx % 10 == 0:
      safety_model, safety_param = get_safety_config(params)
      dat = messaging.new_message("pandaStates", 1)
      dat.valid = True
      dat.pandaStates[0] = {
        "ignitionLine": True,
        "pandaType": "blackPanda",
        "controlsAllowed": True,
        "safetyModel": safety_model,
        "safetyParam": safety_param,
      }
      pm.send("pandaStates", dat)

    idx += 1
    rk.keep_time()


if __name__ == "__main__":
  main()
