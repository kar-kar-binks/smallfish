#!/usr/bin/env bash
set -e

BUSES="${1:-0 1 2}"

sudo modprobe vcan

for n in $BUSES; do
  iface="vcan${n}"
  if ! ip link show "$iface" &> /dev/null; then
    sudo ip link add dev "$iface" type vcan
  fi
  sudo ip link set up "$iface"
  echo "up: $iface"
done
