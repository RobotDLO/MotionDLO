#!/usr/bin/env python3
"""
Synchronized RGB + Event Recorder + Depth + Rosbag (TrackDLO edition)
===================================
Records IDS Peak (frame) and Prophesee (event) cameras simultaneously.
After each captured frame, IDS Peak pulses Line 3 → event camera logs a
trigger timestamp.  Output includes a manifest.json that maps each saved
frame file to its trigger timestamp in the event camera's microsecond clock.

Output layout
-------------
OUTPUT_DIR/
    events.raw          — event stream (trigger timestamps embedded)
    frames/
        frame_000001.jpg
        frame_000002.jpg
        ...
    manifest.json       — {"frame_000001.jpg": 123456, ...}  (µs timestamps)

Usage
-----
    # Standalone (no robot controller):
    python synchronized_recorder.py

    # With robot controller sending start signal on Line 2:
    python synchronized_recorder.py --wait-for-robot

    # Fixed duration:
    python synchronized_recorder.py --duration 30
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
import time

import cv2
import ids_peak.ids_peak as ids_peak
import ids_peak_ipl.ids_peak_ipl as ids_ipl
import metavision_hal
import numpy as np
from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device

# ── Configuration ──────────────────────────────────────────────────────────────



OUTPUT_DIR       = "/path/to/Dataset/Single Cable/Green_Cable/Speed_5/3"
TRACKDLO_DATASET = "/path/to/trackdlo/dataset/single_cable"
ROSBAG_TOPICS    = [
    "/camera/color/image_raw",
    "/camera/aligned_depth_to_color/image_raw",
    "/camera/color/camera_info",
    "/camera/aligned_depth_to_color/camera_info",
]
JPEG_QUALITY     = 95       # JPEG compression (0-100)
PULSE_DURATION_S = 0.001    # 1 ms LOW pulse on Line 3
FRAME_TIMEOUT_MS = 5000     # IDS Peak buffer wait timeout
BUFFER_COUNT     = 16       # IDS Peak DataStream buffer pool
EVENT_DELTA_T_US = 50_000   # event batch window for trigger polling (50 ms)

# ──────────────────────────────────────────────────────────────────────────────


class FrameCaptureThread(threading.Thread):
    """
    Captures frames from the IDS Peak camera.
    After each frame: saves JPEG, pulses Line 3 to trigger the event camera,
    and puts (frame_index, filename, ids_timestamp_ns) into frame_queue.
    """

    def __init__(
        self,
        frames_dir:  str,
        frame_queue: queue.Queue,
        stop_event:  threading.Event,
        wait_for_robot: bool,
    ):
        super().__init__(daemon=True, name="FrameCaptureThread")
        self.frames_dir     = frames_dir
        self.frame_queue    = frame_queue
        self.stop_event     = stop_event
        self.wait_for_robot = wait_for_robot
        self.frame_count    = 0

    def run(self):
        ids_peak.Library.Initialize()
        dev = ds = nodemap = None

        try:
            dm = ids_peak.DeviceManager.Instance()
            dm.Update()
            if dm.Devices().empty():
                raise RuntimeError("No IDS Peak camera found.")

            dev     = dm.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
            nodemap = dev.RemoteDevice().NodeMaps()[0]

            # ── Line 3: output → trigger pulse to event camera ────────────
            nodemap.FindNode("LineSelector").SetCurrentEntry("Line3")
            nodemap.FindNode("LineMode").SetCurrentEntry("Output")
            nodemap.FindNode("LineSource").SetCurrentEntry("UserOutput0")
            nodemap.FindNode("UserOutputSelector").SetCurrentEntry("UserOutput0")
            nodemap.FindNode("UserOutputValue").SetValue(True)   # idle HIGH

            # ── Optionally wait for robot controller on Line 2 ────────────
            if self.wait_for_robot:
                print("[Frame] Waiting for robot start signal on Line 2 ...")
                nodemap.FindNode("LineSelector").SetCurrentEntry("Line2")
                nodemap.FindNode("LineMode").SetCurrentEntry("Input")
                while not self.stop_event.is_set():
                    if nodemap.FindNode("LineStatus").Value():
                        print("[Frame] Robot start signal received on Line 2.")
                        break
                    time.sleep(0.01)
                if self.stop_event.is_set():
                    return
            else:
                print("[Frame] Standalone mode — starting immediately (no Line 2 wait).")

            # ── Freerun acquisition (49 fps) ──────────────────────────────
            nodemap.FindNode("TriggerSelector").SetCurrentEntry("ExposureStart")
            nodemap.FindNode("TriggerMode").SetCurrentEntry("Off")
            nodemap.FindNode("AcquisitionMode").SetCurrentEntry("Continuous")

            ds = dev.DataStreams()[0].OpenDataStream()
            payload_size = nodemap.FindNode("PayloadSize").Value()
            for _ in range(BUFFER_COUNT):
                ds.QueueBuffer(ds.AllocAndAnnounceBuffer(payload_size))

            nodemap.FindNode("TLParamsLocked").SetValue(1)
            ds.StartAcquisition()
            nodemap.FindNode("AcquisitionStart").Execute()
            nodemap.FindNode("AcquisitionStart").WaitUntilDone()
            print("[Frame] Acquisition started at freerun.")

            while not self.stop_event.is_set():
                try:
                    buf = ds.WaitForFinishedBuffer(FRAME_TIMEOUT_MS)
                except Exception:
                    continue

                self.frame_count += 1
                ids_ts_ns = buf.Timestamp_ns()

                # ── Save frame as JPEG ────────────────────────────────────
                img = ids_ipl.Image.CreateFromSizeAndBuffer(
                    buf.PixelFormat(),
                    buf.BasePtr(),
                    buf.Size(),
                    buf.Width(),
                    buf.Height(),
                )
                # This IDS Peak IPL version uses PixelFormatName_* string constants
                converted = img.ConvertTo(ids_ipl.PixelFormatName_BGRa8)
                raw_arr   = converted.get_numpy_3D()
                bgr = raw_arr[:, :, :3] if raw_arr.shape[2] == 4 else raw_arr
                fname = f"frame_{self.frame_count:06d}.jpg"
                cv2.imwrite(
                    os.path.join(self.frames_dir, fname), bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                )

                # ── Pulse Line 3 LOW → event camera trigger ───────────────
                nodemap.FindNode("UserOutputSelector").SetCurrentEntry("UserOutput0")
                nodemap.FindNode("UserOutputValue").SetValue(False)
                time.sleep(PULSE_DURATION_S)
                nodemap.FindNode("UserOutputValue").SetValue(True)

                # ── Notify event thread ───────────────────────────────────
                self.frame_queue.put((self.frame_count, fname, ids_ts_ns))
                ds.QueueBuffer(buf)

        except Exception as e:
            print(f"[Frame] Error: {e}")
        finally:
            if nodemap:
                try:
                    nodemap.FindNode("AcquisitionStop").Execute()
                    nodemap.FindNode("AcquisitionStop").WaitUntilDone()
                    nodemap.FindNode("TLParamsLocked").SetValue(0)
                except Exception:
                    pass
            if ds:
                try:
                    ds.StopAcquisition()
                    ds.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
                    for b in ds.AnnouncedBuffers():
                        ds.RevokeBuffer(b)
                except Exception:
                    pass
            ids_peak.Library.Close()
            print(f"[Frame] Stopped — {self.frame_count} frames saved.")


def run_event_camera(
    output_dir:  str,
    frames_dir:  str,
    frame_queue: queue.Queue,
    stop_event:  threading.Event,
    duration_s:  float | None,
):
    """
    Opens the event camera, enables trigger input, records to .raw,
    matches trigger timestamps to frame files (FIFO order), writes manifest.
    """
    raw_path      = os.path.join(output_dir, "events.raw")
    manifest_path = os.path.join(output_dir, "manifest.json")

    device = initiate_device("")

    i_trigger_in = device.get_i_trigger_in()
    if i_trigger_in is None:
        raise RuntimeError("Event camera has no trigger input.")
    i_trigger_in.enable(metavision_hal.I_TriggerIn.Channel.MAIN)

    # Record raw events to disk (trigger events are embedded automatically)
    i_events_stream = device.get_i_events_stream()
    i_events_stream.log_raw_data(raw_path)
    print(f"[Event] Recording → {raw_path}")

    mv_it  = EventsIterator.from_device(device=device, delta_t=EVENT_DELTA_T_US)
    reader = mv_it.reader

    # FIFO matching: trigger N ↔ frame N (same order, guaranteed by hardware)
    pending_frames:   list[tuple[int, str, int]] = []
    pending_triggers: list[int] = []
    manifest: dict[str, int]    = {}

    t_start = time.perf_counter()
    print("[Event] Listening for trigger events ...")

    try:
        for evs in mv_it:
            if stop_event.is_set():
                break
            if duration_s and (time.perf_counter() - t_start) >= duration_s:
                stop_event.set()
                break

            # Drain frame queue
            while True:
                try:
                    pending_frames.append(frame_queue.get_nowait())
                except queue.Empty:
                    break

            # Read trigger events — falling edge (polarity=0) = our pulse
            for trg in reader.get_ext_trigger_events():
                if trg["p"] == 0:
                    pending_triggers.append(int(trg["t"]))
                    print(f"[Event] Trigger @ {trg['t']} µs")
            reader.clear_ext_trigger_events()

            # Match FIFO
            while pending_frames and pending_triggers:
                _, fname, _ = pending_frames.pop(0)
                t_us        = pending_triggers.pop(0)
                manifest[fname] = t_us
                print(f"[Event] Matched {fname} ↔ {t_us} µs")

    finally:
        i_events_stream.stop_log_raw_data()

        # Drain any remaining unmatched items
        while True:
            try:
                pending_frames.append(frame_queue.get_nowait())
            except queue.Empty:
                break
        while pending_frames and pending_triggers:
            _, fname, _ = pending_frames.pop(0)
            t_us        = pending_triggers.pop(0)
            manifest[fname] = t_us

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        n = len(manifest)
        print(f"\n[Event] {n} frame↔trigger pairs matched.")
        print(f"[Event] Manifest → {manifest_path}")
        print(f"[Event] Raw file  → {raw_path}")
        if n == 0:
            print("[Event] WARNING — no pairs matched. Check trigger wiring.")


def start_rosbag(cable_color: str, speed_folder: str, run: str) -> subprocess.Popen:
    """Start rosbag record inside the trackdlo Docker container."""
    bag_path = f"{TRACKDLO_DATASET}/{cable_color}/{speed_folder}/{run}/recording.bag"
    cmd = [
        "docker", "exec", DOCKER_CONTAINER,
        "bash", "-c",
        f"source /ros_entrypoint.sh && "
        f"source /root/tracking_ws/devel/setup.bash && "
        f"rosbag record -O {bag_path} " + " ".join(ROSBAG_TOPICS),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(f"[Bag] Recording → {bag_path}")

    def _log():
        for line in proc.stdout:
            print(f"[Bag] {line}", end="")
    threading.Thread(target=_log, daemon=True).start()

    return proc


def stop_rosbag(proc: subprocess.Popen):
    """Stop rosbag record by sending SIGINT inside the container (writes index cleanly)."""
    # Kill rosbag inside the container directly — sending signal to docker exec
    # process from outside does not reliably propagate into the container.
    subprocess.run(
        ["docker", "exec", DOCKER_CONTAINER, "bash", "-c", "pkill -SIGINT -f 'rosbag record'"],
        check=False,
    )
    if proc and proc.poll() is None:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("[Bag] Stopped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration",       type=float, default=None,
                        help="Recording duration in seconds (default: Ctrl+C to stop)")
    parser.add_argument("--output-dir",     type=str,   default=OUTPUT_DIR)
    parser.add_argument("--wait-for-robot", action="store_true",
                        help="Wait for robot controller start signal on Line 2 before capturing")
    args = parser.parse_args()

    
    norm       = os.path.normpath(args.output_dir)
    parts      = norm.replace("\\", "/").split("/")
    run        = parts[-1]
    speed_folder = parts[-2]
    cable_color  = parts[-3]

    frames_dir  = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    frame_queue = queue.Queue()
    stop_event  = threading.Event()

    print(f"[Recorder] Output dir : {args.output_dir}")
    print(f"[Recorder] Robot wait : {args.wait_for_robot}")
    print(f"[Recorder] Duration   : {args.duration or 'until Ctrl+C'}")
    print(f"[Recorder] Cable      : {cable_color}  Speed: {speed_folder}  Run: {run}")
    print()

    frame_thread = FrameCaptureThread(
        frames_dir     = frames_dir,
        frame_queue    = frame_queue,
        stop_event     = stop_event,
        wait_for_robot = args.wait_for_robot,
    )
    frame_thread.start()

    bag_proc = start_rosbag(cable_color, speed_folder, run)

    try:
        run_event_camera(
            output_dir  = args.output_dir,
            frames_dir  = frames_dir,
            frame_queue = frame_queue,
            stop_event  = stop_event,
            duration_s  = args.duration,
        )
    except KeyboardInterrupt:
        print("\n[Recorder] Ctrl+C — stopping ...")
    finally:
        stop_event.set()
        stop_rosbag(bag_proc)
        frame_thread.join(timeout=5.0)
        print("[Recorder] Done.")


if __name__ == "__main__":
    main()
