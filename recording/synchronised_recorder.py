#!/usr/bin/env python3
"""
Synchronized RGB + Event Recorder 
=========================================================
Records IDS Peak (frame) and Prophesee (event) cameras simultaneously.

Shutter / trigger strategy
---------------------------
SensorShutterMode = GlobalReset requires TriggerMode = On on the AR0521.
We therefore run the IDS camera in SOFTWARE-TRIGGERED mode:

  1. Pulse Line 3 LOW  → EVK4 logs a falling-edge timestamp           (~0 µs)
  2. TriggerSoftware.Execute()  → all pixels begin exposure simultaneously
                                   (GlobalReset: same start, rolling end)  (~1 ms)
  3. WaitForFinishedBuffer()    → readout complete
  4. ConvertTo / imwrite        → JPEG saved (latency irrelevant here)

The EVK4 timestamp therefore leads the actual exposure start by the USB
round-trip latency of one SetValue call (~1 ms), which is constant and
small enough for both e2calib and MotionDLO tracking.

Hardware wiring
---------------
  IDS Hirose HR10A-10P-12S          EVK4 HD GPIO header (2.54 mm)
    Line 3 pin — UserOutput0 ──────► Trigger IN (Channel 0 / MAIN)
    GND        ───────────────────► GND
  Both sides 3.3 V LVTTL, direct connection. Optional 100 Ω series resistor.

Output layout
-------------
OUTPUT_DIR/
    events.raw          — full event stream (.raw), trigger timestamps embedded
    frames/
        <t_us>.jpg      — frames renamed to their EVK4 trigger timestamp (µs)
    ids_timestamps.txt  — one timestamp per line (µs), for e2calib --timestamps_file
    manifest.json       — {"<t_us>.jpg": t_us, ...}

Usage
-----
    python synchronized_recorder.py
    python synchronized_recorder.py --wait-for-robot
    python synchronized_recorder.py --duration 30
    python synchronized_recorder.py --fps 10
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time

import cv2
import ids_peak.ids_peak as ids_peak
import ids_peak_ipl.ids_peak_ipl as ids_ipl
import metavision_hal
from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device

# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_DIR       = "/path/to/dataset/Single Cable/Black_cable/Speed_50/1"
JPEG_QUALITY     = 95       # JPEG compression (0–100)
PULSE_DURATION_S = 0.001    # 1 ms LOW pulse on Line 3
FRAME_TIMEOUT_MS = 5000     # IDS Peak buffer wait timeout (ms)
BUFFER_COUNT     = 16       # IDS Peak DataStream buffer pool size
EVENT_DELTA_T_US = 50_000   # event batch window for trigger polling (50 ms)
DEFAULT_FPS      = 20.0     # software trigger rate (frames per second)

# ──────────────────────────────────────────────────────────────────────────────


class FrameCaptureThread(threading.Thread):
    """
    Captures frames from the IDS Peak camera in SOFTWARE-TRIGGERED mode
    with SensorShutterMode = GlobalReset.

    Acquisition loop per frame:
      1. Pulse Line 3 LOW  → EVK4 falling-edge trigger timestamp
      2. TriggerSoftware.Execute()  → all pixels begin exposure simultaneously
      3. WaitForFinishedBuffer()    → wait for readout
      4. Save JPEG
    """

    def __init__(
        self,
        frames_dir:     str,
        frame_queue:    queue.Queue,
        stop_event:     threading.Event,
        wait_for_robot: bool,
        fps:            float,
    ):
        super().__init__(daemon=True, name="FrameCaptureThread")
        self.frames_dir     = frames_dir
        self.frame_queue    = frame_queue
        self.stop_event     = stop_event
        self.wait_for_robot = wait_for_robot
        self.frame_period   = 1.0 / fps
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

            # ── Line 3: UserOutput0, idle HIGH ────────────────────────────
            nodemap.FindNode("LineSelector").SetCurrentEntry("Line3")
            nodemap.FindNode("LineMode").SetCurrentEntry("Output")
            nodemap.FindNode("LineSource").SetCurrentEntry("UserOutput0")
            nodemap.FindNode("UserOutputSelector").SetCurrentEntry("UserOutput0")
            nodemap.FindNode("UserOutputValue").SetValue(True)   # idle HIGH
            print("[Frame] Line 3 → UserOutput0 idle HIGH.")

            # ── Software trigger mode — required for GlobalReset ──────────
            nodemap.FindNode("TriggerSelector").SetCurrentEntry("ExposureStart")
            nodemap.FindNode("TriggerMode").SetCurrentEntry("On")
            nodemap.FindNode("TriggerSource").SetCurrentEntry("Software")
            print("[Frame] TriggerMode = On (Software), required for GlobalReset.")

            # ── GlobalReset shutter: all pixels start exposure together ───
            nodemap.FindNode("SensorShutterMode").SetCurrentEntry("GlobalReset")
            print("[Frame] SensorShutterMode = GlobalReset ✓")

            # ── Optionally wait for robot controller on Line 2 ────────────
            if self.wait_for_robot:
                print("[Frame] Waiting for robot start signal on Line 2 ...")
                nodemap.FindNode("LineSelector").SetCurrentEntry("Line2")
                nodemap.FindNode("LineMode").SetCurrentEntry("Input")
                while not self.stop_event.is_set():
                    if nodemap.FindNode("LineStatus").Value():
                        print("[Frame] Robot start signal received.")
                        break
                    time.sleep(0.01)
                if self.stop_event.is_set():
                    return
            else:
                print("[Frame] Standalone mode — starting immediately.")

            # ── Open data stream ──────────────────────────────────────────
            nodemap.FindNode("AcquisitionMode").SetCurrentEntry("Continuous")
            ds = dev.DataStreams()[0].OpenDataStream()
            payload_size = nodemap.FindNode("PayloadSize").Value()
            for _ in range(BUFFER_COUNT):
                ds.QueueBuffer(ds.AllocAndAnnounceBuffer(payload_size))

            nodemap.FindNode("TLParamsLocked").SetValue(1)
            ds.StartAcquisition()
            nodemap.FindNode("AcquisitionStart").Execute()
            nodemap.FindNode("AcquisitionStart").WaitUntilDone()
            print(f"[Frame] Acquisition started (software trigger, {1/self.frame_period:.1f} fps).")

            next_trigger = time.perf_counter()

            while not self.stop_event.is_set():

                # Rate-limit to target fps
                now = time.perf_counter()
                if now < next_trigger:
                    time.sleep(next_trigger - now)
                next_trigger += self.frame_period
                
                
                # ── 1. Pulse Line 3 LOW → EVK4 records falling-edge timestamp ──
                nodemap.FindNode("UserOutputSelector").SetCurrentEntry("UserOutput0")
                nodemap.FindNode("UserOutputValue").SetValue(False)
                time.sleep(PULSE_DURATION_S)
                nodemap.FindNode("UserOutputValue").SetValue(True)
                
                # ── 2. Fire software trigger → GlobalReset exposure starts ──
                # All pixels begin collecting photons simultaneously here.
                # Latency from pulse to this call is ~1 ms (one USB round-trip).
                nodemap.FindNode("TriggerSoftware").Execute()



                # ── 3. Wait for readout ────────────────────────────────────
                try:
                    buf = ds.WaitForFinishedBuffer(FRAME_TIMEOUT_MS)
                except Exception:
                    continue

                self.frame_count += 1
                ids_ts_ns = buf.Timestamp_ns()

                # ── 4. Save JPEG ───────────────────────────────────────────
                img = ids_ipl.Image.CreateFromSizeAndBuffer(
                    buf.PixelFormat(),
                    buf.BasePtr(),
                    buf.Size(),
                    buf.Width(),
                    buf.Height(),
                )
                converted = img.ConvertTo(ids_ipl.PixelFormatName_BGRa8)
                raw_arr   = converted.get_numpy_3D()
                bgr       = raw_arr[:, :, :3] if raw_arr.shape[2] == 4 else raw_arr
                fname     = f"frame_{self.frame_count:06d}.jpg"
                cv2.imwrite(
                    os.path.join(self.frames_dir, fname), bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                )

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
    Opens the EVK4, enables trigger input on Channel 0 (MAIN), records to
    .raw, matches trigger timestamps to frame files (FIFO), renames frames
    to their trigger timestamp, writes manifest + timestamps file.

    Trigger polarity: FALLING EDGE (p == 0) = Line 3 LOW pulse.
    The pulse fires ~1 ms before TriggerSoftware.Execute(), so the EVK4
    timestamp leads the actual GlobalReset exposure start by ~1 ms.
    """
    raw_path      = os.path.join(output_dir, "events.raw")
    manifest_path = os.path.join(output_dir, "manifest.json")
    ts_path       = os.path.join(output_dir, "ids_timestamps.txt")

    device = initiate_device("")

    i_trigger_in = device.get_i_trigger_in()
    if i_trigger_in is None:
        raise RuntimeError("EVK4 has no trigger input interface.")
    i_trigger_in.enable(metavision_hal.I_TriggerIn.Channel.MAIN)
    print("[Event] Trigger input enabled on EVK4 Channel 0 (MAIN).")

    i_events_stream = device.get_i_events_stream()
    i_events_stream.log_raw_data(raw_path)
    print(f"[Event] Recording → {raw_path}")

    mv_it  = EventsIterator.from_device(device=device, delta_t=EVENT_DELTA_T_US)
    reader = mv_it.reader

    pending_frames:   list[tuple[int, str, int]] = []
    pending_triggers: list[int] = []
    manifest: dict[str, int]    = {}
    ts_list:  list[int]         = []

    t_start = time.perf_counter()
    print("[Event] Listening for trigger events (falling edge = pre-exposure pulse) ...")

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

            # Falling edge (p == 0) = Line 3 LOW pulse, fired just before exposure
            for trg in reader.get_ext_trigger_events():
                if trg["p"] == 0:
                    pending_triggers.append(int(trg["t"]))
                    print(f"[Event] Trigger @ {trg['t']} µs")
            reader.clear_ext_trigger_events()

            # FIFO match and rename
            while pending_frames and pending_triggers:
                _, old_fname, _ = pending_frames.pop(0)
                t_us            = pending_triggers.pop(0)

                new_fname = f"{t_us}.jpg"
                old_path  = os.path.join(frames_dir, old_fname)
                new_path  = os.path.join(frames_dir, new_fname)
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)

                manifest[new_fname] = t_us
                ts_list.append(t_us)
                print(f"[Event] Matched {old_fname} → {new_fname} ({t_us} µs)")

            # Write timestamps incrementally
            with open(ts_path, "w") as f:
                for ts in sorted(ts_list):
                    f.write(f"{ts}\n")

    finally:
        i_events_stream.stop_log_raw_data()

        # Drain remaining unmatched items
        while True:
            try:
                pending_frames.append(frame_queue.get_nowait())
            except queue.Empty:
                break
        while pending_frames and pending_triggers:
            _, old_fname, _ = pending_frames.pop(0)
            t_us            = pending_triggers.pop(0)
            new_fname       = f"{t_us}.jpg"
            old_path        = os.path.join(frames_dir, old_fname)
            new_path        = os.path.join(frames_dir, new_fname)
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
            manifest[new_fname] = t_us
            ts_list.append(t_us)

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        with open(ts_path, "w") as f:
            for ts in sorted(ts_list):
                f.write(f"{ts}\n")

        n = len(manifest)
        print(f"\n[Event] {n} frame↔trigger pairs matched.")
        print(f"[Event] Manifest    → {manifest_path}")
        print(f"[Event] Timestamps  → {ts_path}")
        print(f"[Event] Raw file    → {raw_path}")
        if n == 0:
            print("[Event] WARNING — no pairs matched. Check:")
            print("        • Line 3 pin wired to EVK4 trigger IN")
            print("        • GND connected between both cameras")
            print("        • Trigger polarity: falling edge (p == 0)")


def main():
    parser = argparse.ArgumentParser(
        description="Synchronized IDS Peak + EVK4 recorder (GlobalReset, software trigger)"
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Recording duration in seconds (default: Ctrl+C to stop)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--wait-for-robot", action="store_true",
        help="Wait for robot controller start signal on Line 2 before capturing",
    )
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS,
        help=f"Software trigger rate in frames/sec (default: {DEFAULT_FPS})",
    )
    args = parser.parse_args()

    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    frame_queue = queue.Queue()
    stop_event  = threading.Event()

    print(f"[Recorder] Output dir  : {args.output_dir}")
    print(f"[Recorder] Robot wait  : {args.wait_for_robot}")
    print(f"[Recorder] Duration    : {args.duration or 'until Ctrl+C'}")
    print(f"[Recorder] FPS         : {args.fps}")
    print(f"[Recorder] Shutter     : GlobalReset (software trigger)")
    print(f"[Recorder] Sync        : Line 3 pulse → EVK4 falling edge, then TriggerSoftware")
    print()

    frame_thread = FrameCaptureThread(
        frames_dir     = frames_dir,
        frame_queue    = frame_queue,
        stop_event     = stop_event,
        wait_for_robot = args.wait_for_robot,
        fps            = args.fps,
    )
    frame_thread.start()

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
        frame_thread.join(timeout=5.0)
        print("[Recorder] Done.")


if __name__ == "__main__":
    main()