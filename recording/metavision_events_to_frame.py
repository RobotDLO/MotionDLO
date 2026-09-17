# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Sample code that demonstrates how to use Metavision SDK to visualize events from a live camera or an event file
"""

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
import argparse
import json
import os
import sys
import cv2

# Make EventBased.Filter importable (lives in ../src/)
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import EventBased.Filter as ev_filter  # noqa: E402


# Spatio-temporal contrast filter settings (match hybrid_pipeline.DEFAULTS).
# Only STC is used here; trail and AFK are disabled.
class FilterArgs:
    disable_trail      = True
    disable_afk        = True
    disable_stc        = False
    stc_filter_thr     = 16000
    stc_cut_trail      = False
    # Unused (trail/afk disabled) but required by filter_events signature:
    trail_us           = 0
    afk_min_freq       = 0.0
    afk_max_freq       = 0.0
    afk_filter_length  = 0
    afk_diff_thresh_us = 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision Simple Viewer sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-event-file', dest='event_file_path', default="",
        help="Path to input event file (RAW, DAT or HDF5). If not specified, the camera live stream is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    parser.add_argument(
        '-o', '--output-dir', dest='output_dir', default="",
        help="If set, generated frames are written as PNG files into this directory.")
    parser.add_argument(
        '--fps', dest='fps', type=float, default=25.0,
        help="Output frame rate for the periodic frame generator.")
    parser.add_argument(
        '--stc-filter-thr', dest='stc_filter_thr', type=int, default=16000,
        help="Spatio-temporal contrast filter threshold (µs). Larger = more permissive.")
    parser.add_argument(
        '--delta-t', dest='delta_t', type=int, default=7000,
        help="EventsIterator delta_t in µs — events are read in batches of this duration.")
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    # Events iterator on Camera or event file
    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=args.delta_t)
    height, width = mv_iterator.get_size()  # Camera Geometry

    # Helper iterator to emulate realtime
    if not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Event filter (spatio-temporal contrast only)
    filter_args = FilterArgs()
    filter_args.stc_filter_thr = args.stc_filter_thr
    _, _, _, _, stc_filter, stc_out = ev_filter.filter_events(
        mv_iterator, width, height, filter_args
    )

    # Window - Graphical User Interface
    with MTWindow(title="Metavision Events Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:
        def keyboard_cb(key, scancode, action, mods):
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        # Event Frame Generator
        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height, fps=args.fps,
                                                           palette=ColorPalette.Dark)

        frame_idx = [0]
        manifest = {}

        def on_cd_frame_cb(ts, cd_frame):
            window.show_async(cd_frame)
            if args.output_dir:
                basename = f"frame_{frame_idx[0]:06d}_{ts}.png"
                cv2.imwrite(os.path.join(args.output_dir, basename), cd_frame)
                manifest[basename] = ts
                frame_idx[0] += 1

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        # Process events
        for evs in mv_iterator:
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()

            stc_filter.process_events(evs, stc_out)
            event_frame_gen.process_events(stc_out.numpy())

            if window.should_close():
                break

        if args.output_dir:
            with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
