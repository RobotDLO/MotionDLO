
import cv2
import numpy as np

class Visualization: 

    def visualize_wire(R, mask=None, skel=None, win="debug", scale=1.0):

        # R in [0,1] -> 8-bit
        R_vis = np.clip(R * 255.0, 0, 255).astype(np.uint8)

        # Nice colormap for contrast
        R_color = cv2.applyColorMap(R_vis, cv2.COLORMAP_TURBO)

        # Overlay mask (green)
        if mask is not None:
            R_color[mask] = (0, 255, 0)

        # Overlay skeleton (red)
        if skel is not None:
            R_color[skel] = (0, 0, 255/2)

        if scale != 1.0:
            h, w = R_color.shape[:2]
            R_color = cv2.resize(R_color, (int(w * scale), int(h * scale)))

        cv2.imshow(win, R_color)

def visualize_wire_with_polylines(R, mask=None, polylines=None, win="debug", scale=1.0):

    R_vis = np.clip(R * 255.0, 0, 255).astype(np.uint8)
    R_color = cv2.applyColorMap(R_vis, cv2.COLORMAP_TURBO)

    if mask is not None:
        R_color[mask] = (0, 255, 0)

    if polylines is not None:
        for pl in polylines:
            if pl is None or len(pl) < 1:
                continue
            # default OpenCV color ordering is BGR
            cv2.polylines(R_color, [pl], isClosed=False, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    if scale != 1.0:
        h, w = R_color.shape[:2]
        R_color = cv2.resize(R_color, (int(w * scale), int(h * scale)))

    cv2.imshow(win, R_color)


def visualize_hybrid(R, mask=None, event_polyline=None, correction_polyline=None,
                     win="hybrid", scale=1.0):
    """
    Hybrid visualisation showing both event-tracked and frame-corrected polylines.

    """
    R_vis = np.clip(R * 255.0, 0, 255).astype(np.uint8)
    R_color = cv2.applyColorMap(R_vis, cv2.COLORMAP_TURBO)

    if mask is not None:
        R_color[mask] = (0, 255, 0)

    if correction_polyline is not None and len(correction_polyline) > 1:
        cv2.polylines(R_color, [correction_polyline], isClosed=False,
                      color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    if event_polyline is not None and len(event_polyline) > 1:
        cv2.polylines(R_color, [event_polyline], isClosed=False,
                      color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    if scale != 1.0:
        h, w = R_color.shape[:2]
        R_color = cv2.resize(R_color, (int(w * scale), int(h * scale)))

    cv2.imshow(win, R_color)
