"""Motion overlay monitor for WoW — bboxes + Frangi mask visualization.

Captures the WoW window, tracks motion boxes with XOR-thresholding,
accumulates a grow-only char bbox region, and runs the Frangi filter
when 'x' is pressed, displaying the binary mask on the overlay.

Hotkeys:
    Ctrl+Q — exit
    X      — run Frangi on char region, show mask
    C      — reset accumulated char bbox
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from typing import Optional

import cv2 as cv
import numpy as np
from dotenv import load_dotenv
from graphics import Rect, crop_image

import ahk
from ahk.directives import NoTrayIcon
import dxcam
import win32gui  # type: ignore
from overlay import overlay, Scene
from frangi_test import apply_frangi_filter
from macro_graph_analysis import macro_graph_analysis
from skimage.morphology import thin
from skimage.filters import frangi

WOW_WINDOW_NAME = "World of Warcraft"

# ── tunables ────────────────────────────────────────────────────────────────

XOR_THRESHOLD = 18
DILATE_PIXELS = 9
MIN_AREA = 80
MAX_BOXES = 64
FRAME_INTERVAL = 0.05
LOG_BOX_EVERY = 20

AGE_SHORT_SEC = 0.5
AGE_LONG_SEC = 2.0
IOU_MATCH_THRESHOLD = 0.25
MAX_TRACK_AGE_SEC = 3.0
CHAR_MIN_AREA_FRAC = 0.04
CHAR_CENTER_FRAC = 0.30

COLOR_SHORT = (80, 120, 255, 220)
COLOR_MEDIUM = (0, 255, 80, 220)
COLOR_LONG = (255, 60, 60, 220)
COLOR_CHAR = (255, 220, 0, 235)


# ── IoU helper ──────────────────────────────────────────────────────────────

def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


# ── box classification ──────────────────────────────────────────────────────

def classify_box(box, img_w, img_h, age_sec):
    x, y, w, h = box
    area = w * h
    is_large = area >= CHAR_MIN_AREA_FRAC * img_w * img_h
    cx, cy = x + w / 2, y + h / 2
    in_center = (
        CHAR_CENTER_FRAC * img_w <= cx <= (1 - CHAR_CENTER_FRAC) * img_w
        and CHAR_CENTER_FRAC * img_h <= cy <= (1 - CHAR_CENTER_FRAC) * img_h
    )
    if is_large and in_center:
        return "char", COLOR_CHAR
    if age_sec >= AGE_LONG_SEC:
        return "long", COLOR_LONG
    if age_sec < AGE_SHORT_SEC:
        return "short", COLOR_SHORT
    return "medium", COLOR_MEDIUM


# ── motion tracker ──────────────────────────────────────────────────────────

class MotionTracker:
    def __init__(self, iou_threshold=IOU_MATCH_THRESHOLD,
                 max_unseen_sec=MAX_TRACK_AGE_SEC):
        self._next_id = 0
        self._tracks: dict[int, dict] = {}
        self._iou_threshold = iou_threshold
        self._max_unseen_sec = max_unseen_sec

    def update(self, boxes, dt_sec, img_w, img_h):
        matched: set[int] = set()
        for box in boxes:
            best_id, best_iou = -1, 0.0
            for tid, tr in self._tracks.items():
                if tid in matched:
                    continue
                iou = _iou(box, tr["box"])
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_iou >= self._iou_threshold and best_id >= 0:
                tr = self._tracks[best_id]
                tr["box"] = box
                tr["age"] += dt_sec
                tr["unseen"] = 0.0
                matched.add(best_id)
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "box": box, "age": dt_sec, "unseen": 0.0,
                    "born": time.monotonic(),
                }
                matched.add(tid)
        for tid in list(self._tracks.keys()):
            if tid not in matched:
                self._tracks[tid]["unseen"] += dt_sec
                if self._tracks[tid]["unseen"] >= self._max_unseen_sec:
                    del self._tracks[tid]
        out = []
        for tid in matched:
            tr = self._tracks.get(tid)
            if tr is None:
                continue
            label, color = classify_box(tr["box"], img_w, img_h, tr["age"])
            out.append((tr["box"], label, color, tr["age"], tid))
        out.sort(key=lambda r: (r[1] != "char", -r[3]))
        return out


# ── overlay drawing ─────────────────────────────────────────────────────────

DrawRecord = tuple[tuple[int, int, int, int], str, tuple[int, int, int, int],
                   float, int]


def draw_boxes(scene: Scene, records: list[DrawRecord], ox: int, oy: int):
    with scene.batch() as s:
        for (x, y, w, h), label, color, age, _tid in records:
            sx, sy = ox + x, oy + y
            pw = 3 if label == "char" else 2
            s.rect(sx, sy, w, h, pen_color=color, pen_width=pw,
                   brush_color=None)
            if label in ("long", "char"):
                tag = "char" if label == "char" else f"{age:.1f}s"
                s.text(sx + 2, max(sy, sy - 14), tag, color=color,
                       font="Consolas", size=11)


def draw_char_region(scene: Scene, region: Rect, ox: int, oy: int):
    with scene.batch() as s:
        rx, ry, rw, rh = region.xywh()
        s.rect(ox + rx, oy + ry, rw, rh, pen_color=COLOR_CHAR, pen_width=2,
               brush_color=None)
    scene.show()


def draw_frangi_mask(scene: Scene, png_bytes: Optional[bytes], edges, centroid, bb,
                     char_region: Optional[Rect], ox: int, oy: int):
    """Draw Frangi binary mask as an image scoped to the char region."""
    if png_bytes is None or char_region is None:
        scene.hide()
        return
    rx, ry, rw, rh = char_region.xywh()
    with scene.batch() as s:
        # s.image(ox + rx, oy + ry, rw, rh, png_bytes)
        # print(edges, bb)
        if len(edges) > 0:
            # print('ok')
            edges = np.array(edges) + np.array([ox+rx,oy+ry])
            edges = edges.tolist()
            if centroid is not None:
                centroid = (np.array(centroid) + np.array([ox+rx,oy+ry])).tolist()
                s.ellipse(*centroid, 21, 21,
                            pen_color=(255, 255, 255, 255), pen_width=3)
            if bb is not None:
                bb = (np.array(bb) + np.array([ox+rx,oy+ry]) - [10, 10]).tolist()
                s.ellipse(*bb, 21, 21,
                            pen_color=(0, 255, 255, 255), pen_width=3)
            for edge in edges:
                s.line(*edge[0], *edge[1], color=(255, 255, 0, 255), width=3)
            
    scene.show()


def draw_bobber_mask(scene: Scene, png: Optional[bytes],
                     cx: int, cy: int, ox: int, oy: int):
    """Draw bobber crop Frangi mask centered at (cx, cy) in frame coords."""
    if png is None:
        scene.hide()
        return
    with scene.batch() as s:
        s.image(ox + cx - 32, oy + cy - 32, 64, 96, png)
    scene.show()


def draw_status(scene: Scene, records: list[DrawRecord], frame_no: int,
                has_frangi: bool, ox: int, oy: int, fps: float, looping: bool):
    if not records and not has_frangi and not looping:
        scene.hide()
        return
    with scene.batch() as s:
        lines = [f"f{frame_no}  fps {fps:.0f}  boxes {len(records)}"]
        loop_str = "AUTO-LOOP ON" if looping else "AUTO-LOOP OFF"
        lines.append(f"Ctrl+X = {loop_str}  X = cast  C = reset")
        for i, line in enumerate(lines):
            s.text(ox + 8, oy + 8 + i * 16, line,
                   color=(0, 255, 0, 220), font="Consolas", size=11)
    scene.show()


# ── motion detection ────────────────────────────────────────────────────────

def detect_motion_boxes(prev, cur, threshold=XOR_THRESHOLD,
                        dilate=DILATE_PIXELS, min_area=MIN_AREA,
                        max_boxes=MAX_BOXES):
    if prev is None or cur is None or prev.shape != cur.shape:
        return []
    diff = cv.absdiff(prev, cur)
    changed = (diff.max(axis=2) > threshold).astype(np.uint8) * 255
    if dilate > 0:
        k = cv.getStructuringElement(cv.MORPH_RECT,
                                     (dilate * 2 + 1, dilate * 2 + 1))
        changed = cv.dilate(changed, k)
    contours, _ = cv.findContours(changed, cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_SIMPLE)
    boxes = [(x, y, w, h) for c in contours
             for x, y, w, h in [cv.boundingRect(c)] if w * h >= min_area]
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:max_boxes]


def get_client_rect(ahk_inst, window_name):
    w = ahk_inst.find_window(title=window_name)
    if w is None:
        return None
    wid = int(w.id)
    p = win32gui.ClientToScreen(wid, (0, 0))
    cr = win32gui.GetClientRect(wid)
    return Rect(p[0], p[1], cr[2], cr[3])


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── main ────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    setup_logging()
    edges = []
    centroid = [0,0]
    bobber_estimate = [0,0]

    with overlay(server="rust", dirty_tracking=False) as ov:
        ov.set_capture_exclusion(True)
        ahk_inst = ahk.AHK(
            version="v2",
            directives=[NoTrayIcon(apply_to_hotkeys_process=True)],
        )
        ahk_inst.set_coord_mode("Mouse", "Screen")

        client_rect = get_client_rect(ahk_inst, WOW_WINDOW_NAME)
        if client_rect is None:
            logging.error("WoW window not found")
            return
        zero_base = Rect(0, 0, client_rect.w, client_rect.h)
        ox, oy, cw, ch = client_rect.xywh()
        logging.info("WoW: (%d,%d) %dx%d", ox, oy, cw, ch)

        cam = dxcam.create(backend="dxgi", output_color="BGR",
                           region=(ox, oy, ox + cw, oy + ch))
        for _ in range(10):
            cam.grab()

        # overlay scenes
        boxes_scene = ov.scene("motion_boxes")
        region_scene = ov.scene("char_region")
        frangi_mask_scene = ov.scene("frangi_mask")
        bobber_scene = ov.scene("bobber")
        indicator_scene = ov.scene("indicator")
        status_scene = ov.scene("status")

        # hotkey queues
        exit_q: list[bool] = []
        frangi_q: list[bool] = []
        reset_q: list[bool] = []
        loop_q: list[bool] = []

        def _exit():
            exit_q.append(True)

        def _frangi():
            frangi_q.append(True)

        def _reset():
            reset_q.append(True)

        def _loop():
            loop_q.append(True)

        ahk_inst.add_hotkey("^q", _exit)
        ahk_inst.add_hotkey("x", _frangi)
        ahk_inst.add_hotkey("c", _reset)
        ahk_inst.add_hotkey("^x", _loop)
        ahk_inst.start_hotkeys()

        # state
        tracker = MotionTracker()
        char_region: Optional[Rect] = None
        frangi_mask_png: Optional[bytes] = None
        bobber_mask_png: Optional[bytes] = None
        bobber_estimate: Optional[tuple[float, float]] = None
        bobber_rect: Optional[Rect] = None
        from bobber_tracker import BobberTracker
        bobber_tracker: Optional[BobberTracker] = None
        cnt: Optional[int] = None

        cast_start_t: Optional[float] = None
        pending_frangi_t: Optional[float] = None
        pending_catch_t: Optional[float] = None
        pending_click_pos: Optional[tuple[int, int]] = None
        looping_fishing: bool = False
        flash_event: Optional[dict] = None

        prev_frame = cam.grab()
        t0 = time.perf_counter()
        prev_t = t0
        frames = 0

        cam.start(region=client_rect.xyxy(), target_fps=60)
        prev_frame = cam.get_latest_frame()

        while not exit_q:
            frame = cam.get_latest_frame()
            if frame is None:
                time.sleep(FRAME_INTERVAL)
                continue

            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now

            boxes = detect_motion_boxes(prev_frame, frame)
            records = tracker.update(boxes, dt, cw, ch)

            # accumulate char region
            char_rec = next((r for r in records if r[1] == "char"), None)
            char_box = char_rec[0] if char_rec else None
            if char_box is not None:
                if char_region is None:
                    char_region = Rect(*char_box)
                else:
                    char_region = char_region.union(Rect(*char_box))

            # Ctrl+X — toggle auto-loop fishing
            if loop_q:
                loop_q.clear()
                looping_fishing = not looping_fishing
                logging.info("Auto-loop fishing %s", "ENABLED" if looping_fishing else "DISABLED")
                if looping_fishing and cast_start_t is None:
                    ahk_inst.send("z")
                    logging.info("Sent Z (fishing key)")
                    cast_start_t = now
                    pending_frangi_t = now + 2.0
                    pending_catch_t = None
                    pending_click_pos = None
                    bobber_tracker = None
                    bobber_rect = None
                    bobber_mask_png = None
                    bobber_estimate = None

            # C — reset
            if reset_q:
                reset_q.clear()
                char_region = None
                frangi_mask_png = None
                bobber_mask_png = None
                bobber_estimate = None
                bobber_rect = None
                bobber_tracker = None
                cast_start_t = None
                pending_frangi_t = None
                pending_catch_t = None
                pending_click_pos = None
                looping_fishing = False
                flash_event = None
                logging.info("char region reset")

            # X — start fishing cycle (simulate Z, wait 2s, run Frangi & track)
            run_frangi = False
            if frangi_q:
                frangi_q.clear()
                ahk_inst.send("z")
                logging.info("Sent Z (fishing key)")
                cast_start_t = now
                pending_frangi_t = now + 2.0
                pending_catch_t = None
                pending_click_pos = None
                bobber_tracker = None
                bobber_rect = None
                bobber_mask_png = None
                bobber_estimate = None

            # 20s hard reset timer: timeout -> fail flash (red), re-cast if looping
            if cast_start_t is not None and (now - cast_start_t >= 20.0):
                logging.info("20s fishing timeout reached (no catch)")
                if char_region is not None:
                    rx, ry, rw, rh = char_region.xywh()
                    fail_x = ox + (int(rx + bobber_estimate[0]) if bobber_estimate is not None else rx + rw // 2)
                    fail_y = oy + (int(ry + bobber_estimate[1]) if bobber_estimate is not None else ry + rh // 2)
                    flash_event = {"color": (255, 0, 0), "pos": (fail_x, fail_y), "start_t": now, "duration": 1.0}

                if looping_fishing:
                    ahk_inst.send("z")
                    logging.info("Auto re-casting (Z)")
                    cast_start_t = now
                    pending_frangi_t = now + 2.0
                    pending_catch_t = None
                    pending_click_pos = None
                    bobber_tracker = None
                    bobber_rect = None
                    bobber_mask_png = None
                    bobber_estimate = None
                else:
                    cast_start_t = None
                    pending_frangi_t = None
                    bobber_tracker = None
                    bobber_rect = None
                    bobber_mask_png = None
                    bobber_estimate = None

            # 2s post-cast timer: trigger Frangi detection
            if pending_frangi_t is not None and now >= pending_frangi_t:
                pending_frangi_t = None
                run_frangi = True

            if run_frangi:
                edges = []
                bobber_estimate = None
                bobber_mask_png = None
                if char_region is not None:
                    cr = char_region.clip(zero_base)
                    if cr.w > 0 and cr.h > 0:
                        crop = crop_image(frame, cr)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        cv.imwrite(f"tmp/frangi_{ts}_crop.png", crop)
                        mask, response, centroid = apply_frangi_filter(crop)
                        cv.imwrite(f"tmp/frangi_{ts}_mask.png", mask)
                        cv.imwrite(f"tmp/frangi_{ts}_response.png", response)

                        edges, centroid = macro_graph_analysis(mask)
                        # convert mask to RGBA: black → transparent,
                        # white → magenta semi-opaque
                        rgba = np.zeros((mask.shape[0], mask.shape[1], 4),
                                        dtype=np.uint8)
                        rx, ry, _, _ = char_region.xywh()

                        if len(edges) > 0 and centroid is not None:
                            cent = np.array(centroid)
                            pts = np.array([pt for edge in edges for pt in edge])
                            a = pts - cent
                            bobber_estimate = cent + sorted(zip(a, map(np.linalg.norm, a.tolist())), key=lambda x: x[1])[-1][0]
                            # logging.info(f'updated bobber estimate: {bobber_estimate}')

                            bobber_rect1 = Rect.from_centdims(ox + rx + bobber_estimate[0], oy + ry + bobber_estimate[1], 96, 96)
                            bobber_rect = Rect(int(rx + bobber_estimate[0]-32), int(ry+ bobber_estimate[1]-32), 64, 96)
                            logging.info(bobber_rect)
                            bobber_tracker = BobberTracker(movement_threshold=8, smoothing_factor=0.05)
                            cnt = 0


                        

                        white = mask > 0
                        rgba[white, 0] = 255   # R
                        rgba[white, 1] = 0     # G
                        rgba[white, 2] = 255   # B
                        rgba[white, 3] = 255   # A (semi-transparent)
                        _, png = cv.imencode(".png", rgba)
                        frangi_mask_png = png.tobytes()
                        cv.imwrite(f"tmp/frangi_{ts}_mask_semi.png", mask)
                        logging.info("Frangi: %d mask px  saved tmp/frangi_%s_*",
                                     np.count_nonzero(mask), ts)

            if bobber_rect is not None:
                if cnt is None:
                    cnt = 0
                bobber_crop = crop_image(frame, bobber_rect)
                # with ov.scene("bob") as bobscn:
                #     bobscn.rect(*bobber_rect1.xywh(), pen_color=(255,255,255,255), pen_width=1)
                bgray = cv.cvtColor(bobber_crop, cv.COLOR_BGR2GRAY)
                bfr = frangi(bgray, sigmas=range(1, 4, 1), black_ridges=False)
                bfn = cv.normalize(bfr, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)  # type: ignore
                # bfn = dilate(bfn, 2, cv.MORPH_ELLIPSE)
                # bfn = erode(bfn, 2, cv.MORPH_ELLIPSE)
                _, bmask = cv.threshold(bfn, 20, 255, cv.THRESH_BINARY)
                # bmask = leave_large_ccs(bmask, cc_num=2)
                bmask = thin(bmask > 0)
                bmask = (bmask * 255).astype(np.uint8)

                ba = np.zeros((bmask.shape[0], bmask.shape[1], 4),
                                dtype=np.uint8)
                bw = bmask > 0
                # ba[bw, 0] = 255 
                ba[bw, 1] = 255 
                # ba[bw, 2] = 255
                ba[bw, 3] = 255

                if bobber_tracker is not None and bobber_tracker.update(bmask):
                    logging.info('!!!!!!!!!!!!!!!!! Splash detected !!!!!!!!!!!!!!!!!')
                    bobber_tracker = None
                    bobber_rect = None
                    bobber_mask_png = None
                    if char_region is not None and bobber_estimate is not None:
                        rx, ry, _, _ = char_region.xywh()
                        click_x = ox + int(rx + bobber_estimate[0])
                        click_y = oy + int(ry + bobber_estimate[1])
                        pending_click_pos = (click_x, click_y)
                        pending_catch_t = now + 0.5
                        flash_event = {"color": (0, 255, 0), "pos": (click_x, click_y), "start_t": now, "duration": 1.0}
                # cv.circle(ba, (30, 30), 10, (255,0,0,255), -1)
                _, bpng = cv.imencode(".png", ba)
                cnt += 1
                bobber_mask_png = bpng.tobytes()

            # 0.5s post-catch RMB click execution
            if pending_catch_t is not None and now >= pending_catch_t:
                pending_catch_t = None
                if pending_click_pos is not None:
                    cx, cy = pending_click_pos
                    logging.info("Catch! RMB click at bobber (%d, %d)", cx, cy)
                    ahk_inst.mouse_move(x=cx, y=cy)
                    ahk_inst.right_click()
                    pending_click_pos = None
                    if looping_fishing:
                        logging.info("Auto-loop: casting next line (Z)")
                        ahk_inst.send("z")
                        cast_start_t = now
                        pending_frangi_t = now + 2.0
                    else:
                        cast_start_t = None

            elapsed = now - t0
            fps = frames / max(elapsed, 1e-3)

            draw_boxes(boxes_scene, records, ox, oy)
            if char_region is not None:
                draw_char_region(region_scene, char_region, ox, oy)

            # bobber mask: positioned at frame coords
            if bobber_estimate is not None and bobber_mask_png is not None and char_region is not None:
                rx, ry, _, _ = char_region.xywh()
                bfx = int(bobber_estimate[0] + rx)
                bfy = int(bobber_estimate[1] + ry)
                draw_bobber_mask(bobber_scene, bobber_mask_png, bfx, bfy, ox, oy)

            # status graphics: yellow circle around bobber & fade out circles (green/red)
            with indicator_scene.batch() as s:
                if bobber_estimate is not None and char_region is not None and bobber_rect is not None:
                    rx, ry, _, _ = char_region.xywh()
                    bx = ox + int(rx + bobber_estimate[0])
                    by = oy + int(ry + bobber_estimate[1])
                    s.ellipse(bx, by, 20, 20, pen_color=(255, 255, 0, 230), pen_width=3, brush_color=None)

                if flash_event is not None:
                    fe_elapsed = now - flash_event["start_t"]
                    dur = flash_event["duration"]
                    if fe_elapsed < dur:
                        alpha = int(255 * (1.0 - fe_elapsed / dur))
                        r, g, b = flash_event["color"]
                        fx, fy = flash_event["pos"]
                        s.ellipse(fx, fy, 25, 25, pen_color=(r, g, b, alpha), pen_width=4, brush_color=None)
                    else:
                        flash_event = None
            indicator_scene.show()

            draw_status(status_scene, records, frames,
                        frangi_mask_png is not None, ox, oy, fps, looping_fishing)

            prev_frame = frame
            frames += 1
            if frames % LOG_BOX_EVERY == 0:
                classes = {r[1] for r in records}
                logging.info("f%5d fps %4.1f boxes %3d tracks %2d classes=%s",
                             frames, fps, len(boxes), len(records),
                             {c: sum(1 for r in records if r[1] == c)
                              for c in classes})
            time.sleep(FRAME_INTERVAL)

        logging.info("shutting down")
        ahk_inst.stop_hotkeys()
        cam.stop()
        for name in ("motion_boxes", "char_region", "frangi_mask", "bobber", "indicator", "status"):
            ov.destroy_scene(name)


if __name__ == "__main__":
    main()
