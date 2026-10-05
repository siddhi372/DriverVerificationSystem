
"""
AI BASED DRIVER RECOGNITION & AUTHENTICATION SYSTEM
----------------------------------------------------
Professional live verification application.

Architecture:
    Camera
       ↓
    AIProcessor
       ↓
    This UI
       ↓
    FastAPI / PostgreSQL
       ↓
    ESP32

AI processing is intentionally kept inside client.ai_processor.AIProcessor.
This file handles camera, UI, ESP32 signal handling and backend submission.
"""

import time
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from client.ai_processor import AIProcessor
from client.esp32_manager import ESP32Manager
from client.verification_result import create_verification_result
from api.api_client import send_verification_result


# =========================================================
# CONFIGURATION
# =========================================================

CAMERA_INDEX = 0

ESP32_PORT = "COM3"
BAUDRATE = 115200

# True  = keyboard testing, no physical ESP32 required
# False = use physical ESP32
TEST_MODE = True

PROCESSING_TIME = 5
RESULT_DISPLAY_TIME = 5

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

WAITING = "WAITING"
VERIFYING = "VERIFYING"
RESULT = "RESULT"


# =========================================================
# PROFESSIONAL UI
# =========================================================

# RGB colors
BG = (7, 14, 25)
SURFACE = (13, 24, 41)
SURFACE_2 = (24, 40, 64)
SURFACE_3 = (24, 40, 63)
BORDER = (55, 88, 125)
BORDER_SOFT = (42, 67, 95)

WHITE = (244, 248, 253)
TEXT = (214, 224, 238)
MUTED = (145, 163, 184)

CYAN = (42, 205, 245)
BLUE = (76, 132, 255)
PURPLE = (202, 88, 232)
GREEN = (50, 220, 92)
YELLOW = (255, 202, 38)
RED = (244, 72, 78)

DESIGN_W = 1600
DESIGN_H = 900


# ---------------------------------------------------------
# FONTS
# ---------------------------------------------------------

FONT_CACHE = {}


def load_font(size, bold=False):
    key = (size, bold)

    if key in FONT_CACHE:
        return FONT_CACHE[key]

    if bold:
        candidates = [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/seguisb.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]

    for filename in candidates:
        try:
            font = ImageFont.truetype(filename, size)
            FONT_CACHE[key] = font
            return font
        except Exception:
            pass

    font = ImageFont.load_default()
    FONT_CACHE[key] = font
    return font


def draw_text(draw, xy, value, size=18, color=TEXT, bold=False):
    draw.text(
        xy,
        str(value),
        font=load_font(size, bold),
        fill=color
    )


def center_text(draw, cx, y, value, size=18, color=TEXT, bold=False):
    font = load_font(size, bold)
    value = str(value)
    box = draw.textbbox((0, 0), value, font=font)
    tw = box[2] - box[0]

    draw.text(
        (int(cx - tw / 2), int(y)),
        value,
        font=font,
        fill=color
    )


def rounded_box(draw, box, fill, outline=None, radius=10, width=1):
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=fill,
        outline=outline,
        width=width
    )


def draw_dot(draw, x, y, radius, color):
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=color
    )


def draw_panel(draw, x1, y1, x2, y2, title, accent):
    """
    Consistent card:
    - fixed header height
    - content starts below header
    - no element is drawn in the header/content boundary
    """

    rounded_box(
        draw,
        (x1, y1, x2, y2),
        SURFACE,
        BORDER,
        radius=10,
        width=2
    )

    # Header
    draw.rectangle(
        (x1 + 2, y1 + 2, x2 - 2, y1 + 54),
        fill=SURFACE_2
    )

    # Accent strip
    draw.rectangle(
        (x1, y1, x1 + 7, y1 + 56),
        fill=accent
    )

    draw_dot(
        draw,
        x1 + 31,
        y1 + 28,
        6,
        accent
    )

    draw_text(
        draw,
        (x1 + 52, y1 + 13),
        title,
        22,
        WHITE,
        True
    )

    draw.line(
        (x1, y1 + 56, x2, y1 + 56),
        fill=BORDER,
        width=2
    )


def draw_label_value(draw, x, y, label, value, value_color=WHITE):
    draw_text(
        draw,
        (x, y),
        label,
        17,
        MUTED,
        False
    )

    draw_text(
        draw,
        (x + 135, y - 1),
        value,
        18,
        value_color,
        True
    )


# ---------------------------------------------------------
# STATUS ICON
# ---------------------------------------------------------

def draw_status_icon(draw, cx, cy, status, color):
    draw.ellipse(
        (cx - 50, cy - 50, cx + 50, cy + 50),
        outline=color,
        width=5
    )

    draw.ellipse(
        (cx - 37, cy - 37, cx + 37, cy + 37),
        outline=BORDER,
        width=2
    )

    if status == "AUTHORIZED":
        draw.line(
            (cx - 23, cy, cx - 7, cy + 17),
            fill=color,
            width=7
        )
        draw.line(
            (cx - 7, cy + 17, cx + 25, cy - 21),
            fill=color,
            width=7
        )

    elif status == "DENIED":
        draw.line(
            (cx - 20, cy - 20, cx + 20, cy + 20),
            fill=color,
            width=7
        )
        draw.line(
            (cx + 20, cy - 20, cx - 20, cy + 20),
            fill=color,
            width=7
        )

    else:
        draw.ellipse(
            (cx - 10, cy - 10, cx + 10, cy + 10),
            fill=color
        )


# ---------------------------------------------------------
# CAMERA
# ---------------------------------------------------------

def put_camera_feed(canvas, camera_frame, ai, x1, y1, x2, y2):
    """
    Insert camera directly into the OpenCV canvas.
    Returns geometry used for sharp PIL overlays.
    """

    feed_x1 = x1 + 12
    feed_y1 = y1 + 70
    feed_x2 = x2 - 12
    feed_y2 = y2 - 38

    target_w = feed_x2 - feed_x1
    target_h = feed_y2 - feed_y1

    source_h, source_w = camera_frame.shape[:2]

    scale = max(
        target_w / float(source_w),
        target_h / float(source_h)
    )

    rw = max(1, int(source_w * scale))
    rh = max(1, int(source_h * scale))

    resized = cv2.resize(
        camera_frame,
        (rw, rh),
        interpolation=cv2.INTER_AREA
    )

    crop_x = max(0, (rw - target_w) // 2)
    crop_y = max(0, (rh - target_h) // 2)

    feed = resized[
        crop_y:crop_y + target_h,
        crop_x:crop_x + target_w
    ]

    if feed.shape[:2] != (target_h, target_w):
        feed = cv2.resize(
            feed,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA
        )

    canvas[
        feed_y1:feed_y2,
        feed_x1:feed_x2
    ] = feed

    cv2.rectangle(
        canvas,
        (feed_x1, feed_y1),
        (feed_x2 - 1, feed_y2 - 1),
        (45, 205, 245),
        2
    )

    # Face box directly on camera frame.
    box = ai.get("bounding_box") if ai else None

    if box is not None:
        bx = int(feed_x1 + box["x"] * scale - crop_x)
        by = int(feed_y1 + box["y"] * scale - crop_y)
        bw = int(box["width"] * scale)
        bh = int(box["height"] * scale)

        if ai.get("decision") == "ALLOW":
            face_color = (50, 220, 92)
        elif ai.get("liveness") == "SPOOF":
            face_color = (244, 72, 78)
        else:
            face_color = (255, 202, 38)

        cv2.rectangle(
            canvas,
            (max(feed_x1, bx), max(feed_y1, by)),
            (
                min(feed_x2 - 1, bx + bw),
                min(feed_y2 - 1, by + bh)
            ),
            face_color,
            3
        )

    return {
        "x1": feed_x1,
        "y1": feed_y1,
        "x2": feed_x2,
        "y2": feed_y2,
        "scale": scale,
        "crop_x": crop_x,
        "crop_y": crop_y,
    }


# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------

def draw_dashboard(camera_frame, ai):
    """Clean, production-style 1600x900 dashboard with strict card boundaries."""

    W, H = DESIGN_W, DESIGN_H

    canvas = np.full(
        (H, W, 3),
        (BG[2], BG[1], BG[0]),
        dtype=np.uint8
    )

    img = Image.fromarray(
        cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    )
    d = ImageDraw.Draw(img)

    # -----------------------------------------------------
    # CARD
    # -----------------------------------------------------

    def card(x1, y1, x2, y2, title, accent):
        rounded_box(
            d,
            (x1, y1, x2, y2),
            SURFACE,
            BORDER,
            radius=9,
            width=2
        )

        d.rectangle(
            (x1 + 2, y1 + 2, x2 - 2, y1 + 55),
            fill=SURFACE_2
        )

        d.rectangle(
            (x1, y1, x1 + 7, y1 + 55),
            fill=accent
        )

        draw_dot(
            d,
            x1 + 30,
            y1 + 28,
            6,
            accent
        )

        draw_text(
            d,
            (x1 + 51, y1 + 12),
            title,
            21,
            WHITE,
            True
        )

        # Explicit two-point line: never diagonal.
        d.line(
            [(x1, y1 + 55), (x2, y1 + 55)],
            fill=BORDER,
            width=2
        )

    def field(x, y, label, value, value_color=WHITE):
        draw_text(d, (x, y), label, 16, MUTED)
        draw_text(
            d,
            (x + 145, y - 1),
            str(value),
            17,
            value_color,
            True
        )

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    rounded_box(
        d,
        (24, 18, 1576, 92),
        SURFACE,
        BORDER,
        radius=10,
        width=2
    )

    d.ellipse(
        (43, 34, 82, 73),
        outline=CYAN,
        width=4
    )

    d.line([(51, 53), (60, 63)], fill=GREEN, width=5)
    d.line([(60, 63), (75, 43)], fill=GREEN, width=5)

    draw_text(
        d,
        (102, 29),
        "AI DRIVER AUTHENTICATION",
        27,
        WHITE,
        True
    )

    draw_text(
        d,
        (103, 62),
        "INTELLIGENT VEHICLE ACCESS CONTROL",
        13,
        CYAN,
        True
    )

    if state == VERIFYING:
        header_status = "AUTHENTICATING"
        header_color = YELLOW
    elif state == RESULT and ai and ai.get("decision") == "ALLOW":
        header_status = "SYSTEM AUTHORIZED"
        header_color = GREEN
    elif state == RESULT:
        header_status = "ACCESS DENIED"
        header_color = RED
    else:
        header_status = "SYSTEM READY"
        header_color = GREEN

    rounded_box(
        d,
        (1330, 34, 1555, 76),
        (13, 40, 34),
        header_color,
        radius=8,
        width=2
    )

    draw_dot(d, 1352, 55, 6, header_color)

    draw_text(
        d,
        (1370, 43),
        header_status,
        14,
        header_color,
        True
    )

    # -----------------------------------------------------
    # GRID
    # -----------------------------------------------------

    left_x1, left_x2 = 24, 522
    mid_x1, mid_x2 = 536, 1034
    right_x1, right_x2 = 1048, 1576

    top_y1, top_y2 = 112, 500
    bottom_y1, bottom_y2 = 514, 805

    card(left_x1, top_y1, left_x2, top_y2, "LIVE CAMERA", CYAN)
    card(mid_x1, top_y1, mid_x2, top_y2, "DRIVER INFORMATION", PURPLE)
    card(right_x1, top_y1, right_x2, top_y2, "AUTHENTICATION", GREEN)

    card(left_x1, bottom_y1, left_x2, bottom_y2, "VEHICLE / LOCATION", CYAN)
    card(mid_x1, bottom_y1, mid_x2, bottom_y2, "VERIFICATION ACTIVITY", YELLOW)
    card(right_x1, bottom_y1, right_x2, bottom_y2, "SYSTEM INFORMATION", BLUE)

    # -----------------------------------------------------
    # CAMERA
    # -----------------------------------------------------

    canvas = cv2.cvtColor(
        np.array(img),
        cv2.COLOR_RGB2BGR
    )

    cam = put_camera_feed(
        canvas,
        camera_frame,
        ai,
        left_x1,
        top_y1,
        left_x2,
        top_y2
    )

    img = Image.fromarray(
        cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    )
    d = ImageDraw.Draw(img)

    # Camera badge
    rounded_box(
        d,
        (
            cam["x1"] + 12,
            cam["y1"] + 12,
            cam["x1"] + 92,
            cam["y1"] + 43
        ),
        (8, 22, 35),
        CYAN,
        radius=7,
        width=1
    )

    draw_dot(
        d,
        cam["x1"] + 28,
        cam["y1"] + 27,
        5,
        GREEN
    )

    draw_text(
        d,
        (cam["x1"] + 40, cam["y1"] + 15),
        "LIVE",
        14,
        WHITE,
        True
    )

    if ai is None:
        message = "READY FOR VERIFICATION"
        message_color = MUTED
    elif state == VERIFYING:
        message = str(ai.get("guidance", "ANALYZING..."))
        message_color = GREEN if message == "GOOD" else YELLOW
    else:
        message = "CAMERA ONLINE"
        message_color = GREEN

    draw_text(
        d,
        (cam["x1"] + 6, cam["y2"] + 8),
        message,
        14,
        message_color,
        True
    )

    # Constrained driver label.
    if ai is not None and ai.get("bounding_box") is not None:
        box = ai["bounding_box"]

        bx = int(
            cam["x1"]
            + box["x"] * cam["scale"]
            - cam["crop_x"]
        )

        by = int(
            cam["y1"]
            + box["y"] * cam["scale"]
            - cam["crop_y"]
        )

        name = ai.get("driver_name", "UNKNOWN")

        if name != "UNKNOWN":
            label_color = (
                GREEN
                if ai.get("decision") == "ALLOW"
                else YELLOW
            )

            label_x = max(
                cam["x1"] + 4,
                min(bx, cam["x2"] - 185)
            )

            label_y = max(
                cam["y1"] + 48,
                min(by - 34, cam["y2"] - 32)
            )

            rounded_box(
                d,
                (
                    label_x,
                    label_y,
                    label_x + 175,
                    label_y + 29
                ),
                (8, 20, 32),
                label_color,
                radius=5,
                width=1
            )

            draw_text(
                d,
                (label_x + 9, label_y + 5),
                name,
                14,
                label_color,
                True
            )

    # -----------------------------------------------------
    # DRIVER INFORMATION
    # -----------------------------------------------------

    if ai is None:
        name = "WAITING"
        driver_id = "--"
        age = "--"
        identity = "NOT VERIFIED"
        liveness = "WAITING"
        live_score = 0.0
        match_score = 0.0
    else:
        name = ai.get("driver_name", "UNKNOWN")
        driver_id = ai.get("driver_id") or "--"

        age_value = ai.get("estimated_age")
        age = "--" if age_value is None else str(age_value)

        identity = (
            "VERIFIED"
            if ai.get("matched", False)
            else "NOT VERIFIED"
        )

        liveness = ai.get("liveness", "WAITING")
        live_score = float(
            ai.get("liveness_confidence", 0.0)
        )
        match_score = float(
            ai.get("driver_confidence", 0.0)
        )

    x = mid_x1 + 30

    field(
        x,
        top_y1 + 82,
        "Driver",
        name,
        GREEN if identity == "VERIFIED" else WHITE
    )

    field(
        x,
        top_y1 + 121,
        "Driver ID",
        driver_id
    )

    field(
        x,
        top_y1 + 160,
        "Age",
        age
    )

    field(
        x,
        top_y1 + 199,
        "Identity",
        identity,
        GREEN if identity == "VERIFIED" else YELLOW
    )

    d.line(
        [(x, top_y1 + 230), (mid_x2 - 30, top_y1 + 230)],
        fill=BORDER,
        width=2
    )

    live_color = (
        GREEN
        if liveness == "LIVE"
        else RED
        if liveness == "SPOOF"
        else WHITE
    )

    field(
        x,
        top_y1 + 254,
        "Liveness",
        "REAL" if liveness == "LIVE" else liveness,
        live_color
    )

    field(
        x,
        top_y1 + 293,
        "Live score",
        f"{live_score * 100:.2f} %"
    )

    field(
        x,
        top_y1 + 332,
        "Match score",
        f"{match_score * 100:.2f} %"
    )

    # -----------------------------------------------------
    # AUTHENTICATION
    # -----------------------------------------------------

    if ai is not None and ai.get("decision") == "ALLOW":
        auth_status = "AUTHORIZED"
        access = "GRANTED"
        auth_color = GREEN
    elif state == VERIFYING:
        auth_status = "VERIFYING"
        access = "CHECKING"
        auth_color = YELLOW
    elif state == RESULT:
        auth_status = "DENIED"
        access = "LOCKED"
        auth_color = RED
    else:
        auth_status = "READY"
        access = "LOCKED"
        auth_color = CYAN

    auth_cx = (right_x1 + right_x2) // 2

    draw_status_icon(
        d,
        auth_cx,
        top_y1 + 140,
        auth_status,
        auth_color
    )

    center_text(
        d,
        auth_cx,
        top_y1 + 205,
        auth_status,
        31,
        auth_color,
        True
    )

    draw_text(
        d,
        (right_x1 + 55, top_y1 + 258),
        "VEHICLE ACCESS",
        15,
        MUTED,
        True
    )

    lock_x = right_x1 + 58
    lock_y = top_y1 + 298

    d.rectangle(
        (
            lock_x,
            lock_y + 12,
            lock_x + 30,
            lock_y + 42
        ),
        outline=auth_color,
        width=3
    )

    d.arc(
        (
            lock_x + 6,
            lock_y - 8,
            lock_x + 24,
            lock_y + 23
        ),
        180,
        360,
        fill=auth_color,
        width=3
    )

    draw_text(
        d,
        (lock_x + 48, lock_y + 12),
        access,
        22,
        auth_color,
        True
    )

    # -----------------------------------------------------
    # VEHICLE / LOCATION
    # -----------------------------------------------------

    vehicle_id = (
        ai.get("vehicle_id") or "--"
        if ai is not None
        else "--"
    )

    field(
        left_x1 + 30,
        bottom_y1 + 72,
        "Vehicle ID",
        vehicle_id
    )

    field(
        left_x1 + 30,
        bottom_y1 + 111,
        "GPS Latitude",
        str(GPS_LATITUDE)
    )

    field(
        left_x1 + 30,
        bottom_y1 + 150,
        "GPS Longitude",
        str(GPS_LONGITUDE)
    )

    d.line(
        [
            (left_x1 + 30, bottom_y1 + 180),
            (left_x2 - 30, bottom_y1 + 180)
        ],
        fill=BORDER,
        width=2
    )

    gps_online = (
        GPS_LATITUDE != "--"
        and GPS_LONGITUDE != "--"
    )

    draw_dot(
        d,
        left_x1 + 40,
        bottom_y1 + 208,
        5,
        GREEN if gps_online else YELLOW
    )

    draw_text(
        d,
        (left_x1 + 55, bottom_y1 + 196),
        "GPS ONLINE"
        if gps_online
        else "GPS HARDWARE PENDING",
        14,
        GREEN if gps_online else YELLOW,
        True
    )

    if TEST_MODE and not getattr(esp32, "connected", False):

        rounded_box(
            d,
            (
                left_x1 + 30,
                bottom_y1 + 238,
                left_x1 + 164,
                bottom_y1 + 269
            ),
            (45, 38, 12),
            YELLOW,
            radius=6,
            width=1
        )

        draw_dot(
            d,
            left_x1 + 43,
            bottom_y1 + 254,
            4,
            YELLOW
        )

        draw_text(
            d,
            (left_x1 + 54, bottom_y1 + 246),
            "TEST MODE",
            12,
            YELLOW,
            True
        )

    # -----------------------------------------------------
    # VERIFICATION ACTIVITY
    # -----------------------------------------------------

    if ai is None:
        activity_title = "Ready for driver verification"
        activity_color = GREEN
        steps = [
            ("FACE DETECTION", "WAIT"),
            ("LIVENESS", "WAIT"),
            ("IDENTITY MATCH", "WAIT")
        ]
    else:

        face_status = (
            "PASS"
            if ai.get("face_count") == 1
            else "FAIL"
            if ai.get("face_count", 0) > 1
            else "WAIT"
        )

        live_status = (
            "PASS"
            if ai.get("liveness") == "LIVE"
            else "FAIL"
            if ai.get("liveness") == "SPOOF"
            else "WAIT"
        )

        identity_status = (
            "PASS"
            if ai.get("matched", False)
            else "FAIL"
            if ai.get("driver_name")
            not in (None, "UNKNOWN")
            else "WAIT"
        )

        all_pass = (
            face_status == "PASS"
            and live_status == "PASS"
            and identity_status == "PASS"
        )

        activity_title = (
            "All security checks passed"
            if all_pass
            else "Verification in progress"
        )

        activity_color = GREEN if all_pass else YELLOW

        steps = [
            ("FACE DETECTION", face_status),
            ("LIVENESS", live_status),
            ("IDENTITY MATCH", identity_status)
        ]

    rounded_box(
        d,
        (
            mid_x1 + 25,
            bottom_y1 + 68,
            mid_x2 - 25,
            bottom_y1 + 108
        ),
        (18, 34, 44),
        None,
        radius=8
    )

    draw_dot(
        d,
        mid_x1 + 43,
        bottom_y1 + 88,
        5,
        activity_color
    )

    draw_text(
        d,
        (mid_x1 + 58, bottom_y1 + 76),
        activity_title,
        16,
        activity_color,
        True
    )

    row_y = bottom_y1 + 132

    for label, status_value in steps:

        if status_value == "PASS":
            c = GREEN
        elif status_value == "FAIL":
            c = RED
        else:
            c = MUTED

        draw_dot(
            d,
            mid_x1 + 38,
            row_y + 8,
            5,
            c
        )

        draw_text(
            d,
            (mid_x1 + 55, row_y),
            label,
            14,
            TEXT
        )

        font = load_font(14, True)
        bbox = d.textbbox(
            (0, 0),
            status_value,
            font=font
        )

        sw = bbox[2] - bbox[0]

        d.text(
            (
                mid_x2 - 35 - sw,
                row_y
            ),
            status_value,
            font=font,
            fill=c
        )

        row_y += 38

    # -----------------------------------------------------
    # SYSTEM INFORMATION
    # -----------------------------------------------------

    now = datetime.now()
    system_cx = (right_x1 + right_x2) // 2

    center_text(
        d,
        system_cx,
        bottom_y1 + 75,
        now.strftime("%d %b %Y").upper(),
        22,
        WHITE,
        True
    )

    center_text(
        d,
        system_cx,
        bottom_y1 + 116,
        now.strftime("%I:%M:%S %p"),
        29,
        WHITE,
        True
    )

    center_text(
        d,
        system_cx,
        bottom_y1 + 157,
        "LOCAL SYSTEM TIME",
        13,
        MUTED
    )

    rounded_box(
        d,
        (
            right_x1 + 48,
            bottom_y1 + 196,
            right_x1 + 173,
            bottom_y1 + 228
        ),
        (19, 47, 36),
        GREEN,
        radius=7,
        width=1
    )

    draw_dot(
        d,
        right_x1 + 62,
        bottom_y1 + 212,
        4,
        GREEN
    )

    draw_text(
        d,
        (right_x1 + 75, bottom_y1 + 203),
        "AI ONLINE",
        13,
        GREEN,
        True
    )

    rounded_box(
        d,
        (
            right_x1 + 185,
            bottom_y1 + 196,
            right_x1 + 310,
            bottom_y1 + 228
        ),
        (20, 38, 55),
        CYAN,
        radius=7,
        width=1
    )

    draw_dot(
        d,
        right_x1 + 199,
        bottom_y1 + 212,
        4,
        CYAN
    )

    draw_text(
        d,
        (right_x1 + 212, bottom_y1 + 203),
        "EDGE AI",
        13,
        CYAN,
        True
    )

    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------

    d.line(
        [(24, 838), (1576, 838)],
        fill=BORDER_SOFT,
        width=1
    )

    draw_text(
        d,
        (30, 858),
        "EDGE AI SECURITY SYSTEM  •  FACE ID  •  LIVENESS  •  VEHICLE ACCESS",
        11,
        MUTED
    )

    footer_status = (
        "READY"
        if state == WAITING
        else "VERIFYING"
        if state == VERIFYING
        else "COMPLETE"
    )

    font = load_font(13, True)
    bbox = d.textbbox(
        (0, 0),
        footer_status,
        font=font
    )

    d.text(
        (
            1570 - (bbox[2] - bbox[0]),
            858
        ),
        footer_status,
        font=font,
        fill=GREEN if state != VERIFYING else YELLOW
    )

    result = cv2.cvtColor(
        np.array(img),
        cv2.COLOR_RGB2BGR
    )

    actual_h, actual_w = camera_frame.shape[:2]

    if (
        actual_w != DESIGN_W
        or actual_h != DESIGN_H
    ):
        result = cv2.resize(
            result,
            (actual_w, actual_h),
            interpolation=cv2.INTER_AREA
        )

    return result


def send_result_to_backend(ai):
    """
    Uses the project's existing FastAPI client/helper.

    This keeps the current backend integration intact.
    """

    verification_data = create_verification_result(
        face_detected=ai["face_detected"],
        face_count=ai["face_count"],
        bounding_box=ai["bounding_box"],
        image_quality=ai["image_quality"],
        liveness=ai["liveness"],
        liveness_confidence=ai["liveness_confidence"],
        driver_name=ai["driver_name"],
        driver_confidence=ai["driver_confidence"],
        decision=ai["decision"]
    )

    print()
    print("=" * 55)
    print("VERIFICATION RESULT")
    print("=" * 55)
    print(verification_data)
    print("=" * 55)

    try:

        response = send_verification_result(
            verification_data
        )

        print(
            "API RESPONSE:",
            response
        )

    except Exception as exc:

        print(
            "API ERROR:",
            exc
        )


# =========================================================
# RUNTIME INITIALIZATION
# =========================================================

WINDOW_NAME = "AI Driver Authentication"

GPS_LATITUDE = "--"
GPS_LONGITUDE = "--"

print("Initializing AI processor...")
processor = AIProcessor()

print("Initializing camera...")
camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

camera.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

if not camera.isOpened():
    raise RuntimeError(
        f"Could not open camera {CAMERA_INDEX}. "
        "Check that the webcam is connected and not being used by another application."
    )

print("Camera ready.")

esp32 = None

if not TEST_MODE:
    print(
        f"Connecting to ESP32 on {ESP32_PORT}..."
    )

    esp32 = ESP32Manager(
        port=ESP32_PORT,
        baudrate=BAUDRATE
    )

    esp32.connect()

    print("ESP32 connected.")

state = WAITING
verification_start_time = None
result_start_time = None
latest_result = None
result_sent = False

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

print()
print("=" * 60)
print("AI DRIVER AUTHENTICATION SYSTEM")
print("=" * 60)

if TEST_MODE:
    print("TEST MODE: Press S to START, X to STOP, Q to QUIT")
else:
    print("Waiting for ESP32 START signal...")

print("=" * 60)


# =========================================================
# MAIN APPLICATION
# =========================================================

try:

    while True:

        # -------------------------------------------------
        # CAMERA
        # -------------------------------------------------

        success, frame = camera.read()

        if not success:

            print(
                "Camera frame failed."
            )

            break

        frame = cv2.flip(
            frame,
            1
        )

        # -------------------------------------------------
        # ESP32 / TEST INPUT
        # -------------------------------------------------

        signal = None

        if not TEST_MODE:

            signal = esp32.read_signal()

        else:

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):

                signal = "START"

                print(
                    "TEST ESP32 -> START"
                )

            elif key == ord("x"):

                signal = "STOP"

                print(
                    "TEST ESP32 -> STOP"
                )

            elif key == ord("q"):

                print(
                    "QUIT requested."
                )

                break

        # -------------------------------------------------
        # START SIGNAL
        # -------------------------------------------------

        if signal == "START":

            if state == WAITING:

                print(
                    "START received"
                )

                state = VERIFYING

                verification_start_time = time.time()
                result_start_time = None

                latest_result = None
                result_sent = False

                processor.reset()

        # -------------------------------------------------
        # STOP SIGNAL
        # -------------------------------------------------

        elif signal == "STOP":

            print(
                "STOP received"
            )

            state = WAITING

            verification_start_time = None
            result_start_time = None

            latest_result = None
            result_sent = False

            processor.reset()

        # -------------------------------------------------
        # WAITING
        # -------------------------------------------------

        if state == WAITING:

            dashboard = draw_dashboard(
                frame,
                None
            )

        # -------------------------------------------------
        # VERIFYING
        # -------------------------------------------------

        elif state == VERIFYING:

            # All AI processing happens here.
            # UI only displays the structured result.
            ai = processor.process(
                frame
            )

            latest_result = ai

            dashboard = draw_dashboard(
                frame,
                ai
            )

            elapsed = (
                time.time()
                - verification_start_time
            )

            # ---------------------------------------------
            # FINAL DECISION
            # ---------------------------------------------

            if elapsed >= PROCESSING_TIME:

                final_decision = ai.get(
                    "decision",
                    "DENY"
                )

                if not result_sent:

                    send_result_to_backend(
                        ai
                    )

                    result_sent = True

                    # Physical ESP32 receives only the final decision.
                    if not TEST_MODE:

                        if final_decision == "ALLOW":
                            esp32.send_command(
                                "ALLOW"
                            )
                        else:
                            esp32.send_command(
                                "DENY"
                            )

                    print(
                        "FINAL RESULT:",
                        final_decision
                    )

                state = RESULT
                result_start_time = time.time()

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        elif state == RESULT:

            if latest_result is not None:

                dashboard = draw_result_dashboard(
                    frame,
                    latest_result
                )

            else:

                dashboard = draw_dashboard(
                    frame,
                    None
                )

            if (
                result_start_time is not None
                and
                time.time() - result_start_time
                >= RESULT_DISPLAY_TIME
            ):

                state = WAITING

                verification_start_time = None
                result_start_time = None

                latest_result = None
                result_sent = False

                processor.reset()

        # -------------------------------------------------
        # DISPLAY
        # -------------------------------------------------

        cv2.imshow(
            WINDOW_NAME,
            dashboard
        )


finally:

    camera.release()

    cv2.destroyAllWindows()

    if not TEST_MODE:

        esp32.close()

    print(
        "Application closed."
    )
