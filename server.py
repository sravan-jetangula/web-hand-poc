
import cv2
import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

FRAME_W = 640
FRAME_H = 480
VOBJ_RECT = (int(FRAME_W*0.55), int(FRAME_H*0.2), 200, 200)
DANGER_THRESH = 40
WARNING_THRESH = 140

LOW_HSV = np.array([0, 30, 60])
HIGH_HSV = np.array([25, 200, 255])
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_SAFE = (50,205,50)
COLOR_WARNING = (0,215,255)
COLOR_DANGER = (0,0,255)

def get_skin_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOW_HSV, HIGH_HSV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=1)
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    return mask

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 2000:
        return None
    return largest

def fingertip_from_contour(cnt):
    M = cv2.moments(cnt)
    if M.get('m00',0) == 0:
        return tuple(cnt[cnt[:,:,1].argmin()][0])
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    centroid = np.array([cx, cy])
    hull = cv2.convexHull(cnt, returnPoints=True)
    hull_pts = hull.reshape(-1,2)
    dists = np.linalg.norm(hull_pts - centroid, axis=1)
    idx = np.argmax(dists)
    fingertip = tuple(hull_pts[idx])
    if fingertip[1] > cy + 30:
        return tuple(cnt[cnt[:,:,1].argmin()][0])
    return fingertip

def point_to_rect_distance(pt, rect):
    x,y,w,h = rect
    px, py = pt
    rx1, ry1, rx2, ry2 = x, y, x+w, y+h
    if rx1 <= px <= rx2 and ry1 <= py <= ry2:
        return 0
    dx = max(rx1 - px, 0, px - rx2)
    dy = max(ry1 - py, 0, py - ry2)
    return int(np.hypot(dx, dy))

def annotate_frame(frame):
    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    mask = get_skin_mask(frame)
    largest = find_largest_contour(mask)
    fingertip = None
    state_text = "SAFE"
    state_color = COLOR_SAFE
    dist = None

    if largest is not None:
        fingertip = fingertip_from_contour(largest)
        cv2.circle(frame, fingertip, 8, (255,255,255), -1)
        dist = point_to_rect_distance(fingertip, VOBJ_RECT)
        if dist <= DANGER_THRESH:
            state_text = "DANGER"
            state_color = COLOR_DANGER
        elif dist <= WARNING_THRESH:
            state_text = "WARNING"
            state_color = COLOR_WARNING

    x,y,w,h = VOBJ_RECT
    cv2.rectangle(frame, (x,y), (x+w, y+h), (180,180,180), 2)
    cv2.rectangle(frame, (6,6), (260,90), (0,0,0), -1)
    cv2.putText(frame, f'STATE: {state_text}', (12,36), FONT, 1.0, state_color, 2, cv2.LINE_AA)
    if dist is not None:
        cv2.putText(frame, f'D={dist}px', (12,70), FONT, 0.8, (200,200,200), 1, cv2.LINE_AA)

    if state_text == 'DANGER':
        t = cv2.getTickCount() / cv2.getTickFrequency()
        alpha = 0.2 + 0.2*np.sin(t*10)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (FRAME_W, FRAME_H), (0,0,255), -1)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        big_txt = "DANGER DANGER"
        (w_txt, h_txt), _ = cv2.getTextSize(big_txt, FONT, 2.0, 6)
        pos = (int((FRAME_W-w_txt)/2), int(FRAME_H/2))
        cv2.putText(frame, big_txt, pos, FONT, 2.0, (255,255,255), 6, cv2.LINE_AA)
    return frame

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            annotated = annotate_frame(img)
            ret, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                continue
            await websocket.send_bytes(buf.tobytes())
    except:
        return

@app.get("/")
def home():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
