import cv2
import time

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka. Coba index 1/2.")

frames, t0 = 0, time.time()

while True:
    ok, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ok: 
        break
    
    frames += 1
    
    if time.time() - t0 >= 1.0:
        cv2.setWindowTitle("Preview", f"Preview (FPS ~ {frames})")
        frames, t0 = 0, time.time()
        
    cv2.imshow("Preview", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()