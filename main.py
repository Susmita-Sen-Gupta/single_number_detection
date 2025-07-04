import cv2
import numpy as np
import torch
from model import DigitCNN
from utils import preprocess

model = DigitCNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

canvas = np.ones((400, 400), dtype="uint8") * 0
drawing = False

def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(canvas, (x, y), 12, (255, 255, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow("Draw Digit")
cv2.setMouseCallback("Draw Digit", draw)

while True:
    display = canvas.copy()
    cv2.putText(display, "Press 'P' to Predict | 'C' to Clear | 'Q' to Quit", (10, 390),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (127), 1)
    cv2.imshow("Draw Digit", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        digit_img = preprocess(canvas)
        with torch.no_grad():
            output = model(digit_img)
            pred = torch.argmax(output, 1).item()
        print(f"Predicted Digit: {pred}")
        cv2.putText(display, f"Prediction: {pred}", (130, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255), 2)
        cv2.imshow("Draw Digit", display)
        cv2.waitKey(1500)

    elif key == ord('c'):
        canvas[:] = 0
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
