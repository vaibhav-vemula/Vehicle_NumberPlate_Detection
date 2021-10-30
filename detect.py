import cv2
import pytesseract

image = cv2.imread('images/car.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
filter = cv2.bilateralFilter(gray,11,19,19)
edged = cv2.Canny(filter, 170, 200)
cont, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cont = sorted(cont, key = cv2.contourArea, reverse = True)[:10]
cnt = None
count = 0
for c in cont:
    ep = 0.05*cv2.arcLength(c, True)
    app = cv2.approxPolyDP(c, ep, True)
    if len(app) == 4:  
        cnt = app
        x,y,w,h = cv2.boundingRect(c)
        break

cropped = image[y:y+h, x:x+w]
tt = pytesseract.image_to_string(cropped)

image = cv2.putText(image, tt[:-3], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2, cv2.LINE_AA)

cv2.drawContours(image, [cnt], -1, (0,0,255), 2)
cv2.imshow("Output", image)
# cv2.imshow("number", cropped)
cv2.waitKey(0) 