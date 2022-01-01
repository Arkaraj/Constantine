import cv2


def drawRectangle(img, faceCoordinates):
    (x, y, w, h) = faceCoordinates
    thinckness = 2
    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thinckness)


# trained xml data
trainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# default/system webcam if we put 0
io = input("Enter the path of the video else press 0: ")
path = 0 if io == "0" else io
webcam = cv2.VideoCapture(path)

while True:
    _, frame = webcam.read()
    grayScaledImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCoordinates = trainedFaceData.detectMultiScale(grayScaledImg)
    for fc in faceCoordinates:
        drawRectangle(frame, fc)
    cv2.imshow('Image', frame)
    # every millisecond
    key = cv2.waitKey(1)
    # 113 is q letter - ascii
    if key == 113:
        break

webcam.release()
