import cv2


def drawRectangle(img, faceCoordinates):
    (x, y, w, h) = faceCoordinates
    thinckness = 2
    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thinckness)


# loads the xml data
# trained xml data
trainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

me = cv2.imread("me.jpg")
img = cv2.imread("group_of_people.jpg")

# grays the image
grayScaledImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# list of coordinates
faceCoordinates = trainedFaceData.detectMultiScale(grayScaledImg)
# print(faceCoordinates)

for fc in faceCoordinates:
    drawRectangle(img, fc)

cv2.imshow('Image', img)
# wait until a key is pressed
cv2.waitKey()
