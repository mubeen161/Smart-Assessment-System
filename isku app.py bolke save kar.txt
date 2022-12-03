import cv2
import numpy as np
import math
import mediapipe as mp
import csv
import time
from flask import Flask, render_template,request

app=Flask(__name__)
@app.route("/")
@app.route("/home")
def home():
    return render_template('index.html')
@app.route("/contact",methods =['POST','GET'])
def contact():
    output =request.form.to_dict()
    name= output["name"]
    return render_template('index.html')
cx, cy, w, h = 100, 100, 200, 200
colorRect = (255, 0, 255)
alpha = 145
startDist = None
scale = 75
out = None
scaling_sensitivity = 200

image = cv2.imread('C:/Users/ziaulqamar/Desktop/New folder/quizcv/2.jpg')
resized_image = cv2.resize(image, (590, 440))
image_height = resized_image.shape[0]
image_width = resized_image.shape[1]

background_image = cv2.imread('C:/Users/ziaulqamar/Desktop/New folder/quizcv/1.png')
background_image = cv2.resize(background_image, (1280, 720))

pathCSV = "C:/Users/ziaulqamar/Desktop/New folder/quizcv/mcq.csv"
questionNo = 0
totalQuestion = 0
score = 0
isQuizEnded = False


def errorDisplay(error_description):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, error_description, (10, 30),
                font, 1, (0, 0, 255), 1, cv2.LINE_AA)


def fingerDistance(finger1, finger2, threshold, img, handNo=0):
    l, _, _ = detector.findDistance(finger1, finger2, img, handNo, draw=False)

    if l < threshold:
        return True, l
    else:
        return False, l


def cornerRect(img, bbox, l=30, t=5, rt=1, colorR=(255, 0, 255), colorC=(0, 255, 0)):
    """
    :param img: Image to draw on.
    :param bbox: Bounding box [x, y, w, h]
    :param l: length of the corner line
    :param t: thickness of the corner line
    :param rt: thickness of the rectangle
    :param colorR: Color of the Rectangle
    :param colorC: Color of the Corners
    :return:
    """
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    if rt != 0:
        cv2.rectangle(img, bbox, colorR, rt)
    # Top Left  x,y
    cv2.line(img, (x, y), (x + l, y), colorC, t)
    cv2.line(img, (x, y), (x, y + l), colorC, t)
    # Top Right  x1,y
    cv2.line(img, (x1, y), (x1 - l, y), colorC, t)
    cv2.line(img, (x1, y), (x1, y + l), colorC, t)
    # Bottom Left  x,y1
    cv2.line(img, (x, y1), (x + l, y1), colorC, t)
    cv2.line(img, (x, y1), (x, y1 - l), colorC, t)
    # Bottom Right  x1,y1
    cv2.line(img, (x1, y1), (x1 - l, y1), colorC, t)
    cv2.line(img, (x1, y1), (x1, y1 - l), colorC, t)

    return img


def isClicked(lmList, lmList1):
    rectList = mcqList[questionNo].rectList
    if lmList:
        condition1, l1 = fingerDistance(8, 12, 55, img, 0)
        if condition1:
            cursor = lmList[8]
            for rect in rectList:
                rect.clickUpdate(cursor)

    if lmList1:
        condition2, l2 = fingerDistance(8, 12, 55, img, 1)
        if condition2:
            cursor = lmList1[8]
            for rect in rectList:
                rect.clickUpdate(cursor)


def isDragged(lmList, lmList1):
    rectList = mcqList[questionNo].rectList
    if lmList:
        condition1, l1 = fingerDistance(4, 8, 40, img, 0)
        if condition1:
            cursor = lmList[8]
            for rect in rectList:
                rect.dragUpdate(cursor)
        else:
            colorRect = (255, 0, 255)

    if lmList1:
        condition2, l2 = fingerDistance(4, 8, 40, img, 1)
        if condition2:
            cursor = lmList1[8]
            for rect in rectList:
                rect.dragUpdate(cursor)
        else:
            colorRect = (255, 0, 255)


def isScaled(lmList, lmList1):
    global startDist, scale
    rectList = mcqList[questionNo].rectList
    if lmList1:
        if detector.fingersUp(img, 0) == [1, 1, 0, 0, 0] and detector.fingersUp(img, 1) == [1, 1, 0, 0, 0]:
            x1, y1 = lmList[8]
            x2, y2 = lmList1[8]
            length = x2 - x1

            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            if startDist == None:
                startDist = length
            else:
                scale = int((length - startDist)//(100 - scaling_sensitivity))

            if scale != 0:
                for rect in rectList:
                    rect.scaleUpdate()
        else:
            startDist = None
            scale = 0


def showScoreScreen():
    global score
    scorePercentage = round((score / totalQuestion) * 100, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, "Quiz Completed", (400, 300),
                font, 2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"Score: {scorePercentage}%",
                (500, 400), font, 2, (0, 255, 0), 2, cv2.LINE_AA)


class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        hands = 0
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                hands = hands+1
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img, hands

    def findPosition(self, img, handNo=0, draw=True):
        """
        Finds landmarks of a single hand and puts them in a list
        in pixel format. Also finds the bounding box around the hand.

        :param img: main image to find hand in
        :param handNo: hand id if more than one hand detected
        :param draw: Flag to draw the output on the image.
        :return: list of landmarks in pixel format; bounding box
        """

        xList = []
        yList = []
        bbox = []
        bboxInfo = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                px, py = int(lm.x * w), int(lm.y * h)
                xList.append(px)
                yList.append(py)
                self.lmList.append([px, py])
                if draw:
                    cv2.circle(img, (px, py), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            boxW, boxH = xmax - xmin, ymax - ymin
            bbox = xmin, ymin, boxW, boxH
            cx, cy = bbox[0] + (bbox[2] // 2), \
                bbox[1] + (bbox[3] // 2)
            bboxInfo = {"id": id, "bbox": bbox, "center": (cx, cy)}

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                              (0, 255, 0), 2)

        return self.lmList, bboxInfo

    def fingersUp(self, img, handNo=0):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        fingers = []
        lmList, _ = self.findPosition(img, handNo)
        if self.results.multi_hand_landmarks:
            myHandType = self.handType(lmList)

            # Thumb
            if myHandType == "Right":
                if lmList[self.tipIds[0]][0] > lmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if lmList[self.tipIds[0]][0] < lmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if lmList[self.tipIds[id]][1] < lmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, handNo=0, draw=True):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1 - Index of Landmark 1.
        :param p2: Point2 - Index of Landmark 2.
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        lmList, _ = self.findPosition(img, handNo)

        if self.results.multi_hand_landmarks[handNo]:
            x1, y1 = lmList[p1][0], lmList[p1][1]
            x2, y2 = lmList[p2][0], lmList[p2][1]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if draw:
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            return length, img, [x1, y1, x2, y2, cx, cy]

    def handType(self, lmList):
        """
        Checks if the hand is left or right
        :return: "Right" or "Left"
        """
        if self.results.multi_hand_landmarks:
            if lmList[17][0] < lmList[5][0]:
                return "Right"
            else:
                return "Left"


class SelfiSegmentation():

    def __init__(self, model=1):
        """
        :param model: model type 0 or 1. 0 is general 1 is landscape(faster)
        """

        self.model = model
        self.mpDraw = mp.solutions.drawing_utils
        self.mpSelfieSegmentation = mp.solutions.selfie_segmentation
        self.selfieSegmentation = self.mpSelfieSegmentation.SelfieSegmentation(
            self.model)

    def removeBG(self, img, imgBg=(255, 255, 255), threshold=0.8):
        """
        :param img: image to remove background from
        :param imgBg: BackGround Image
        :param threshold: higher = more cut, lower = less cut
        :return:
        """

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.selfieSegmentation.process(imgRGB)
        condition = np.stack((results.segmentation_mask,)
                             * 3, axis=-1) > threshold
        if isinstance(imgBg, tuple):
            _imgBg = np.zeros(img.shape, dtype=np.uint8)
            _imgBg[:] = imgBg
            imgOut = np.where(condition, img, _imgBg)
        else:
            imgOut = np.where(condition, img, imgBg)
        return imgOut


class Rect():
    def __init__(self, posCenter, size=[200, 200], text=None, obType=None, defaultColor=(255, 0, 0)):
        self.posCenter = posCenter
        self.size = size
        self.clicked = False
        self.text = text
        self.obType = obType
        self.color = defaultColor

    def clickUpdate(self, cursor):
        global scale, resized_image

        cx, cy = self.posCenter
        w, h = self.size

        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.clicked = True
            time.sleep(0.4)

    def dragUpdate(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor
            global colorRect
            colorRect = (0, 255, 0)
            cornerRect(imgNew, (cx - w // 2, cy - h //
                                2, w, h), 20, 8, 5, colorRect, (0, 0, 0))

    def scaleUpdate(self):
        try:
            global image, resized_image
            w, h = self.size
            newW, newH = (w+scale)//2, (h+scale)//2
            resized_image = cv2.resize(image, (newW * 2, newH * 2))
            self.size = [newW * 2, newH * 2]
        except:
            errorDisplay("error in zoom function")

    def imagePlacement(self):
        cx, cy = self.posCenter
        w, h = self.size

        try:
            img[cy - (h//2): cy + (h//2), cx - (w//2): cx + (w//2)] = cv2.addWeighted(img[cy - (h//2)                                                                                          : cy + (h//2), cx - (w//2): cx + (w//2), :], alpha, resized_image[0:h, 0:w, :], 1 - alpha, 0)
        except:
            errorDisplay("image out-of-bounds")

        global out
        out = img.copy()
        mask = imgNew.astype(bool)
        out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    def textPlacement(self, offset=50, scale=3, thickness=3, font=cv2.FONT_HERSHEY_PLAIN):
        if self.text is not None:
            cx, cy = self.posCenter
            (t_w, t_h), _ = cv2.getTextSize(self.text, font, scale, thickness)
            self.size = t_w + offset, t_h + offset
            w, h = self.size
            x1, y1, x2, y2 = cx - w//2, cy + h//2, cx + w//2, cy - h//2

            cv2.rectangle(img, (x1, y1), (x2, y2), self.color, cv2.FILLED)
            cv2.putText(img, self.text, (x1 + offset//2, y1 - offset//2),
                        font, scale, (255, 255, 255), thickness)

        global out
        out = img.copy()


class MCQ():
    def __init__(self, data):
        self.question = data[0]
        self.choice1 = data[1]
        self.choice2 = data[2]
        self.choice3 = data[3]
        self.choice4 = data[4]
        self.answer = int(data[5])

        self.rectList = []
        self.rectList.append(Rect([640, 100], text=self.question, obType=0))
        self.rectList.append(Rect([360, 200], text=self.choice1, obType=1))
        self.rectList.append(Rect([920, 200], text=self.choice2, obType=2))
        self.rectList.append(Rect([360, 300], text=self.choice3, obType=3))
        self.rectList.append(Rect([920, 300], text=self.choice4, obType=4))

        self.userAns = None


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=1, maxHands=2)
segmentor = SelfiSegmentation()

with open(pathCSV, newline="\n") as f:
    reader = csv.reader(f, delimiter=";")
    data = list(reader)[2:]
totalQuestion = len(data)
mcqList = []
for x in data:
    mcqList.append(MCQ(x))

imgList = []
for x in range(1):
    imgList.append(Rect([x * 250 + 300, 300], [image_width, image_height]))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = segmentor.removeBG(img, background_image, threshold=0.3)
    img, hands = detector.findHands(img, draw=False)
    lmList, _ = detector.findPosition(img, handNo=0, draw=False)
    lmList1 = []
    if hands == 2:
        lmList1, _ = detector.findPosition(img, handNo=1, draw=False)

    imgNew = np.zeros_like(img, np.uint8)

    isClicked(lmList, lmList1)

    isDragged(lmList, lmList1)

    isScaled(lmList, lmList1)

    out = img.copy()

    if isQuizEnded == False:
        rectList = mcqList[questionNo].rectList
        for rect in rectList:
            if rect.clicked == True:
                if rect.obType == mcqList[questionNo].answer:
                    rect.color = (0, 255, 0)
                    if score < totalQuestion:
                        score += 1
                else:
                    rect.color = (0, 0, 255)

                if questionNo < totalQuestion - 1:
                    questionNo += 1
                else:
                    isQuizEnded = True

            rect.textPlacement(scale=2, thickness=2)
        barValue = 150 + (950 // totalQuestion) * score
        cv2.rectangle(out, (150, 600), (barValue, 650),
                      (0, 255, 0), cv2.FILLED)
        cv2.rectangle(out, (150, 600), (1100, 650), (255, 0, 255), 5)
    else:
        showScoreScreen()

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

    cv2.imshow("Python Quiz", out)
    cv2.waitKey(1)
if __name__ == '__main__':
    app.run(debug=True,port=5555 )