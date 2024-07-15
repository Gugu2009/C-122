import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller

keyboard = Controller()
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence= 0.5)
tipIds = [4, 8, 12, 16, 20]

def countFingers(image, hand_landmarks, handNo=0):
    if hand_landmarks:
        landmarks = hand_landmarks[handNo].landmark
        fingers  = []

        for lm_index in tipIds:
            finger_tip_y = landmarks[lm_index].y
            finger_bottom_y =landmarks[lm_index-2].y

            if lm_index !=4 :
                if finger_tip_y < finger_bottom_y:
                    fingers.append(1)
                else:
                    fingers.append(0)   
        totalFingers = fingers.count(1)

def drawHandLandmarks(image,hand_Landmarks):
    if hand_Landmarks:
        for landmarks in hand_Landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)
    print(hand_Landmarks)

while True:
    sucess,image =cap.read()
    image = cv2.flip(image, 1)

    results = hands.process(image)

    if results.multi_hand_landmarks:
        hands_landmarks = results.multi_hand_landmarks

        drawHandLandmarks(image, hands_landmarks)
        countFingers(image, hands_landmarks)
    cv2.imshow("Video legal", image)

    key = cv2.waitKey(1)
    if key == 32:
        break
cv2.destroyAllWindows
