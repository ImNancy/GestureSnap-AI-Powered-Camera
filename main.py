import cv2
import mediapipe as mp
import numpy as np
import time


class GestureCam:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

        self.cap = cv2.VideoCapture(0)
        self.zoom_factor = 1.0
        self.zoom_speed = 0.05
        self.photo_count = 0
        self.countdown_start = 0
        self.countdown_duration = 5

    def zoom_image(self, image, factor):
        h, w = image.shape[:2]
        crop_h, crop_w = int(h / factor), int(w / factor)
        y_start, x_start = (h - crop_h) // 2, (w - crop_w) // 2
        cropped = image[y_start:y_start + crop_h, x_start:x_start + crop_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def detect_gesture(self, hand_landmarks):
        landmarks = hand_landmarks.landmark
        fingers = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        finger_states = [landmarks[finger].y < landmarks[finger - 2].y for finger in fingers]

        # Victory sign (only index and middle fingers up)
        if finger_states[1] and finger_states[2] and not finger_states[3] and not finger_states[4]:
            return "victory"
        # Open hand (all fingers up)
        elif all(finger_states[1:]):
            return "open"
        # Closed hand (no fingers up)
        elif not any(finger_states[1:]):
            return "closed"

        return "none"

    def run(self):
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.flip(image, 1)  # Mirror the image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = self.hands.process(image_rgb)

            image = self.zoom_image(image, self.zoom_factor)

            gesture = "none"
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    gesture = self.detect_gesture(hand_landmarks)

            current_time = time.time()

            if gesture == "victory" and self.countdown_start == 0:
                self.countdown_start = current_time
                cv2.putText(image, "Victory Sign Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif gesture == "open":
                self.zoom_factor = max(1.0, self.zoom_factor - self.zoom_speed)
                cv2.putText(image, "Zooming Out", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif gesture == "closed":
                self.zoom_factor = min(3.0, self.zoom_factor + self.zoom_speed)
                cv2.putText(image, "Zooming In", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if self.countdown_start > 0:
                elapsed_time = current_time - self.countdown_start
                if elapsed_time < self.countdown_duration:
                    remaining_time = int(self.countdown_duration - elapsed_time)
                    cv2.putText(image, f"Capturing in: {remaining_time}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    self.photo_count += 1
                    cv2.imwrite(f"victory_photo_{self.photo_count}.jpg", image)
                    print(f"Photo captured! (Total: {self.photo_count})")
                    self.countdown_start = 0

            cv2.putText(image, f"Zoom: {self.zoom_factor:.2f}x", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Gesture-Controlled Camera', image)

            if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = GestureCam()
    cam.run()