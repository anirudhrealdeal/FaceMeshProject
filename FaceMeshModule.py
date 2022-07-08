# Module to detect 468 points on a face
import cv2.cv2 as cv
import mediapipe as mp
import time
# id numbers can be gound in  mediapipe papers
class FaceMesh():
    def __init__(self, mode=False, maxNum=2, refine=False, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.maxNum = maxNum
        self.refine = refine
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.mode, self.maxNum, self.refine,
                                                 self.detectionCon, self.trackingCon)

    def findFaceMesh(self, image, draw=True):
        imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        # Frame rate dramatically reduces once the results line is put in
        faces = []
        if self.results.multi_face_landmarks:

            for landmark in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, landmark, self.mpFaceMesh.FACEMESH_CONTOURS,
                                           self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(landmark.landmark):
                    # print(lm)
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id== 150 or id==6:
                        cv.putText(image, str(id), (cx, cy), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                    # print(f'ID:{id}\nCoordinates:\nx-coordinate:{cx}\ny-coordinate:{cy}\n\n')
                    face.append([id,cx,cy])
                faces.append(face)
        return image,faces






def main():
    cap = cv.VideoCapture('Videos/1.mp4')
    pTime = 0
    detector = FaceMesh()
    while True:
        success, image = cap.read()
        image, faces = detector.findFaceMesh(image)
        if len(faces)!=0:
            print(len(faces),faces) #Prints no of faces along with the landmarks in each face
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(image, f'FPS:{int(fps)}', (30, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv.imshow("Image", image)
        cv.waitKey(5)


if __name__ == '__main__':
    main()