import cv2.cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture('Videos/1.mp4')


mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec( thickness=1, circle_radius=1, color=(0,255,0))
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
'''
def __init__(self,
               static_image_mode=False,
               max_num_faces=1,
               refine_landmarks=False,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5)
'''

pTime = 0
while True:
    success, image = cap.read()
    imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    # Frame rate dramatically reduces once the results line is put in
    if results.multi_face_landmarks:
        for landmark in results.multi_face_landmarks:
            # print(id,landmark)
            mpDraw.draw_landmarks(image,landmark,mpFaceMesh.FACEMESH_CONTOURS,drawSpec,drawSpec)
            for id,lm in enumerate(landmark.landmark):
                # print(lm)
                h,w,c = image.shape
                cx, cy= int(lm.x*w), int(lm.y*h)
                print(f'ID:{id}\nCoordinates:\nx-coordinate:{cx}\ny-coordinate:{cy}\n\n')
    '''
    def draw_landmarks(
        image: np.ndarray,
        landmark_list: landmark_pb2.NormalizedLandmarkList,
        connections: Optional[List[Tuple[int, int]]] = None,
        landmark_drawing_spec: Union[DrawingSpec,
                                     Mapping[int, DrawingSpec]] = DrawingSpec(
                                         color=RED_COLOR),
        connection_drawing_spec: Union[DrawingSpec,
                                       Mapping[Tuple[int, int],
                                               DrawingSpec]] = DrawingSpec()):
    '''
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(image, f'FPS:{int(fps)}', (30,50), cv.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

    cv.imshow("Image", image)
    cv.waitKey(5)
