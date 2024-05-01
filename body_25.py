import cv2
import os 
import json



BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
    "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
    "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23, "RHeel": 24,
    "Background": 25
}


POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "MidHip"], ["MidHip", "RHip"], ["RHip", "RKnee"],
    ["RKnee", "RAnkle"], ["MidHip", "LHip"], ["LHip", "LKnee"],
    ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"],
    ["RHeel", "RBigToe"], ["RHeel", "RSmallToe"], ["LHeel", "LBigToe"],
    ["LHeel", "LSmallToe"]
]

# 각 파일 path
protoFile = "/Users/junghyunkim/HR-VITON/openpose_models/pose_deploy.prototxt"
weightsFile = "/Users/junghyunkim/HR-VITON/openpose_models/pose_iter_584000.caffemodel"
 
# 위의 path에 있는 network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 동영상 파일 불러오기
params = dict()
# params["model_folder"] = "C:\\Users\\DATATREE\\Downloads\\openpose-master\\models"
video_folder = "/Users/junghyunkim/HR-VITON/test/image"  # 비디오 파일들이 저장된 폴더
# video_folder = "C:\\Users\\DATATREE\\Downloads\\사람동작 영상\\비디오\\1_걷기\\1-1"
result_folder = "/Users/junghyunkim/HR-VITON/openpose_json"


for filename in os.listdir(video_folder):
    if filename.endswith(".jpg"):
        video_path = os.path.join(video_folder, filename)
        print("Processing video: ", video_path)

        params["video"] = video_path
        params["write_json"] = os.path.join(result_folder, os.path.splitext(filename)[0])

        # 비디오 읽기
        cap = cv2.VideoCapture(params["video"])
        if not cap.isOpened():
            print("Unable to open video file: ", video_path)
            continue

        while(cap.isOpened()):
            ret, image = cap.read()
            # frame_count +=1

            if not ret:
                break

            # if frame_count==1 or frame_count % 30 !=0:
            #     continue
        
            # frame.shape = 불러온 이미지에서 height, width, color 받아옴
            imageHeight, imageWidth, _ = image.shape

            # network에 넣기위해 전처리
            inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)

            # network에 넣어주기
            net.setInput(inpBlob)

            # 결과 받아오기
            output = net.forward()

            # output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
            
            H = output.shape[2]
            W = output.shape[3]
            #print("이미지 ID : ", len(output[0]), ", H : ", output.shape[2], ", W : ",output.shape[3]) # 이미지 ID

            # 키포인트 검출시 이미지에 그려줌
            points = []

            # json 용 포인트들
            points_json = []
            for i in range(0,25):
                # 해당 신체부위 신뢰도 얻음.
                probMap = output[0, i, :, :]

                # global 최대값 찾기
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                # 원래 이미지에 맞게 점 위치 변경
                x = (imageWidth * point[0]) / W
                y = (imageHeight * point[1]) / H

                # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
                if prob > 0.05 :    
                    cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)       # circle(그릴곳, 원의 중심, 반지름, 색)
                    cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                    points.append((int(x), int(y)))
                    points_json.append(x)
                    points_json.append(y)
                    points_json.append(prob)
                else :
                    points.append(None)
                    for _ in range(3):
                        points_json.append(0)

            
            cv2.imshow("Output-Keypoints",image)
            cv2.waitKey(1)

            json_data = {"pose_keypoints_2d": points_json}

            with open(params["write_json"] + "_keypoints" + '.json', 'w') as outfile:
                json.dump(json_data, outfile)

            # 이미지 복사
            imageCopy = image

            # 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
            for pair in POSE_PAIRS:
                partA = pair[0]             # Head
                partA = BODY_PARTS[partA]   # 0
                partB = pair[1]             # Neck
                partB = BODY_PARTS[partB]   # 1

                if points[partA] and points[partB]:
                    cv2.line(imageCopy, points[partA], points[partB], (0, 255, 0), 2)
            new_filename = filename.split(".")[0]
            cv2.imwrite(f'{new_filename}_rendered.png',imageCopy)
            cv2.imshow("Output-Keypoints",imageCopy)
            cv2.waitKey(1)
            

cap.release()
cv2.destroyAllWindows()
exit()