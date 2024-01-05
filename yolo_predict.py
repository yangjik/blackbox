from ultralytics import YOLO
import torch
from shapely.geometry import Polygon, Point

# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
path = './12_13_67_thumbnail.jpg'
results = model.predict(path, save=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs

# 클래스 id
object_type = []
for _ in boxes.cls:
    object_type.append(int(_))

# 객체의 좌표
yolo_point = []
for _ in boxes.xywh:
    center_x = int(_[0])
    center_y = int(_[1])
    width = int(_[2])
    hight = int(_[3])

    yolo_point.append((center_x, center_y, width, hight))




# 불필요한 클래스id 인덱스 받기
drop_index = []

# 0 : 사람, 1 : 자전거, 2 : 자동차
# 3 : 오토바이, 5 : 버스, 7 : 트럭
for _ in range(0, len(object_type)):
    if object_type[_] == 0 or object_type[_] == 1 or object_type[_] == 2 or object_type[_] == 3 or object_type[_] == 5 or object_type[_] == 7:
        pass
    else:
        drop_index.append(_)

# 불필요한 클래스 인덱스 삭제
# 인덱스 번호를 안 뒤집을 경우 앞에서부터 삭제하므로
# 뒤로갈수록 해당 인덱스 존재 안해서.
drop_index.reverse()
for _ in drop_index:
    object_type.pop(_)
    yolo_point.pop(_)


# 클래스id -> 한글로 변환
for _ in range(0, len(object_type)):
    if object_type[_] == 0:
        object_type[_] = '사람'

    elif object_type[_] == 1:
        object_type[_] = '자전거'

    elif object_type[_] == 2:
        object_type[_] = '자동차'

    elif object_type[_] == 3:
        object_type[_] = '오토바이'

    elif object_type[_] == 5:
        object_type[_] = '버스'

    elif object_type[_] == 7:
        object_type[_] = '트럭'       

# 사각형의 4개의 점 좌표 구하기.
# x1 : 왼쪽 상단, x2 : 오른쪽 상단
# x3 : 왼쪽 하단, x4 : 오른쪽 하단

# 하나의 클래스 기준 4개의 점 좌표 저장.
point = []

for raw in yolo_point:
    x = raw[0]
    y = raw[1]
    w = raw[2]
    h = raw[3]
    x1 = (int(x - (w/2)), int(y + (h/2)))
    x2 = (int(x + (w/2)), int(y + (h/2)))
    x3 = (int(x - (w/2)), int(y - (h/2)))
    x4 = (int(x + (w/2)), int(y - (h/2)))

    point.append((x1, x2, x3, x4))

# opencv식 워닝존 좌표
warning_point = [(270, 310), (12, 445), (610, 445), (383, 310)]

# 변환 워닝존 좌표
warning_point = [(12, 310), (270, 445), (383, 445), (610, 310)]

def warning_zone(point, polygon):
    """
    point: [x, y] 좌표 값을 가진 리스트
    polygon: 다각형의 꼭지점 좌표를 가진 리스트
    """

    # 점의 좌표를 사용하여 점의 객체를 만듭니다.
    point_obj = Point(point)

    # 다각형의 꼭지점 좌표를 사용하여 다각형의 경계를 구합니다.
    polygon_path = Polygon(polygon)

    # 점의 객체가 다각형의 경계 안에 있는지 확인합니다.
    return polygon_path.contains(point_obj)

# 워닝존
rectangle = [(12, 310), (270, 445), (383, 445), (610, 310)]

# 데이터 클래스 하나 4개의 점
# square = point

result_case = []
# 좌표가 사다리꼴 안에 있는지 확인합니다.
for square, result_type in zip(point, object_type):
    for p in square:
        point_obj = Point(p)
        if warning_zone(point_obj, rectangle):
            # print(_, "포함됨")
            # 조건문
            if p[0] < int((rectangle[3][0] - rectangle[0][0]) / 2):
                result_txt = '왼쪽'
                print(f'{result_txt}에 {result_type}있습니다.')
                result_case.append([result_txt, result_type])
                break
            elif int((rectangle[3][0] - rectangle[0][0]) / 2) < p[0]:
                result_txt = '오른쪽'
                result_case.append([result_txt, result_type])
                print(f'{result_txt}에 {result_type}있습니다.')
                break
        else:
            pass
            # print("포함되지 않음")

result_case