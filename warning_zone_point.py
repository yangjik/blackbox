import json

# json 파일 로드
with open('./PT.json', 'r') as file:
    data = json.load(file)

# json파일에 있는 좌표값 추출
warning_points = []
for shape in data["shapes"]:
    warning_points = shape['points']

# warning zone 좌표 변환 함수 설정
def convert_coordinates(warning_points):
    Transformed_warning_points = []
    # Warning zone coordinates are rounded
    for point in warning_points:
        x = int(round(point[0], 0))
        y = int(round(point[1], 0))
        Transformed_warning_points.append((x, y))

    # 각 좌표의 x와 y 값을 분리하여 별도의 변수에 할당
    x1, y1 = Transformed_warning_points[0]
    x2, y2 = Transformed_warning_points[1]
    x3, y3 = Transformed_warning_points[2]
    x4, y4 = Transformed_warning_points[3]

    # y1, y4 와 y2, y3 값을 비교해서 더 큰 값으로 통일
    max_y1_y4 = max(y1, y4)
    max_y2_y3 = max(y2, y3)

    # x1과 x2, x3와 x4의 x 좌표를 바꿈
    Transformed_warning_points[0] = (x2, y1)
    Transformed_warning_points[1] = (x1, y2)
    Transformed_warning_points[2] = (x4, y3)
    Transformed_warning_points[3] = (x3, y4)


    Transformed_warning_points[0] = (Transformed_warning_points[0][0], max_y1_y4)    
    Transformed_warning_points[1] = (Transformed_warning_points[1][0], max_y2_y3)
    Transformed_warning_points[2] = (Transformed_warning_points[2][0], max_y2_y3)
    Transformed_warning_points[3] = (Transformed_warning_points[3][0], max_y1_y4)

    return Transformed_warning_points



print(convert_coordinates(warning_points))


# 최종 워닝존 좌표
result_points = {
    '민감': ([[0, 280], [260, 480], [400, 480], [640, 280]]),
    '보통': ([[20, 290], [280, 470], [380, 470], [620, 290]]),
    '둔한': ([[40, 300], [300, 460], [360, 460], [600, 300]])
}

result_points