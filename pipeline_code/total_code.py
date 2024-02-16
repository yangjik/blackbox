import os, shutil, cv2, sys, math, playsound, threading

# 개인정보 - 사운드 제거시 필요
from moviepy.editor import VideoFileClip
# Autolabeling
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
# Yolo predcit and train
from ultralytics import YOLO
# 워닝존 알고리즘 사다리꼴
from shapely.geometry import Polygon, Point
# gTTS 생성
from gtts import gTTS
import numpy as np
import pandas as pd
# 윈도우 내장 사운드
import winsound as wsd
import matplotlib.pyplot as plt

#########################################
'''
    Part1 : 디렉토리 만들기 및 데이터 생성
'''
#########################################
# 디렉토리만들기 최초 실행 필요
def make_dir(ls_dir):
    for _ in ls_dir:
        # 폴더 존재시 내부 폴더 데이터 삭제 후 폴더 생성.
        if os.path.exists(_) == True:
            shutil.rmtree(_)
            name = _.split('/')[1]
            print(f'기존 {name} 폴더 삭제')

            os.mkdir(_)
            print(f'폴더 삭제 후 {name} 폴더 생성\n')

        # 폴더 없을 땐 생성.
        else:
            os.mkdir(_)
            name = ls_dir[0].split('/')[1]
            print(f'{name} 폴더 생성\n')

# 동영상 원본 있는 디렉토리 설정하면 
def mk_nosound_resize(video_path):

    # 사운드제거 후 저장할 디렉토리
    nosound_dir = '/'.join(video_path.split('/')[:3]) + '/nosound/'
    if not os.path.exists(nosound_dir):
        print('nosound 폴더 생성.')
        os.mkdir(nosound_dir)
    else:
        pass
    
    # resize 후 저장할 디렉토리
    blackbox_dir = video_path.replace('video', 'blackbox')
    blackbox_dir = '/'.join(blackbox_dir.split('/')[:3]) + '/'
    if not os.path.exists(blackbox_dir):
        print('blackbox 폴더 생성.')
        os.mkdir(blackbox_dir)
    else:
        pass 
    
    # 파일 저장시 m_d
    day = video_path.split('/')[2]

    # 해당경로 파일 읽기
    ls_org = os.listdir(video_path)
   
    num = 1
    for _ in range(0, len(ls_org)):
        # 사운드 제거 파일
        nosound_path = f'./video/{day}/nosound/nosound_' + day.replace('.', '_') + f'_{num}.mp4'
        # resize 동영상
        save_path = f'./blackbox/{day}/' + day.replace('.', '_') + f'_{num}.mp4'
        # video 원본
        org_path = video_path + ls_org[_]
        # 파일명 숫자 붙이기
        num += 1

        # nosound, resize동영상 만듬.
        convert_video(org_path, nosound_path, save_path)

# make_nosound_path 이 함수 실행 하면 밑에 함수 실행.
# 이 함수는 개인정보때문에 nosound 한 후 동영상 resize 진행.
def convert_video(org_path, nosound_path ,save_path):

    # 사운드 제거 할 동영상 읽기.
    videoclip = VideoFileClip(org_path)

    # 사운드 제거.
    new_clip = videoclip.without_audio()
    
    # 사운드 제거 파일 저장.
    new_clip.write_videofile(nosound_path)

    # 동영상 읽기 <-- 사운드 제거된 영상
    cap = cv2.VideoCapture(nosound_path)

    # 동영상 프레임
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # 저장할 동영상 - 소리 제거 안되어있음
    out_cap = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), cap.get(cv2.CAP_PROP_FPS), (640, 640))

    # 동영상 프레임 반복
    while cap.isOpened():
        # 프레임 읽기
        ret, frame = cap.read()

        # 프레임 읽기 성공 여부 확인
        if not ret:
            # 다음 프레임 없음
            break

        # 프레임 크기 변경
        frame = cv2.resize(frame, (640, 640))

        # 프레임 출력
        out_cap.write(frame)

        # 키 입력 대기
        k = cv2.waitKey(1)
        if k == 27:
            break

    # 캡처 종료
    cap.release()
    out_cap.release()


# 동영상 파일 있는 폴더 지정.
# ex) './blackbox/1.27/'
def mk_video_img(video_path):

    # 해당 경로 비디오파일 리스트 가져오기
    ls_video = os.listdir(video_path)

    # 이미지 저장할 디렉토리정보 가져오기
    day = video_path.split('/')[2]
    save_dir = f'./auto/images/{day}/'

    # mm.dd 폴더 없으면 만들기
    if not os.path.exists(save_dir):
        print(f'{day} 폴더 생성')
        os.mkdir(save_dir)

    # video_fps_mkimg 함수를 통해서 프레임별 이미지 만들기
    for _ in ls_video:
        
        if not video_path.endswith('/'):
            video_path = video_path + '/'
        
        org_path = f'{video_path}{_}'
        
        video_fps_mkimg(org_path, save_dir)

# 예시
# video_path = './1_20_1.mp4'
# save_dir = './image/'
# fps = 70
def video_fps_mkimg(video_path, save_dir, fps=70):

    file_name = video_path.split('.')[1].split('/')[1]

    # 동영상 파일 읽기
    video = cv2.VideoCapture(video_path)

    # 동영상이 안열리면 해당 코드 정지.
    if not video.isOpened():
        print(f'동영상 경로 및 파일 확인해주세요. {video_path}')
        sys.exit()

    # 동영상에서 저장될 이미지
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print(save_dir, ' 폴더 생성 완')

    # 동영상의 전체 프레임 확인
    print('현재 동영상 total frame : ', video.get(cv2.CAP_PROP_FRAME_COUNT))

    count = 0
    save_count = 0  # 이미지 저장할때 파일명에 붙음.

    # 동영상 읽고 프레임별 저장
    while video.isOpened():

        ret, img = video.read()

        # 다음 프레임 없으면 중단
        if not ret:
            break

        if count % fps == 0:
            img_save = f'{save_dir}{file_name}_{save_count}.jpg'
            cv2.imwrite(img_save, img)
            save_count += 1
            
        count += 1

    video.release()


#########################################
'''
    Part2 : Auto Labeling
'''
#########################################

# grounding-dino 오토라벨링
def label_dino(org_img_path):

    # 라벨 지정 <- 추후에 변수로 수정가능.
    ontology=CaptionOntology({
        "person": "person",
        "bicycle": "bicycle",
        'car' : 'car',
        'motorcycle' : 'motorcycle',
        'bus' : 'bus',
        'truck' : 'truck' 
    })

    base_model = GroundingDINO(ontology=ontology)

    # 오토 라벨링 진행
    try:
        base_model.label(
            input_folder = org_img_path,
            extension=".jpg",
            output_folder ='./dino_dataset/')
    except:
        pass

# coco data
def label_custom(img_path, model='yolov8n.pt'):

    # 기존에 predict 폴더가 있으면 삭제.
    if os.path.exists('./runs/detect/predict/') == True:
        shutil.rmtree('./runs/detect/predict/')

    # img_path 경로가 './auto/images/1.27' 이런 형식이 아닐때. '/'맨마지막에 붙인다.
    if not img_path.endswith('/'):
        img_path = img_path + '/'

    # 라벨명 지정 <- 여기서 변수로 받을 수 있음
    we_want = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']

    model = YOLO(model)

    img_dir = os.listdir(img_path)

    # 경로 + 파일명 리스트 만들기.
    path = []
    for _ in img_dir:
        path.append(img_path + _)

    for _ in path:
        model.predict(_, save=True, save_txt=True, imgsz=640)
        yolo_id = model.names

    # txt 파일 리스트 받아오기.
    ls_txt = os.listdir('./runs/detect/predict/labels/')

    for ls in ls_txt:
        
        txt_path = './runs/detect/predict/labels/' + ls

        # txt파일 읽기
        with open(txt_path, 'r', encoding='utf-8') as f:
            txt = f.readlines()
    
        now_index = 0       # 초기 인덱스
        drop_index = []     # 삭제할 인덱스 <- txt파일 정보 받아올때는 리스트로 받아와서
        for _ in txt:

            # yolo_id <- 클래스id 에서 우리가 필요하지 않는 id들은 삭제
            a = yolo_id[int(_.split(' ')[0])]

            # 우리가 원하는 라벨 리스트에서 있는지 확인
            if a in we_want:
                # print(a, '는 있습니다.')
                now_index += 1
            elif a not in we_want:
                # print(a, '는 없습니다.')
                drop_index.append(now_index)
                now_index += 1

        # 뒤집는 이유는 기존 파일 역순으로 삭제해야 문제 발생안함.
        # 상세한 설명은 리스트.. 슬라이싱 검색하면 이해가 됨.
        drop_index.reverse()

        # 위에서 얻은 인덱스로 txt정보 삭제
        for _ in drop_index:
            txt.pop(_)

        # 밑에 정보가 최종적으로 txt파일로 저장될 내용.
        re_txt = []
        for _ in txt:

            # 클래스id는 맨앞이므로
            _ = _.split(' ')
            
            # coco는 5 : 버스, 7 : 트럭 -> 4 : 버스, 5 : 트럭 으로 변경
            if _[0] == '5':
                _[0] = '4'
            elif _[0] == '7':
                _[0] = '5'
            re_txt.append(' '.join(_))
            
        # re_txt 최종 저장될 내용 작성
        with open(txt_path, 'w') as f:
            f.writelines(re_txt)


# txt파일, 이미지 파일(단일 파일)을 이용해서 시각화 
def show_label(txt_path, img_path):

    # 원본 txt 파일읽기 - 비율
    with open(txt_path, 'r', encoding='utf-8') as f:
        txt_org = f.readlines()

    # 원본 읽고 불필요 문자 '\n' 제거
    txt_re = []
    for _ in txt_org:
        txt_re.append(_.replace('\n', ''))

    # 이미지 읽기 - 여기서 좌표 변환 후 박스 그리기
    img = cv2.imread(img_path)

    if img is None:
        raise print('이미지 없어 경로 다시확인')

    img_w, img_h, img_c = img.shape

    # 박스 그리는 위치
    for _ in txt_re:
        class_id, x, y, w, h = _.split(' ')
        class_id, x, y, w, h = int(class_id), float(x), float(y), float(w), float(h)

        # yolo식 좌표
        re_x, re_y, re_w, re_h = int(x * img_h), int(y * img_w), int(w * img_h), int(h * img_w)

        # opencv 식 좌표
        x1 = (int(re_x - (re_w/2)), int(re_y + (re_h/2)))
        x2 = (int(re_x + (re_w/2)), int(re_y + (re_h/2)))
        x3 = (int(re_x - (re_w/2)), int(re_y - (re_h/2)))
        x4 = (int(re_x + (re_w/2)), int(re_y - (re_h/2)))

        # print(x1, x4)

        # 클래스 보고 색상변경
        if class_id == 0:
            # 빨 : 사람
            cv2.rectangle(img=img, pt1=x1, pt2=x4, color=(0, 0, 255), thickness=3)

        elif class_id == 1:
            # 파 : 자전거
            cv2.rectangle(img=img, pt1=x1, pt2=x4, color=(255, 0, 0), thickness=3)

        elif class_id == 2:
            # 녹 : 차
            cv2.rectangle(img=img, pt1=x1, pt2=x4, color=(0, 255, 0), thickness=3)

        elif class_id == 3:
            # 핑크 : 오토바이
            cv2.rectangle(img=img, pt1=x1, pt2=x4, color=(255, 0, 255), thickness=3)

        elif class_id == 4:
            # 주황 : 버스
            cv2.rectangle(img=img, pt1=x1, pt2=x4, color=(255, 102, 0), thickness=3)

        elif class_id == 5:
            # 노랑 : 트럭
            cv2.rectangle(img=img, pt1=x1, pt2=x4, color=(255, 0, 255), thickness=3)

    # 클래스 개수 적어두기
    cv2.putText(img=img, text=f'total : {len(txt_re)}', org=(0,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0,0,255), thickness=3)

    cv2.imshow('label', img)

    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()

#########################################
'''
    Part3 : Yolov8 train
'''
#########################################

# 학습 전 클래스 분포도 확인
# txt파일 있는 폴더 지정
def class_pie_eda(txt_dir):
    ls_coco = os.listdir(txt_dir)

    coco_path = []
    for _ in ls_coco:
        coco_path.append(txt_dir + _)

    class_id = []
    label = []

    for _ in coco_path:
        with open(_, 'r', encoding='utf-8') as f:
            txt = f.readlines()
        
        for _ in txt:
            # _.split(' ')[0] 는 txt파일에서 라벨
            class_id.append(_.split(' ')[0])
            if _.split(' ')[0] == '0':
                label.append('person')

            elif _.split(' ')[0] == '1':
                label.append('bicycle')

            elif _.split(' ')[0] == '2':
                label.append('car')

            elif _.split(' ')[0] == '3':
                label.append('motorcycle')

            elif _.split(' ')[0] == '4':
                label.append('bus')

            elif _.split(' ')[0] == '5':
                label.append('truck')            

    data = {
        'num' : class_id,
        'label' : label
    }

    df = pd.DataFrame(data)
    value = list(df.label.value_counts().sort_index().values)
    index = list(df.label.value_counts().sort_index().index)
    explode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

    plt.title(f'data : {len(ls_coco)}')
    plt.pie(value, labels=index, explode=explode, autopct='%.1f%%', shadow=True)
    plt.show()

#########################################
'''
    Part4 : Auto labeling 
'''
#########################################

# check_txt 이 함수를 만들게 된 배경은 predict 한 결과 txt파일이 없는 경우도 많아서 그땐 사람 검수 필요.

# os.listdir 해서 나온 리스트 값만 필요
# correct_ls = 정답 txt리스트
# predict_ls = predict 한 txt 리스트
# copy_path = 이미지 원본있는 폴더dir <- 버전이 계속 바뀌다보니.. 지정필요
# ex) ./datas_v4/test/images/
def check_txt(correct_ls, predict_ls, org_img_path):
    check = []
    
    # copy_path 경로가 './datas_v4/test/images/' 이런 형식이 아닐때. '/'맨마지막에 붙인다.
    if not org_img_path.endswith('/'):
        org_img_path = org_img_path + '/'

    # prdict txt파일 없는거 파악
    for _ in correct_ls:
        
        if _ not in predict_ls:
            check.append(_)

    # txt파일이 동일하므로 종료
    check = []
    if len(check) == 0:
        sys.exit('txt파일개수 동일합니다.')
    
    for _ in check:
        img_copy = org_img_path + _.replace('.txt', '.jpg')
        shutil.copy(img_copy, './human_check/img/')


# prdict txt파일이랑 test txt파일 비교해서 동일하지 않으면 human_check 폴더로 이동.
# 이미지는 원본이 아니라 predict 한 결과 이미지 이동.
# ex) one_path = './runs/detect/predict3/labels/MP_KSC_000041.txt'
#     two_path = './test_v4/test/labels/MP_KSC_000041.txt'
def model_verify(predict_path, correct_path, error_rate=5):

    file_name = predict_path.split('/')[-1]
    predict_img = '/'.join(predict_path.split('/')[:-2]) + '/' + file_name.replace('txt', 'jpg')

    # 첫번째 프레임
    with open(predict_path, 'r', encoding='utf-8') as f:
        one_txt = f.readlines()
    one_txt = sorted(one_txt, key=lambda x: float(x.split(' ')[0]))

    # 두번째 프레임
    with open(correct_path, 'r', encoding='utf-8') as f:
        two_txt = f.readlines()
    two_txt = sorted(two_txt, key=lambda x: float(x.split(' ')[0]))

    if len(one_txt) == len(two_txt):
        for one, two in zip(one_txt, two_txt):
            false_count = 0
         
            # txt 파일 정보
            one_cl = one.split(' ')[0]
            one_x = float(one.split(' ')[1])
            one_y = float(one.split(' ')[2])
            one_w = float(one.split(' ')[3])
            one_h = float(one.split(' ')[4])

            two_cl = two.split(' ')[0]
            two_x = float(two.split(' ')[1])
            two_y = float(two.split(' ')[2])
            two_w = float(two.split(' ')[3])
            two_h = float(two.split(' ')[4])

            # 클래스명이 같을때
            if one_cl == two_cl:

                # 오차율 공식
                x_acc = int(abs(two_x - one_x) / one_x * 100)
                y_acc = int(abs(two_y - one_y) / one_y * 100)
                w_acc = int(abs(two_w - one_w) / one_w * 100)
                h_acc = int(abs(two_h - one_h) / one_h * 100)

                if x_acc <= error_rate and y_acc <= error_rate and w_acc <= error_rate and h_acc <= error_rate: 

                    # print('동일합니다.')
                    # print(x_acc, y_acc, w_acc, h_acc)
                    pass
                               
                else:
                    # print('다릅니다.')
                    # print(x_acc, y_acc, w_acc, h_acc)
                    false_count += 1              

        # txt 파일 한 행씩 비교 후 same_count가 0이면 동일한 이미지가 아니므로 카피.
        if false_count > 0:   
            shutil.copy(predict_img, './human_check/img/')
            shutil.copy(predict_path, './human_check/label/')
            shutil.copy(predict_path, './human_check/correct_label/')

    else:
        shutil.copy(predict_img, './human_check/img/')
        shutil.copy(predict_path, './human_check/label/')
        shutil.copy(correct_path, './human_check/correct_label/')



#########################################
'''
    Part4 : Supervision and Warning zone and gTTS
'''
#########################################

# 한글을 영어로 변환함수
def en_to_ko(result_case):

    # 방향 한글 -> 영어
    way = result_case[0][0]
    if way == '오른쪽':
        en_way = 'right'
    elif way == '왼쪽':
        en_way = 'left'
    
    # 객체 한글 -> 영어
    id = result_case[0][1]
    if id == '사람':
        en_id = 'person'
    elif id == '자전거':
        en_id = 'bicycle'
    elif id == '자동차':
        en_id = 'car'
    elif id == '오토바이':
        en_id = 'motorcycle'
    elif id == '자전거':
        en_id = 'person'
    elif id == '버스':
        en_id = 'bus'
    elif id == '트럭':
        en_id = 'truck'
    
    return en_way, en_id

# 비프음
def play_beep():
    wsd.Beep(2000, 500)
    # print('beep 쓰레드 종료')
# tts
def play_sound(path):
    try:
        playsound.playsound(path)
        # print('tts 쓰레드 종료')
    except:
        pass
# 워닝존 in 탐지
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
    result = polygon_path.contains(point_obj)
    # print(result)
    # print('워닝존 : ', polygon)
    return result

t_gtts = None
t_beep = None
def warning(model, video_path):

    global t_gtts
    global t_beep

    model = YOLO(model)

    video = cv2.VideoCapture(video_path)

    if not video:
        raise(f'동영상 못 읽지 못했습니다.\n경로는 {video_path} 입니다.')

    # 비디오 정보
    video_w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # 왼쪽 위 [0], 왼쪽 밑 [1], 오른쪽 위 [3], 오른쪽 밑 [2]
    # rectangle = [(180, 480), (20, 570), (620, 570), (480, 480)]
    # rectangle = [(180, 480), (70, 570), (570, 570), (480, 480)]

    roatio_yolo = [[0.28125, 0.75], [0.109375, 0.890625], [0.890625, 0.890625], [0.75, 0.75]]

    # 워닝존 비율 좌표로 변경 <- 640 x 640 일때 워닝존 좌표이므로
    # 변수가 있을때
    if 'rectangle' in locals():    
        ratio_point = []
        for _ in rectangle:
            x, y = _[0], _[1]
            convert_x = x / video_w
            convert_y = y / video_h
            ratio_point.append([convert_x, convert_y])
    else:
        pass

    # warning_zone 함수에는 해당 비율 좌표가 아닌 절대 좌표가 필요하므로 convert 작업 필요.
    if 'ratio_point' in locals():
        abs_point = []
        for _ in ratio_point:
            x, y = _[0], _[1]

            convert_x = int(x * video_w)
            convert_y = int(y * video_h)
            abs_point.append([convert_x, convert_y])
    else:
        pass

    if 'roatio_yolo' in locals():
        abs_point = []
        for _ in roatio_yolo:
            x, y = _[0], _[1]

            convert_x = int(x * video_w)
            convert_y = int(y * video_h)
            abs_point.append([convert_x, convert_y])
    else:
        pass

    while True:
        # 동영상 정보 가져옴
        ret, frame = video.read()

        # 동영상 다음 프레임이 없다는것이므로 종료
        if not ret:
            break
        
        results = model.predict(frame)

        # yolo식 클래스id
        yolo_class = model.names

        # box 정보 리스트화 <- 여기서 나오는 데이터는 텐서배열
        ls_point = []
        ls_id = []
        for _ in results:
            # box 정보 얻기
            for a, b in zip(_.boxes.xywh.tolist(), _.boxes.cls.tolist()):
                ls_point.append(a)
                ls_id.append(b)

        # 불필요한 인덱스 파악
        drop_index = []
        index = 0
        for _ in ls_id:
            id = int(_)
            if id == 0 or id == 1 or id == 2\
                or id == 3 or id == 5 or id == 7:
                pass
            else:
                drop_index.append(index)
            index += 1
        # 불필요한 클래스가 있으면 삭제 해야하므로 인덱스는 뒤집기
        drop_index.reverse()

        if len(drop_index) > 0:
        # 불필요한 인덱스 삭제
            for _ in drop_index:
                ls_point.pop(_)
                ls_id.pop(_)

        # 필요한 정보 <- 비율 좌표가 아닌 절대좌표
        re_id = []
        rectangle_point = []
        for one, two in zip(ls_point, ls_id):

            # class_id
            id = int(two)
            # 변경하는 이유는 gtts에서 한글로 변환해야되서.
            if id == 0:
                id = '사람'
            elif id == 1:
                id = '자전거'
            elif id == 2:
                id = '자동차'
            elif id == 3:
                id = '오토바이'
            elif id == 5:
                id = '버스'
            elif id == 7:
                id = '트럭'     
            re_id.append(id)

            # yolo식 절대좌표
            cen_x = int(one[0])
            cen_y = int(one[1])
            w = int(one[2])
            h = int(one[3])

            # 사각형의 4개의 점 좌표 구하기.
            # x1 : 왼쪽 상단, x2 : 오른쪽 상단
            # x3 : 왼쪽 하단, x4 : 오른쪽 하단
            x1 = (int(cen_x - (w/2)), int(cen_y + (h/2)))
            x2 = (int(cen_x + (w/2)), int(cen_y + (h/2)))
            x3 = (int(cen_x - (w/2)), int(cen_y - (h/2)))
            x4 = (int(cen_x + (w/2)), int(cen_y - (h/2)))
            
            rectangle_point.append((x1, x2, x3, x4))

        result_case = []

        # 좌표가 사다리꼴 안에 있는지 확인합니다.
        detection_left = 0
        detection_right = 0
        for point, result_type in zip(rectangle_point, re_id):
            x1 = point[0]
            x4 = point[3]

            for p in point:
                point_obj = Point(p)
                # yolo 박스 전체 시각화 (빨강색)
                cv2.rectangle(frame, x1, x4, color=(0,0,255), thickness=2)
                if warning_zone(point_obj, abs_point):
                    object_in = True
                    object_x = p[0]
                    zone_x = int((abs_point[2][0] + abs_point[1][0]) / 2)   # 점과 점사이의 중심
                    # 조건문
                    if object_x < zone_x:
                        # 워닝존 들어온 박스만 시각화(녹색)
                        cv2.rectangle(frame, x1, x4, color=(0,255,0), thickness=2)
                        detection_left += 1
                        result_txt = '왼쪽'
                        # print(f'{result_txt}에 {result_type}')
                        result_case.append([result_txt, result_type])
                        break
                    elif object_x > zone_x:
                        cv2.rectangle(frame, x1, x4, color=(0,255,0), thickness=2)
                        detection_right += 1
                        result_txt = '오른쪽'
                        # print(f'{result_txt}에 {result_type}')
                        result_case.append([result_txt, result_type])
                        break
                else:  
                    pass

        # center puttxt
        if detection_left > 0 and detection_right > 0:
            txt_point = (int(video_w * (2/5)), int(video_h * (1/3)))
            cv2.putText(frame, 'both Warning', org=txt_point, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

        elif detection_left > 0 and detection_right == 0:
            txt_point = (int(video_w * (1/10)), int(video_h * (1/3)))
            cv2.putText(frame, 'left Warning', org=txt_point, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

        elif detection_left == 0 and detection_right > 0:
            txt_point = (int(video_w * (2/3)), int(video_h * (1/3)))
            cv2.putText(frame, 'right Warning', org=txt_point, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
        
        detection_left = 0
        detection_right = 0

        zone = [np.array(abs_point)]
        cv2.polylines(frame, zone, True ,(255,255,255), thickness=3)
        cv2.imshow("YOLOv8", frame)

        key = cv2.waitKey(60)

        if key == 27:
            break

        # 동영상에서 워닝존안에 객체가 안들어오면 result_case의 정보는 없다 
        if len(result_case) == 0:
            pass
        else:
            txt = result_case[0][0] + ' ' + result_case[0][1]
            way, id = en_to_ko(result_case)
            path = f'./blackbox_sound/{way}_{id}.mp3'

            # 해당 파일이 없으면 비프음
            if os.path.exists(path) == False:
                # 파이썬 내장 사운드 삐이익 재생
                if t_beep !=None and t_beep.is_alive():
                    continue
                else:
                    t_beep = threading.Thread(target=play_beep)
                    t_beep.start()
                # tts = gTTS(txt, lang='ko')
                # tts.save(path)
            else:
                if t_gtts !=None and t_gtts.is_alive():
                    continue
                else:
                    t_gtts = threading.Thread(target=play_sound, args=(path, ))
                    t_gtts.start()

    cv2.destroyAllWindows()