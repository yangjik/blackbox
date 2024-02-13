import os, shutil, cv2, sys, math

# 개인정보 - 사운드 제거시 필요
from moviepy.editor import VideoFileClip
# Autolabeling
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

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




#########################################
'''
    Part3 : Yolov8 train
'''
#########################################



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

