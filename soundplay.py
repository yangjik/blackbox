from gtts import gTTS
import playsound
import pymysql
import os
import subprocess

conn = pymysql.connect(
    host = 'localhost',
    user = 'root',
    password = '0000',
    database = 'sound_db',
    charset = 'utf8'
)
cur = conn.cursor()

def get_docker_data(di, ob):
    case = f'{di} {ob} 있습니다.'
    query = 'select docker_data from sound where txt=\"{}\";'.format(case)
    cur.execute(query)
    data = cur.fetchall()
    if data:
        docker_data = data[0][0]
        return docker_data, case
    else:
        return None, case

def get_dieg_obeg(result_case):
    if result_case[0] == "왼쪽":
        dieg = "left"
    elif result_case[0] == "오른쪽":
        dieg = "right"

    if result_case[1] == "사람":
        obeg = "person"
    elif result_case[1] == "자전거":
        obeg = "bicycle"
    elif result_case[1] == "자동차":
        obeg = "car"
    elif result_case[1] == "오토바이":
        obeg = "motorcycle"
    elif result_case[1] == "버스":
        obeg = "bus"
    elif result_case[1] == "트럭":
        obeg = "truck"

    return dieg, obeg

def docker_download_file(docker_data, download_path, case, dieg, obeg):
    if os.path.exists(download_path):
        return

    if docker_data:
        subprocess.run(["docker", "run", "--name", "blackboxsound", "sangho011/blackboxsound:blackbox"])
        subprocess.run(["docker", "cp", f"blackboxsound:/blackboxsound/{dieg}{obeg}.mp3", "../blackboxsound"])
        subprocess.run(["docker", "rm", "blackboxsound"])

    else:
        txt_path = case
        tts = gTTS(text=txt_path, lang='ko')
        tts.save(download_path)
        query = 'UPDATE sound SET docker_data = \'O\', writer = \'han\', update_day = now() WHERE txt =\"{}\";'.format(case)
        cur.execute(query)
        conn.commit()

def download_file(dieg, obeg, docker_data, case):
    download_path=f"../blackboxsound/{dieg}{obeg}.mp3"
    docker_download_file(docker_data, download_path, case, dieg, obeg)
    playsound.playsound(download_path)

def play(result_case):
    di = result_case[0]
    ob = result_case[1]
    dieg, obeg = get_dieg_obeg(result_case)
    docker_data, case = get_docker_data(di, ob)
    download_file(dieg, obeg, docker_data, case)