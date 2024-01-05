-- 스키마 생성 
CREATE SCHEMA `sound_db` ;

-- 해당 스키마 사용
use sound_db;

-- 사용자 추가
CREATE USER 'han'@'%' IDENTIFIED BY '0000';

-- 권한 부여
GRANT ALL PRIVILEGES ON mydb.* TO 'han'@'%';
FLUSH PRIVILEGES;

-- 테이블 삭제
DROP TABLE sound;

-- 테이블 생성  <- 이슈 발생전
CREATE TABLE sound(
	file_num VARCHAR(100) PRIMARY KEY,
    txt VARCHAR(200) NOT NULL,
    object VARCHAR(15) NOT NULL,
    sound_url VARCHAR(200),
    sound_file longblob,
    writer VARCHAR(10) NOT NULL,
    regist_day TIMESTAMP,
    update_day TIMESTAMP
) DEFAULT CHARSET = utf8mb4;
-- sound_file mediumblob

-- 테이블 생성 이슈 발생 후
CREATE TABLE sound(
	file_num VARCHAR(100) PRIMARY KEY,
    txt VARCHAR(200) NOT NULL,
    object VARCHAR(15) NOT NULL,
    sound_url VARCHAR(200),
    writer VARCHAR(10) NOT NULL,
    regist_day TIMESTAMP,
    update_day TIMESTAMP
);

-- 테스트 파일 - insert into
INSERT INTO sound VALUES ('1', '오른쪽 객체있습니다.', '객체', LOAD_FILE('D:/opencv_study/voice.mp3'), 'root', now(), now());
-- 이미지
INSERT INTO sound VALUES ('2', 'ㅁㅁㅁ', '객체', LOAD_FILE('D:/opencv_study/test12.jpg'), 'root', now(), now());
-- txt 파일
INSERT INTO sound VALUES ('3', 'ㅅㄷㄴㅅ', '객체', LOAD_FILE('D:/opencv_study/testtxt.txt'), 'root', now(), now());
-- 깃 링크
INSERT INTO sound VALUES ('12_24_01', '오른쪽 차 있습니다.', '차', 'https://github.com/yangjik/blackbox/raw/main/sound/voice.mp3', 'root', now(), now());

--  데이터 저장 위치 조회
show variables like 'datadir';

-- 파일 존재 확인(local pc)
SELECT load_file('D:/sound1.mp3');

-- 이미지로 테스트
SELECT load_file('D:/opencv_study/test12.jpg');

-- 데이터 삭제
delete from sound where file_num='1';

-- DB 조회
SELECT * FROM sound;

-- 파일 조회하기
select sound_url from sound where file_num='12_24_01';


-- 파일 다운
-- 음성
SELECT sound_file INTO OUTFILE 'D:/opencv_study/dbvoice.mp3' FROM sound;
-- 이미지
SELECT sound_file INTO OUTFILE 'D:/opencv_study/dbtest.jpg' FROM sound;
-- txt
SELECT sound_file INTO OUTFILE 'D:/opencv_study/dbtest.txt' FROM sound;