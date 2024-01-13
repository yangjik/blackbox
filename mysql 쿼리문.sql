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

-- 테이블 생성 이슈 발생 후
CREATE TABLE sound(
    txt VARCHAR(200) NOT NULL,
    object VARCHAR(15) NOT NULL,
    docker_data VARCHAR(200),
    writer VARCHAR(10) NOT NULL,
    regist_day TIMESTAMP,
    update_day TIMESTAMP
);

-- 깃 링크
INSERT INTO sound VALUES ('12_24_01', '오른쪽 차 있습니다.', '차', NULL, 'root', now(), now());

--  데이터 저장 위치 조회
show variables like 'datadir';

-- 데이터 삭제
delete from sound where file_num='1';

-- DB 조회 및 조건문
SELECT * FROM sound;

select sound_url from sound where file_num='12_24_01';