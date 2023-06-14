# Readme

### [Streamlit Link](https://junhyeok1002-samsung-sec-samsung-securities-project-main-2lkjqy.streamlit.app/)
위 혹은 아래의 링크를 클릭하면 저희가 구축한 Streamlit Share에 접근할 수 있습니다. 실시간 정보를 가져오는 검색엔진이므로 와이파이가 보장된 환경에서 실행하는 것을 권장드립니다.

(https://junhyeok1002-samsung-sec-samsung-securities-project-main-2lkjqy.streamlit.app/)

### 제출 파일 설명

폴더구조

- data 폴더 : 구축한 데이터 베이스 폴더
    - info.json : 코스피 200 종목 사전, 이벤트 사전 정보
    - 배당.csv : 배당관련 정보 데이터
- gif 폴더 : README.md 에 들어갈 Gif파일을 담은 폴더
- user_funtions 폴더 : Main에서 사용할 함수들을 담은 코드 폴더
    - Database_Input.py : 데이터불러오고 인풋을 받는 함수들을 구현한 코드
    - Markowitz_Portfolio.py : 포트폴리오 이론 관련 함수들을 구현한 코드
    - News_Crawler.py : 뉴스 크롤러 함수들을 구현한 코드
    - Stock_Information.py : 주식종목을 실시간 스크래핑 및 크롤링하는 함수들을 구현한 코드
- packages.txt (스트림릿 환경설정용)
- requirements.txt (참고 : python은 3.9.16 버전입니다)
- Samsung_Securities_Database.ipynb (데이터 베이스 구축 코드파일)
- Samsung_Securities_Project_Main.py (Streamlit Main파일)
- Readme.md : 본 파일
- 삼성증권 디지털기술팀 4조 발표자료.ppt : 발표자료

### 스트림릿 기능 간단 설명

- 목표
    
     흔히 주린이(주식+어린이)라고 불리우는 개인 투자자들이 비합리적인 투자를 하게된다. 그 이유는 금융지식에 대한 진입장벽과 Dart, KRX, 뉴스 등등에 파편화되어 있는 주가정보를 매번 검색하기가 번거롭기 때문이다 
    
    따라서 본 검색 엔진을 통해 구현하고자 하는 목표는 다음과 같다
    
    1. 파편화되어 있는 주가 정보를 한 번의 검색을 통해 얻을 수 있는 기능
    2. 금융용어에 대한 정보를 뉴스 사례와 함께 확인할 수 있는 기능
    3. 포트폴리오 백테스팅이나 개인화 맞춤 추천을 받을 수 있도록하는 기능

- 종목 이벤트 검색 기능
    - 관련 종목의 주가/재무/펀더멘탈 정보를 시각화하고 관련이벤트기반 뉴스를 정리해서 보여줌으로써 파편화된 주가 정보를 하나의 플랫폼에서 한번의 검색으로 집약하는 기능
        1. ex) 삼성전자의 신제품관련 이벤트를 알려줘 
        2. ex) 한섬, 삼성전자의 주가 이벤트를 정리해줘

        ![Untitled](/gif/KakaoTalk_20230606_214030780_01.gif)
        
    1. 이벤트 용어 검색 기능
    - 모르는 금융용어를 설명과 뉴스사례들과 함께 제시함으로써 금융지식 함양에 도움이 되는 기능
        1. ex) 당기순이익, 영업이익, 부채비율이 뭐야?

        ![Untitled](/gif/KakaoTalk_20230606_214030780.gif)
        
        
    1. 포트폴리오 백테스팅 및 markowitz portfolio theory기반 추천 기능
    - 일반인이 접근하기 어려운 백테스팅 기능과 개인의 리스크 선호(기피)도에 기반한 맞춤형 포트폴리오 추천하는 기능
    1. ex) 삼성전자, 현대차, 카카오의 포트폴리오 수익률을 알려줘
        
        (추가로 슬라이브 바에 포트폴리오 비중, 리스크 선호도를 입력하기)
       
        ![Untitled](/gif/KakaoTalk_20230606_214536939.gif)
        
    

### 역할분담

공통 : 데이터 수집 및 주가 이벤트 검토 

안형준 : 이벤트 사전 정의 및 정리

오재은 : 크롤러, API를 이용한 DB구축

전종윤 : PPT 내용 구축 및 주석처리

최윤서 : Streamit 및 PPT 디자인

최준혁 : 코드 정리 및 Streamlit 함수 구현
