# Import Modules
import json
import pandas as pd
import numpy as np
import re

# DataBase에서 kospi200 사전(종목명:티커)과 이벤트 사전 불러오는 함수
def db_loading():
    with open('./data/info.json', 'r',encoding = 'utf-8') as f:
        info = json.load(f)
    
    event_dict = info['event_dict']
    kospi200_name_code = info['kospi200_name_code']
    kospi200_code_name = info['kospi200_code_name']
    
    return event_dict, kospi200_name_code, kospi200_code_name

# 검색 받은 정보들(검색 시작종료일, 검색된 이름티커, 검색된 이벤트들)을 확인하고 입력된 정보를 사용하기에 적합하게 전처리하는 함수
def search_input(start_date, end_date, namecode, events, kospi200_name_code, kospi200_code_name, event_dict):
    # 검색가능 범위
    able_name = list(kospi200_name_code.keys())
    able_code = list(kospi200_name_code.values())
    
    # 검색어
    search = namecode[:] # copy  
    
    # namecode 입력부 확인 및 name, code 변수 분리
    name = '' ; code = ''
    if re.match('[\d]{5}[\d\w]{1}',str(namecode)): 
        code = namecode
        if code in able_code: name =  kospi200_code_name[code]
        else :st.write('검색 가능한 코드가 아닙니다. KOSPI200 종목에서 선택해주세요')
    else : 
        name = namecode
        if name in able_name:  code = kospi200_name_code[name]
        else : st.write('검색 가능한 코드가 아닙니다. KOSPI200 종목에서 선택해주세요')

    # 검색된 이벤트 목록처리 : 입력되지 않으면 모든 이벤트, 1개 이상 입력되면 해당 이벤트로
    intersect = set(events) & set(event_dict.keys())
    event = list(intersect)
    if len(intersect) > 0: pass 
    else : event = 'All'
        
    return name, code, event, search, start_date, end_date, able_name, able_code, intersect