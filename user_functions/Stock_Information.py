# Import Modules
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as ms
from pykrx import stock
from pykrx import bond
import plotly.express as px

# 날짜와 기간(N일)을 입력하면 해당 날짜로 부터 N일 전의 날짜를 반환
def change_date(date , period):
    new_date = datetime.strptime(date, '%Y%m%d')
    new_date = new_date - timedelta(days=period)
    new_date = new_date.strftime('%Y%m%d')
    return new_date

# 검색시작일, 검색종료일, 종목코드를 입력 받아 주가 차트를 그리기 위한 정보들을 긁어오는 함수
def scrapping_info(start_date,end_date, code):
    #시가, 종가, 저가, 고가
    temp1 = stock.get_market_ohlcv(start_date,end_date, code, adjusted = True); time.sleep(0.5) 
    
    #펀더멘탈(BPS,EPS,PER,PBR,DPS,DIV)
    temp2 = stock.get_market_fundamental(start_date, end_date, code) ; time.sleep(0.5)
    
    #외국인 소진율
    temp3 = stock.get_exhaustion_rates_of_foreign_investment(start_date, end_date, code)[['한도소진률']] 
    
    # 시가총액 
    merged_df = temp1.merge(temp2, left_index=True, right_index=True, how='outer')
    merged_df = merged_df.merge(temp3, left_index=True, right_index=True, how='outer')
    merged_df['시가총액'] = ''
    nearest = stock.get_nearest_business_day_in_a_week(end_date)
    idx = datetime.strptime(nearest, '%Y%m%d').strftime('%Y-%m-%d')
    cap = stock.get_market_cap(nearest).loc[code, '시가총액']
    merged_df.loc[idx,'시가총액'] = cap
    
    # 5, 20, 60일 이동평균
    merged_df['5일_이동평균'] = merged_df['종가'].rolling(window=5).mean()
    merged_df['20일_이동평균'] = merged_df['종가'].rolling(window=20).mean()
    merged_df['60일_이동평균'] = merged_df['종가'].rolling(window=60).mean()
    
    return merged_df

# 앞선 scrapping_info를 활용하여 검색기간이 길어 주가 정보불러오는 것이 문제가 있을 경우 최근 한달의 정보만 가져오도록 예외처리하는 함수
def making_stock_df(start_date,end_date,code):
    try : 
        start_date1 = change_date(start_date, 90) #시작일 90일 전부터 시작하는 이유는 60개장일전의 정보를 이용해 이동평균선을 그리기 위함
        merged_df = scrapping_info(start_date1,end_date, code)
    except : 
        st.write("검색하신 기간이 길어 주가 정보를 불러올 수 없으므로 최근 한달의 정보를 불러옵니다")
        start_date = change_date(end_date, 120) #종료일 120일 전부터 시작하는 이유는 한달의 정보를 불러오되 60개장일 이동평균을 계산하기 위함
        merged_df = scrapping_info(start_date,end_date, code)
        start_date = change_date(end_date, 30)
        
    stock_df = merged_df.sort_index(ascending=False)
    ago = stock.get_nearest_business_day_in_a_week(start_date)
    stock_df = stock_df[stock_df.index > ago]
    return stock_df #stock_df.iloc[n] -> n 개장일 전의 정보

# 주가 dataframe을 입력받아 캔들차트를 그려 그림 객체를 반환하는 함수
def plot_candle(stock_df):
    # 캔들 차트 객체 생성
    candle = go.Candlestick(
        x=stock_df.index,
        open=stock_df['시가'],
        high=stock_df['고가'],
        low=stock_df['저가'],
        close=stock_df['종가'],
        increasing_line_color = 'red', # 상승봉 스타일링
        decreasing_line_color = 'blue', # 하락봉 스타일링
        name = 'Price'
    )

    # 스무스한 이동평균선 객체 생성(5, 20, 60일 이동평균선)
    ma_5 = go.Scatter(
        x=stock_df.index,
        y=stock_df['5일_이동평균'],
        mode='lines',
        line=dict(color="#98FB98", width=2),
        name=f'{5}Days Moving Average'
    )
    ma_20 = go.Scatter(
        x=stock_df.index,
        y=stock_df['20일_이동평균'],
        mode='lines',
        line=dict(color='#FF9999', width=2),
        name=f'{20}Days Moving Average'
    )
    ma_60 = go.Scatter(
        x=stock_df.index,
        y=stock_df['60일_이동평균'],
        mode='lines',
        line=dict(color='#FFB347', width=2),
        name=f'{60}Days Moving Average'
    )

    # 바 차트(거래량) 객체 생성
    volume_bar = go.Bar(x=stock_df.index, 
                        y=stock_df['거래량'], 
                        name = 'Volume',
                        marker=dict(color='#D3D3D3'))

    fig = ms.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[3,1])

    fig.add_trace(candle, row=1, col=1)
    fig.add_trace(ma_5, row=1, col=1)
    fig.add_trace(ma_20, row=1, col=1)
    fig.add_trace(ma_60, row=1, col=1)
    fig.add_trace(volume_bar, row=2, col=1)

    fig.update_layout(
        # title='??',
        yaxis1_title='Stock Price',
        yaxis2_title='Volume',
        # xaxis2_title='periods',
        xaxis1_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,    
    )
    # 여백 설정
    fig.update_layout(
        margin=dict(t=0),  # 상단, 하단, 좌측, 우측 여백 값을 0으로 설정
        legend=dict(orientation='h')
    )
    return fig

# 금액 출력 형식 변환 코드 : 숫자를 입력받아 '조, 억, 만, 원' 중 2개까지 표현하는 문자열로 반환하는 함수
def format_currency(number): 
    if number < 10000:
        return f"{number}원"

    units = ["조", "억", "만"]
    scales = [1000000000000, 100000000, 10000]

    result = ""
    for unit, scale in zip(units, scales):
        if number >= scale:
            quotient = number // scale
            number %= scale
            result += f"{quotient:d}{unit} "

    result += f"{number:d}원"
    test = re.findall("(\d+.\s\d+[\w])",result)
    if test: return test[0]
    else : return result
    return result

# 네이버에서 주식 재무 정보를 불러오기 위한 크롤러 
# 매출액, 영업이익, 당기순이익, 영업이익률, 순이익률, ROE, 부채비율, 당좌비율, 유보율을 불러오기 위함
def get_finance_info(code):
    url = f'https://finance.naver.com/item/main.naver?code={code}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    sub_section = soup.find_all(class_='sub_section')
    text = sub_section[4].get_text().replace(',','').replace('\t','\n') # .replace('  ',' - ')

    title = re.findall('\d{4}[.].+', text)[:10]
    status = soup.find_all(class_='date')[0].text

    sub_section = soup.find_all(class_='sub_section')
    text = sub_section[4].get_text().replace(',','').replace('\t','\n').replace('\xa0' , '0')

    idx = re.findall('.+', text).index('매출액')
    text = ' '.join(re.findall('.+', text)[idx:])
    body = re.findall('[^\s\d.-]+[\d.\-\s]+',text)[:9]

    data = [b.split() for b in body]
    rows = data[:]
    df = pd.DataFrame(rows)
    df = df.set_index(0).T
    df.index = title
    df = df.iloc[4:9, :]
    df.index.name = None
    return df, status

# 배당정보를 데이터베이스에서 불러와 종목티커와 날짜를 입력하면 해당 종목, 날짜 기준 다음 배당락일과 지불일을 반환하는 함수
# 배당이 없는 주식은 데이터 베이스에 20300000으로 처리 되어있음 + 날짜(E)표시로 반환되는 날짜는 기존 배당정보를 기반으로 예측된 날짜임 
def dividend(code, end_date):
    div = pd.read_csv('./data/배당.csv', dtype = str)
    div['stock_code'] = div['stock_code'].apply(lambda x : f'{int(x):06}')
    div['다음배당락일'][div['다음배당락일'] > '20230601'] = div['다음배당락일']+"(E)"
    div['다음지불일'][div['다음지불일'] > '20230605'] = div['다음지불일']+"(E)"
    
    temp = div[div['stock_code']==code]
    temp['다음배당락일isin'] = end_date < temp['다음배당락일']
    temp['다음지불일isin'] = end_date < temp['다음지불일']
    lock = temp[temp['다음배당락일isin']].sort_values('다음배당락일').iloc[0]['다음배당락일']
    pay = temp[temp['다음지불일isin']].sort_values('다음지불일').iloc[0]['다음지불일']
    return lock, pay