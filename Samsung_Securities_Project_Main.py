# 삼성증권 프로젝트 : MAIN 파일
# Import module
import urllib.request
import requests
import streamlit as st
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from konlpy.tag import Okt
from collections import Counter
from itertools import islice
import re
import time
from tqdm import tqdm
import sys
from collections import OrderedDict
import io
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.subplots as ms
from pykrx import stock
from pykrx import bond
import mplfinance as mpf
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
import seaborn as sns; sns.set()
import scipy.optimize as opt

# 사용자 정의 함수들 import
from user_functions.Database_Input import *
from user_functions.News_Crawler import *
from user_functions.Stock_Information import *
from user_functions.Markowitz_Portfolio import *

# Main 파트
if __name__ == "__main__":
    # 전체 페이지 제목 출력
    st.title('Finance Search Engine')
    st.subheader('Samsung Securities Digital Tech. Project') 

    case = None # 검색 case 분류를 위한 변수
    event_dict, kospi200_name_code, kospi200_code_name = db_loading() #DB에서 데이터 로딩
    
    # 사이드탭을 이용하여 검색 도우미 기능 구현
    with st.sidebar :
        st.header("Search Helper")
        search_kind = st.radio("검색 형태를 설정해주십시오",('Event-Based Search', 'Portfolio Return'))
        
        # 이벤트 기반 검색 : 1. 특정 종목들의 이벤트 정리기능, 2. 이벤트 설명 및 뉴스사례 검색 
        if search_kind == 'Event-Based Search':
            st.subheader("Searchable Stocks & Events List")
            option1= st.multiselect(f"검색 가능 종목 사전(Kospi200)",list(kospi200_name_code.keys())) ; option1_end = ""
            option2= st.multiselect(f"이벤트 사전",list(event_dict.keys())) ; option2_end = "" 
            search_str = ', '.join(option1)+("의 " if option1 else '')+', '.join(option2)+\
                (" 이벤트에 대해 알려줘" if option1 or option2 else '') 
        # 구성 포트폴리오의 수익률 백테스팅 및 추천 기능
        else:
            st.subheader("Searchable Stocks & PortFolio Return")
            option1= st.multiselect(f"검색 가능 종목 사전(Kospi200)",list(kospi200_name_code.keys())) ; option1_end = ""
            search_str = ', '.join(option1)+("의 " if option1 else '')+\
            ("포트폴리오 수익률에 대해 알려줘" if option1 else '') 
    
    # 검색하는 폼 생성
    with st.form("텍스트 입력 폼"):
        # 검색 시작 / 종료일 검색
        left_column, center_column,right_column = st.columns([4.25,4.25,1.5])
        with left_column: # 왼쪽 열에 컨텐츠 추가
            start_date = st.date_input('검색 시작일', value = datetime.now().date() - timedelta(days=100),\
                help = "기본 값은 당일로부터 100일 전") 
        with center_column: # 오른쪽 열에 컨텐츠 추가
            end_date = st.date_input('검색 종료일' , \
                help = "기본 값은 당일") 
        with right_column:
            page_num = st.selectbox(label = 'page 개수', options = list(range(2,11)), \
                help = "불러올 뉴스 페이지 수(페이지당 10개, 중복되는 기사는 제거), \
                뉴스검색 아닐 시 사용되지 않는 변수, 추천하는 기본 값은 2page이며 페이지가 많을 수록 검색 시간이 오래걸립니다")

        # 검색어 입력
        left_column, right_column = st.columns([9,1])
        with left_column: # 검색창
            search_block = str(st.text_input('검색어를 입력해주세요', search_str, \
            help = "검색 가능 종목 목록, 이벤트 목록의 단어 포함 여부로 검색을 진행하며 \
                    추가적으로 '수익률(율)' 단어가 포함될 시 종목들을 이용하여 수익률을 계산해줍니다\
                    입력을 마치면 '검색'버튼을 눌러주십시오"))
        with right_column:# 검색버튼
            st.markdown("<h1 style='font-size: 10px;'></h1>", unsafe_allow_html=True)
            submit_button = st.form_submit_button(label = "검색", use_container_width =True, \
                help = "검색은 값을 전부 입력하고 버튼을 누르는 시점에 진행됩니다.")
        
        # 검색내용 중 종목과 이벤트를 구분해서 사전 및 리스트 생성 : 띄어쓰기 오류도 잡아낼 수 있도록 처리함
        search_block = ''.join([char for char in search_block if char != ' ']) 
        namecode = {name : search_block.find(name) if (search_block.find(name)) >= 0 else search_block.find(code)\
                               for name, code in kospi200_name_code.items() if (search_block.find(name) >= 0 or search_block.find(code) >= 0)}

        event_list = set([''.join([char for char in event if char != ' ']) for event in event_dict.keys()])    # 띄어쓰기 없애기
        events = [event for event in event_list if (event in search_block)]
        
        # 검색 창에 종목들과 수익률(율) or 포트폴리오 단어가 있으면 각 종목에 대한 포트폴리오 구성 비중과 개인의 리스크 선호도를 입력받음
        if len(list(namecode.keys())) >= 1 and ('수익률' in  search_block or '수익율' in  search_block or '포트폴리오' in  search_block):
            # 포트폴리오 구성 비율 선택
            st.markdown(f"**각 종목의 포트폴리오 비중을 선택해주세요**",\
                         help = '종목 비중 합이 1이 아니더라도 비중을 합쳐서 나누어 비율에 맞게 할당합니다 ')
            name_code = list(sorted(namecode, key=namecode.get))
            total = 1
            st.session_state = dict()
            if 'weight' not in st.session_state: st.session_state['weight'] = dict()
            for i, name in enumerate(name_code):
                try : test_slider = st.slider(f'{name}의 포트폴리오 비중을 선택하시오', float(0),float(total),st.session_state['weight'][name])
                except : test_slider = st.slider(f'{name}의 포트폴리오 비중을 선택하시오', float(0),float(total),0.00)
                st.session_state['weight'][name] = test_slider
                
            # 리스크 선호도 선택 : 추후 이를 기반으로 포트폴리오 비중 추천
            risk_preference = 1.00
            with st.sidebar:
                risk_slider = st.slider(f'리스크 선호도를 선택해주세요', float(0),float(1), risk_preference,\
                                        help = '(1:리스크 선호, 0: 리스키 기피)')
                risk_preference = str(risk_slider)
            
            # 비중이 존재하는지 여부를 이용해 예외처리하여 streamlit 흐름을 제어하는 부분
            try : weight = {kospi200_name_code[k] : v/sum(st.session_state['weight'].values()) for k, v  in st.session_state['weight'].items()}
            except : pass
            st.session_state = dict()
    
    # 값을 다 입력하고 버튼을 눌렀을 때 실행한다 (이벤트는 미입력 가능, 미입력 시 해당하는 모든 이벤트 정보를 가져옴)
    if  start_date and end_date and search_block and submit_button:  
        # 검색된 정보를 보고 어떤 유형의 검색인지 파악하는 부분
        if len(list(namecode.keys())) >= 1 and ('수익률' in  search_block or '수익율' in  search_block): case = 'ret' #포트폴리오 검색 유형
        elif len(list(namecode.keys())) >= 1 : case = 'event'  # 종목 이벤트 검색 유형
        elif len(list(namecode.keys())) == 0 and len(events) > 0 : case = 'explain' # 이벤트 설명 유형
        else : case = 'blank' #빈 유형 : streamlit 흐름 제어를 위해 선언
    
    # 종목 이벤트 검색 유형의 경우
    if case == 'event':
        search_kind = 'Event-Based Search'
        namecode = list(sorted(namecode, key=namecode.get))
        st.header(f"Event Overview") # ; st.markdown("---")
        try : 
            # 검색된 주식종목 별로 탭화면 구분
            st.subheader("Stock List")
            tabs= st.tabs(namecode)
            for i, name_code in enumerate(namecode):
                with tabs[i]:
                    name, code, event, search, start_date, end_date, able_name, able_code,  intersect= \
                    search_input(start_date, end_date, name_code, events, kospi200_name_code, kospi200_code_name, event_dict)
                    stock_df = making_stock_df(start_date.strftime('%Y%m%d'),end_date.strftime('%Y%m%d'),code)
                    finance_df,status = get_finance_info(code)

                    # 각 주식 종목 별 주가정보/펀더멘탈/재무정보를 탭을 구분하여 보여줌
                    tab1, tab2, tab3 = st.tabs(["주가 정보", "펀더멘탈 지표", "재무 정보"])
                    with tab1 : #주가 정보 탭
                        st.subheader("Price Information")

                        # 현재가, 시가, 고가, 저가, 전일 종가, 외국인소진율, 거래량, 시가총액 표시
                        a,b,c,d = st.columns(4)
                        st.markdown(""" <style> [data-testid="stMetricValue"] {font-size: 30px;text-align: center;} </style>""",unsafe_allow_html=True)
                        with a: 
                            st.metric(f"{name} ({code})", f"{stock_df.iloc[0]['종가']:,}", 
                                        f"{stock_df.iloc[0]['종가']-stock_df.iloc[1]['종가']:,} ({stock_df.iloc[0]['등락률']:.2f}%)")
                            st.metric('전일종가',f"{stock_df.iloc[1]['종가']:,}",f"{stock_df.iloc[1]['종가']-stock_df.iloc[2]['종가']:,}")
                        with b:
                            st.metric('시가',f"{stock_df.iloc[0]['시가']:,}",f"{stock_df.iloc[0]['시가']-stock_df.iloc[1]['시가']:,}")
                            st.metric('외국인소진률',f"{stock_df.iloc[1]['한도소진률']:.2f}%",\
                                      f"{stock_df.iloc[0]['한도소진률']-stock_df.iloc[1]['한도소진률']:.2f}%")
                        with c:
                            st.metric('고가',f"{stock_df.iloc[0]['고가']:,}",f"{stock_df.iloc[0]['고가']-stock_df.iloc[1]['고가']:,}")
                            st.metric('거래량',f"{stock_df.iloc[1]['거래량']:,}",f"{stock_df.iloc[0]['거래량']-stock_df.iloc[1]['거래량']:,}")
                        with d:
                            st.metric('저가',f"{stock_df.iloc[0]['저가']:,}",f"{stock_df.iloc[0]['저가']-stock_df.iloc[1]['저가']:,}")
                            temp = stock_df.iloc[0]['시가총액']
                            st.metric('시가총액',f"{format_currency(temp)}")

                        # 시장 시점 처리
                        date = datetime.strptime(status[:10], '%Y.%m.%d')
                        today = stock.get_nearest_business_day_in_a_week(end_date.strftime('%Y%m%d'))
                        today = datetime.strptime(today, '%Y%m%d')
                        if today == date:pass
                        else : status = today.strftime("%Y.%m.%d")+" 기준(장마감)"

                        styled_text = f"<div style='text-align:right'>{status}</div>"
                        st.markdown(styled_text, unsafe_allow_html=True)

                        # 캔들 차트     
                        fig = plot_candle(stock_df)
                        st.plotly_chart(fig)
                        st.divider()

                    with tab2 : #펀더멘탈 탭
                        st.subheader("Valuation Information")
                        try : lock, pay = dividend(code, end_date.strftime("%Y%m%d"))
                        except : pay , lock = "-" , "-"

                        # PER, PBR, BPS, EPS, DIV, DPS, 다음배당락일, 다음지불일 표시
                        a,b,c,d = st.columns(4)
                        with a: 
                            st.metric('PER(배)',f"{stock_df.iloc[0]['PER']:.2f}",f"{stock_df.iloc[0]['PER']-stock_df.iloc[1]['PER']:.2f}")
                            st.metric('DIV(%)',f"{stock_df.iloc[0]['DIV']:.2f}%",f"{stock_df.iloc[0]['DIV']-stock_df.iloc[1]['DIV']:.2f}")
                        with b: 
                            st.metric('PBR(배)',f"{stock_df.iloc[0]['PBR']:.2f}",f"{stock_df.iloc[0]['PBR']-stock_df.iloc[1]['PBR']:.2f}")
                            st.metric('DPS(원)',f"{stock_df.iloc[0]['DPS']:,}",f"{stock_df.iloc[0]['DPS']-stock_df.iloc[1]['DPS']:.2f}")
                        with c: 
                            st.metric('BPS(원)',f"{stock_df.iloc[0]['BPS']:,}",f"{stock_df.iloc[0]['BPS']-stock_df.iloc[1]['BPS']:,}")
                            st.metric('다음배당락일',f"{lock if lock != '20230000' else '-'}")
                        with d: 
                            st.metric('EPS(원)',f"{stock_df.iloc[0]['EPS']:,}",f"{stock_df.iloc[0]['EPS']-stock_df.iloc[1]['EPS']:,}")
                            st.metric('다음지불일',f"{pay if pay != '20230000' else '-'}")

                        # 펀더멘탈 정보 플랏팅(시각화)
                        with st.container():
                            # 2행 3열의 서브플롯 생성
                            fig = sp.make_subplots(rows=2, cols=3,subplot_titles=("PER","BPS","DIV","PBR","EPS","DPS"))

                            graph = go.Scatter(x=stock_df.index, y=stock_df['PER'], mode='lines', name='PER');fig.add_trace(graph, row=1, col=1)
                            graph = go.Scatter(x=stock_df.index, y=stock_df['BPS'], mode='lines', name='BPS');fig.add_trace(graph, row=1, col=2)
                            graph = go.Scatter(x=stock_df.index, y=stock_df['DIV'], mode='lines', name='DIV');fig.add_trace(graph, row=1, col=3)
                            graph = go.Scatter(x=stock_df.index, y=stock_df['PBR'], mode='lines', name='PBR');fig.add_trace(graph, row=2, col=1)
                            graph = go.Scatter(x=stock_df.index, y=stock_df['EPS'], mode='lines', name='EPS');fig.add_trace(graph, row=2, col=2)
                            graph = go.Scatter(x=stock_df.index, y=stock_df['DPS'], mode='lines', name='DPS');fig.add_trace(graph, row=2, col=3)

                            fig.update_layout(margin=dict(t=20), legend=dict(orientation='h'))
                            st.plotly_chart(fig)
                        st.divider()

                    with tab3 : #재무정보 탭
                        st.subheader("Financial Information")
                        finance_df = finance_df.rename(columns={'ROE(지배주주)': "ROE"})
                        finance_df = finance_df.astype(float)
                        # 재무정보 표시 : 매출액/영업이익/당기순이익/영업이익률/순이익률/ROE/부채비율/당좌비율/유보율
                        with st.container(): st.dataframe(finance_df,use_container_width=True)
                        finance_df.index = [f'{i[2:4]}년 {i[5:7]}월' for i in finance_df.index]

                        # 매출액/영업이익/당기순이익 과 영업이익률/순이익률/ROE 와 부채비율/당좌비율/유보율를 탭으로 나누어 시각화
                        tab11, tab22, tab33 = st.tabs(["매출액/영업이익/당기순이익", "영업이익률/순이익률/ROE", "부채비율/당좌비율/유보율"])
                        with tab11 : 
                            fig1 = make_subplots(rows=1, cols=1)
                            for name in ['매출액','영업이익', '당기순이익']:
                                fig1.add_trace(go.Bar(x=finance_df.index, y=finance_df[name], name=name),row=1, col=1) # 바차트 생성
                            fig1.update_layout(yaxis_title='단위(억)', height=300, margin=dict(t=0,b=0))
                            st.plotly_chart(fig1)
                        with tab22 : 
                            fig2 = make_subplots(rows=1, cols=1)
                            for name in ['영업이익률','순이익률', 'ROE']:
                                fig2.add_trace(go.Bar(x=finance_df.index, y=finance_df[name], name=name),row=1, col=1) # 바차트 생성
                            fig2.update_layout(yaxis_title='단위(%)', height=300, margin=dict(t=0,b=0))
                            st.plotly_chart(fig2)                  
                        with tab33 : 
                            fig3 = make_subplots(rows=1, cols=3) 
                            for i, name in enumerate(['부채비율','당좌비율', '유보율']):
                                fig3.add_trace(go.Bar(x=finance_df.index, y=finance_df[name], name=name),row=1, col=i+1) # 바차트 생성
                            fig3.update_layout(yaxis_title='단위(%)', height=300, margin=dict(t=0,b=0))
                            st.plotly_chart(fig3)                   
                        st.divider()
                filtered_news_df, new_event_dict = news(search, event, start_date.strftime("%Y%m%d"),\
                                                        end_date.strftime("%Y%m%d"), event_dict , page_num = page_num)
                n_print = news_print(filtered_news_df, name, event_dict, intersect ,top_n = 3)                        
        except : 
            st.warning("기간이 너무 짧아 주가 정보를 불러올 수 없습니다.")
                
            filtered_news_df, new_event_dict = news(search, event, start_date.strftime("%Y%m%d"),\
                                                    end_date.strftime("%Y%m%d"), event_dict , page_num = page_num)
            n_print = news_print(filtered_news_df, name, event_dict, intersect ,top_n = 3)


    # 이벤트 설명 및 뉴스사례 검색의 경우
    elif case == 'explain': 
        search_kind = 'Event-Based Search'
        st.header(f"Event Explanation")
        st.subheader("Event List")
        
        # 검색된 이벤트들을 탭으로 나누어 구현
        tabs= st.tabs(events)
        for i, eve in enumerate(events):
            with tabs[i]:
                st.subheader(f"{eve}?")
                st.write(event_dict[eve])
                # 이벤트 별로 뉴스 사례들을 검색하여 보여줌
                filtered_news_df, new_event_dict = news(eve, [eve], start_date.strftime("%Y%m%d"),end_date.strftime("%Y%m%d"), \
                                                        event_dict , page_num = page_num)
                n_print = news_print(filtered_news_df,'', event_dict,[eve] ,top_n = page_num*10, explain = False)
    
    # 포트폴리오 백테스팅 및 포트폴리오 추천 검색의 경우
    elif case == 'ret': 
        try : 
            st.header(f"Portfolio Return")
            
            # 포드폴리오 종목과 비중등으로 백테스팅하는 부분
            namecode = list(sorted(namecode, key=namecode.get))
            codes = list(weight.keys()) 
            ret, portfolio_df, percent_ret, percent_vols, percent_var = back_test(codes, np.array(list(weight.values())), \
                                          start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
            log_ret = ret
            port_rets, port_std = make_random_portfolio(ret, len(codes), log_ret)
            fig_opt, sharpe_ratio, opt_weight = portfolio_opt(port_rets,port_std, len(codes), ret, portfolio_df, log_ret, end_date)

            returns_f = bond.get_otc_treasury_yields(end_date.strftime('%Y%m%d')).loc['국고채 10년','수익률']/100 # 무위험자산 정보
            
            # 백테스팅 결과 출력
            st.subheader("Back Test Results")
            a,b,c,d = st.columns(4)
            st.markdown(""" <style> [data-testid="stMetricValue"] {font-size: 30px;text-align: center;} </style>""",unsafe_allow_html=True)
            with a: st.metric('포트폴리오 수익률',percent_ret)
            with b: st.metric('리스크(Volatility/Std.Dev)',percent_vols)
            with c: st.metric('분산(Variance)',percent_var)
            with d: 
                sharpe = (float(percent_ret.rstrip('%'))/100 - returns_f)/(float(percent_vols.rstrip('%'))/100)
                st.metric('샤프 비율(Sharpe ratio)', f'{sharpe:.2f}',\
                         help = '소수점 2자리로 처리한 변수들로 계산한 수치들이므로 실제와 근사하지만 일치하지는 않을 수 있습니다')
            
            # 포트폴리오 각 종목의 비중 및 수익률을 table로 표시
            with st.container() :                       
                temp_df1 = pd.DataFrame([np.array(list(weight.values()))], columns=list(weight.keys()))
                temp_dict = {kospi200_code_name[k]:v for k,v in dict(log_ret.cumsum(axis = 0).iloc[-1]).items()}
                temp_df2 = pd.DataFrame([np.array(list(temp_dict.values()))], columns=list(weight.keys())) 
                df = pd.concat([temp_df1,temp_df2], axis = 0).round(2)
                df.index = ['비중','수익률']
                df.columns = [f'{kospi200_code_name[c]}({c})' for c in df.columns]
                df = df[[col for col in df.columns if df.loc['비중',col] > 0]]
                st.dataframe(df, use_container_width=True)

            # 백테스팅 결과 : 포트폴리오 각 종목의 비중 및 수익률을 시각화
            with st.container() : 
                aa, bb = st.columns(2)
                with aa : 
                    labels = [kospi200_code_name[code] for code in codes]
                    values = np.array(list(weight.values()))

                    colors = px.colors.qualitative.Pastel # 색상 맵 지정 (Plotly Express의 색상 맵)

                    fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))])
                    fig.update_traces(textinfo='label+percent')
                    fig.update_layout(title='Portfolio Weights',margin=dict(t=30,b=0,l=0,r=0), \
                                          width=300, height=300,legend=dict(orientation='h'))            
                    st.plotly_chart(fig, use_container_width=True) 
                with bb : 
                    temp_dict = {kospi200_code_name[k]:v for k,v in dict(log_ret.cumsum(axis = 0).iloc[-1]).items()}
                    labels = list(temp_dict.keys())
                    values = np.array(list(temp_dict.values()))

                    colors = px.colors.qualitative.Pastel

                    fig = go.Figure(data=go.Bar(x=labels, y=values, marker=dict(color=colors)))
                    # fig.update_traces(textinfo='label+percent')
                    fig.update_layout(title='Return on Individual Assets',margin=dict(t=30,b=0,l=0,r=0), \
                                          width=100, height=300,legend=dict(orientation='h'))            
                    st.plotly_chart(fig, use_container_width=True)         

            st.divider()    
            
            # 포트폴리오 최적화 및 리스크 위험도 맞춤형 추천파트
            st.subheader("Portfolio Optimization & Recommendation")
            st.markdown("<span style='font-size: 20px;'>**Markowitz Portfolio Optimization**</span>", unsafe_allow_html=True)
            st.pyplot(fig_opt)

            codes = list(opt_weight.keys()) 
            ret, portfolio_df, percent_ret, percent_vols, percent_var = \
                back_test(codes, np.array(list(opt_weight.values())), start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))

            # 최대 샤프비율 추천 탭과 리스크 위험도 맞춤형 추천탭으로 화면 구분
            st.markdown("<span style='font-size: 20px;'>**Portfolio Recommendations**</span>", unsafe_allow_html=True)
            t1, t2 = st.tabs(['Recommendation 1 : 최대 샤프 비율 포트폴리오','Recommendation 2 : 리스크 선호도 맞춤형 포트폴리오']) 
            with t1: # 최대 샤프비율 추천 탭
                st.markdown(f"**Recommendation 1 : Highest Sharpe-ratio Portfolio**")
                
                # 백테스팅 결과 출력
                w,x,y,z = st.columns(4)
                st.markdown(""" <style> [data-testid="stMetricValue"] {font-size: 30px;text-align: center;} </style>""",unsafe_allow_html=True)
                with w: st.metric('포트폴리오 수익률',percent_ret)
                with x: st.metric('리스크(Volatility/Std.Dev)',percent_vols)
                with y: st.metric('분산(Variance)',percent_var) 
                with z: st.metric('샤프 비율(Sharpe ratio)',round(sharpe_ratio,3))
                
                # 포트폴리오 각 종목의 비중 및 수익률을 table로 표시
                with st.container() : 
                    temp_df1 = pd.DataFrame([np.array(list(opt_weight.values()))], columns=list(opt_weight.keys()))
                    temp_dict = {kospi200_code_name[k]:v for k,v in dict(log_ret.cumsum(axis = 0).iloc[-1]).items()}
                    temp_df2 = pd.DataFrame([np.array(list(temp_dict.values()))], columns=list(opt_weight.keys())) 
                    df = pd.concat([temp_df1,temp_df2], axis = 0).round(4)
                    df.index = ['비중','수익률']
                    df.columns = [f'{kospi200_code_name[c]}({c})' for c in df.columns]
                    df = df[[col for col in df.columns if df.loc['비중',col] > 0]]
                    st.dataframe(df, use_container_width=True)
                
                # 포트폴리오 추천 결과 : 포트폴리오 각 종목의 비중 및 수익률을 시각화
                with st.container() : 
                    aaa, bbb = st.columns(2)
                    with aaa : 
                        labels = [kospi200_code_name[code] for code in codes]
                        values = np.array(list(opt_weight.values()))

                        # 색상 맵 지정 (Plotly Express의 색상 맵)
                        colors = px.colors.qualitative.Set3

                        fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))])
                        fig.update_traces(textinfo='label+percent')
                        fig.update_layout(title='Portfolio Weights',margin=dict(t=30,l=0,r=0), \
                                              width=400, height=300,legend=dict(orientation='h'))            
                        st.plotly_chart(fig, use_container_width=True) 
                    with bbb : 
                        temp_dict = {kospi200_code_name[k]:v for k,v in dict(log_ret.cumsum(axis = 0).iloc[-1]).items()}
                        labels = list(temp_dict.keys())
                        values = np.array(list(temp_dict.values()))

                        colors = px.colors.qualitative.Set3

                        fig = go.Figure(data=go.Bar(x=labels, y=values, marker=dict(color=colors)))
                        # fig.update_traces(textinfo='label+percent')
                        fig.update_layout(title='Return on Individual Assets',margin=dict(t=30,l=0,r=0), \
                                              width=400, height=300,legend=dict(orientation='h'))            
                        st.plotly_chart(fig, use_container_width=True)  
                st.divider()   
            with t2: # 리스크 위험도 맞춤형 추천 탭
                st.markdown(f"**Recommendation 2 : Risk Preference Tailored Portfolio**",\
                    help = "최적 포트폴리오와 무위험자산(10년 국채)의 비율을 조정하면 리스크 관리를 할 수 있습니다")

                returns_f = bond.get_otc_treasury_yields(end_date.strftime('%Y%m%d')).loc['국고채 10년','수익률']/100 # 무위험자산 정보
                w_f = round(1 - float(risk_preference),2) # 무위험자산 비중

                # 무위험자산과 혼합한 포트폴리오의 분산 계산
                portfolio_var_mixed = (float(risk_preference)) * float(percent_var.rstrip("%"))/100  # 위험자산 포트폴리오의 분산
                # 무위험자산과 혼합한 포트폴리오의 표준편차 계산
                portfolio_std_mixed = np.sqrt(portfolio_var_mixed)
                # 무위험자산과 혼합한 포트폴리오 수익률 계산
                portfolio_return_mixed = returns_f * w_f + (float(percent_ret.rstrip("%"))/100) * (1 - w_f)
                
                # 백테스팅 결과 출력
                xx,yy,zz,ww = st.columns(4)
                st.markdown(""" <style> [data-testid="stMetricValue"] {font-size: 30px;text-align: center;} </style>""",unsafe_allow_html=True)
                with xx: st.metric('포트폴리오 수익률',f'{portfolio_return_mixed*100:.2f}%')
                with yy: st.metric('리스크(Volatility/Std.Dev)',f'{portfolio_std_mixed*100:.2f}%')
                with zz: st.metric('분산(Variance)',f'{portfolio_var_mixed*100:.2f}%') 
                with ww: st.metric('샤프 비율(Sharpe ratio)',round((portfolio_return_mixed - returns_f)/portfolio_std_mixed,2),\
                                help = '소수점 2자리로 처리한 변수들로 계산한 수치들이므로 실제와 근사하지만 일치하지는 않을 수 있습니다')
                
                # 포트폴리오 각 종목의 비중 및 수익률을 table로 표시
                with st.container() : 
                    temp_df1 = pd.DataFrame([np.array(list(opt_weight.values()))], columns=list(opt_weight.keys()))
                    temp_dict = {kospi200_code_name[k]:v for k,v in dict(log_ret.cumsum(axis = 0).iloc[-1]).items()}
                    temp_df2 = pd.DataFrame([np.array(list(temp_dict.values()))], columns=list(opt_weight.keys())) 
                    df = pd.concat([temp_df1,temp_df2], axis = 0)
                    df.index = ['비중','수익률']
                    df.columns = [f'{kospi200_code_name[c]}({c})' for c in df.columns]
                    df = df[[col for col in df.columns if df.loc['비중',col] > 0]]
                    # 위험자산 비중 조정
                    df.loc['비중',:] *= float(risk_preference)

                    # 무위험 자산 추가
                    df['국고채 10년'] = [w_f, returns_f]
                    df = df.round(4)
                    st.dataframe(df, use_container_width=True)
                
                # 포트폴리오 추천 결과 : 포트폴리오 각 종목의 비중 및 수익률을 시각화
                with st.container() : 
                    AA, BB = st.columns(2)
                    with AA : 
                        labels = [col[:-8] if col != '국고채 10년' else col for col in df.columns]
                        values = np.array(df.loc['비중'])

                        # 색상 맵 지정 (Plotly Express의 색상 맵)
                        colors = px.colors.qualitative.Set3

                        fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))])
                        fig.update_traces(textinfo='label+percent')
                        fig.update_layout(title='Portfolio Weights',margin=dict(t=30,l=0,r=0), \
                                              width=400, height=300,legend=dict(orientation='h'))            
                        st.plotly_chart(fig, use_container_width=True) 
                    with BB : 
                        labels = [col[:-8] if col != '국고채 10년' else col for col in df.columns]
                        values = np.array(df.loc['수익률'])

                        colors = px.colors.qualitative.Set3

                        fig = go.Figure(data=go.Bar(x=labels, y=values, marker=dict(color=colors)))
                        # fig.update_traces(textinfo='label+percent')
                        fig.update_layout(title='Return on Individual Assets',margin=dict(t=30,l=0,r=0), \
                                              width=400, height=300,legend=dict(orientation='h'))            
                        st.plotly_chart(fig, use_container_width=True)  
                st.divider() 
        
        # 포트폴리오 추천탭 예외 처리
        except Exception as e : 
            st.warning(str(e))
    
    # Streamlit 흐름 제어를 위한 부분
    elif case == 'blank':
        st.error('검색 결과가 없습니다')
    else :
        pass