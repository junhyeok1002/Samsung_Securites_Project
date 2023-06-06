# Import Modules
import urllib.request
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from konlpy.tag import Okt
from collections import Counter
from itertools import islice
import re
import time
from tqdm import tqdm
from collections import OrderedDict
import streamlit as st

#이벤트 기사 추출을 위한 기사본문 띄어쓰기 없애는 함수
def noblank(string): 
    # 텍스트에서 명사,알파벳 추출
    tokenizer = Okt()
    # 형태소 분석
    morphs = tokenizer.pos(string)

    noblank = list()
    for morph in morphs:
        if morph[1] in ['Noun','Alpha']: #형태소가 명사, 알파벳일 경우
            noblank.append(morph[0])
    noblank = ''.join(noblank)     
    return noblank

# 이벤트 단어들이 포함된 기사들만 필터링하기 위한 함수
def filter_rows_by_words(df, column, words): 
    df['event'] = df[column].apply(lambda x: [word for word in words if word in str(x)])
    filtered_df = df[df['event'].apply(lambda x: len(x) > 0)]
    return filtered_df

# 리스트에 어떤 값이 존재하는 지 확인하기 위한 함수
def is_contain(value, value_list):
    flag = False
    for v in value_list:
        if (value in v) and (value != v):
            flag = True
    return flag

# 키워드 추출 자연어 처리 함수 
# 긴 텍스트 : 형태소가 명사인 것이 이어나오면 하나의 키워도로 묶고 구성 명사들의 평균빈도수를 구해 키워드 점수 부여 
# -> 맥락을 알 수 있는 키워드를 반환해주지만 텍스트가 짧으면 효과가 안좋음

# 짧은 텍스트 : 명사를 이어 붙이지 않고 단순 빈도수를 이용하여 키워드 점수 부여 
# -> 짧은 텍스트에서 위의 방법이 효과적이지 않으므로 텍스트가 짧을 경우 단순 빈도수로 처리
def keyword_extract(string):
    # 형태소 분석기 초기화
    tokenizer = Okt()

    # 텍스트에서 명사 추출
    nouns = tokenizer.nouns(string)

    # 키워드 추출을 위한 불용어 처리 (선택적)
    stopwords = [] # ['기사', '텍스트', '예시']
    nouns = [noun for noun in nouns if noun not in stopwords]

    # 키워드 빈도수 카운트
    keyword_counter = Counter(nouns)

    # 빈도수가 가장 높은 상위 키워드 추출
    top_keywords = keyword_counter.most_common(20)  # 상위 5개 

    # 텍스트에서 형태소 분석
    morphs = tokenizer.pos(string)

    # 연속된 명사를 하나의 명사로 처리
    nouns = []
    temp_noun = ''
    for morph in morphs:
        if morph[1] in ['Noun','Alpha']: # , 'Alpha']:  # 명사 또는 알파벳인 경우
            temp_noun += (morph[0]+' ')
        else:
            if temp_noun:  # 이전에 연속된 명사가 있었으면
                nouns.append(temp_noun.strip())  # 하나의 명사로 처리하여 추가
                temp_noun = ''  # 임시 명사 초기화

    # 마지막에 연속된 명사가 있을 경우 처리
    if temp_noun:
        nouns.append(temp_noun)

    # 추출된 명사 출력
    # print(nouns)

    # 사전에 입력
    keywords= OrderedDict()
    for i in nouns:
        keywords[i] = 0
        for spt in i.split():
            try : keywords[i] += keyword_counter[spt] 
            except : pass
        keywords[i] = round(keywords[i]/len(i.split()),2)
    keywords = OrderedDict(sorted(keywords.items(), reverse = True,key=lambda x: x[1]))
    top_keywords = OrderedDict(islice(keywords.items(), 10))

    # 큰 범위의 키워드 탐색을 위해 필터링함
    # 필터링 방법 : 키워드를 포함하는 상위 키워드가 있다면 그 중 키워드 점수가 가장 높은 것 하나만 선정
    words = list(keywords.keys())
    filtered_keywords= OrderedDict()
    
    max_value = '' ; max_keys = ''
    for key in keywords:
        temp_dict = OrderedDict()
        for word in words:
            if (is_contain(key, [word])== True) and (is_contain(word, words)== False):
                temp_dict[word] = keywords[word]
        try:        
            max_value = max(temp_dict.values())  # 사전의 값들 중 최대값을 구함
            max_keys = [key for key, value in temp_dict.items() if value == max_value]  # 최대값과 일치하는 키들을 리스트로 저장
        except : 
            pass

        for k in max_keys: 
            filtered_keywords[k] = keywords[k]

    filtered_keywords = OrderedDict(sorted(filtered_keywords.items(), reverse = True , key=lambda x: x[1]))
    top_filtered_keywords = OrderedDict(islice(filtered_keywords.items(), 10))
    
    # 기사 내용이 적으면 filtering 기법이 효용성이 떨어지므로 기사의 필터링키워드 사이즈에 따라 다른 방식으로 출력
    if len(list(top_filtered_keywords.keys())) > 5: 
        return top_filtered_keywords
    else :
        return top_keywords

    
# 뉴스 크롤링 함수들
# 입력된 수를 1, 11, 21, 31 ...만들어 주는 함수
def makePgNum(num):
    if num == 1:   return num
    elif num == 0: return num+1
    else:          return num+9*(num-1)

# 크롤링할 url 생성하는 함수 만들기(검색어, 크롤링 시작 페이지, 크롤링 종료 페이지)
def makeUrl(search, start_pg, end_pg,ds,de):
    date = f'&sort=0&photo=0&nso=so%3Ar%2Cp%3Afrom{ds}to{de}'
    if start_pg == end_pg:
        start_page = makePgNum(start_pg)
        url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + search + "&start=" + str(start_page)+date
        return url
    else:
        urls = []
        for i in range(start_pg, end_pg + 1):
            page = makePgNum(i)
            url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=" + search + "&start=" + str(page)+date
            urls.append(url)
        return urls    

# html에서 원하는 속성 추출하는 함수 만들기 (기사, 추출하려는 속성값)
def news_attrs_crawler(articles,attrs):
    attrs_content=[]
    for i in articles:
        attrs_content.append(i.attrs[attrs])
    return attrs_content
# ConnectionError방지


#html생성해서 기사크롤링하는 함수 만들기(url): 링크를 반환
def articles_crawler(url, i):
    #html 불러오기
    original_html = requests.get(i,headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"})
    html = BeautifulSoup(original_html.text, "html.parser")
    url_naver = html.select("div.group_news > ul.list_news > li div.news_area > div.news_info > div.info_group > a.info")
    url = news_attrs_crawler(url_naver,'href')
    return url

#제목, 링크, 내용 1차원 리스트로 꺼내는 함수 생성
def makeList(newlist, content):
    for i in content:
        for j in i:
            newlist.append(j)
    return newlist

# 앞선 뉴스크롤링 함수들을 이용하여 실질적인 뉴스크롤링을 진행하는 함수
def news(search, event, ds,de, event_dict ,page_num = 3):
    # naver url 생성
    page = 1
    page2 = page_num
    url = makeUrl(search, page, page2, ds,de)
    
    # 뉴스 크롤러 실행
    news_titles = []
    news_url = []
    news_contents = []
    news_dates = []
    
    for i in url:
        url = articles_crawler(url, i)
        news_url.append(url)

    # 제목, 링크, 내용 담을 리스트 생성
    news_url_1 = []

    # 1차원 리스트로 만들기(내용 제외)
    makeList(news_url_1, news_url)

    # NAVER 뉴스만 남기기
    final_urls = []
    for i in tqdm(range(len(news_url_1))):
        if "news.naver.com" in news_url_1[i]:
            final_urls.append(news_url_1[i])
        else:
            pass

    # 뉴스 내용 크롤링
    # 네이버 뉴스 크롤링 : 검색, page1, page2 입력
    # page당 10개 기사
    # 출처 : https://wonhwa.tistory.com/52
    for i in tqdm(final_urls):
        # 각 기사 html get하기
        news = requests.get(i, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"})
        news_html = BeautifulSoup(news.text, "html.parser")

        # 뉴스 제목 가져오기
        title = news_html.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
        if title is None:
            title = news_html.select_one("#content > div.end_ct > div > h2")

        # 뉴스 본문 가져오기
        content = news_html.select("div#dic_area")
        if not content:
            content = news_html.select("#articeBody")
            
        # 기사 텍스트만 가져오기 + 줄바꿈 처리
        content = ''.join(str(content))
        content = re.sub('([\s]{2,})', '', content)
        content = content.replace('<br/>','\n\n')
        content = re.sub('([~])', r'\\\1', content)
        
        
        # html 태그 제거 및 텍스트 다듬기
        pattern1 = '<[^>]*>'
        title = re.sub(pattern=pattern1, repl='', string=str(title))
        content = re.sub(pattern=pattern1, repl='', string=content)
        pattern2 = """[\n\n\n\n\n// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}"""
        content = content.replace(pattern2, '')

        news_titles.append(title)
        news_contents.append(content)
        

        try:
            html_date = news_html.select_one("div#ct> div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp > div > span")
            news_date = html_date.attrs['data-date-time']
        except AttributeError:
            news_date = news_html.select_one("#content > div.end_ct > div > div.article_info > span > em")
            news_date = re.sub(pattern=pattern1, repl='', string=str(news_date))

        # 날짜 가져오기
        news_dates.append(news_date)

    print("검색된 기사 갯수: 총 ", (page2 + 1 - page) * 10, '개')

    # 데이터 프레임 만들기
    news_df = pd.DataFrame({'date': news_dates, 'title': news_titles, 'link': final_urls, 'content': news_contents})

    # 중복 행 지우기
    news_df = news_df.drop_duplicates(keep='first', ignore_index=True)
    print("중복 제거 후 행 개수: ", len(news_df))

    # 키워드 필터링
    news_df['keyword'] = np.nan
    news_df['keyword'] = news_df['content'].apply(keyword_extract)
    news_df['noblank'] = np.nan
    news_df['noblank'] = news_df['content'].apply(noblank)
    news_df['len'] = news_df['noblank'].apply(len)

    if event == 'All' :
        event = list(event_dict.keys())
    filtered_news_df = filter_rows_by_words(df=news_df, column='noblank', words=event)
    new_event_dict = dict()


    if filtered_news_df['event'].sum(): # 검색된 이벤트가 0개가 아니어야함
        for event in set(filtered_news_df['event'].sum()):
            temp_df = filtered_news_df[filtered_news_df['event'].apply(lambda x: event in x)]
            temp_df = temp_df.sort_values(by='date', ascending=False).head(2)
            temp_df.index = range(len(temp_df))
            temp_df = temp_df.to_dict()
            new_event_dict[event] = temp_df

    return filtered_news_df, new_event_dict

# 뉴스 기사에서 이벤트 키워드를 하이라이팅하는 코드
def highlight_word(match): 
    word = match.group(0)
    return f'<span style="background-color:#EBEBEB; color: #FF4B4B;border-radius: 3px;padding: 1px;">{word}</span>'


# 크롤링한 뉴스 정보들을 streamlit에 디자인하여 표현하기 위한 함수
def news_print_detail(event, news_print, explain, event_dict):
    if event in list(news_print.keys()):
        news_title = [news_print[event][k]['title'][:10].strip()+"..." for k in range(len(news_print[event]))]               
        news_tabs= st.tabs(news_title)                        
        for i in range(len(news_print[event])):
            with news_tabs[i]:
                # 표의 너비 설정
                table_width = 100  # 퍼센트로 설정하려면 숫자를 백분율로 계산해야 합니다.

                # 표의 스타일 및 레이아웃을 위한 HTML 코드 생성
                table_html = f"""
                <table style="width: {table_width}%">
                  <colgroup><col style="width: 75%"><col style="width: 25%"></colgroup>                        
                  <tr><th>기사 제목</th><th>날짜</th></tr>
                  <tr><td><a href={news_print[event][i]['link']}>{news_print[event][i]['title']}</a></td>
                    <td>{news_print[event][i]["date"]}</td></tr>
                </table>
                """

                # 표 출력
                st.markdown(table_html, unsafe_allow_html=True)


                # 데이터프레임 생성 후 키워드 점수 기준으로 정렬
                temp_df = pd.DataFrame(list(news_print[event][i]['keyword'].items()), columns=['Keyword', 'Score'])
                temp_df['Keyword'] = temp_df['Keyword'].astype(str) + "(" + temp_df['Score'].apply(lambda x: str(x)) + ")"
                temp_df['Score'].astype(float)
                temp_df['News Keywords with Score'] = 'News Keywords with Score'
                temp_df = temp_df.sort_values('Score', ascending=False)

                highlighted_keyword = [ f"<span style='background-color:#EBEBEB; color: #FF4B4B;border-radius: 3px;\
                font-size: 15px;padding: 1px;'> {keyword}</span>" for keyword in list(temp_df['Keyword'].unique())]
                
                # border: 2px solid #F08F8F;
                # 표 스타일 설정
                table_style = f"""
                <table style="width: {table_width}%">                  
                  <tr><th>키워드 및 빈도점수</th></tr> <tr><td>{'  '.join(highlighted_keyword)}</td></tr>
                </table>
                """
                # 표 출력
                st.markdown(table_style, unsafe_allow_html=True)
                
                # 글자 키워드 하이라이팅
                all_text = news_print[event][i]['content'] 
                pattern = '[\s]{0,}'.join(list(event))
                highlighted_text = re.sub(pattern, highlight_word, all_text)

                # 표 스타일 설정
                table_style_news_text = f"""
                <table style="width: {table_width}%">                  
                  <tr><th>기사 전문</th></tr> <tr><td>{highlighted_text.strip('[ ]')}</td></tr>
                </table>
                """
                st.markdown(table_style_news_text, unsafe_allow_html=True)
    else :
        st.error(f'검색 조건 내 해당 이벤트 관련된 뉴스가 없습니다.')
        st.warning(f'찾으시는 정보가 있으시다면 page를 늘리고 검색기간을 구체적으로 좁혀보십시오')
    if explain == True:
        with st.expander(f"{event}에 대해 궁금하신가요?"): st.write(event_dict[event])
    
    return news_print 

# 크롤링한 뉴스들 중에서 해당 종목, 이벤트에 관련된 뉴스들을 필터링하여 news_print_detail를 이용해 출력하는 함수
def news_print(filtered_news_df, name , event_dict, intersect ,top_n = 3, explain = True):
    news_print = dict()
    if filtered_news_df['event'].sum(): event = list(set(filtered_news_df['event'].sum()))
    else : event = []
    
    for i in range(len(event)):
        new_print = dict()
        temp = filtered_news_df[filtered_news_df['event'].apply(lambda x: event[i] in x)]
        new_print[event[i]] = temp.sort_values(by = 'date', ascending = False).head(top_n)
        new_print[event[i]].index = range(len(new_print[event[i]]))
        news_print[event[i]] = new_print[event[i]].T.to_dict()
    
    st.subheader("Event-Based News List")
    if len(intersect) == 0:
        if len(news_print) > 0 : 
            event_tabs= st.tabs(news_print)
            for i, event in enumerate(news_print): 
                with event_tabs[i]:
                    news_print = news_print_detail(event, news_print, explain, event_dict)
        else :
            st.warning('정의한 모든 이벤트에 대해 해당되는 뉴스가 없습니다')
    else : 
        event_tabs= st.tabs(list(intersect))
        for j, event in enumerate(list(intersect)):
            with event_tabs[j]: # event, news_print
                news_print = news_print_detail(event, news_print, explain, event_dict)
    return news_print