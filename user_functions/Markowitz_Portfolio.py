# Import Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import streamlit as st
from datetime import datetime, timedelta
from pykrx import stock
from pykrx import bond
import seaborn as sns; sns.set()
import scipy.optimize as opt

# plot 설정
plt.rc('font', family ='Malgun Gothic') # 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False #마이너스 기호 깨짐 방지

# 포트폴리오 종목들, 각 비중, 매수일, 매도일 입력시 백테스팅한 결과를 반환하는 함수 : (로그)수익률, 분산(표준편차), 일별 종가 df를 반환
def back_test(codes, weight, start_date, end_date):
    # 포트폴리오 df 생성
    portfolio_df = pd.DataFrame()
    for code in codes :
        temp = stock.get_market_ohlcv(start_date,end_date, code, adjusted = True)[['종가']] ; time.sleep(0.5)
        portfolio_df =  pd.concat([portfolio_df, temp], axis = 1)
    portfolio_df.columns = codes

    # 로그 수익률 -> 이걸 쓰는게 맞다
    log_ret = portfolio_df.apply(np.log).diff().dropna()
    log_returns = log_ret.cumsum(axis = 0).iloc[-1]
    portfolioLogReturn = np.sum(log_returns*weight)

    # 변동성 위험
    cov_matrix = log_ret.cov() * len(log_ret)
    port_variance = np.dot(weight.T, np.dot(cov_matrix, weight))
    port_volatility = np.sqrt(port_variance)

    percent_var = str(round(port_variance* 100, 2)) + '%'
    percent_vols = str(round(port_volatility* 100, 2)) + '%'
    percent_ret = str(round(portfolioLogReturn*100, 2))+'%'

    return log_ret, portfolio_df, percent_ret, percent_vols, percent_var

#평균수익률과 표준편차 반환하기 위한 함수
def ret_std(weight, ret, log_ret): 
    port_mean = np.sum(weight * ret.mean() *len(log_ret))               #포트폴리오 수익률의 평균구하기
    port_var = np.dot(weight.T, np.dot(ret.cov()*len(log_ret), weight)) #포트폴리오 수익률의 분산구하기
    port_std = np.sqrt(port_var)                                        #분산에 루트를 취해 표준편차 구한 모습
    return port_mean, port_std                                         #평균수익률과 표준편차 반환

#포트폴리오 수익률과 volatility, sharpe ratio반환하는 함수
def statistics(weights,ret, log_ret, rf=0): # statistics 선언(포트폴리오 수익률, volatility, sharpe ratio반환)
    weights = np.array(weights)             #weight를 array변환
    pret = np.sum(ret.mean() * weights) * len(log_ret) - rf  #포트폴리오 수익률 생성
    pvol = np.sqrt(np.dot(weights.T, np.dot(ret.cov() * len(log_ret), weights))) #포트폴리오 volatility 생성
    return np.array([pret, pvol, pret / pvol])  #포트폴리오 수익률, volatility, sharpe ratio반환

#-sharpe ratio 반환하는 함수 
def min_func_sharpe(weights,ret, log_ret, rf=0): 
    return -statistics(weights,ret, log_ret, rf)[2]

#volatility^2 반환하는 함수
def min_func_volatility(weights,ret, log_ret): 
    return statistics(weights,ret, log_ret)[1] **2 

# 랜덤하게 5000개의 포트폴리오를 생성하고 구성종목의 비중을 랜덤하게 부여하는 함수 : 최적포트폴리오 탐색 및 시각화 목적으로 생성
def make_random_portfolio(ret, noa, log_ret):
    port_rets = [] ; port_std = []
    for w in range(5000): #랜덤하게 나머지 비중을 부여하여 5000개 포트만들기
        weight = np.random.random(noa) #종목개수만큼 랜덤한 숫자를 부여함.(랜덤하게 비중 부여하는 작업)
        weight /= np.sum(weight) #포트전체비율에서 5종목이 랜덤하게 나누어 가지도록 부여
        mu, sig = ret_std(np.array(weight), ret, log_ret) #ret_std함수로 수익률의 평균과 표준편차 반환
        port_rets.append(mu) #ret 리스트에 수익률 평균 append
        port_std.append(sig) #std 리스트에 수익률 표준편차 append
    return port_rets, port_std

# 포트폴리오이론을 기반으로 최적포트폴리오 그래프, 비율, 샤프지수 등을 반환하는 함수
def portfolio_opt(port_rets, port_std, noa, ret, portfolio_df, log_ret, end_date):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    # plt.style.use('seaborn')

    #다음은 최적화 함수에 들어가는 argument들을 모아놓은 것임
    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})  #가중치의 합이 1이되도록(100%)가 되도록 제약
    bnds = tuple((0,1) for i in range((noa)))   # ((0, 1), (0, 1), (0, 1), (0, 1)) 0~1까지의 값을 가질 수 있음
    k = noa * [1. / noa,] # initial guess 

    #다음은 -sharpe ratio를 최소화하는 작업임(sharpe ratio 최대화)
    opts = opt.minimize(min_func_sharpe, k, method='SLSQP',
                           bounds=bnds, constraints=cons, args=(ret, log_ret))
    #다음은 volatility^2를 최소화하는 작업임
    optv = opt.minimize(min_func_volatility, k, method='SLSQP',
                           bounds=bnds, constraints=cons, args=(ret, log_ret))

    axes[0].scatter(port_std, port_rets, #x축에 포트폴리오 표준편차, y축에 평균 수익률을 점(.)으로 표시
                c=np.array(port_rets) / np.array(port_std), marker='.', cmap = 'Greys')#  sr을 좌측에 하양,회색의 cmap으로 표시
                # random portfolio composition

    axes[0].plot(statistics(opts['x'],ret, log_ret)[1], statistics(opts['x'],ret, log_ret)[0], #sharpe ratio가 가장 높은 port를 빨간 별
             'r*', markersize=15.0, label = 'Portfolio with highest Sharpe Ratio') #라벨링 'Portfolio with highest Sharpe Ratio'
                # portfolio with highest Sharpe ratio
    axes[0].plot(statistics(optv['x'],ret, log_ret)[1], statistics(optv['x'],ret, log_ret)[0],#variance가 가장 낮은 port를 노란 별
             'y*', markersize=15.0, label = 'Minimum variance portfolio') #라벨링 'Minimum variance portfolio'
                # minimum variance portfolio

    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xlabel('variance')
    axes[0].set_ylabel('expected return')

    rf = bond.get_otc_treasury_yields(end_date.strftime('%Y%m%d')).iloc[3,0]/100  #risk free asset 수익률: 10년물 국채 수익률 실시간 스크래핑
    slope = (statistics(opts['x'],ret, log_ret)[0] - rf) / statistics(opts['x'],ret, log_ret)[1] #mean variance frontier의 기울기 구하기
    x =  np.linspace(0.0,0.24,5000) #0.0부터 0.24까지 일정간격으로 5000개 x 생성
    y = [x*slope + rf for x in np.linspace(0.0,0.24,5000)]#mean variance frontier의 y값들 설정(x에 기울기 곱하고 rf더한 값들)


    axes[1].scatter(port_std, port_rets,  #x축에 포트폴리오 표준편차, y축에 평균 수익률을 점(.)으로 표시
                c=np.array(port_rets) / np.array(port_std), marker='.',cmap = 'Greys') #  sr을 좌측에 흰색,회색의 cmap으로 표시  

    axes[1].plot(x,y, label = 'mean-variance frontier with riskfree asset') #mean variance frontier를 그린다. 
    axes[1].plot(statistics(opts['x'],ret, log_ret)[1], statistics(opts['x'],ret, log_ret)[0],#sharpe ratio가 가장 높은 port를 빨간 별
             'r*', markersize=15.0, label = 'Portfolio with highest Sharpe Ratio')#라벨링 'Portfolio with highest Sharpe Ratio'
                # portfolio with highest Sharpe ratio

    axes[1].legend() #상단에 범례 출력
    axes[1].grid(True) # 격자 생성
    axes[1].set_xlabel('expected volatility')#x축에 라벨링 'expected volatility'     
    axes[1].set_ylabel('expected return') #y축에 라벨링 'expected return'     
    
    sharpe_ratio = statistics(opts['x'],ret, log_ret,rf)[2]
    opt_weight = {portfolio_df.columns[i] : round(opts['x'][i],10)  for i in range(noa)}
    
    return fig, sharpe_ratio, opt_weight