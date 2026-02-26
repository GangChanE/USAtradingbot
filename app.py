import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
import matplotlib.pyplot as plt

# 페이지 설정
st.set_page_config(page_title="Magnificent 7 Beast Strategy", layout="wide")
st.title("🦁 Magnificent 7: 1x 야수 조련 대시보드")

# ---------------------------------------------------------
# 1. 전략 파라미터 (Method B)
# ---------------------------------------------------------
STRATEGY = {
    'SMH':  {'Drop': 1.5, 'Ent': 1.4, 'Ext': 2.7},
    'SLV':  {'Drop': 0.7, 'Ent': 1.3, 'Ext': 3.6},
    'URA':  {'Drop': 0.6, 'Ent': 1.9, 'Ext': 3.6},
    'USO':  {'Drop': 0.3, 'Ent': 1.1, 'Ext': 1.5},
    'FXI':  {'Drop': 1.0, 'Ent': 2.0, 'Ext': -0.1},
    'ARKK': {'Drop': 3.3, 'Ent': 3.1, 'Ext': 3.1},
    'TAN':  {'Drop': 2.8, 'Ent': 2.8, 'Ext': 2.3}
}

# ---------------------------------------------------------
# 2. 데이터 로드 및 분석 함수
# ---------------------------------------------------------
@st.cache_data
def analyze_market():
    tickers = list(STRATEGY.keys())
    data = yf.download(tickers, period="1y", progress=False)
    
    # 멀티인덱스 처리
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    report = []
    
    for ticker in tickers:
        try:
            df = data[[ticker]].dropna()
            closes = df[ticker].values
            p = STRATEGY[ticker]
            
            win = 20
            sigmas = np.full(len(closes), 999.0)
            slopes = np.full(len(closes), -999.0)
            x = np.arange(win)
            
            hold = False
            ent_slope = 0.0
            
            # 시뮬레이션
            for i in range(win, len(closes)):
                y = closes[i-win:i]
                s, inter, _, _, _ = linregress(x, y)
                std = np.std(y - (s*x + inter))
                
                curr_sigma = 999.0
                curr_slope = -999.0
                
                if std > 0: curr_sigma = (closes[i] - (s*(win-1)+inter)) / std
                if closes[i] > 0: curr_slope = (s / closes[i]) * 100
                
                sigmas[i] = curr_sigma
                slopes[i] = curr_slope
                
                if i < len(closes) - 1:
                    if not hold:
                        if curr_sigma <= -p['Ent']:
                            hold = True; ent_slope = curr_slope
                    else:
                        if curr_sigma >= p['Ext'] or curr_slope < (ent_slope - p['Drop']):
                            hold = False
            
            # 오늘자 상태
            today_sigma = sigmas[-1]
            today_slope = slopes[-1]
            today_price = closes[-1]
            
            status = "HOLDING" if hold else "WAITING"
            action = "HOLD"
            
            if hold:
                cut_slope = ent_slope - p['Drop']
                if today_sigma >= p['Ext']: action = "SELL (익절)"
                elif today_slope < cut_slope: action = "SELL (손절)"
                else: action = "HOLD (보유)"
                info = f"Stop Slope: {cut_slope:.2f}%"
            else:
                if today_sigma <= -p['Ent']: action = "BUY (진입)"
                else: action = "WAIT (대기)"
                info = f"Target Sigma: -{p['Ent']}"

            report.append({
                'Ticker': ticker,
                'Price': today_price,
                'Sigma': today_sigma,
                'Slope': today_slope,
                'Status': status,
                'Action': action,
                'Detail': info
            })
            
        except Exception as e:
            continue
            
    return pd.DataFrame(report)

# ---------------------------------------------------------
# 3. 화면 출력 (UI)
# ---------------------------------------------------------
if st.button('🚀 야수 신호 분석 시작'):
    with st.spinner('데이터 수집 및 3D 연산 중...'):
        df_res = analyze_market()
        
        st.success("분석 완료!")
        
        # 스타일링 함수
        def color_action(val):
            color = 'black'
            if 'BUY' in val: color = 'red'
            elif 'SELL' in val: color = 'blue'
            elif 'HOLD' in val: color = 'green'
            return f'color: {color}; font-weight: bold;'

        st.dataframe(df_res.style.map(color_action, subset=['Action']), use_container_width=True)
        
        st.markdown("### 📊 전략 가이드")
        st.info("""
        * **BUY:** 과매도 구간 진입 (즉시 매수 후 Slope 기록)
        * **SELL (손절):** 진입 시점보다 기울기(Slope)가 Drop Limit 이상 꺾임
        * **SELL (익절):** 과매수 Sigma 도달
        * **Method B:** 매도 자금은 즉시 다른 보유 종목에 재투자하십시오.
        """)
