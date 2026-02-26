import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
import time

# 페이지 설정 (가장 위에 있어야 함)
st.set_page_config(page_title="Magnificent 7 Beast Strategy", layout="wide")

# 제목
st.title("🦁 Magnificent 7: 1x 야수 조련 대시보드")
st.markdown("---")

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
# 2. 데이터 로드 및 분석 함수 (캐싱 적용)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)  # 1시간마다 갱신
def analyze_market():
    tickers = list(STRATEGY.keys())
    
    # 데이터 다운로드 (에러 방지용 예외처리 강화)
    try:
        data = yf.download(tickers, period="1y", progress=False)
    except Exception as e:
        st.error(f"데이터 다운로드 중 치명적 오류 발생: {e}")
        return pd.DataFrame()

    # 데이터가 비어있으면 빈 DF 반환
    if data.empty:
        return pd.DataFrame()
    
    # 멀티인덱스 컬럼 처리 (yfinance 버전 호환성)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            # 'Close' 레벨이 있으면 가져오고, 없으면 레벨 0을 가져옴
            if 'Close' in data.columns.get_level_values(0):
                data = data['Close']
            else:
                data.columns = data.columns.get_level_values(0)
        except:
            pass # 구조가 단순하면 패스
    
    # Close 컬럼만 남기기 (가끔 Adj Close 등이 섞일 때 대비)
    # yfinance 최신 버전은 'Close' 밑에 티커가 있는 구조일 수 있음
    
    report = []
    
    for ticker in tickers:
        try:
            # 해당 티커의 데이터만 추출
            if ticker in data.columns:
                series = data[ticker].dropna()
            else:
                # 데이터 구조가 다를 경우 (단일 종목 다운로드 등)
                continue
                
            closes = series.values
            if len(closes) < 20: continue # 데이터 너무 적으면 패스
            
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
                
                # 어제까지의 상태 추적
                if i < len(closes) - 1:
                    if not hold:
                        if curr_sigma <= -p['Ent']:
                            hold = True; ent_slope = curr_slope
                    else:
                        if curr_sigma >= p['Ext'] or curr_slope < (ent_slope - p['Drop']):
                            hold = False
            
            # 오늘자 상태 판단
            today_sigma = sigmas[-1]
            today_slope = slopes[-1]
            today_price = closes[-1]
            
            status = "HOLDING" if hold else "WAITING"
            action = "HOLD"
            info = ""
            
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
            # 개별 종목 에러는 무시하고 진행
            continue
            
    return pd.DataFrame(report)

# ---------------------------------------------------------
# 3. 메인 실행 로직 (자동 실행)
# ---------------------------------------------------------

# 로딩 표시
with st.spinner('🦁 야수들이 깨어나고 있습니다... (데이터 수집 중)'):
    df_res = analyze_market()

# 결과 출력
if not df_res.empty:
    st.success(f"분석 완료! ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    
    # 스타일링 함수 (오류 방지를 위해 컬럼 존재 여부 확인)
    def color_action(val):
        color = 'black'
        if 'BUY' in val: color = 'red'
        elif 'SELL' in val: color = 'blue'
        elif 'HOLD' in val: color = 'green'
        return f'color: {color}; font-weight: bold;'

    # 데이터프레임 표시 (높이 자동 조절)
    st.dataframe(
        df_res.style.map(color_action, subset=['Action']).format({
            'Price': '{:.2f}',
            'Sigma': '{:.2f}',
            'Slope': '{:.2f}%'
        }), 
        use_container_width=True,
        hide_index=True
    )
    
    # 전략 가이드 표시
    with st.expander("📊 전략 운용 가이드 (Click to open)", expanded=True):
        st.info("""
        * **BUY (진입):** 과매도 구간 진입 (즉시 매수 후 Slope 기록)
        * **SELL (손절):** 진입 시점보다 기울기(Slope)가 Drop Limit 이상 꺾임
        * **SELL (익절):** 과매수 Sigma 도달
        * **HOLD (보유):** 아직 청산 신호 없음
        * **운용 원칙 (Method B):** 매도 자금은 즉시 다른 보유 종목에 재투자하십시오. (현금 보유 0원 원칙)
        """)

else:
    st.error("⚠️ 데이터를 불러오지 못했습니다.")
    st.warning("""
    **가능한 원인:**
    1. 야후 파이낸스(yfinance) 서버 일시적 오류
    2. 데이터 다운로드 속도가 너무 느림
    3. 티커명이 변경됨
    
    **해결책:** 잠시 후 새로고침(F5)을 눌러주세요.
    """)
