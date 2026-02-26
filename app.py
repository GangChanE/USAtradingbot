import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
import time

# 페이지 설정
st.set_page_config(page_title="Magnificent 7 Pro Dashboard", layout="wide")

st.title("🦁 Magnificent 7: 1x 야수 조련 프로 대시보드")
st.markdown("---")

# ---------------------------------------------------------
# 1. 전략 파라미터 (Method B)
# ---------------------------------------------------------
STRATEGY = {
    'SOXQ':  {'Drop': 1.5, 'Ent': 1.4, 'Ext': 2.7},
    'SLV':  {'Drop': 0.7, 'Ent': 1.3, 'Ext': 3.6},
    'URA':  {'Drop': 0.6, 'Ent': 1.9, 'Ext': 3.6},
    'USO':  {'Drop': 0.3, 'Ent': 1.1, 'Ext': 1.5},
    'FXI':  {'Drop': 1.0, 'Ent': 2.0, 'Ext': -0.1},
    'ARKK': {'Drop': 3.3, 'Ent': 3.1, 'Ext': 3.1},
    'TAN':  {'Drop': 2.8, 'Ent': 2.8, 'Ext': 2.3}
}

# ---------------------------------------------------------
# 2. 데이터 분석 및 가격 역산 함수
# ---------------------------------------------------------
@st.cache_data(ttl=1800) # 30분 캐시
def analyze_market_pro():
    tickers = list(STRATEGY.keys())
    
    try:
        data = yf.download(tickers, period="1y", progress=False)
    except Exception as e:
        st.error(f"데이터 다운로드 실패: {e}")
        return pd.DataFrame()

    if data.empty: return pd.DataFrame()

    # 컬럼 정리
    if isinstance(data.columns, pd.MultiIndex):
        try:
            if 'Close' in data.columns.get_level_values(0):
                data = data['Close']
            else:
                data.columns = data.columns.get_level_values(0)
        except: pass

    report = []
    
    for ticker in tickers:
        try:
            if ticker not in data.columns: continue
            
            series = data[ticker].dropna()
            closes = series.values
            if len(closes) < 30: continue
            
            p = STRATEGY[ticker]
            win = 20
            x = np.arange(win)
            
            # 시뮬레이션 상태 변수
            hold = False
            entry_slope = 0.0 # 진입 시점의 기울기
            
            # 전체 히스토리 시뮬레이션 (현재 보유 상태 추적용)
            for i in range(win, len(closes)-1):
                y = closes[i-win:i]
                s, inter, _, _, _ = linregress(x, y)
                std = np.std(y - (s*x + inter))
                
                curr_sigma = 999.0
                curr_slope = -999.0
                if std > 0: curr_sigma = (closes[i] - (s*(win-1)+inter)) / std
                if closes[i] > 0: curr_slope = (s / closes[i]) * 100
                
                if not hold:
                    if curr_sigma <= -p['Ent']:
                        hold = True; entry_slope = curr_slope
                else:
                    # 익절 or 손절(Slope Drop)
                    if curr_sigma >= p['Ext'] or curr_slope < (entry_slope - p['Drop']):
                        hold = False

            # --- [오늘자 데이터 정밀 분석] ---
            # 오늘 기준 지표 계산
            y_last = closes[-win:]
            s, inter, _, _, _ = linregress(x, y_last)
            L = s*(win-1) + inter # 회귀선 상의 오늘 적정가
            std = np.std(y_last - (s*x + inter)) # 표준편차
            
            today_price = closes[-1]
            today_slope = (s / today_price) * 100 if today_price > 0 else 0
            today_sigma = (today_price - L) / std if std > 0 else 0
            
            # --- [가격 역산 로직] ---
            # Sigma = (Price - L) / std  =>  Price = L + (Sigma * std)
            target_buy_price = L + (-p['Ent'] * std)
            target_sell_price_sigma = L + (p['Ext'] * std)
            
            # 상태 판단
            status = "HOLDING" if hold else "WAITING"
            action = "HOLD" # 기본값
            
            # 화면 표시용 변수들
            display_ent_sigma = f"-{p['Ent']} (${target_buy_price:.2f})"
            display_ext_sigma = f"{p['Ext']} (${target_sell_price_sigma:.2f})"
            display_ent_slope = "-"
            display_stop_slope = "-"
            
            if hold:
                # 보유 중일 때
                cut_slope_limit = entry_slope - p['Drop']
                display_ent_slope = f"{entry_slope:.2f}%"
                display_stop_slope = f"{cut_slope_limit:.2f}%"
                
                if today_sigma >= p['Ext']:
                    action = "SELL (익절)"
                elif today_slope < cut_slope_limit:
                    action = "SELL (손절)"
                else:
                    action = "HOLD (보유)"
            else:
                # 대기 중일 때
                if today_sigma <= -p['Ent']:
                    action = "BUY (진입)"
                    # 진입 신호 발생 시, 오늘 슬로프가 진입 슬로프가 됨
                    display_ent_slope = f"{today_slope:.2f}% (New)"
                    display_stop_slope = f"{today_slope - p['Drop']:.2f}% (Est)"
                else:
                    action = "WAIT (대기)"

            report.append({
                'Ticker': ticker,
                'Action': action,          # 매매 신호 (가장 중요)
                'Price': today_price,      # 현재가
                'Cur Sigma': today_sigma,  # 현재 시그마
                'Cur Slope': today_slope,  # 현재 기울기
                'Entry Sigma (Price)': display_ent_sigma, # 목표 진입가
                'Exit Sigma (Price)': display_ext_sigma,  # 목표 청산가
                'Entry Slope': display_ent_slope,         # 진입 시 기울기
                'Stop Slope': display_stop_slope          # 손절 기준 기울기
            })
            
        except Exception as e:
            continue
            
    return pd.DataFrame(report)

# ---------------------------------------------------------
# 3. UI 렌더링
# ---------------------------------------------------------

with st.spinner('🦁 야수들의 맥박(Slope)을 측정 중입니다...'):
    df_res = analyze_market_pro()

if not df_res.empty:
    st.success(f"데이터 업데이트: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 컬러링 함수 정의
    def highlight_rows(row):
        action = row['Action']
        color = ''
        if 'BUY' in action:
            return ['background-color: #d4edda; color: #155724; font-weight: bold'] * len(row) # 연한 초록 배경
        elif 'SELL' in action:
            return ['background-color: #f8d7da; color: #721c24; font-weight: bold'] * len(row) # 연한 빨강 배경
        elif 'HOLD' in action:
            return ['background-color: #e2e3e5; color: #383d41'] * len(row) # 회색 배경
        return [''] * len(row)

    def text_color_action(val):
        if 'BUY' in val: return 'color: green; font-weight: bold;'
        if 'SELL' in val: return 'color: red; font-weight: bold;'
        if 'HOLD' in val: return 'color: blue; font-weight: bold;'
        return 'color: black;'

    # 데이터프레임 스타일 적용
    st.dataframe(
        df_res.style
        .map(text_color_action, subset=['Action'])
        .format({
            'Price': '${:.2f}',
            'Cur Sigma': '{:.2f}',
            'Cur Slope': '{:.2f}%'
        })
        .set_properties(**{'text-align': 'center'})
        .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
        , 
        use_container_width=True,
        hide_index=True
    )
    
    # 범례 및 설명
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**🟢 BUY (진입)**\n\n현재 시그마가 'Entry Sigma' 이하로 떨어졌습니다. 즉시 매수하고 자금을 태우세요.")
    with c2:
        st.error("**🔴 SELL (청산)**\n\n익절(Sigma 도달)하거나 손절(Slope 꺾임) 신호입니다. 즉시 매도하여 현금을 확보하거나 리밸런싱하세요.")
    with c3:
        st.warning("**🔵 HOLD (보유)**\n\n아직 추세가 살아있습니다. 'Stop Slope' 밑으로 기울기가 떨어지지 않는지 매일 체크하세요.")

else:
    st.error("데이터 수집 실패. 잠시 후 새로고침 해주세요.")
