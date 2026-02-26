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
# 1. 전략 파라미터 (SMH는 로직용, 실제 매매는 SOXQ)
# ---------------------------------------------------------
STRATEGY = {
    'SMH':  {'Drop': 1.5, 'Ent': 1.4, 'Ext': 2.7}, # 로직은 SMH 기준
    'SLV':  {'Drop': 0.7, 'Ent': 1.3, 'Ext': 3.6},
    'URA':  {'Drop': 0.6, 'Ent': 1.9, 'Ext': 3.6},
    'USO':  {'Drop': 0.3, 'Ent': 1.1, 'Ext': 1.5},
    'FXI':  {'Drop': 1.0, 'Ent': 2.0, 'Ext': -0.1},
    'ARKK': {'Drop': 3.3, 'Ent': 3.1, 'Ext': 3.1},
    'TAN':  {'Drop': 2.8, 'Ent': 2.8, 'Ext': 2.3}
}

# 대체재 매핑 (로직 종목 -> 실제 매매 종목)
PROXY_MAP = {
    'SMH': 'SOXQ'
}

# ---------------------------------------------------------
# 2. 데이터 분석 및 가격 역산 함수
# ---------------------------------------------------------
@st.cache_data(ttl=1800) # 30분 캐시
def analyze_market_pro():
    # 로직용 티커와 실제 매매용 티커 모두 다운로드
    logic_tickers = list(STRATEGY.keys())
    trade_tickers = list(PROXY_MAP.values())
    all_tickers = list(set(logic_tickers + trade_tickers))
    
    try:
        data = yf.download(all_tickers, period="1y", progress=False)
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
    
    for ticker in logic_tickers:
        try:
            if ticker not in data.columns: continue
            
            # --- [1] SMH(로직 종목) 데이터로 신호 계산 ---
            series = data[ticker].dropna()
            closes = series.values
            if len(closes) < 30: continue
            
            p = STRATEGY[ticker]
            win = 20
            x = np.arange(win)
            
            # 히스토리 시뮬레이션 (Slope/Sigma 계산)
            hold = False
            entry_slope = 0.0
            
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
                    if curr_sigma >= p['Ext'] or curr_slope < (entry_slope - p['Drop']):
                        hold = False

            # --- [2] 오늘자 지표 계산 ---
            y_last = closes[-win:]
            s, inter, _, _, _ = linregress(x, y_last)
            L = s*(win-1) + inter 
            std = np.std(y_last - (s*x + inter))
            
            today_price_logic = closes[-1] # SMH 가격
            today_slope = (s / today_price_logic) * 100 if today_price_logic > 0 else 0
            today_sigma = (today_price_logic - L) / std if std > 0 else 0
            
            # --- [3] 가격 변환 로직 (SMH -> SOXQ) ---
            display_ticker = ticker
            today_price_display = today_price_logic
            conversion_ratio = 1.0
            
            # 대체재가 있는 경우 (SMH -> SOXQ)
            if ticker in PROXY_MAP:
                proxy_ticker = PROXY_MAP[ticker]
                if proxy_ticker in data.columns:
                    proxy_price = data[proxy_ticker].dropna().iloc[-1]
                    conversion_ratio = proxy_price / today_price_logic # 가격 비율 계산
                    
                    display_ticker = f"{proxy_ticker} (via {ticker})"
                    today_price_display = proxy_price
            
            # 목표가 역산 (로직 종목 기준 -> 비율 적용 -> 실제 종목 가격)
            # Target Price = (L + (Target_Sigma * std)) * Ratio
            target_buy_price = (L + (-p['Ent'] * std)) * conversion_ratio
            target_sell_price_sigma = (L + (p['Ext'] * std)) * conversion_ratio
            
            # 상태 판단
            status = "HOLDING" if hold else "WAITING"
            action = "HOLD"
            
            display_ent_sigma = f"-{p['Ent']} (${target_buy_price:.2f})"
            display_ext_sigma = f"{p['Ext']} (${target_sell_price_sigma:.2f})"
            display_ent_slope = "-"
            display_stop_slope = "-"
            
            if hold:
                cut_slope_limit = entry_slope - p['Drop']
                display_ent_slope = f"{entry_slope:.2f}%"
                display_stop_slope = f"{cut_slope_limit:.2f}%"
                
                if today_sigma >= p['Ext']: action = "SELL (익절)"
                elif today_slope < cut_slope_limit: action = "SELL (손절)"
                else: action = "HOLD (보유)"
            else:
                if today_sigma <= -p['Ent']:
                    action = "BUY (진입)"
                    display_ent_slope = f"{today_slope:.2f}% (New)"
                    display_stop_slope = f"{today_slope - p['Drop']:.2f}% (Est)"
                else:
                    action = "WAIT (대기)"

            report.append({
                'Ticker': display_ticker,
                'Action': action,
                'Price': today_price_display,
                'Cur Sigma': today_sigma,
                'Cur Slope': today_slope,
                'Entry Sigma (Price)': display_ent_sigma,
                'Exit Sigma (Price)': display_ext_sigma,
                'Entry Slope': display_ent_slope,
                'Stop Slope': display_stop_slope
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
    
    def text_color_action(val):
        if 'BUY' in val: return 'color: green; font-weight: bold;'
        if 'SELL' in val: return 'color: red; font-weight: bold;'
        if 'HOLD' in val: return 'color: blue; font-weight: bold;'
        return 'color: black;'

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
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**🟢 BUY (진입)**\n\n즉시 매수. SOXQ의 경우 SMH의 신호를 받아 자동 계산된 가격입니다.")
    with c2:
        st.error("**🔴 SELL (청산)**\n\n익절/손절 신호. 보유 종목을 전량 매도하고 리밸런싱하세요.")
    with c3:
        st.warning("**🔵 HOLD (보유)**\n\n추세 지속 중. Stop Slope 이탈 여부만 체크하세요.")

else:
    st.error("데이터 수집 실패. 잠시 후 새로고침 해주세요.")
