import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
import time

# ---------------------------------------------------------
# ⚙️ 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="2x Real Beast Strategy",
    page_icon="🦁",
    layout="wide"
)

# ---------------------------------------------------------
# ⚙️ 1. 전략 파라미터 (Logic: 1배수 기준)
# ---------------------------------------------------------
# 시그널은 여기서 찾습니다 (역사가 긴 1배수 형님들)
STRATEGY = {
    'SMH':  {'Drop': 1.5, 'Ent': 1.4, 'Ext': 2.7, 'Theme': '반도체'},
    'SLV':  {'Drop': 0.7, 'Ent': 1.3, 'Ext': 3.6, 'Theme': '은(Silver)'},
    'URA':  {'Drop': 0.6, 'Ent': 1.9, 'Ext': 3.6, 'Theme': '우라늄'},
    'USO':  {'Drop': 0.3, 'Ent': 1.1, 'Ext': 1.5, 'Theme': '원유'},
    'FXI':  {'Drop': 1.0, 'Ent': 2.0, 'Ext': -0.1,'Theme': '중국'},
    'ARKK': {'Drop': 3.3, 'Ent': 3.1, 'Ext': 3.1, 'Theme': '혁신기업'},
    'TAN':  {'Drop': 2.8, 'Ent': 2.8, 'Ext': 2.3, 'Theme': '태양광'}
}

# ---------------------------------------------------------
# ⚙️ 2. 매매 매핑 (Trade: 2배수 야수들)
# ---------------------------------------------------------
# 실제 계좌에 담을 2배수 동생들
PROXY_MAP = {
    'SMH':  {'Ticker': 'USD',  'Name': 'ProShares Ultra Semi (2x)'},
    'SLV':  {'Ticker': 'AGQ',  'Name': 'ProShares Ultra Silver (2x)'},
    'URA':  {'Ticker': 'URA',  'Name': 'Global X Uranium (1x)'}, # 2배수 없음 (그대로)
    'USO':  {'Ticker': 'UCO',  'Name': 'ProShares Ultra Oil (2x)'},
    'FXI':  {'Ticker': 'XPP',  'Name': 'ProShares Ultra China (2x)'},
    'ARKK': {'Ticker': 'TARK', 'Name': 'AXS 2X Innovation (2x)'}, # TARK 추가 완료!
    'TAN':  {'Ticker': 'TAN',  'Name': 'Invesco Solar (1x)'}     # 2배수 없음 (그대로)
}

# ---------------------------------------------------------
# ⚙️ 3. 데이터 분석 및 가격 역산 함수
# ---------------------------------------------------------
@st.cache_data(ttl=900) # 15분마다 갱신 (실전용)
def analyze_market_real_beast():
    # 로직용(1x)과 매매용(2x) 티커 모두 수집
    logic_tickers = list(STRATEGY.keys())
    trade_tickers = [v['Ticker'] for v in PROXY_MAP.values()]
    all_tickers = list(set(logic_tickers + trade_tickers))
    
    try:
        data = yf.download(all_tickers, period="2y", progress=False) # 넉넉하게 2년
    except Exception as e:
        st.error(f"데이터 다운로드 실패: {e}")
        return pd.DataFrame()

    if data.empty: return pd.DataFrame()

    # 멀티인덱스 처리
    if isinstance(data.columns, pd.MultiIndex):
        try:
            if 'Close' in data.columns.get_level_values(0):
                data = data['Close']
            else:
                data.columns = data.columns.get_level_values(0)
        except: pass

    report = []
    
    for logic_tk in logic_tickers:
        try:
            # 1. 데이터 유효성 체크
            if logic_tk not in data.columns: continue
            
            # 매매할 티커 정보 가져오기
            trade_info = PROXY_MAP[logic_tk]
            trade_tk = trade_info['Ticker']
            
            # 매매 티커 데이터가 없으면 로직 티커로 대체 (혹시 모를 오류 방지)
            if trade_tk not in data.columns:
                current_price = 0.0
            else:
                current_price = data[trade_tk].dropna().iloc[-1]

            # 2. 로직(1x) 계산
            series = data[logic_tk].dropna()
            closes = series.values
            if len(closes) < 30: continue
            
            p_params = STRATEGY[logic_tk]
            win = 20
            x = np.arange(win)
            
            # 히스토리 시뮬레이션 (현재 보유 상태 추적)
            hold = False
            entry_slope = 0.0
            
            for i in range(win, len(closes)-1):
                y = closes[i-win:i]
                s, inter, _, _, _ = linregress(x, y)
                std = np.std(y - (s*x + inter))
                
                # 당시 지표
                curr_sigma = 999.0
                curr_slope = -999.0
                if std > 0: curr_sigma = (closes[i] - (s*(win-1)+inter)) / std
                if closes[i] > 0: curr_slope = (s / closes[i]) * 100
                
                if not hold:
                    if curr_sigma <= -p_params['Ent']:
                        hold = True; entry_slope = curr_slope
                else:
                    if curr_sigma >= p_params['Ext'] or curr_slope < (entry_slope - p_params['Drop']):
                        hold = False

            # 3. 오늘자 지표 (Live)
            y_last = closes[-win:]
            s, inter, _, _, _ = linregress(x, y_last)
            L = s*(win-1) + inter 
            std = np.std(y_last - (s*x + inter))
            
            logic_price = closes[-1]
            today_slope = (s / logic_price) * 100 if logic_price > 0 else 0
            today_sigma = (logic_price - L) / std if std > 0 else 0
            
            # 4. 가격 변환 (Logic Price -> Trade Price)
            # 목표 Sigma 도달 시점의 예상 가격을 2배수 ETF 가격으로 환산
            conversion_ratio = 1.0
            if logic_price > 0 and current_price > 0:
                conversion_ratio = current_price / logic_price
            
            # 목표가 역산 (1x 기준) -> 환산 (2x 기준)
            target_buy_price_1x = L + (-p_params['Ent'] * std)
            target_sell_price_1x = L + (p_params['Ext'] * std)
            
            display_buy_price = target_buy_price_1x * conversion_ratio
            display_sell_price = target_sell_price_1x * conversion_ratio
            
            # 5. 상태 판단 및 메시지
            status = "HOLDING" if hold else "WAITING"
            action = "HOLD"
            
            display_ent_sigma = f"-{p_params['Ent']} (${display_buy_price:.2f})"
            display_ext_sigma = f"{p_params['Ext']} (${display_sell_price:.2f})"
            display_ent_slope = "-"
            display_stop_slope = "-"
            
            if hold:
                cut_slope_limit = entry_slope - p_params['Drop']
                display_ent_slope = f"{entry_slope:.2f}%"
                display_stop_slope = f"{cut_slope_limit:.2f}%"
                
                if today_sigma >= p_params['Ext']:
                    action = "SELL (익절)"
                elif today_slope < cut_slope_limit:
                    action = "SELL (손절)"
                else:
                    action = "HOLD (보유)"
            else:
                if today_sigma <= -p_params['Ent']:
                    action = "BUY (진입)"
                    display_ent_slope = f"{today_slope:.2f}% (New)"
                    display_stop_slope = f"{today_slope - p_params['Drop']:.2f}% (Est)"
                else:
                    action = "WAIT (대기)"

            # 6. 리포트 데이터 생성
            # 이름 포맷: "USD (SMH/반도체)"
            display_name = f"{trade_tk} ({logic_tk}/{p_params['Theme']})"
            
            report.append({
                'Display Name': display_name,
                'Ticker': trade_tk,
                'Action': action,
                'Price': current_price,
                'Cur Sigma': today_sigma,
                'Cur Slope': today_slope,
                'Target Buy': display_ent_sigma,
                'Target Sell': display_ext_sigma,
                'Entry Slope': display_ent_slope,
                'Stop Slope': display_stop_slope,
                'Theme': p_params['Theme']
            })
            
        except Exception as e:
            continue
            
    return pd.DataFrame(report)

# ---------------------------------------------------------
# ⚙️ 4. UI 렌더링
# ---------------------------------------------------------
st.title("🦁 2x Real Beast Strategy")
st.markdown("### `MDD -41%`를 견디는 자에게 `CAGR 61%`가 있으라.")
st.caption(f"Last Update: {time.strftime('%Y-%m-%d %H:%M:%S')} | Data Source: Yahoo Finance")
st.markdown("---")

# 데이터 로딩
with st.spinner('야수의 심장으로 시장을 스캔 중입니다... (2배수 로딩)'):
    df_res = analyze_market_real_beast()

if not df_res.empty:
    
    # 스타일링 함수
    def text_color_action(val):
        if 'BUY' in val: return 'color: #2ecc71; font-weight: bold;' # 밝은 초록
        if 'SELL' in val: return 'color: #e74c3c; font-weight: bold;' # 밝은 빨강
        if 'HOLD' in val: return 'color: #3498db; font-weight: bold;' # 밝은 파랑
        return 'color: #95a5a6;'

    # 메인 테이블 출력
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
        hide_index=True,
        column_order=['Display Name', 'Action', 'Price', 'Cur Sigma', 'Cur Slope', 'Target Buy', 'Target Sell', 'Entry Slope', 'Stop Slope']
    )
    
    # 요약 카드 (Summary Cards)
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    
    buy_list = df_res[df_res['Action'].str.contains('BUY')]['Ticker'].tolist()
    sell_list = df_res[df_res['Action'].str.contains('SELL')]['Ticker'].tolist()
    hold_list = df_res[df_res['Action'].str.contains('HOLD')]['Ticker'].tolist()
    
    with c1:
        st.success(f"**🟢 BUY ({len(buy_list)})**")
        if buy_list:
            for t in buy_list: st.write(f"- {t} 진입!")
        else: st.caption("진입 신호 없음")
        
    with c2:
        st.error(f"**🔴 SELL ({len(sell_list)})**")
        if sell_list:
            for t in sell_list: st.write(f"- {t} 청산/손절!")
        else: st.caption("청산 신호 없음")
        
    with c3:
        st.info(f"**🔵 HOLD ({len(hold_list)})**")
        if hold_list:
            for t in hold_list: st.write(f"- {t} 보유 중")
        else: st.caption("보유 종목 없음")

    # 가이드
    with st.expander("📖 **[Real Beast] 매매 원칙 (필독)**", expanded=True):
        st.markdown("""
        1.  **시그널과 매매의 분리:** * 신호는 안정적인 `1배수 ETF(SMH, SLV 등)`에서 찾습니다.
            * 매매는 화끈한 `2배수 ETF(USD, AGQ 등)`로 실행합니다.
        2.  **이벤트 드리븐 리밸런싱 (Event-Driven):**
            * **`BUY`** 또는 **`SELL`** 신호가 뜬 날에만 계좌를 확인합니다.
            * 신호가 뜨면 **[전량 매도(현금화) + 신규/기존 종목 1/N 재매수]**를 수행하여 비중을 맞춥니다.
            * 신호가 없는 날은 HTS를 켜지 마십시오. (세금 절약 + 멘탈 관리)
        3.  **야수의 심장:** * MDD -40%는 시스템 오류가 아니라 **'스프링이 눌리는 과정'**입니다. 절대 쫄지 마십시오.
            * 2배수 레버리지는 횡보장에서 녹습니다. 하지만 추세가 터지면 그 모든 손실을 한 방에 만회합니다.
        """)

else:
    st.error("데이터 수집 실패. 잠시 후 새로고침(F5) 해주세요.")
