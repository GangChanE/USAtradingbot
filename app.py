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
    page_title="US 5 Beasts V18.0 (Alpha Hunter)",
    page_icon="🦅",
    layout="wide"
)

# ---------------------------------------------------------
# ⚙️ 1. 전략 파라미터 (V18.0 미국 Set 1: Alpha Hunter)
# 시그널(sig)은 1배수로 분석하고, 실제 매매는 3배수로 진행
# ---------------------------------------------------------
BEASTS = {
    'TECL': {'sig': 'XLK',  'name': 'TECL (미국기술주 3x)', 'ent': 3.8, 'ext': 2.3, 'drop': 0.7, 'r_ent': 0.02, 'r_ext': 0.02, 'theme': '기술주'},
    'SOXL': {'sig': 'SOXX', 'name': 'SOXL (반도체 3x)',   'ent': 3.9, 'ext': 2.1, 'drop': 0.4, 'r_ent': 0.02, 'r_ext': 0.02, 'theme': '반도체'},
    'NAIL': {'sig': 'XHB',  'name': 'NAIL (주택건설 3x)', 'ent': 2.6, 'ext': 1.8, 'drop': 0.2, 'r_ent': 0.02, 'r_ext': 0.02, 'theme': '건설'},
    'YINN': {'sig': 'FXI',  'name': 'YINN (중국대형주 3x)', 'ent': 3.8, 'ext': 1.9, 'drop': 1.4, 'r_ent': 0.03, 'r_ext': 0.02, 'theme': '중국'},
    'TMF':  {'sig': 'TLT',  'name': 'TMF (장기국채 3x)',  'ent': 3.2, 'ext': -0.1,'drop': 0.1, 'r_ent': 0.02, 'r_ext': 0.02, 'theme': '안전판'}
}

# ---------------------------------------------------------
# ⚙️ 사이드바 (사용자 상태 및 세금 관리)
# ---------------------------------------------------------
with st.sidebar:
    st.header("💼 내 포트폴리오 상태")
    st.markdown("현재 실제 계좌에 보유 중인 야수와 **투입 비율**을 선택해 주세요.")
    
    user_portfolio = {}
    for tk, info in BEASTS.items():
        st.markdown(f"**{info['name']}**")
        status = st.radio(
            f"상태 선택 ({info['name']})",
            options=["미보유 (0%)", "1차 진입 완료 (50%)", "2차 추매 완료 (100%)", "1차 익절 완료 (50% 남음)"],
            key=tk,
            label_visibility="collapsed"
        )
        if status != "미보유 (0%)":
            user_portfolio[tk] = status
            
    st.markdown("---")
    st.header("💸 양도소득세(22%) 관리기")
    st.markdown("올해 1월 1일부터 현재까지 확정된 **누적 실현 수익(달러)**을 입력하세요. 1~5월 매매 시 세금을 빼고 재투자해야 합니다.")
    annual_profit = st.number_input("올해 실현 수익 ($)", min_value=0.0, value=0.0, step=100.0)
    
    tax_liability = annual_profit * 0.22
    st.metric("내년 5월 예상 양도세 (Reserve)", f"${tax_liability:,.2f}")
    if tax_liability > 0:
        st.warning(f"⚠️ 야수 익절 시 **${tax_liability:,.2f}**는 재투자하지 말고 달러 예수금(KOFR 등)으로 묶어두세요!")

# ---------------------------------------------------------
# ⚙️ 2. 데이터 분석 엔진 (1x 시그널 기반 3x 타점 계산)
# ---------------------------------------------------------
@st.cache_data(ttl=900)
def analyze_us_beasts(portfolio):
    # 1배수(시그널)와 3배수(타격) 티커 모두 수집
    sig_tickers = [v['sig'] for v in BEASTS.values()]
    trd_tickers = list(BEASTS.keys())
    all_tickers = list(set(sig_tickers + trd_tickers))
    
    try:
        data_close = yf.download(all_tickers, period="1y", progress=False)['Close'].ffill()
        data_high = yf.download(all_tickers, period="1y", progress=False)['High'].ffill()
        data_low = yf.download(all_tickers, period="1y", progress=False)['Low'].ffill()
    except Exception as e:
        st.error(f"데이터 다운로드 에러: {e}")
        return pd.DataFrame(), []
        
    if data_close.empty: 
        return pd.DataFrame(), []

    report = []
    missing_beasts = [] 
    
    for tk in trd_tickers:
        sig_tk = BEASTS[tk]['sig']
        
        if tk not in data_close.columns or sig_tk not in data_close.columns: 
            missing_beasts.append(BEASTS[tk]['name'])
            continue
            
        # 기준 지표는 모두 시그널(1배수) 차트로 계산
        sig_closes = data_close[sig_tk].dropna().values
        sig_highs = data_high[sig_tk].dropna().values
        sig_lows = data_low[sig_tk].dropna().values
        
        # 가격 표시는 유저가 실제로 사는 3배수 차트로 표시
        trd_closes = data_close[tk].dropna().values
        
        if len(sig_closes) < 30: 
            missing_beasts.append(BEASTS[tk]['name'])
            continue
        
        p = BEASTS[tk]
        win = 20
        x = np.arange(win)
        
        # 오늘자(마지막 거래일) 1배수 지표 계산
        y_last = sig_closes[-win:]
        s, inter, _, _, _ = linregress(x, y_last)
        L = s*(win-1) + inter 
        std = np.std(y_last - (s*x + inter))
        
        today_sig_price = sig_closes[-1]
        today_trd_price = trd_closes[-1] # 실제 3x 가격
        
        today_slope = (s / today_sig_price) * 100 if today_sig_price > 0 else 0
        today_sigma = (today_sig_price - L) / std if std > 0 else 0
        
        # 🌟 V18.0 미국장 동적 스케일링 판별 🌟
        action = "WAIT (대기)"
        target_info = f"진입대기 (목표 Sig -{p['ent']:.1f})"
        
        if tk in portfolio:
            status = portfolio[tk]
            
            # [최근 10일 기울기 평균 - 손절 기준선 계산]
            recent_slopes = []
            for i in range(len(sig_closes)-10, len(sig_closes)):
                sy, _inter, _, _, _ = linregress(x, sig_closes[i-win:i])
                recent_slopes.append((sy/sig_closes[i])*100)
            avg_ent_slope = np.mean(recent_slopes)
            
            # 1. 1차 진입 완료 상태 (50% 보유)
            if status == "1차 진입 완료 (50%)":
                recent_low = np.min(sig_lows[-10:]) 
                bounce_rate = (today_sig_price - recent_low) / recent_low
                
                if today_slope < (avg_ent_slope - p['drop']):
                    action = "🛑 SELL ALL (손절)"
                    target_info = f"기울기 꺾임 (현재 {today_slope:.2f}% < 기준 {avg_ent_slope - p['drop']:.2f}%)"
                elif bounce_rate >= p['r_ent']:
                    action = "🔥 BUY 50% (2차 추매)"
                    target_info = f"저점대비 +{bounce_rate*100:.1f}% 반등 (목표 {p['r_ent']*100}%)"
                else:
                    action = "HOLD 50% (관망)"
                    target_info = f"반등 대기 (현재 +{bounce_rate*100:.1f}%)"

            # 2. 2차 진입 완료 상태 (100% 보유)
            elif status == "2차 추매 완료 (100%)":
                if today_slope < (avg_ent_slope - p['drop']):
                    action = "🛑 SELL ALL (손절)"
                    target_info = f"기울기 꺾임 (현재 {today_slope:.2f}% < 기준 {avg_ent_slope - p['drop']:.2f}%)"
                elif today_sigma >= p['ext']:
                    action = "💰 SELL 50% (1차 익절)"
                    target_info = f"과매수 도달 (Sig {today_sigma:.2f} >= {p['ext']:.1f})"
                else:
                    action = "HOLD 100% (관망)"
                    target_info = f"슈팅 대기 (현재 Sig {today_sigma:.2f})"

            # 3. 1차 익절 완료 상태 (50% 보유)
            elif status == "1차 익절 완료 (50% 남음)":
                recent_high = np.max(sig_highs[-10:])
                drop_rate = (recent_high - today_sig_price) / recent_high
                
                if drop_rate >= p['r_ext']:
                    action = "📉 SELL ALL (최종 익절)"
                    target_info = f"고점대비 -{drop_rate*100:.1f}% 하락 (목표 {p['r_ext']*100}%)"
                else:
                    action = "HOLD 50% (관망)"
                    target_info = f"하락 대기 (현재 -{drop_rate*100:.1f}%)"
                    
        else:
            # 미보유 상태
            if today_sigma <= -p['ent']:
                action = "🛒 BUY 50% (1차 진입)"
                target_info = f"과매도 도달 (Sig {today_sigma:.2f} <= -{p['ent']:.1f})"
            else:
                action = "WAIT (대기)"
                target_info = f"진입 대기 (현재 Sig {today_sigma:.2f})"
                
        report.append({
            'Theme': p['theme'],
            'Signal(1x)': sig_tk,
            'Trade(3x)': p['name'],
            'Action': action,
            'Price(3x)': float(today_trd_price),
            'Sigma(1x)': float(today_sigma),
            'Slope(1x)': float(today_slope),
            'Status / Target': target_info
        })

    return pd.DataFrame(report), missing_beasts

# ---------------------------------------------------------
# ⚙️ 3. 웹 UI 렌더링
# ---------------------------------------------------------
st.title("🦅 The Quantum Oracle V18.0 (US Alpha Hunter)")
st.caption(f"Last Update: {time.strftime('%Y-%m-%d %H:%M:%S')} | Logic: 1x Signal -> 3x Execution & Tax Reserve")
st.markdown("---")

with st.spinner("월스트리트 야수들의 실시간 시그널을 분석 중입니다..."):
    df_res, missing_beasts = analyze_us_beasts(user_portfolio) 

if missing_beasts:
    st.warning(f"⚠️ 야후 파이낸스 서버 오류로 데이터 누락: **{', '.join(missing_beasts)}**")

if not df_res.empty:
    st.subheader("📊 미국 5야수 실시간 시그널 대시보드")
    
    def text_color_action(val):
        if 'BUY' in val: return 'color: #155724; background-color: #d4edda; font-weight: bold;'
        if 'SELL' in val: return 'color: #721c24; background-color: #f8d7da; font-weight: bold;'
        if 'HOLD' in val: return 'color: #004085; background-color: #cce5ff; font-weight: bold;'
        if 'WAIT' in val: return 'color: #856404; background-color: #fff3cd;'
        return 'color: #383d41; background-color: #e2e3e5;'

    st.dataframe(
        df_res.style
        .map(text_color_action, subset=['Action'])
        .format({
            'Price(3x)': '${:,.2f}',
            'Sigma(1x)': '{:.2f}',
            'Slope(1x)': '{:.2f}%'
        })
        .set_properties(**{'text-align': 'center'})
        .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
        , 
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.subheader("📝 미국장 행동 강령 (Action Plan)")
    
    action_items = df_res[df_res['Action'].str.contains('BUY|SELL')]
    
    if not action_items.empty:
        for _, row in action_items.iterrows():
            if 'SELL ALL' in row['Action']:
                if '손절' in row['Action']:
                    st.error(f"🚨 **[전량 손절]** {row['Trade(3x)']} : {row['Status / Target']} -> 오늘 밤 프리장/본장에서 남은 물량 100% 매도!")
                else:
                    st.error(f"📉 **[최종 익절]** {row['Trade(3x)']} : {row['Status / Target']} -> 추세 꺾임 확인. 오늘 밤 남은 물량 100% 매도!")
            elif 'SELL 50%' in row['Action']:
                st.warning(f"💰 **[절반 익절]** {row['Trade(3x)']} : {row['Status / Target']} -> 과매수 도달! 오늘 밤 보유 물량의 50% 1차 익절!")
            elif 'BUY 50% (2차' in row['Action']:
                st.success(f"🔥 **[불타기 추매]** {row['Trade(3x)']} : {row['Status / Target']} -> 반등 확인! 오늘 밤 남은 현금 100% 투입!")
            elif 'BUY 50% (1차' in row['Action']:
                st.info(f"🛒 **[1차 진입]** {row['Trade(3x)']} : {row['Status / Target']} -> 피의 바다 도달. 오늘 밤 할당 비중의 50%만 매수!")
    else:
        if user_portfolio:
            st.success("▶️ **[HOLD]** 현재 야수가 사냥 중입니다. 미국장은 변동성이 큽니다. 시그널이 뜰 때까지 푹 주무십시오.")
        else:
            st.success("🏦 **[100% CASH PARKING]** 달러 예수금 파킹 통장(KOFR 등)에서 연 4% 이자를 받으며 편안하게 관망하십시오.")
