import streamlit as st
import pandas as pd
from pathlib import Path
from apps.temp_forecast import run as run_temp_forecast
from apps.supply_forecast import run as run_supply_forecast
from apps.temp_model_anal import run as run_temp_model_anal
from apps.supply_model_anal import run as run_supply_model_anal

# 마지막 날짜 반환 함수
def get_last_date():
    DATA_PATH = Path.cwd() / "data" / "weather_supply.csv"
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    return df["date"].iloc[-1]


# 페이지 구성은 반드시 첫 줄에서 설정
st.set_page_config(layout="wide", page_title="기온 및 공급량 예측")

# ✅ 사이드바 메뉴 (라디오 버튼 사용)
st.sidebar.title("📌 페이지 선택")
selected_page = st.sidebar.radio(
    "이동할 페이지를 선택하세요:",
    (
        "일별 기온 예측",
        "일별 공급량 예측",
        "기온 예측 모델 분석",
        "공급량 예측 모델 분석"
    )
)

# ✅ 페이지 라우팅
if selected_page == "일별 기온 예측":
    run_temp_forecast()
elif selected_page == "일별 공급량 예측":
    run_supply_forecast()
elif selected_page == "기온 예측 모델 분석":
    run_temp_model_anal()
elif selected_page == "공급량 예측 모델 분석":
    run_supply_model_anal()

# ✅ 마지막 날짜 정보 표시
last_date = get_last_date()
st.sidebar.markdown(f"<hr><small>📅 데이터 최신 날짜: <b>{last_date}</b></small>", unsafe_allow_html=True)
