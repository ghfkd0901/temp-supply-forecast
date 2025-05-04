import os
import streamlit as st
import pandas as pd
import numpy as np
import holidays
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from pathlib import Path

def run():
    # ✅ 페이지 설정
    st.title("🌡️ 일별 기온 예측")

    # ✅ 경로 설정
    BASE_DIR = Path(os.getcwd())
    DATA_PATH = BASE_DIR / "data" / "weather_supply.csv"
    MODEL_DIR = Path("D:/Streamlit_Project/temp_suppy_forecast/models")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)  # 모델 저장 폴더 생성

    # ✅ 1️⃣ 데이터 로드 함수
    @st.cache_data
    def load_data():
        df = pd.read_csv(DATA_PATH, encoding='utf-8', sep=',')
        column_mapping = {
            'date': '날짜',
            'avg_temp': '평균기온',
            'max_temp': '최고기온',
            'min_temp': '최저기온',
            'supply_m3': '공급량(M3)',
            'supply_mj': '공급량(MJ)',
        }
        df.rename(columns=column_mapping, inplace=True)
        return df[['날짜', '평균기온', '최고기온', '최저기온', '공급량(M3)', '공급량(MJ)']]

    # ✅ 2️⃣ 컬럼 추가 함수
    def add_columns(df):
        df = df.copy()
        df['날짜'] = pd.to_datetime(df['날짜'])
        df['연'] = df['날짜'].dt.year
        df['월'] = df['날짜'].dt.month
        df['일'] = df['날짜'].dt.day
        weekday_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
        df['요일'] = df['날짜'].dt.weekday.map(weekday_map)
        kr_holidays = holidays.KR(years=df['연'].unique())
        df['공휴일'] = df['날짜'].apply(lambda x: kr_holidays.get(x, ""))
        return df

    # ✅ 3️⃣ 데이터 로드 및 컬럼 추가
    data = load_data()
    data = add_columns(data)

    # ✅ 4️⃣ 모델 훈련 및 로드
    def train_and_save_models():
        data_clean = data[['최고기온', '최저기온', '평균기온']].dropna()
        X_temp = data_clean[['최고기온', '최저기온']]
        y_temp = data_clean['평균기온']
        
        temp_model_linear = LinearRegression().fit(X_temp, y_temp)
        with open(MODEL_DIR / 'temp_model_linear.pkl', 'wb') as f:
            pickle.dump(temp_model_linear, f)

        temp_model_rf = RandomForestRegressor(random_state=42).fit(X_temp, y_temp)
        with open(MODEL_DIR / 'temp_model_rf.pkl', 'wb') as f:
            pickle.dump(temp_model_rf, f)

        return temp_model_linear, temp_model_rf

    try:
        with open(MODEL_DIR / 'temp_model_linear.pkl', 'rb') as f:
            temp_model_linear = pickle.load(f)
        with open(MODEL_DIR / 'temp_model_rf.pkl', 'rb') as f:
            temp_model_rf = pickle.load(f)
        st.success("✅ 모델 로드 완료")
    except:
        st.warning("⚠️ 모델 파일이 없어 훈련을 시작합니다...")
        temp_model_linear, temp_model_rf = train_and_save_models()
        st.success("✅ 모델 훈련 및 저장 완료!")

    # ✅ 5️⃣ 외부 링크 안내
    st.markdown("""
        <h5 style='text-align:left'>🌤️ 참조 가능한 외부 날씨 예측 사이트:</h5>
        <ul>
            <li><a href='https://www.accuweather.com/ko/kr/daegu/223347/may-weather/223347' target='_blank'>아큐웨더</a></li>
            <li><a href='https://www.kr-weathernews.com/mv4/html/weekly.html?loc=2700000000' target='_blank'>웨더뉴스</a></li>
            <li><a href='https://www.weather.go.kr/w/weather/forecast/short-term.do#dong/2729057600/35.8416384/128.5029888/%EB%8C%80%EA%B5%AC%EA%B4%91%EC%97%AD%EC%8B%9C%20%EB%8B%AC%EC%84%9C%EA%B5%AC%20%EC%9D%B4%EA%B3%A11%EB%8F%99/LOC/%EC%9C%84%EA%B2%BD%EB%8F%84(35.84,128.50)' target='_blank'>기상청</a></li>
            <li><a href='https://weather.com/ko-KR/weather/monthly/l/f4d5899d30e9ae766d244a8dffa5a0b7392144e65066032b688e4b84b25643a5' target='_blank'>웨더채널</a></li>
        </ul>
    """, unsafe_allow_html=True)

    # ✅ 6️⃣ 예측 UI
    st.sidebar.title("📅 예측 기간 설정")
    today = datetime.today()
    start_date = st.sidebar.date_input("시작일", today)
    end_date = st.sidebar.date_input("종료일", datetime(today.year, today.month + 1, 1) - timedelta(days=1))

    if start_date > end_date:
        st.sidebar.error("시작일은 종료일보다 이전이어야 합니다.")

    date_range = pd.date_range(start=start_date, end=end_date)
    date_df = pd.DataFrame({'날짜': date_range})
    date_df['최고기온'] = None
    date_df['최저기온'] = None

    st.write("### 🔢 최고기온, 최저기온 입력")
    edited_df = st.data_editor(date_df, num_rows="dynamic")

    # ✅ 7️⃣ 예측 실행
    if st.button("예측하기"):
        if edited_df[['최고기온', '최저기온']].isnull().any().any():
            st.error("❌ 모든 날짜의 최고기온과 최저기온을 입력해주세요.")
        else:
            X_pred = edited_df[['최고기온', '최저기온']]
            edited_df['평균기온(선형회귀)'] = temp_model_linear.predict(X_pred).round(1)
            edited_df['평균기온(랜덤포레스트)'] = temp_model_rf.predict(X_pred).round(1)
            st.session_state['result_temp_df'] = edited_df.copy()

    # ✅ 8️⃣ 결과 표시
    if 'result_temp_df' in st.session_state:
        st.write("### 📈 예측 결과")
        st.dataframe(st.session_state['result_temp_df'])

if __name__ == "__main__":
    run()
