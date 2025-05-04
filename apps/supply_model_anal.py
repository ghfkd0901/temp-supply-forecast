import os
import streamlit as st
import pandas as pd
import holidays
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from datetime import datetime
from pathlib import Path
from plotly.colors import sample_colorscale

def run():
    st.markdown("""
        <h1 style='text-align: center;'>📊 최적의 예측 모델 학습 데이터 결정</h1>
    """, unsafe_allow_html=True)

    # ✅ 데이터 로드
    BASE_DIR = Path(os.getcwd())
    DATA_PATH = BASE_DIR / "data" / "weather_supply.csv"

    @st.cache_data
    def load_data():
        df = pd.read_csv(DATA_PATH, encoding='utf-8')
        column_mapping = {
            'date': '날짜',
            'avg_temp': '평균기온',
            'max_temp': '최고기온',
            'min_temp': '최저기온',
            'supply_m3': '공급량(M3)',
            'supply_mj': '공급량(MJ)',
        }
        df.rename(columns=column_mapping, inplace=True)
        df.dropna(subset=['평균기온', '최고기온', '최저기온', '공급량(M3)'], inplace=True)
        return df[['날짜', '평균기온', '최고기온', '최저기온', '공급량(M3)', '공급량(MJ)']]

    def add_columns(df):
        df = df.copy()
        df['날짜'] = pd.to_datetime(df['날짜'])
        df['연'] = df['날짜'].dt.year
        df['월'] = df['날짜'].dt.month
        df['요일'] = df['날짜'].dt.weekday.map({0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"})
        kr_holidays = holidays.KR(years=df['연'].unique())
        df['공휴일'] = df['날짜'].apply(lambda x: kr_holidays.get(x, ""))
        return df

    data = add_columns(load_data())

    # ✅ 필터링
    st.sidebar.header("필터링 옵션")
    selected_year = st.sidebar.multiselect("연도", sorted(data["연"].unique()), default=sorted(data["연"].unique()))
    selected_month = st.sidebar.multiselect("월", sorted(data["월"].unique()), default=sorted(data["월"].unique()))
    selected_weekday = st.sidebar.multiselect("요일", ["월","화","수","목","금","토","일"], default=["월","화","수","목","금","토","일"])
    exclude_holidays = st.sidebar.checkbox("공휴일 제외", value=False)

    data_filtered = data[
        data["연"].isin(selected_year) &
        data["월"].isin(selected_month) &
        data["요일"].isin(selected_weekday)
    ]
    if exclude_holidays:
        data_filtered = data_filtered[data_filtered["공휴일"] == ""]
    data_filtered = data_filtered.dropna(subset=["평균기온", "공급량(M3)"])

    # ✅ 시각화 함수
    def plot_all_models(df):
        x = df["평균기온"].values.reshape(-1, 1)
        y = df["공급량(M3)"].values

        models = {
            "다항회귀 (3차)": PolynomialFeatures(degree=3),
            "KNN": KNeighborsRegressor(n_neighbors=5),
            "결정트리": DecisionTreeRegressor(max_depth=5, random_state=0),
            "랜덤포레스트": RandomForestRegressor(max_depth=5, random_state=0)
        }

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"<b style='font-size:16px'>{name}</b>" for name in models.keys()]
        )

        # ✅ 연도별 색상 (이산형 but 그라데이션)
        year_list = sorted(df["연"].unique())
        colors = sample_colorscale("Viridis", [i / (len(year_list)-1) for i in range(len(year_list))])
        color_map = {year: colors[i] for i, year in enumerate(year_list)}

        for i, (name, model) in enumerate(models.items()):
            row = i // 2 + 1
            col = i % 2 + 1

            # ✅ 연도별 산점도 trace 생성
            for year in year_list:
                year_data = df[df["연"] == year]
                fig.add_trace(go.Scatter(
                    x=year_data["평균기온"],
                    y=year_data["공급량(M3)"],
                    mode='markers',
                    name=str(year),
                    marker=dict(color=color_map[year], size=4),
                    legendgroup=str(year),
                    showlegend=(i == 0)  # 첫번째 subplot만 legend 표시
                ), row=row, col=col)

            # ✅ 예측선 + 회귀 계산
            x_model = df["평균기온"].values.reshape(-1, 1)
            y_model = df["공급량(M3)"].values
            x_range = np.linspace(x_model.min(), x_model.max(), 100).reshape(-1, 1)

            if name == "다항회귀 (3차)":
                poly = PolynomialFeatures(degree=3)
                x_poly = poly.fit_transform(x_model)
                linreg = LinearRegression().fit(x_poly, y_model)
                y_pred = linreg.predict(poly.transform(x_range))
                r2 = r2_score(y_model, linreg.predict(x_poly))
                coef = linreg.coef_
                intercept = linreg.intercept_
                equation = f"y = {coef[3]:.3f}x³ + {coef[2]:.3f}x² + {coef[1]:.3f}x + {intercept:.3f}"

                fig.add_trace(go.Scatter(
                    x=x_range.flatten(), y=y_pred,
                    mode='lines', line=dict(color='red'), name="3차 회귀", showlegend=False
                ), row=row, col=col)

                fig.add_annotation(
                    text=equation,
                    xref=f"x{i+1}", yref=f"y{i+1}",
                    x=x_model.min(), y=y_model.max(),
                    showarrow=False,
                    font=dict(size=12, color="red"),
                    row=row, col=col
                )

            else:
                model.fit(x_model, y_model)
                y_pred = model.predict(x_range)
                r2 = model.score(x_model, y_model)
                fig.add_trace(go.Scatter(
                    x=x_range.flatten(), y=y_pred,
                    mode='lines', line=dict(color='blue'), showlegend=False
                ), row=row, col=col)

            # ✅ R² 값 표시
            fig.add_annotation(
                text=f"<b>R² = {r2:.3f}</b>",
                xref=f"x{i+1}", yref=f"y{i+1}",
                x=x_model.max(), y=y_model.max(),
                showarrow=False,
                font=dict(size=18, color="black"),
                xanchor="right",
                row=row, col=col
            )

        fig.update_layout(
            height=900,
            width=1000,
            title_text="📈 모델별 평균기온 vs 공급량(M3) 예측 결과",
            legend_title="연도",
            legend=dict(itemsizing="constant")
        )
        st.plotly_chart(fig, use_container_width=True)

    # ✅ 실행
    plot_all_models(data_filtered)

if __name__ == "__main__":
    run()
