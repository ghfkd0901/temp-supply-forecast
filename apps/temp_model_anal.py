import os
import streamlit as st
import pandas as pd
import holidays
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from pathlib import Path

def run():
    st.markdown("""
        <h1 style='text-align: center;'>🌡️ 3D 평균기온 예측 (최저/최고기온 → 평균기온)</h1>
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
        df.dropna(subset=['평균기온', '최고기온', '최저기온'], inplace=True)
        return df[['날짜', '평균기온', '최고기온', '최저기온']]

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
    data_filtered = data_filtered.dropna(subset=["최저기온", "최고기온", "평균기온"])

    # ✅ 3D 시각화 함수
    def plot_3d_models(df):
        X = df[["최저기온", "최고기온"]].values
        y = df["평균기온"].values
        year_list = sorted(df["연"].unique())

        # 연도별 색상 (이산형 그라데이션)
        colors = sample_colorscale("Viridis", [i / (len(year_list)-1) for i in range(len(year_list))])
        color_map = {year: colors[i] for i, year in enumerate(year_list)}

        models = {
            "선형회귀": LinearRegression(),
            "랜덤포레스트": RandomForestRegressor(max_depth=5, random_state=0)
        }

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=[f"<b style='font-size:16px'>{name}</b>" for name in models.keys()]
        )

        for i, (name, model) in enumerate(models.items()):
            row, col = 1, i + 1

            # ✅ 연도별 산점도 trace
            for year in year_list:
                df_year = df[df["연"] == year]
                fig.add_trace(go.Scatter3d(
                    x=df_year["최저기온"],
                    y=df_year["최고기온"],
                    z=df_year["평균기온"],
                    mode='markers',
                    name=str(year),
                    marker=dict(size=3, color=color_map[year]),
                    legendgroup=str(year),
                    showlegend=(i == 0)
                ), row=row, col=col)

            # ✅ 모델 예측 surface
            model.fit(X, y)
            r2 = model.score(X, y)

            x_range = np.linspace(df["최저기온"].min(), df["최저기온"].max(), 30)
            y_range = np.linspace(df["최고기온"].min(), df["최고기온"].max(), 30)
            x_grid, y_grid = np.meshgrid(x_range, y_range)
            pred_input = np.c_[x_grid.ravel(), y_grid.ravel()]
            z_pred = model.predict(pred_input).reshape(x_grid.shape)

            fig.add_trace(go.Surface(
                x=x_grid, y=y_grid, z=z_pred,
                colorscale='Reds',
                showscale=False,
                opacity=0.6,
                name=f"{name} 예측면",
                hoverinfo="skip"
            ), row=row, col=col)

            # ✅ R² 값 표시
            fig.add_annotation(
                text=f"<b>R² = {r2:.3f}</b>",
                xref="paper", yref="paper",
                x=0.23 + 0.5 * i, y=1.02,
                showarrow=False,
                font=dict(size=16, color="black")
            )

        # ✅ 축 제목 설정 (scene, scene2)
        fig.update_layout(
            height=850,
            width=1150,
            title="📈 최저/최고기온 기반 평균기온 예측 (3D 모델 시각화)",
            legend_title="연도",
            scene=dict(
                xaxis_title="최저기온",
                yaxis_title="최고기온",
                zaxis_title="평균기온"
            ),
            scene2=dict(
                xaxis_title="최저기온",
                yaxis_title="최고기온",
                zaxis_title="평균기온"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # ✅ 실행
    plot_3d_models(data_filtered)

if __name__ == "__main__":
    run()
