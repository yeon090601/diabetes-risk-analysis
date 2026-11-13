
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="당뇨병 데이터 분석",
    layout="wide"
)

st.title("건강 데이터를 활용한 당뇨병 진단 여부 분석")
st.write("데이터 전처리 및 랜덤 포레스트를 활용한 주요 속성 분석")

# 1. CSV 파일 불러오기
df = pd.read_csv("diabetes.csv")

st.subheader("1. 데이터 탐색")

st.write("전체 데이터 크기:", df.shape)
st.write("데이터 미리보기")
st.dataframe(df.head())

# 2. 분석에 사용할 속성 설정
features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

target = "Outcome"

# 3. 결측값 처리
st.subheader("2. 결측값 처리")

data = df.copy()

zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# 0값을 결측값으로 변환
data[zero_columns] = data[zero_columns].replace(
    0, float("nan")
)

# 결측값을 중앙값으로 대체
for col in features:
    data[col] = data[col].fillna(data[col].median())

st.success("0값을 결측값으로 처리하고 중앙값으로 대체했습니다.")

# 4. 이상치 탐지 및 제거
st.subheader("3. 이상치 탐지 및 제거")

Q1 = data[features].quantile(0.25)
Q3 = data[features].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outlier_condition = (
    (data[features] < lower) |
    (data[features] > upper)
).any(axis=1)

clean_data = data.loc[~outlier_condition].copy()

st.write("전처리 전 데이터 수:", len(df))
st.write("제거된 이상치 데이터 수:", int(outlier_condition.sum()))
st.write("전처리 후 데이터 수:", len(clean_data))

st.success("IQR 방법으로 이상치 제거를 완료했습니다.")

# 5. 랜덤 포레스트 모델 학습
st.subheader("4. 속성별 중요도 분석")

X = clean_data[features]
y = clean_data[target]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# 6. 속성 중요도 계산
importance = model.feature_importances_

importance_df = pd.DataFrame({
    "속성": [
        "임신 횟수",
        "혈당",
        "혈압",
        "피부 두께",
        "인슐린",
        "BMI",
        "가족력 관련 지표",
        "나이"
    ],
    "중요도": importance
})

importance_df = importance_df.sort_values(
    by="중요도",
    ascending=True
)

# 7. 막대그래프 시각화
st.subheader("5. 당뇨병 진단 여부 예측에 중요한 속성")

fig = px.bar(
    importance_df,
    x="중요도",
    y="속성",
    orientation="h",
    text=importance_df["중요도"].round(3),
    title="건강 속성별 중요도 비교",
    labels={
        "중요도": "속성 중요도",
        "속성": "건강 속성"
    }
)

fig.update_traces(textposition="outside")

st.plotly_chart(fig, use_container_width=True)

# 8. 중요도 순위 출력
top_feature = importance_df.iloc[-1]

st.success(
    f"예측에 가장 중요한 속성은 "
    f"{top_feature['속성']}입니다. "
    f"(중요도: {top_feature['중요도']:.3f})"
)

st.dataframe(
    importance_df.sort_values(
        by="중요도",
        ascending=False
    ).reset_index(drop=True)
)

st.caption(
    "※ 속성 중요도는 모델의 예측 기준이며 "
    "당뇨병의 직접적인 원인을 의미하지 않습니다."
)
