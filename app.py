
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="당뇨병 관련 속성 분석", layout="wide")

st.title("당뇨병 진단 여부와 관련된 주요 속성 분석")
st.write("건강 데이터를 활용하여 당뇨병 진단 여부 예측에 중요한 속성을 분석합니다.")

# 1. 데이터 불러오기
df = pd.read_csv("diabetes.csv")

# 2. 분석에 사용할 속성
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

# 3. 데이터 전처리
data = df.copy()

zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

data[zero_columns] = data[zero_columns].replace(0, float("nan"))

# 4. 데이터 확인
st.subheader("데이터 확인")
st.write("전체 데이터 수:", len(data))
st.dataframe(data.head())

# 5. 학습 데이터와 테스트 데이터 분리
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. 결측값 처리 및 랜덤 포레스트 모델
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# 7. 속성 중요도 계산
importance = model.named_steps["classifier"].feature_importances_

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

# 8. 중요도 그래프
st.subheader("당뇨병 진단 여부 예측에 중요한 속성")

fig = px.bar(
    importance_df,
    x="중요도",
    y="속성",
    orientation="h",
    text=importance_df["중요도"].round(3),
    labels={
        "중요도": "속성 중요도",
        "속성": "건강 속성"
    },
    title="건강 속성별 중요도 비교"
)

fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True)

# 9. 가장 중요한 속성 표시
top_feature = importance_df.iloc[-1]

st.success(
    f"예측에 가장 중요한 속성: {top_feature['속성']} "
    f"(중요도: {top_feature['중요도']:.3f})"
)

st.caption(
    "※ 속성 중요도는 모델의 예측 기준이며 "
    "당뇨병의 직접적인 원인을 의미하지 않습니다."
)
