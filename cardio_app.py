import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ✅ XGBoost 모델 불러오기
@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.set_page_config(page_title="심혈관 위험 예측기", layout="wide")
st.title("💓 당신의 심혈관 건강은 안전한가요?")
st.markdown("""
**XGBoost 기반 심혈관 질환 위험 예측 & 건강 개선 제안**  
사용자의 건강 정보 입력 시 위험도를 예측하고, 주요 개선 항목도 제시합니다.
""")

# --- Sidebar Inputs ---
st.sidebar.header("📝 건강 정보 입력")
age = st.sidebar.slider("나이", 20, 80, 60)
gender = st.sidebar.radio("성별", ["남성", "여성"])
height = st.sidebar.number_input("키 (cm)", 140, 200, 170)
weight = st.sidebar.number_input("몸무게 (kg)", 40, 150, 65)
ap_hi = st.sidebar.slider("수축기 혈압", 90, 200, 120)
ap_lo = st.sidebar.slider("이완기 혈압", 40, 130, 80)
cholesterol = st.sidebar.selectbox("콜레스테롤 등급", [1, 2, 3])
gluc = st.sidebar.selectbox("혈당 등급", [1, 2, 3])
smoke = st.sidebar.checkbox("흡연 여부")
alco = st.sidebar.checkbox("음주 여부")
active = st.sidebar.checkbox("운동을 규칙적으로 하나요?")

# --- Data Preparation ---
bmi = weight / ((height / 100) ** 2)

# --- Data Preparation ---
bmi = weight / ((height / 100) ** 2)

input_data = {
    'age': age * 365,  # 모델 학습 당시 일 단위였다면
    'gender': 1 if gender == "남성" else 2,
    'ap_hi': ap_hi,
    'ap_lo': ap_lo,
    'cholesterol': cholesterol,
    'gluc': gluc,
    'smoke': int(smoke),
    'alco': int(alco),
    'active': int(active),
    'BMI': bmi  # ✅ XGBoost 학습에 사용된 피처!
}

input_df = pd.DataFrame([input_data])


input_df = pd.DataFrame([input_data])

# --- Risk Prediction ---
st.subheader("📊 예측 결과")
proba = model.predict_proba(input_df)[0][1]  # cardio=1 확률
st.metric(label="심혈관 질환 위험도", value=f"{proba * 100:.2f}%")

# --- 개선 제안 ---
suggestions = []
if ap_hi > 120:
    suggestions.append("💡 수축기 혈압 낮추기")
if ap_lo > 80:
    suggestions.append("💡 이완기 혈압 낮추기")
if bmi > 24.9:
    suggestions.append("💡 체중 감량")
if cholesterol > 1:
    suggestions.append("💡 콜레스테롤 개선")
if gluc > 1:
    suggestions.append("💡 혈당 개선")
if smoke:
    suggestions.append("💡 금연하기")
if alco:
    suggestions.append("💡 음주 줄이기")
if not active:
    suggestions.append("💡 운동 시작하기")

if suggestions:
    st.markdown("#### 개선 제안:")
    for s in suggestions:
        st.write(s)
else:
    st.success("현재 건강 상태는 양호합니다! 👍")

# --- 통합 혈압 시뮬레이터: 수축기 + 이완기 ---
st.subheader("시뮬레이터: 혈압 조정 시 위험도 변화")

# 슬라이더로 수축기, 이완기 혈압 조정
sim_ap_hi = st.slider("수축기 혈압 (mmHg)", 90, 200, ap_hi)
sim_ap_lo = st.slider("이완기 혈압 (mmHg)", 40, 130, ap_lo)

# 조정된 데이터로 예측
sim_data_combined = input_data.copy()
sim_data_combined['ap_hi'] = sim_ap_hi
sim_data_combined['ap_lo'] = sim_ap_lo
sim_df_combined = pd.DataFrame([sim_data_combined])
sim_proba_combined = model.predict_proba(sim_df_combined)[0][1]

# 결과 표시
st.info(f"혈압을 {ap_hi}/{ap_lo} → {sim_ap_hi}/{sim_ap_lo} mmHg로 조정하면\n\n위험도는 {sim_proba_combined*100:.2f}%입니다.")
