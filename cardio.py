# cardio.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pickle
import shap
import google.generativeai as genai

# ───────────────────────────────────────────────────────────────
# 기본 설정
# ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="심혈관 위험 예측기", layout="wide")

# ───────────────────────────────────────────────────────────────
# Gemini 설정: 키 로드 + 경량 모델 강제 + 캐시 + 폴백
# ───────────────────────────────────────────────────────────────
def _load_api_key():
    try:
        return st.secrets["google"]["api_key"]
    except Exception:
        return os.getenv("GOOGLE_API_KEY", "")

API_KEY = _load_api_key()
genai.configure(api_key=API_KEY)

GEMINI_MODEL_NAME = "gemini-1.5-flash-002"  # 쿼터/비용 안전

def make_prompt(input_data: dict, proba: float) -> str:
    # 토큰 절약: 꼭 필요한 정보만
    readable = {
        "나이(세)": int(input_data["age"] // 365),
        "성별": "남" if input_data["gender"] == 1 else "여",
        "혈압": f'{input_data["ap_hi"]}/{input_data["ap_lo"]}',
        "콜레스테롤": input_data["cholesterol"],
        "혈당": input_data["gluc"],
        "흡연": int(input_data["smoke"]),
        "음주": int(input_data["alco"]),
        "운동": int(input_data["active"]),
        "BMI": round(input_data["BMI"], 1),
    }
    return (
        "역할: 당신은 예방의학 전문의입니다.\n"
        "지침: 아래 건강정보를 바탕으로 심혈관 위험 감소를 위한 생활습관 조언을 "
        "간결한 bullet 3~5개로, 근거 중심으로 제시하세요. 전문용어는 쉽게.\n"
        f"건강정보: {readable}\n"
        f"예측 위험도(퍼센트): {proba*100:.2f}\n"
        "출력형식: '-'로 시작하는 bullet, 각 1문장.\n"
    )

def _rule_based_fallback(input_data: dict, proba: float) -> str:
    tips = []
    ap_hi, ap_lo = input_data.get("ap_hi", 0), input_data.get("ap_lo", 0)
    if ap_hi >= 140 or ap_lo >= 90:
        tips.append("혈압: 주 5일 30분 유산소(빠른 걷기·자전거) + 소금 5g 이하로 제한.")
    else:
        tips.append("혈압: 주 3~5회 30분 유산소 유지, 카페인·염분 과다 섭취 주의.")
    bmi = input_data.get("BMI", 0)
    if bmi >= 25:
        tips.append("체중: 일일 500kcal 감산으로 주당 0.5kg 감량 목표.")
    tips.append("식단: 채소·통곡물·생선 중심 DASH 패턴, 가공육·포화지방 줄이기.")
    if int(input_data.get("active", 0)) == 0:
        tips.append("운동: 일일 8000보 이상 목표로 활동량 추적.")
    if int(input_data.get("smoke", 0)) == 1:
        tips.append("흡연: 보조제·상담 연계로 금연 도전(4주 유지 성공률↑).")
    tips.append("관리: 가정용 혈압계로 아침/저녁 측정, 2주 평균으로 추세 확인.")
    head = f"예측 위험도: {proba*100:.2f}% → 생활습관 조언 (폴백)"
    return head + "\n\n- " + "\n- ".join(tips[:5])

@st.cache_data(show_spinner=False)
def _cached_gemini_reply(model_name: str, prompt: str) -> str:
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()

# ───────────────────────────────────────────────────────────────
# 한글 레이블 & 값 변환
# ───────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    'age': '나이 (세)', 'gender': '성별', 'ap_hi': '수축기 혈압', 'ap_lo': '이완기 혈압',
    'cholesterol': '콜레스테롤 등급', 'gluc': '혈당 등급', 'smoke': '흡연 여부',
    'alco': '음주 여부', 'active': '운동 여부', 'BMI': '체질량지수'
}

def translate_value(feature, value):
    if feature == "cholesterol":
        return ['안전', '양호', '위험'][int(value) - 1]
    if feature == "gluc":
        return ['안전', '양호', '위험'][int(value) - 1]
    if feature == "gender":
        return "남성" if int(value) == 1 else "여성"
    if feature == "smoke":
        return "흡연" if int(value) == 1 else "비흡연"
    if feature == "alco":
        return "음주" if int(value) == 1 else "비음주"
    if feature == "active":
        return "운동함" if int(value) == 1 else "운동 안함"
    if feature == "age":
        return f"{int(value // 365)}세"
    if feature == "BMI":
        return f"{float(value):.1f}"
    return value

# ───────────────────────────────────────────────────────────────
# 모델 로드
# ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("xgb_best_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ───────────────────────────────────────────────────────────────
# UI
# ───────────────────────────────────────────────────────────────
st.title("💓 당신의 심혈관 건강은 안전한가요?")
st.caption(f"Gemini model: `{GEMINI_MODEL_NAME}`")
st.markdown("""
**XGBoost 기반 심혈관 질환 위험 예측 & 건강 개선 제안**  
사용자의 건강 정보를 바탕으로 위험도를 예측하고,  
개인별로 중요한 위험 요인을 설명하며 건강 개선 조언도 제공합니다.
""")

# 입력 사이드바
st.sidebar.header("📝 건강 정보 입력")
age = st.sidebar.slider("나이", 20, 80, 60)
gender = st.sidebar.radio("성별", ["남성", "여성"])
height = st.sidebar.number_input("키 (cm)", 140, 200, 170)
weight = st.sidebar.number_input("몸무게 (kg)", 40, 150, 65)
ap_hi = st.sidebar.slider("수축기 혈압", 90, 200, 120)
ap_lo = st.sidebar.slider("이완기 혈압", 40, 130, 80)
cholesterol = st.sidebar.selectbox("콜레스테롤 등급", ["안전", "양호", "위험"])
gluc = st.sidebar.selectbox("혈당 등급", ["안전", "양호", "위험"])
smoke = st.sidebar.checkbox("흡연 여부")
alco = st.sidebar.checkbox("음주 여부")
active = st.sidebar.checkbox("운동을 규칙적으로 하나요?")

cholesterol_map = {"안전": 1, "양호": 2, "위험": 3}
gluc_map = {"안전": 1, "양호": 2, "위험": 3}
bmi = weight / ((height / 100) ** 2) if height else 0.0

input_data = {
    'age': age * 365,
    'gender': 1 if gender == "남성" else 2,
    'ap_hi': ap_hi,
    'ap_lo': ap_lo,
    'cholesterol': cholesterol_map[cholesterol],
    'gluc': gluc_map[gluc],
    'smoke': int(smoke),
    'alco': int(alco),
    'active': int(active),
    'BMI': bmi
}
input_df = pd.DataFrame([input_data])

# ───────────────────────────────────────────────────────────────
# 예측
# ───────────────────────────────────────────────────────────────
st.subheader("📊 예측 결과")
proba = float(model.predict_proba(input_df)[0][1])
st.metric(label="심혈관 질환 위험도", value=f"{proba * 100:.2f}%")

# ───────────────────────────────────────────────────────────────
# SHAP: TreeExplainer 우선, 실패 시 callable+masker 백업
# ───────────────────────────────────────────────────────────────
st.markdown("#### 📌 예측 근거 (개인별 변수 기여도 기준)")

def _get_1d_contrib(sv):
    vals = sv.values
    if getattr(vals, "ndim", None) == 3:
        return vals[0, :, 1]  # (n_samples, n_features, n_classes) → 양성클래스
    if getattr(vals, "ndim", None) == 2:
        return vals[0, :]
    return np.array(vals).ravel()

try:
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        f = lambda X: model.predict_proba(pd.DataFrame(X, columns=input_df.columns)).astype(np.float64)
        masker = shap.maskers.Independent(input_df)
        explainer = shap.Explainer(f, masker=masker)

    sv = explainer(input_df)
    contrib = _get_1d_contrib(sv)

    shap_df = pd.DataFrame({
        "변수": input_df.columns,
        "입력값": [translate_value(c, v) for c, v in input_df.iloc[0].items()],
        "기여도": contrib
    })
    shap_df["기여도절댓값"] = shap_df["기여도"].abs()
    shap_df["기여도비율(%)"] = shap_df["기여도절댓값"] / shap_df["기여도절댓값"].sum() * 100
    shap_top3 = shap_df.sort_values(by="기여도절댓값", ascending=False).head(3)

    for _, row in shap_top3.iterrows():
        sign = "높였습니다" if row["기여도"] > 0 else "낮췄습니다"
        st.markdown(
            f"• **{row['변수']}** 값이 **{row['입력값']}**로 입력되어, "
            f"심혈관 위험 예측 확률을 **{abs(row['기여도']):.3f}만큼 {sign}** "
            f"(전체 영향 {row['기여도비율(%)']:.1f}%).",
            unsafe_allow_html=True
        )

    with st.expander("🔍 전체 변수 영향 보기 (SHAP 방향성 시각화)"):
        shap_df_full = shap_df.sort_values(by="기여도", key=np.abs, ascending=False).head(10)
        colors = ['red' if v > 0 else 'blue' for v in shap_df_full["기여도"]]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(shap_df_full["변수"], shap_df_full["기여도"], color=colors, edgecolor='black')
        ax.axvline(0, color='gray', linewidth=1)
        ax.set_xlabel("SHAP Value")
        ax.set_title("SHAP Value Contribution (Red = Increase ↑ / Blue = Decrease ↓)")
        ax.invert_yaxis()
        max_val = max(abs(shap_df_full["기여도"].max()), abs(shap_df_full["기여도"].min()))
        ax.set_xlim(-max_val * 1.2, max_val * 1.2)
        for bar, val in zip(bars, shap_df_full["기여도"]):
            txt = f"{val:.3f}" if val < 0 else f"+{val:.3f}"
            ax.text(val + (0.02 if val > 0 else -0.02),
                    bar.get_y() + bar.get_height() / 2,
                    txt, va='center',
                    ha='left' if val > 0 else 'right',
                    fontsize=10, color='black')
        plt.tight_layout()
        st.pyplot(fig)

except Exception as e:
    st.warning("SHAP 값을 계산하는 중 오류가 발생했습니다.")
    st.exception(e)

# ───────────────────────────────────────────────────────────────
# Gemini 개선 제안 (캐시 + 폴백)
# ───────────────────────────────────────────────────────────────
st.subheader("🛠️ 개선 제안 (Gemini 기반)")
if st.button("Gemini에게 조언 요청하기 🧠", type="primary"):
    prompt = make_prompt(input_data, proba)
    try:
        text = _cached_gemini_reply(GEMINI_MODEL_NAME, prompt)
        st.markdown(text)
    except Exception as e:
        st.warning("Gemini 호출이 제한되었습니다(쿼터/네트워크/권한). 폴백 조언을 제공합니다.")
        st.markdown(_rule_based_fallback(input_data, proba))
       # with st.expander("📄 오류 상세"):
            #st.exception(e)

# ───────────────────────────────────────────────────────────────
# 시뮬레이터
# ───────────────────────────────────────────────────────────────
st.subheader("⚙️ 시뮬레이터: 혈압 조정 시 위험도 변화")
sim_ap_hi = st.slider("수축기 혈압 (mmHg)", 90, 200, ap_hi)
sim_ap_lo = st.slider("이완기 혈압 (mmHg)", 40, 130, ap_lo)

sim_data = input_data.copy()
sim_data['ap_hi'] = sim_ap_hi
sim_data['ap_lo'] = sim_ap_lo
sim_df = pd.DataFrame([sim_data])
sim_proba = float(model.predict_proba(sim_df)[0][1])
st.info(f"혈압을 {ap_hi}/{ap_lo} → {sim_ap_hi}/{sim_ap_lo} mmHg로 조정하면, 위험도는 {sim_proba*100:.2f}%로 바뀝니다.")
