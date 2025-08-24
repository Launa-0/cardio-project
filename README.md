Streamlit 심혈관 위험 예측기 사용 방법

1. 가상환경 설치 (선택)
   python -m venv .venv
   .venv\Scripts\activate  (윈도우)
   source .venv/bin/activate (맥/리눅스)

2. 라이브러리 설치
   pip install -r requirements.txt

3. 실행 방법
   streamlit run cardio_app.py

4. 사용 방법
   - 왼쪽 사이드바에서 나이, 혈압, 콜레스테롤 등 입력
   - 위험도 예측 결과 확인
   - 수축기/이완기 혈압을 조절해 시뮬레이션 가능

구성 파일 설명
- cardio_app.py : Streamlit 웹앱 코드
- xgb_model.pkl : 학습된 XGBoost 모델
- 심혈관계 모델 : 전처리,시각화~ 전이학습까지 구글 코랩에서 진행

autogluon을 사용하려 했지만 자꾸 충돌이 일어나서 학습했던 모델 중 가장 높은 정확도와 민감도를 보인 xgb_model 사용

최종 배포 사이트
https://cardio-project-q2btpa92nykgwv7ooabl6k.streamlit.app/


+++)) 8월 24일 최종발표회 교수님 개선사항
하이퍼 파라미터 튜닝할떄 최적화
그리드 서치 튜닝으로
7만이면 굳이 전이학습 안해도 됨

개인별 개선 제안 이걸 하이라이트로
생성형 ai랑 연결해 개선재안
9월 학술대회 내보자
