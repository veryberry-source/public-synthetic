import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
import re
import time
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit.components.v1 as components

# ─── 페이지 설정 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PersonaLab AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── 글로벌 CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

/* ── 전역 리셋 ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A0C10 !important;
    color: #E2E8F0 !important;
}

[data-testid="stSidebar"] {
    background-color: #0D1017 !important;
    border-right: 1px solid #1E2533 !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

/* ── 헤더 히어로 ── */
.hero-wrap {
    background: linear-gradient(135deg, #0D1117 0%, #111827 50%, #0A0C10 100%);
    border: 1px solid #1E2533;
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(168,85,247,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 100px;
    padding: 4px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #38BDF8;
    letter-spacing: 0.08em;
    margin-bottom: 16px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: #F1F5F9;
    margin: 0 0 8px 0;
    line-height: 1.2;
    letter-spacing: -0.02em;
}
.hero-title span {
    background: linear-gradient(90deg, #38BDF8, #818CF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-family: 'Noto Sans KR', sans-serif;
    font-size: 14px;
    color: #64748B;
    margin: 0;
    font-weight: 300;
    letter-spacing: 0.01em;
}

/* ── 섹션 헤더 ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.15em;
    color: #38BDF8;
    text-transform: uppercase;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1E2533, transparent);
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #E2E8F0;
    margin: 0 0 20px 0;
}

/* ── 카드 컨테이너 ── */
.card {
    background: #0D1117;
    border: 1px solid #1E2533;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
}
.card:hover { border-color: #2D3748; }

/* ── 입력 필드 전체 재정의 ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
.stSelectbox > div > div,
[data-testid="stNumberInput"] input {
    background-color: #111827 !important;
    border: 1px solid #1E2533 !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
    font-family: 'Noto Sans KR', sans-serif !important;
    font-size: 13px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.08) !important;
}

/* ── 라벨 ── */
label, .stSelectbox label, .stSlider label {
    font-family: 'Noto Sans KR', sans-serif !important;
    font-size: 12px !important;
    color: #94A3B8 !important;
    font-weight: 500 !important;
}

/* ── 버튼 ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0EA5E9, #6366F1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.02em !important;
    height: 48px !important;
    transition: opacity 0.2s, transform 0.1s !important;
    box-shadow: 0 4px 24px rgba(14,165,233,0.25) !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── 다운로드 버튼 ── */
[data-testid="stDownloadButton"] > button {
    background: #111827 !important;
    color: #38BDF8 !important;
    border: 1px solid #1E2533 !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    height: 44px !important;
    transition: all 0.2s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: #38BDF8 !important;
    background: rgba(56,189,248,0.06) !important;
}

/* ── 탭 ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #0D1117 !important;
    border-bottom: 1px solid #1E2533 !important;
    gap: 0 !important;
    padding: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748B !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 12px 20px !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #38BDF8 !important;
    border-bottom-color: #38BDF8 !important;
    background: transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    background: #0D1117 !important;
    padding: 24px !important;
    border: 1px solid #1E2533 !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
}

/* ── 데이터프레임/테이블 ── */
[data-testid="stDataFrame"], .stTable {
    background: #0D1117 !important;
    border: 1px solid #1E2533 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] th {
    background: #111827 !important;
    color: #94A3B8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stDataFrame"] td {
    color: #CBD5E1 !important;
    font-size: 13px !important;
}

/* ── 진행바 ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #0EA5E9, #6366F1) !important;
    border-radius: 100px !important;
}
[data-testid="stProgress"] > div {
    background: #1E2533 !important;
    border-radius: 100px !important;
}

/* ── Status ── */
[data-testid="stStatusWidget"] {
    background: #0D1117 !important;
    border: 1px solid #1E2533 !important;
    border-radius: 10px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0D1117 !important;
    border: 1px solid #1E2533 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #94A3B8 !important;
}
[data-testid="stExpander"] summary:hover {
    color: #E2E8F0 !important;
}

/* ── Multiselect ── */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: rgba(56,189,248,0.1) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: 4px !important;
    color: #38BDF8 !important;
    font-size: 11px !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [role="slider"] {
    background: #38BDF8 !important;
}

/* ── Alert / Info ── */
[data-testid="stAlert"] {
    background: rgba(56,189,248,0.05) !important;
    border: 1px solid rgba(56,189,248,0.15) !important;
    border-radius: 8px !important;
    color: #94A3B8 !important;
    font-family: 'Noto Sans KR', sans-serif !important;
}
[data-testid="stAlert"][data-type="success"] {
    background: rgba(34,197,94,0.05) !important;
    border-color: rgba(34,197,94,0.2) !important;
}
[data-testid="stAlert"][data-type="error"] {
    background: rgba(239,68,68,0.05) !important;
    border-color: rgba(239,68,68,0.2) !important;
}

/* ── 사이드바 섹션 레이블 ── */
.sidebar-section {
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.18em;
    color: #334155;
    text-transform: uppercase;
    padding: 16px 0 6px 0;
    border-top: 1px solid #1E2533;
    margin-top: 8px;
}

/* ── 통계 칩 ── */
.stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(56,189,248,0.06);
    border: 1px solid rgba(56,189,248,0.12);
    border-radius: 8px;
    padding: 8px 14px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #38BDF8;
    margin: 4px 4px 4px 0;
}
.stat-chip strong {
    font-size: 16px;
    font-weight: 600;
    color: #F1F5F9;
}

/* ── Q 레이블 ── */
.q-label {
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #E2E8F0;
    margin: 24px 0 12px 0;
    padding: 12px 16px;
    background: #111827;
    border-left: 3px solid #38BDF8;
    border-radius: 0 8px 8px 0;
}

/* ── 구분선 ── */
hr { border-color: #1E2533 !important; }

/* ── 차트 배경 ── */
[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"] {
    background: transparent !important;
}

/* ── 복사 알림 토스트 ── */
#copy-notice {
    position: fixed;
    top: 24px;
    right: 24px;
    z-index: 99999;
    background: linear-gradient(135deg, #0EA5E9, #6366F1);
    color: white;
    padding: 12px 20px;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 600;
    box-shadow: 0 8px 32px rgba(14,165,233,0.3);
    display: none;
    animation: slideIn 0.2s ease;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── 사이드바 이미지 가운데 ── */
[data-testid="stSidebar"] img { display: block; margin: 0 auto 8px; }

/* ── 전체 폰트 폴백 ── */
p, li, span, div, td, th, input, textarea, select, button {
    font-family: 'Noto Sans KR', sans-serif;
}
</style>
<div id="copy-notice">✓ 클립보드에 복사되었습니다</div>
""", unsafe_allow_html=True)


# ─── 클립보드 복사 헬퍼 ─────────────────────────────────────────────────────────
def st_copy_to_clipboard(text, button_label, key):
    escaped = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    components.html(f"""
    <script>
    function copyClip_{key}() {{
        navigator.clipboard.writeText(`{escaped}`).then(() => {{
            const n = window.parent.document.getElementById('copy-notice');
            n.style.display = 'block';
            setTimeout(() => {{ n.style.display = 'none'; }}, 2200);
        }});
    }}
    </script>
    <button onclick="copyClip_{key}()" style="
        width:100%; height:36px;
        background:#111827; border:1px solid #1E2533;
        border-radius:6px; cursor:pointer;
        font-family:'Syne',sans-serif; font-weight:600;
        font-size:12px; color:#94A3B8;
        letter-spacing:0.03em;
        transition:all 0.2s;
    "
    onmouseover="this.style.borderColor='#38BDF8';this.style.color='#38BDF8';"
    onmouseout="this.style.borderColor='#1E2533';this.style.color='#94A3B8';">
        {button_label}
    </button>
    """, height=42)


# ─── 핵심 로직 ──────────────────────────────────────────────────────────────────
def process_persona(client, model_choice, row_data, scenario, persona_tmpl,
                    question_tmpl, num_questions, questions_block, temp):
    persona_ans = ["NA"] * num_questions
    try:
        profile_context = "\n".join([f"- {k}: {v}" for k, v in row_data.items()])
        sys_inst = f"{persona_tmpl}\n\n[현재 상황]\n{scenario}\n\n[상세 프로필]\n{profile_context}"
        user_prompt = question_tmpl.format(
            num_questions=num_questions, questions_block=questions_block)
        resp = client.models.generate_content(
            model=model_choice,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_inst, temperature=temp)
        )
        nums = re.findall(r'\d+', resp.text)
        if len(nums) >= num_questions:
            persona_ans = nums[:num_questions]
        elif nums:
            for i, n in enumerate(nums):
                persona_ans[i] = n
    except:
        pass
    return persona_ans


# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">⬡ &nbsp;AI-POWERED SURVEY SIMULATION</div>
    <h1 class="hero-title">Persona<span>Lab</span> AI</h1>
    <p class="hero-sub">고도화된 언어 모델 기반의 페르소나 합성 데이터 생성 및 사회과학 분석 솔루션 &nbsp;·&nbsp; v4.5 Professional</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0 0 20px 0;">
        <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:800;
                    background:linear-gradient(90deg,#38BDF8,#818CF8);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            Control Center
        </div>
        <div style="font-family:'DM Mono',monospace; font-size:10px;
                    color:#334155; letter-spacing:0.1em; margin-top:4px;">
            PERSONALAB AI v4.5
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔑  API & Model Engine", expanded=True):
        api_key = st.text_input("Gemini API Key", type="password",
                                placeholder="AIza...")
        model_choice = st.selectbox("모델 엔진",
                                    ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
        temp = st.slider("창의성 (Temperature)", 0.0, 1.0, 0.7, step=0.05)
        max_workers = st.slider("병렬 처리 스레드", 1, 40, 20)

    with st.expander("🎭  페르소나 행동 지침", expanded=True):
        persona_tmpl = st.text_area(
            "System Instruction",
            value=(
                "당신은 아래 제공된 프로필 정보를 가진 실존 인물입니다.\n"
                "당신의 사회적 위치, 경제적 상황에 완벽히 빙의하여 설문에 임하세요.\n"
                "추상적인 정답이 아니라, 당신의 삶의 배경에서 나올 법한 "
                "'솔직하고 주관적인' 의견을 선호합니다."
            ),
            height=110,
        )

    with st.expander("🌐  사회적 맥락 (Scenario)", expanded=False):
        common_scenario = st.text_area(
            "현재 시점의 배경/사건",
            value=(
                "오늘은 2026년 4월 27일입니다.\n"
                "최근 주요 사건\n"
                "- 지방선거 선거구 획정; 광역의회 비례대표 의석 확대\n"
                "- 광주 4개 선거구 광역의원 중대 선거구제 최초 도입\n"
                "- 이재명, 인도·베트남 국빈 방문\n"
                "- 코스피 최고 장중 6,557.76\n"
                "- 삼성전자 노조 '성과급 투쟁'\n"
                "- 이란 전쟁 휴전 연장"
            ),
            height=130,
        )

    # 사이드바 하단 상태 표시
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    if 'results_df' in st.session_state:
        n = len(st.session_state['results_df'])
        q = len(st.session_state['questions'])
        st.markdown(f"""
        <div style="background:#0D1117; border:1px solid #1E2533; border-radius:10px;
                    padding:14px 16px; text-align:center;">
            <div style="font-family:'DM Mono',monospace; font-size:9px;
                        color:#334155; letter-spacing:0.15em; margin-bottom:8px;">
                LAST RUN
            </div>
            <div style="font-family:'Syne',sans-serif; font-size:22px;
                        font-weight:800; color:#38BDF8;">{n:,}</div>
            <div style="font-family:'Noto Sans KR',sans-serif; font-size:11px;
                        color:#64748B;">케이스 · {q}개 문항</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 + 2
# ═══════════════════════════════════════════════════════════════════════════════
col_in1, col_in2 = st.columns([1, 1], gap="large")

with col_in1:
    st.markdown('<div class="section-label">01 &nbsp; Data Asset</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">응답자 프로필 로드</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "CSV 또는 XLSX 파일 업로드",
        type=["xlsx", "csv"],
        key="uploader",
        label_visibility="collapsed",
    )

    if uploaded_file:
        df = (pd.read_csv(uploaded_file)
              if uploaded_file.name.endswith('.csv')
              else pd.read_excel(uploaded_file))
        df.columns = [str(c).lower().strip() for c in df.columns]

        # 날짜 컬럼 자동 변환
        for col in df.columns:
            if any(kw in col for kw in ['조사일', '날짜', 'date']):
                def safe_date(x):
                    if pd.isna(x): return x
                    try:
                        v = float(x)
                        if v > 1000:
                            return pd.to_datetime(v, unit='D',
                                                  origin='1899-12-30').strftime('%Y-%m-%d')
                        return str(int(v))
                    except:
                        try: return pd.to_datetime(x).strftime('%Y-%m-%d')
                        except: return str(x)
                df[col] = df[col].apply(safe_date)

        # 요약 칩
        st.markdown(f"""
        <div style="margin:16px 0 12px;">
            <span class="stat-chip"><strong>{len(df):,}</strong> Cases</span>
            <span class="stat-chip"><strong>{len(df.columns)}</strong> Variables</span>
        </div>
        """, unsafe_allow_html=True)

        selected_features = st.multiselect(
            "핵심 분석 변수 선택",
            options=df.columns.tolist(),
            default=[c for c in df.columns
                     if not any(kw in c for kw in ['id', 'no', 'uuid', '가중치'])],
            key="features",
        )

        with st.expander("데이터셋 미리보기 (상위 5행)"):
            st.dataframe(df[selected_features].head(5), use_container_width=True)


with col_in2:
    st.markdown('<div class="section-label">02 &nbsp; Survey Design</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">설문 문항 설계</div>', unsafe_allow_html=True)

    question_input = st.text_area(
        "질문 리스트 (줄바꿈으로 구분)",
        value=(
            "이번 지방선거에 투표하실건가요? "
            "(1. 반드시 할 것이다 2. 아마 할 것 같다 "
            "3. 아마 하지 않을 것 같다 4. 투표하지 않겠다 5. 모름)\n"
            "대구시장 선거에서 누가 당선되는 것이 조금이라도 좋다고 보십니까? "
            "(1. 더불어민주당 김부겸 2. 국민의힘 추경호 3. 모름)"
        ),
        height=160,
        key="q_in",
    )

    question_tmpl_ui = st.text_area(
        "답변 생성 포맷 (프롬프트 템플릿)",
        value=(
            "아래 {num_questions}개의 설문 문항에 대해 당신의 가치관에 근거하여 답변하세요.\n\n"
            "[수행 지침]\n"
            "1. 각 문항을 읽고, 당신의 프로필이라면 어떤 선택을 할지 짧게 추론하세요.\n"
            "2. 최종 결과는 오직 '숫자'만 순서대로 콤마(,)로 구분하여 한 줄로 출력하세요.\n"
            "3. 다른 설명은 절대 하지 마세요.\n\n"
            "문항 리스트:\n{questions_block}"
        ),
        height=160,
        key="q_form",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RUN BUTTON
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.divider()

run_col, _ = st.columns([1, 2])
with run_col:
    run_btn = st.button("⬡  지능형 시뮬레이션 시작", type="primary", key="run_sim",
                        use_container_width=True)

if run_btn:
    if not api_key or 'df' not in locals():
        st.error("설정 오류: API Key와 프로필 데이터를 모두 확인해 주세요.")
    else:
        try:
            client = genai.Client(api_key=api_key)
            questions = [q.strip() for q in question_input.split('\n') if q.strip()]
            q_block = "\n".join([f"Q{i+1}. {q}" for i, q in enumerate(questions)])

            with st.status("⬡  AI 페르소나가 응답을 생성하고 있습니다…", expanded=True) as status:
                results_df = df.copy()
                all_responses: dict = {}

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            process_persona, client, model_choice,
                            row[selected_features].to_dict(),
                            common_scenario, persona_tmpl,
                            question_tmpl_ui, len(questions), q_block, temp
                        ): idx
                        for idx, row in df.iterrows()
                    }
                    pbar = st.progress(0, text="응답 생성 중…")
                    for i, future in enumerate(as_completed(futures)):
                        idx = futures[future]
                        all_responses[idx] = future.result()
                        pbar.progress(
                            (i + 1) / len(df),
                            text=f"응답 생성 중… {i+1}/{len(df)}"
                        )

                for i in range(len(questions)):
                    results_df[f'q_{i+1}'] = [all_responses[idx][i] for idx in range(len(df))]

                st.session_state['results_df'] = results_df
                st.session_state['questions'] = questions
                status.update(label=f"✓  시뮬레이션 완료  ·  {len(df)}개 케이스 처리됨",
                              state="complete")
        except Exception as e:
            st.error(f"실행 오류: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if 'results_df' in st.session_state:
    res_df = st.session_state['results_df']
    q_list = st.session_state['questions']
    q_cols = [c for c in res_df.columns if c.startswith('q_')]
    demo_cols = [c for c in res_df.columns if not c.startswith('q_')]

    st.divider()
    st.markdown('<div class="section-label">03 &nbsp; Analysis Dashboard</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-title">분석 결과</div>', unsafe_allow_html=True)

    # 상단 요약 메트릭
    na_total = sum(
        (res_df[qc] == 'NA').sum() for qc in q_cols if qc in res_df.columns
    )
    total_cells = len(res_df) * len(q_cols)
    completion_rate = round((1 - na_total / max(total_cells, 1)) * 100, 1)

    m1, m2, m3, m4 = st.columns(4)
    for col, val, label in [
        (m1, f"{len(res_df):,}", "총 케이스"),
        (m2, str(len(q_list)), "설문 문항"),
        (m3, f"{completion_rate}%", "응답 완성률"),
        (m4, str(len(demo_cols)), "인구통계 변수"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#0D1117; border:1px solid #1E2533;
                        border-radius:10px; padding:18px 20px; text-align:center;">
                <div style="font-family:'Syne',sans-serif; font-size:26px;
                            font-weight:800; color:#38BDF8;">{val}</div>
                <div style="font-family:'Noto Sans KR',sans-serif; font-size:12px;
                            color:#64748B; margin-top:4px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["📈  빈도 분석", "🔀  교차 분석", "💾  내보내기"])

    # ── 빈도 분석 ──────────────────────────────────────────────────────────────
    with t1:
        for i, q in enumerate(q_list):
            q_col = f'q_{i+1}'
            st.markdown(f'<div class="q-label">Q{i+1}. &nbsp;{q}</div>',
                        unsafe_allow_html=True)

            counts = res_df[q_col].value_counts().sort_index()
            percents = (res_df[q_col].value_counts(normalize=True)
                        .sort_index() * 100).round(1)

            stats_df = pd.DataFrame({
                '사례수 (N)': counts,
                '비율 (%)': percents.apply(lambda x: f"{x:.1f}%"),
            })

            c1, c2 = st.columns([1, 2])
            with c1:
                st.dataframe(stats_df, use_container_width=True)
                st_copy_to_clipboard(
                    stats_df.to_csv(sep='\t'),
                    "📋  Excel로 복사",
                    f"copy_f_{i}"
                )
            with c2:
                st.bar_chart(counts, color="#38BDF8")
            st.divider()

    # ── 교차 분석 ──────────────────────────────────────────────────────────────
    with t2:
        c1, c2 = st.columns(2)
        with c1:
            r_var = st.selectbox(
                "⬇️  행 변수 (인구통계)",
                options=demo_cols, index=None,
                placeholder="행 변수 선택",
                key="cross_row"
            )
        with c2:
            col_var = st.selectbox(
                "➡️  열 변수 (설문 문항)",
                options=q_cols,
                placeholder="문항 선택",
                key="cross_col"
            )

        if r_var and col_var:
            tn, tp = st.tabs(["사례수 (N)", "비율 (%)"])
            with tn:
                ctab_n = pd.crosstab(res_df[r_var], res_df[col_var],
                                     margins=True, margins_name="전체")
                st.dataframe(ctab_n, use_container_width=True)
                st_copy_to_clipboard(ctab_n.to_csv(sep='\t'),
                                     "📋  사례수 표 복사", "copy_cn")
            with tp:
                ctab_pct = (pd.crosstab(res_df[r_var], res_df[col_var],
                                        normalize='index') * 100).round(1)
                st.dataframe(ctab_pct.map(lambda x: f"{x:.1f}%"),
                             use_container_width=True)
                st_copy_to_clipboard(ctab_pct.to_csv(sep='\t'),
                                     "📋  비율 표 복사", "copy_cp")
                st.bar_chart(ctab_pct, color="#818CF8")
        else:
            st.info("변수를 선택하면 교차표가 자동으로 생성됩니다.")

    # ── 내보내기 ────────────────────────────────────────────────────────────────
    with t3:
        st.markdown("""
        <div style="background:#111827; border:1px solid #1E2533; border-radius:10px;
                    padding:24px; margin-bottom:20px;">
            <div style="font-family:'Syne',sans-serif; font-size:16px;
                        font-weight:700; color:#E2E8F0; margin-bottom:8px;">
                📦  결과 데이터셋 다운로드
            </div>
            <div style="font-family:'Noto Sans KR',sans-serif; font-size:13px;
                        color:#64748B; line-height:1.6;">
                시뮬레이션 완료된 원본 프로필 + 생성 응답이 병합된 XLSX 파일을 다운로드합니다.
            </div>
        </div>
        """, unsafe_allow_html=True)

        out = BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as wr:
            res_df.to_excel(wr, index=False, sheet_name="PersonaLab_Results")
        st.download_button(
            "📥  Excel 결과 리포트 다운로드",
            out.getvalue(),
            "persona_lab_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_final",
            use_container_width=True,
        )
