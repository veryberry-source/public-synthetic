import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
import re
import time
from datetime import date
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

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A0C10 !important;
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] {
    background-color: #0D1017 !important;
    border-right: 1px solid #1E2533 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }

/* ── 히어로 ── */
.hero-wrap {
    background: linear-gradient(135deg, #0D1117 0%, #111827 50%, #0A0C10 100%);
    border: 1px solid #1E2533; border-radius: 16px;
    padding: 36px 40px; margin-bottom: 32px;
    position: relative; overflow: hidden;
}
.hero-wrap::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:240px; height:240px;
    background:radial-gradient(circle,rgba(56,189,248,.08) 0%,transparent 70%);
    pointer-events:none;
}
.hero-wrap::after {
    content:''; position:absolute; bottom:-40px; left:30%;
    width:180px; height:180px;
    background:radial-gradient(circle,rgba(168,85,247,.06) 0%,transparent 70%);
    pointer-events:none;
}
.hero-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(56,189,248,.08); border:1px solid rgba(56,189,248,.2);
    border-radius:100px; padding:4px 12px;
    font-family:'DM Mono',monospace; font-size:11px;
    color:#38BDF8; letter-spacing:.08em; margin-bottom:16px;
}
.hero-title {
    font-family:'Syne',sans-serif; font-size:32px; font-weight:800;
    color:#F1F5F9; margin:0 0 8px; line-height:1.2; letter-spacing:-.02em;
}
.hero-title span {
    background:linear-gradient(90deg,#38BDF8,#818CF8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero-sub { font-family:'Noto Sans KR',sans-serif; font-size:14px; color:#64748B; margin:0; font-weight:300; }

/* ── 섹션 ── */
.section-label {
    font-family:'DM Mono',monospace; font-size:10px; letter-spacing:.15em;
    color:#38BDF8; text-transform:uppercase; margin-bottom:12px;
    display:flex; align-items:center; gap:8px;
}
.section-label::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,#1E2533,transparent); }
.section-title { font-family:'Syne',sans-serif; font-size:18px; font-weight:700; color:#E2E8F0; margin:0 0 20px; }

/* ── 입력 ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
.stSelectbox > div > div,
[data-testid="stNumberInput"] input {
    background-color:#111827 !important; border:1px solid #1E2533 !important;
    border-radius:8px !important; color:#E2E8F0 !important;
    font-family:'Noto Sans KR',sans-serif !important; font-size:13px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color:#38BDF8 !important; box-shadow:0 0 0 3px rgba(56,189,248,.08) !important;
}
label, .stSelectbox label, .stSlider label {
    font-family:'Noto Sans KR',sans-serif !important; font-size:12px !important;
    color:#94A3B8 !important; font-weight:500 !important;
}

/* ── 버튼 (primary) ── */
[data-testid="stButton"] > button {
    background:linear-gradient(135deg,#0EA5E9,#6366F1) !important;
    color:#fff !important; border:none !important; border-radius:8px !important;
    font-family:'Syne',sans-serif !important; font-weight:700 !important;
    font-size:14px !important; letter-spacing:.02em !important;
    height:48px !important; transition:opacity .2s,transform .1s !important;
    box-shadow:0 4px 24px rgba(14,165,233,.25) !important;
}
[data-testid="stButton"] > button:hover { opacity:.9 !important; transform:translateY(-1px) !important; }
[data-testid="stButton"] > button:active { transform:translateY(0) !important; }

/* ── AI 생성 버튼 (secondary-like) ── */
button[kind="secondary"] {
    background:#111827 !important; color:#38BDF8 !important;
    border:1px solid rgba(56,189,248,.3) !important; border-radius:8px !important;
    font-family:'Syne',sans-serif !important; font-weight:600 !important;
    font-size:13px !important; height:40px !important;
}
button[kind="secondary"]:hover {
    background:rgba(56,189,248,.06) !important;
    border-color:#38BDF8 !important;
}

/* ── 다운로드 ── */
[data-testid="stDownloadButton"] > button {
    background:#111827 !important; color:#38BDF8 !important;
    border:1px solid #1E2533 !important; border-radius:8px !important;
    font-family:'Syne',sans-serif !important; font-weight:600 !important;
    font-size:13px !important; height:44px !important; transition:all .2s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color:#38BDF8 !important; background:rgba(56,189,248,.06) !important;
}

/* ── 탭 ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background:#0D1117 !important; border-bottom:1px solid #1E2533 !important;
    gap:0 !important; padding:0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background:transparent !important; color:#64748B !important;
    font-family:'Syne',sans-serif !important; font-size:13px !important;
    font-weight:600 !important; padding:12px 20px !important;
    border-bottom:2px solid transparent !important; border-radius:0 !important;
    transition:all .2s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color:#38BDF8 !important; border-bottom-color:#38BDF8 !important;
    background:transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    background:#0D1117 !important; padding:24px !important;
    border:1px solid #1E2533 !important; border-top:none !important;
    border-radius:0 0 12px 12px !important;
}

/* ── 데이터프레임 ── */
[data-testid="stDataFrame"], .stTable {
    background:#0D1117 !important; border:1px solid #1E2533 !important;
    border-radius:8px !important; overflow:hidden !important;
}
[data-testid="stDataFrame"] th {
    background:#111827 !important; color:#94A3B8 !important;
    font-family:'DM Mono',monospace !important; font-size:11px !important;
    letter-spacing:.05em !important;
}
[data-testid="stDataFrame"] td { color:#CBD5E1 !important; font-size:13px !important; }

/* ── 진행바 ── */
[data-testid="stProgress"] > div > div {
    background:linear-gradient(90deg,#0EA5E9,#6366F1) !important;
    border-radius:100px !important;
}
[data-testid="stProgress"] > div { background:#1E2533 !important; border-radius:100px !important; }

/* ── Status / Expander / Alert ── */
[data-testid="stStatusWidget"] {
    background:#0D1117 !important; border:1px solid #1E2533 !important; border-radius:10px !important;
}
[data-testid="stExpander"] {
    background:#0D1117 !important; border:1px solid #1E2533 !important; border-radius:10px !important;
}
[data-testid="stExpander"] summary {
    font-family:'Syne',sans-serif !important; font-size:13px !important;
    font-weight:600 !important; color:#94A3B8 !important;
}
[data-testid="stExpander"] summary:hover { color:#E2E8F0 !important; }
[data-testid="stAlert"] {
    background:rgba(56,189,248,.05) !important; border:1px solid rgba(56,189,248,.15) !important;
    border-radius:8px !important; color:#94A3B8 !important;
    font-family:'Noto Sans KR',sans-serif !important;
}
[data-testid="stAlert"][data-type="success"] {
    background:rgba(34,197,94,.05) !important; border-color:rgba(34,197,94,.2) !important;
}
[data-testid="stAlert"][data-type="error"] {
    background:rgba(239,68,68,.05) !important; border-color:rgba(239,68,68,.2) !important;
}

/* ── Multiselect / Slider ── */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background:rgba(56,189,248,.1) !important; border:1px solid rgba(56,189,248,.2) !important;
    border-radius:4px !important; color:#38BDF8 !important;
    font-size:11px !important; font-family:'DM Mono',monospace !important;
}
[data-testid="stSlider"] [role="slider"] { background:#38BDF8 !important; }

/* ── Checkbox ── */
[data-testid="stCheckbox"] label { color:#94A3B8 !important; font-size:12px !important; }
[data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p { color:#94A3B8 !important; }

/* ── 통계 칩 ── */
.stat-chip {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(56,189,248,.06); border:1px solid rgba(56,189,248,.12);
    border-radius:8px; padding:8px 14px;
    font-family:'DM Mono',monospace; font-size:12px; color:#38BDF8;
    margin:4px 4px 4px 0;
}
.stat-chip strong { font-size:16px; font-weight:600; color:#F1F5F9; }

/* ── Q 레이블 ── */
.q-label {
    font-family:'Syne',sans-serif; font-size:14px; font-weight:700; color:#E2E8F0;
    margin:24px 0 12px; padding:12px 16px;
    background:#111827; border-left:3px solid #38BDF8; border-radius:0 8px 8px 0;
}

/* ── 시나리오 패널 ── */
.scenario-box {
    background:#0D1117; border:1px solid #1E2533; border-radius:12px;
    padding:20px; margin-bottom:12px; position:relative;
}
.scenario-box.ai-generated {
    border-color:rgba(56,189,248,.3);
    background:linear-gradient(135deg,#0D1117,#0f1923);
}
.scenario-meta {
    font-family:'DM Mono',monospace; font-size:10px; color:#334155;
    letter-spacing:.1em; margin-bottom:8px;
    display:flex; align-items:center; gap:8px;
}
.scenario-meta .dot {
    width:6px; height:6px; border-radius:50%;
    background:#38BDF8; display:inline-block;
    box-shadow:0 0 6px #38BDF8;
}
.ai-badge {
    display:inline-flex; align-items:center; gap:4px;
    background:rgba(56,189,248,.1); border:1px solid rgba(56,189,248,.25);
    border-radius:4px; padding:2px 8px;
    font-family:'DM Mono',monospace; font-size:9px;
    color:#38BDF8; letter-spacing:.08em;
}

/* ── 카테고리 토글 ── */
.cat-grid {
    display:grid; grid-template-columns:1fr 1fr;
    gap:6px; margin:10px 0;
}
.cat-item {
    background:#111827; border:1px solid #1E2533; border-radius:6px;
    padding:8px 10px; cursor:pointer; transition:all .2s;
    font-family:'Noto Sans KR',sans-serif; font-size:12px; color:#64748B;
}
.cat-item.active { border-color:#38BDF8; color:#38BDF8; background:rgba(56,189,248,.06); }

/* ── 기타 ── */
hr { border-color:#1E2533 !important; }
[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"] { background:transparent !important; }
[data-testid="stSidebar"] img { display:block; margin:0 auto 8px; }
p, li, span, div, td, th, input, textarea, select, button {
    font-family:'Noto Sans KR',sans-serif;
}

/* ── 복사 토스트 ── */
#copy-notice {
    position:fixed; top:24px; right:24px; z-index:99999;
    background:linear-gradient(135deg,#0EA5E9,#6366F1);
    color:white; padding:12px 20px; border-radius:10px;
    font-family:'Syne',sans-serif; font-size:13px; font-weight:600;
    box-shadow:0 8px 32px rgba(14,165,233,.3);
    display:none; animation:slideIn .2s ease;
}
@keyframes slideIn {
    from { opacity:0; transform:translateY(-8px); }
    to   { opacity:1; transform:translateY(0); }
}

/* ── 스피너 애니메이션 ── */
.pulse-dot {
    display:inline-block; width:8px; height:8px; border-radius:50%;
    background:#38BDF8; animation:pulse 1.2s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:.4; transform:scale(.7); }
}
</style>
<div id="copy-notice">✓ 클립보드에 복사되었습니다</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 헬퍼 함수
# ═══════════════════════════════════════════════════════════════════════════════

def st_copy_to_clipboard(text, button_label, key):
    escaped = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    components.html(f"""
    <script>
    function copyClip_{key}() {{
        navigator.clipboard.writeText(`{escaped}`).then(() => {{
            const n = window.parent.document.getElementById('copy-notice');
            n.style.display='block';
            setTimeout(()=>{{n.style.display='none';}},2200);
        }});
    }}
    </script>
    <button onclick="copyClip_{key}()" style="
        width:100%;height:36px;background:#111827;border:1px solid #1E2533;
        border-radius:6px;cursor:pointer;font-family:'Syne',sans-serif;
        font-weight:600;font-size:12px;color:#94A3B8;letter-spacing:.03em;transition:all .2s;"
    onmouseover="this.style.borderColor='#38BDF8';this.style.color='#38BDF8';"
    onmouseout="this.style.borderColor='#1E2533';this.style.color='#94A3B8';">
        {button_label}
    </button>""", height=42)


# ── AI 사회적 맥락 자동 생성 ──────────────────────────────────────────────────
def generate_scenario_with_grounding(api_key: str, model: str,
                                     categories: list[str],
                                     ref_date: str,
                                     extra_context: str = "") -> str:
    """
    Gemini + Google Search Grounding으로 최신 이슈를 수집해
    페르소나 에이전트용 맥락 텍스트를 생성합니다.
    """
    # 카테고리별 검색 쿼리 매핑
    CATEGORY_SEARCH = {
        "선거·공천":   "2026 지방선거 공천 최신",
        "정당·이념":   "한국 정당 정치 최신 뉴스",
        "정책·입법":   "국회 법안 정책 최신",
        "행정·사법":   "정부 행정 사법 최신",
        "부동산":      "2026 부동산 시장 최신",
        "주식·금융":   "코스피 주식 금융 최신",
        "고용·임금":   "고용 임금 취업 최신",
        "물가·소비":   "물가 소비 경제 최신",
        "교육":        "교육 입시 학교 최신",
        "복지·의료":   "복지 의료 건강 최신",
        "젠더·세대":   "젠더 세대 갈등 최신",
        "범죄·안전":   "범죄 치안 안전 최신",
        "수도권 이슈":  "서울 경기 인천 지역 현안 최신",
        "영남 이슈":    "부산 대구 경북 경남 지역 현안 최신",
        "호남 이슈":    "광주 전남 전북 지역 현안 최신",
        "충청 이슈":    "대전 세종 충남 충북 지역 현안 최신",
        "한미관계":    "한미 관계 외교 최신",
        "한중관계":    "한중 관계 외교 최신",
        "한일관계":    "한일 관계 외교 최신",
        "글로벌 이슈":  "국제 글로벌 이슈 최신",
    }

    # 선택된 카테고리의 검색 지시문 생성
    search_instructions = "\n".join([
        f'- [{cat}] 검색어: "{CATEGORY_SEARCH.get(cat, cat + " 최신 뉴스")}"' 
        for cat in categories
    ])

    # 출력 형식: 선택된 카테고리만 동적 생성
    output_format = "\n\n".join([
        f"[{cat}]\n- (구체적 사건명과 현황)\n- (추가 이슈, 있을 경우)"
        for cat in categories
    ])

    extra_line = f"추가 키워드: {extra_context}" if extra_context else ""

    prompt = f"""오늘은 {ref_date}입니다.

아래 각 카테고리에 대해 지정된 검색어로 Google 검색을 실행하고,
{ref_date} 기준 최근 2주간 한국에서 일반인이 뉴스·SNS를 통해 접했을 만한
주요 사건·이슈를 카테고리별로 간결하게 작성하세요.

[카테고리별 검색 지시]
{search_instructions}

[출력 형식 — 선택된 카테고리만]
{output_format}

[작성 규칙]
- 각 카테고리당 1~2개 이슈만 작성 (과도한 상세 금지)
- 일반인이 포털 헤드라인·TV 뉴스 수준으로 알 법한 내용으로 요약
- 날짜·인물명 등 세부 수치는 꼭 필요한 경우만 포함
- 한 항목은 한 문장으로 작성
- 여론조사·지지율·오차범위 등 조사 수치는 절대 포함하지 말 것
- 여론·민심의 방향을 평가하거나 암시하는 표현 금지
{extra_line}"""

    client = genai.Client(api_key=api_key)

    # Google Search Grounding 활성화
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=0.3,
        )
    )
    raw_text = resp.text.strip()
    cleaned, removed = filter_poll_references_ai(client, model, raw_text)
    return cleaned, removed


def filter_poll_references_ai(client, model: str, text: str) -> tuple[str, list[str]]:
    """
    1차: 정규식으로 명확한 키워드 즉시 제거
    2차: Gemini가 문맥 기반으로 남은 문장 재검수
    반환: (정제된 텍스트, 제거된 줄 목록)
    """
    import re as _re

    # ── 1차: 정규식 하드필터 ──────────────────────────────────────────────────
    # 오탐 없이 명확하게 여론조사임을 드러내는 패턴만 선별
    # 여론조사 관련 핵심 단어 — 이 중 하나라도 포함된 줄은 즉시 제거
    HARD_PATTERNS = [
        r'여론\s*조사',   # "여론조사"
        r'오차\s*범위',   # "오차범위"
        r'지지\s*율',     # "지지율"
        r'당선\s*(유력|확실)',
    ]
    hard_re = _re.compile("|".join(HARD_PATTERNS))

    lines = text.split("\n")
    after_hard, removed_lines = [], []
    for line in lines:
        if hard_re.search(line):
            removed_lines.append(line.strip())
        else:
            after_hard.append(line)

    after_hard_text = "\n".join(after_hard).strip()

    # ── 2차: Gemini 문맥 검수 ────────────────────────────────────────────────
    if not after_hard_text:
        return after_hard_text, removed_lines

    review_prompt = f"""아래 텍스트에서 다음 기준에 해당하는 문장(또는 불릿 항목)을 찾아내세요.

[제거 기준]
- 특정 후보·정당이 경쟁 상대보다 우세하거나 열세라고 서술하는 문장
- 선거 결과나 당락을 예측·암시하는 문장
- 여론·민심의 방향·추세를 평가하는 문장 (단, 정치 활동 사실 묘사는 유지)

[유지 기준 — 절대 제거하지 말 것]
- 사건·발언·정책·행동 등 객관적 사실을 기술하는 문장
- 정치적 활동(유세, 공약 발표, 회의, 방문 등)을 묘사하는 문장
- 경제·사회·외교 사실 정보

[출력 형식]
제거 대상이 있으면:
REMOVE:
<원문 그대로 한 줄씩>

없으면:
REMOVE: NONE

다른 설명 없이 위 형식만 출력하세요.

[검토할 텍스트]
{after_hard_text}"""

    try:
        review_resp = client.models.generate_content(
            model=model,
            contents=review_prompt,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        review_text = review_resp.text.strip()

        if "REMOVE: NONE" not in review_text and "REMOVE:" in review_text:
            block = review_text.split("REMOVE:", 1)[1].strip()
            ai_removed = [s.strip() for s in block.split("\n") if s.strip()]

            final_lines, ai_removed_lines = [], []
            for line in after_hard:
                if any(r in line for r in ai_removed):
                    ai_removed_lines.append(line.strip())
                else:
                    final_lines.append(line)

            removed_lines.extend(ai_removed_lines)
            return "\n".join(final_lines).strip(), removed_lines

    except Exception:
        pass  # 검수 실패 시 1차 결과만 반환

    return after_hard_text, removed_lines


# ── 페르소나 응답 생성 ────────────────────────────────────────────────────────
def process_persona(client, model_choice, row_data, scenario, persona_tmpl,
                    question_tmpl, num_questions, questions_block, temp):
    persona_ans = ["NA"] * num_questions
    try:
        profile_context = "\n".join([f"- {k}: {v}" for k, v in row_data.items()])
        sys_inst = (f"{persona_tmpl}\n\n[현재 상황]\n{scenario}\n\n"
                    f"[상세 프로필]\n{profile_context}")
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
    <p class="hero-sub">고도화된 언어 모델 기반의 페르소나 합성 데이터 생성 및 사회과학 분석 솔루션 &nbsp;·&nbsp; v5.0 Professional</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0 0 20px;">
        <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
                    background:linear-gradient(90deg,#38BDF8,#818CF8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Control Center
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;
                    color:#334155;letter-spacing:.1em;margin-top:4px;">
            PERSONALAB AI v5.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── API & 모델 ──────────────────────────────────────────────────────────
    with st.expander("🔑  API & Model Engine", expanded=True):
        api_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
        model_choice = st.selectbox("모델 엔진",
                                    ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
        temp = st.slider("창의성 (Temperature)", 0.0, 1.0, 0.7, step=0.05)
        max_workers = st.slider("병렬 처리 스레드", 1, 40, 20)

    # ── 페르소나 지침 ────────────────────────────────────────────────────────
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

    # ── 🌐 사회적 맥락 (핵심 개선 영역) ────────────────────────────────────
    with st.expander("🌐  사회적 맥락 (Scenario)", expanded=True):

        # 탭: AI 자동 생성 / 수동 입력
        sc_tab_ai, sc_tab_manual = st.tabs(["🤖  AI 자동 생성", "✏️  수동 입력"])

        with sc_tab_ai:
            st.markdown("""
            <div style="font-family:'Noto Sans KR',sans-serif;font-size:12px;
                        color:#64748B;line-height:1.7;margin-bottom:12px;">
                Gemini + Google Search Grounding으로<br>
                실시간 뉴스를 수집·요약합니다.
            </div>
            """, unsafe_allow_html=True)

            # 기준 날짜
            ref_date = st.date_input(
                "기준 날짜",
                value=date.today(),
                key="sc_date",
            )

            # ── 카테고리 세분화 선택 ──────────────────────────────────────
            st.markdown("""
            <div style='font-size:11px;color:#94A3B8;margin:8px 0 4px;
                        font-family:"DM Mono",monospace;letter-spacing:.08em;'>
                CATEGORY SELECT
            </div>""", unsafe_allow_html=True)

            # 정치
            st.markdown("""<div style='font-size:11px;color:#38BDF8;
                margin:10px 0 4px;font-weight:600;'>🏛️ 정치</div>""",
                unsafe_allow_html=True)
            ca, cb = st.columns(2)
            with ca:
                cat_election = st.checkbox("선거·공천",   value=True,  key="cat_election")
                cat_party    = st.checkbox("정당·이념",   value=True,  key="cat_party")
            with cb:
                cat_policy   = st.checkbox("정책·입법",   value=False, key="cat_policy")
                cat_govt     = st.checkbox("행정·사법",   value=False, key="cat_govt")

            # 경제
            st.markdown("""<div style='font-size:11px;color:#34D399;
                margin:10px 0 4px;font-weight:600;'>📈 경제</div>""",
                unsafe_allow_html=True)
            cc, cd = st.columns(2)
            with cc:
                cat_realestate = st.checkbox("부동산",    value=False, key="cat_real")
                cat_finance    = st.checkbox("주식·금융", value=False, key="cat_fin")
            with cd:
                cat_labor      = st.checkbox("고용·임금", value=False, key="cat_labor")
                cat_price      = st.checkbox("물가·소비", value=False, key="cat_price")

            # 사회
            st.markdown("""<div style='font-size:11px;color:#FB923C;
                margin:10px 0 4px;font-weight:600;'>🌿 사회</div>""",
                unsafe_allow_html=True)
            ce, cf = st.columns(2)
            with ce:
                cat_edu     = st.checkbox("교육",         value=False, key="cat_edu")
                cat_welfare = st.checkbox("복지·의료",    value=False, key="cat_welf")
            with cf:
                cat_gender  = st.checkbox("젠더·세대",    value=False, key="cat_gender")
                cat_crime   = st.checkbox("범죄·안전",    value=False, key="cat_crime")

            # 지역 (지방선거 특화)
            st.markdown("""<div style='font-size:11px;color:#A78BFA;
                margin:10px 0 4px;font-weight:600;'>📍 지역 이슈</div>""",
                unsafe_allow_html=True)
            cg, ch = st.columns(2)
            with cg:
                cat_metro    = st.checkbox("수도권",       value=False, key="cat_metro")
                cat_yeongnam = st.checkbox("영남",         value=False, key="cat_yeong")
            with ch:
                cat_honam    = st.checkbox("호남",         value=False, key="cat_honam")
                cat_chungcheong = st.checkbox("충청",      value=False, key="cat_chung")

            # 국제
            st.markdown("""<div style='font-size:11px;color:#64748B;
                margin:10px 0 4px;font-weight:600;'>🌏 국제·외교</div>""",
                unsafe_allow_html=True)
            ci, cj = st.columns(2)
            with ci:
                cat_us_kr  = st.checkbox("한미관계",      value=False, key="cat_us")
                cat_cn_kr  = st.checkbox("한중관계",       value=False, key="cat_cn")
            with cj:
                cat_jp_kr  = st.checkbox("한일관계",       value=False, key="cat_jp")
                cat_global = st.checkbox("글로벌 이슈",    value=False, key="cat_global")

            # 추가 키워드
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            extra_kw = st.text_input(
                "추가 키워드 (선택)",
                placeholder="예: 이재명, 추경호, 반도체",
                key="sc_extra",
            )

            # 생성 버튼
            gen_btn = st.button(
                "⬡  맥락 자동 생성",
                key="gen_scenario",
                use_container_width=True,
            )

            if gen_btn:
                if not api_key:
                    st.error("API Key를 먼저 입력해 주세요.")
                else:
                    cat_map = {
                        "선거·공천": cat_election, "정당·이념": cat_party,
                        "정책·입법": cat_policy,   "행정·사법": cat_govt,
                        "부동산":    cat_realestate,"주식·금융": cat_finance,
                        "고용·임금": cat_labor,    "물가·소비": cat_price,
                        "교육":      cat_edu,       "복지·의료": cat_welfare,
                        "젠더·세대": cat_gender,   "범죄·안전": cat_crime,
                        "수도권 이슈":    cat_metro,
                        "영남 이슈":      cat_yeongnam,
                        "호남 이슈":      cat_honam,
                        "충청 이슈":      cat_chungcheong,
                        "한미관계":  cat_us_kr,    "한중관계":  cat_cn_kr,
                        "한일관계":  cat_jp_kr,    "글로벌 이슈": cat_global,
                    }
                    selected_cats = [k for k, v in cat_map.items() if v]

                    if not selected_cats:
                        st.warning("카테고리를 하나 이상 선택해 주세요.")
                    else:
                        with st.spinner("🔍 Google Search로 최신 뉴스 수집 중…"):
                            try:
                                generated, removed = generate_scenario_with_grounding(
                                    api_key=api_key,
                                    model=model_choice,
                                    categories=selected_cats,
                                    ref_date=str(ref_date),
                                    extra_context=extra_kw,
                                )
                                st.session_state['auto_scenario'] = generated
                                st.session_state['scenario_date'] = str(ref_date)
                                st.session_state['scenario_cats'] = selected_cats

                                if removed:
                                    st.warning(
                                        f"⚠️  여론조사 관련 표현 {len(removed)}줄이 자동 제거되었습니다."
                                    )
                                    with st.expander("제거된 내용 확인"):
                                        for r in removed:
                                            st.markdown(
                                                f"<div style='font-family:\"DM Mono\",monospace;"
                                                f"font-size:11px;color:#64748B;"
                                                f"text-decoration:line-through;padding:2px 0;'>"
                                                f"✕ {r}</div>",
                                                unsafe_allow_html=True
                                            )
                                else:
                                    st.success("맥락 생성 완료 — 여론조사 표현 없음 ✓")
                            except Exception as e:
                                st.error(f"생성 오류: {e}")

            # 생성 결과 미리보기 & 편집
            if 'auto_scenario' in st.session_state:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;margin:10px 0 6px;">
                    <span class="ai-badge">⬡ AI GENERATED</span>
                    <span style="font-family:'DM Mono',monospace;font-size:10px;color:#334155;">
                        {st.session_state.get('scenario_date','—')}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                edited_scenario = st.text_area(
                    "생성된 맥락 (직접 편집 가능)",
                    value=st.session_state['auto_scenario'],
                    height=200,
                    key="sc_ai_edit",
                )
                # 편집본을 세션에 반영
                st.session_state['auto_scenario'] = edited_scenario

                # 적용 확인
                apply_btn = st.button("✓  시뮬레이션에 적용", key="apply_scenario",
                                      use_container_width=True)
                if apply_btn:
                    st.session_state['active_scenario'] = edited_scenario
                    st.success("적용 완료! 시뮬레이션에 반영됩니다.")

        with sc_tab_manual:
            manual_scenario = st.text_area(
                "현재 시점의 배경/사건",
                value=st.session_state.get('active_scenario',
                    "오늘은 {}입니다.\n최근 주요 사건\n"
                    "- 지방선거 선거구 획정; 광역의회 비례대표 의석 확대\n"
                    "- 이재명, 인도·베트남 국빈 방문\n"
                    "- 코스피 최고 장중 6,557.76\n"
                    "- 삼성전자 노조 '성과급 투쟁'\n"
                    "- 이란 전쟁 휴전 연장".format(date.today())),
                height=200,
                key="sc_manual",
            )
            apply_manual = st.button("✓  수동 입력 적용", key="apply_manual",
                                     use_container_width=True)
            if apply_manual:
                st.session_state['active_scenario'] = manual_scenario
                st.success("적용 완료!")

    # ── 현재 적용된 맥락 상태 표시 ──────────────────────────────────────────
    active_scenario = st.session_state.get('active_scenario', None)
    if active_scenario:
        lines = [l for l in active_scenario.split('\n') if l.strip()]
        preview = lines[0] if lines else ""
        st.markdown(f"""
        <div style="background:#0D1117;border:1px solid rgba(56,189,248,.2);
                    border-radius:8px;padding:10px 14px;margin-top:8px;">
            <div style="font-family:'DM Mono',monospace;font-size:9px;
                        color:#38BDF8;letter-spacing:.12em;margin-bottom:4px;">
                ● ACTIVE SCENARIO
            </div>
            <div style="font-family:'Noto Sans KR',sans-serif;font-size:11px;
                        color:#64748B;line-height:1.5;">
                {preview[:60]}{'…' if len(preview)>60 else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#0D1117;border:1px solid #1E2533;
                    border-radius:8px;padding:10px 14px;margin-top:8px;">
            <div style="font-family:'DM Mono',monospace;font-size:9px;
                        color:#334155;letter-spacing:.12em;margin-bottom:4px;">
                ○ NO SCENARIO APPLIED
            </div>
            <div style="font-family:'Noto Sans KR',sans-serif;font-size:11px;color:#334155;">
                AI 자동 생성 또는 수동 입력 후 적용하세요
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 사이드바 하단 런 요약 ────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    if 'results_df' in st.session_state:
        n = len(st.session_state['results_df'])
        q = len(st.session_state['questions'])
        st.markdown(f"""
        <div style="background:#0D1117;border:1px solid #1E2533;border-radius:10px;
                    padding:14px 16px;text-align:center;">
            <div style="font-family:'DM Mono',monospace;font-size:9px;
                        color:#334155;letter-spacing:.15em;margin-bottom:8px;">LAST RUN</div>
            <div style="font-family:'Syne',sans-serif;font-size:22px;
                        font-weight:800;color:#38BDF8;">{n:,}</div>
            <div style="font-family:'Noto Sans KR',sans-serif;font-size:11px;
                        color:#64748B;">케이스 · {q}개 문항</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 + 2
# ═══════════════════════════════════════════════════════════════════════════════
col_in1, col_in2 = st.columns([1, 1], gap="large")

with col_in1:
    st.markdown('<div class="section-label">01 &nbsp; Data Asset</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-title">응답자 프로필 로드</div>',
                unsafe_allow_html=True)

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
    st.markdown('<div class="section-label">02 &nbsp; Survey Design</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-title">설문 문항 설계</div>',
                unsafe_allow_html=True)

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

# ── 현재 적용된 시나리오 인라인 표시 ─────────────────────────────────────────
active_scenario = st.session_state.get('active_scenario', None)
if active_scenario:
    with st.expander("📌  현재 적용된 사회적 맥락 확인", expanded=False):
        st.markdown(f"""
        <div style="background:#0D1117;border-left:3px solid #38BDF8;
                    padding:14px 18px;border-radius:0 8px 8px 0;
                    font-family:'Noto Sans KR',sans-serif;font-size:13px;
                    color:#CBD5E1;line-height:1.8;white-space:pre-wrap;">
{active_scenario}
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("💡 사이드바 [사회적 맥락] 에서 AI 자동 생성 또는 수동 입력 후 '적용' 버튼을 눌러주세요.")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.divider()

run_col, _ = st.columns([1, 2])
with run_col:
    run_btn = st.button("⬡  지능형 시뮬레이션 시작", type="primary",
                        key="run_sim", use_container_width=True)

if run_btn:
    if not api_key or 'df' not in locals():
        st.error("설정 오류: API Key와 프로필 데이터를 모두 확인해 주세요.")
    else:
        # 적용된 시나리오 우선, 없으면 수동 입력값 사용
        run_scenario = st.session_state.get('active_scenario',
                       st.session_state.get('sc_manual', ""))
        if not run_scenario.strip():
            st.warning("사회적 맥락이 비어 있습니다. 기본값으로 진행합니다.")
            run_scenario = f"오늘은 {date.today()}입니다."

        try:
            client = genai.Client(api_key=api_key)
            questions = [q.strip() for q in question_input.split('\n') if q.strip()]
            q_block = "\n".join([f"Q{i+1}. {q}" for i, q in enumerate(questions)])

            with st.status("⬡  AI 페르소나가 응답을 생성하고 있습니다…",
                           expanded=True) as status:
                results_df = df.copy()
                all_responses: dict = {}

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            process_persona, client, model_choice,
                            row[selected_features].to_dict(),
                            run_scenario, persona_tmpl,
                            question_tmpl_ui, len(questions), q_block, temp
                        ): idx
                        for idx, row in df.iterrows()
                    }
                    pbar = st.progress(0, text="응답 생성 중…")
                    for i, future in enumerate(as_completed(futures)):
                        idx = futures[future]
                        all_responses[idx] = future.result()
                        pbar.progress((i+1)/len(df),
                                      text=f"응답 생성 중… {i+1}/{len(df)}")

                for i in range(len(questions)):
                    results_df[f'q_{i+1}'] = [all_responses[idx][i]
                                               for idx in range(len(df))]

                st.session_state['results_df'] = results_df
                st.session_state['questions'] = questions
                # 사용된 시나리오도 저장 (내보내기 메타용)
                st.session_state['used_scenario'] = run_scenario
                status.update(
                    label=f"✓  완료  ·  {len(df)}개 케이스  ·  {len(questions)}개 문항",
                    state="complete")
        except Exception as e:
            st.error(f"실행 오류: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if 'results_df' in st.session_state:
    res_df = st.session_state['results_df']
    q_list = st.session_state['questions']
    q_cols  = [c for c in res_df.columns if c.startswith('q_')]
    demo_cols = [c for c in res_df.columns if not c.startswith('q_')]

    st.divider()
    st.markdown('<div class="section-label">03 &nbsp; Analysis Dashboard</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-title">분석 결과</div>', unsafe_allow_html=True)

    na_total = sum((res_df[qc] == 'NA').sum() for qc in q_cols if qc in res_df.columns)
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
            <div style="background:#0D1117;border:1px solid #1E2533;
                        border-radius:10px;padding:18px 20px;text-align:center;">
                <div style="font-family:'Syne',sans-serif;font-size:26px;
                            font-weight:800;color:#38BDF8;">{val}</div>
                <div style="font-family:'Noto Sans KR',sans-serif;font-size:12px;
                            color:#64748B;margin-top:4px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["📈  빈도 분석", "🔀  교차 분석", "💾  내보내기"])

    # ── 빈도 분석 ────────────────────────────────────────────────────────────
    with t1:
        for i, q in enumerate(q_list):
            q_col = f'q_{i+1}'
            st.markdown(f'<div class="q-label">Q{i+1}. &nbsp;{q}</div>',
                        unsafe_allow_html=True)
            counts   = res_df[q_col].value_counts().sort_index()
            percents = (res_df[q_col].value_counts(normalize=True)
                        .sort_index() * 100).round(1)
            stats_df = pd.DataFrame({
                '사례수 (N)': counts,
                '비율 (%)': percents.apply(lambda x: f"{x:.1f}%"),
            })
            c1, c2 = st.columns([1, 2])
            with c1:
                st.dataframe(stats_df, use_container_width=True)
                st_copy_to_clipboard(stats_df.to_csv(sep='\t'),
                                     "📋  Excel로 복사", f"copy_f_{i}")
            with c2:
                st.bar_chart(counts, color=["#38BDF8"])
            st.divider()

    # ── 교차 분석 ────────────────────────────────────────────────────────────
    with t2:
        c1, c2 = st.columns(2)
        with c1:
            r_var = st.selectbox("⬇️  행 변수 (인구통계)", options=demo_cols,
                                 index=None, placeholder="행 변수 선택", key="cross_row")
        with c2:
            col_var = st.selectbox("➡️  열 변수 (설문 문항)", options=q_cols,
                                   placeholder="문항 선택", key="cross_col")

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
                palette = ["#818CF8","#38BDF8","#34D399","#FB923C",
                           "#F472B6","#A78BFA","#60A5FA","#4ADE80"]
                n_cols = len(ctab_pct.columns)
                chart_colors = (palette * ((n_cols // len(palette)) + 1))[:n_cols]
                st.bar_chart(ctab_pct, color=chart_colors)
        else:
            st.info("변수를 선택하면 교차표가 자동으로 생성됩니다.")

    # ── 내보내기 ──────────────────────────────────────────────────────────────
    with t3:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1E2533;border-radius:10px;
                    padding:24px;margin-bottom:20px;">
            <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;
                        color:#E2E8F0;margin-bottom:8px;">📦  결과 데이터셋 다운로드</div>
            <div style="font-family:'Noto Sans KR',sans-serif;font-size:13px;
                        color:#64748B;line-height:1.6;">
                원본 프로필 + 생성 응답 + 사용된 사회적 맥락이 포함된 XLSX 파일
            </div>
        </div>
        """, unsafe_allow_html=True)

        out = BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as wr:
            res_df.to_excel(wr, index=False, sheet_name="Results")
            # 사용된 시나리오를 별도 시트로 저장
            used_sc = st.session_state.get('used_scenario', '')
            meta_df = pd.DataFrame({
                "항목": ["생성 일시", "사용된 사회적 맥락"],
                "내용": [str(date.today()), used_sc],
            })
            meta_df.to_excel(wr, index=False, sheet_name="Scenario_Meta")

        st.download_button(
            "📥  Excel 결과 리포트 다운로드",
            out.getvalue(),
            "persona_lab_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_final",
            use_container_width=True,
        )