import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
import re
import json
import base64
import html as _html
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
html, body, [data-testid="stAppViewContainer"] { background-color:#0A0C10 !important; color:#E2E8F0 !important; }
[data-testid="stSidebar"] { background-color:#0D1017 !important; border-right:1px solid #1E2533 !important; }
[data-testid="stSidebar"] > div:first-child { padding-top:2rem; }

.hero-wrap {
    background:linear-gradient(135deg,#0D1117 0%,#111827 50%,#0A0C10 100%);
    border:1px solid #1E2533; border-radius:16px; padding:36px 40px;
    margin-bottom:32px; position:relative; overflow:hidden;
}
.hero-wrap::before { content:''; position:absolute; top:-60px; right:-60px; width:240px; height:240px; background:radial-gradient(circle,rgba(56,189,248,.08) 0%,transparent 70%); pointer-events:none; }
.hero-wrap::after  { content:''; position:absolute; bottom:-40px; left:30%; width:180px; height:180px; background:radial-gradient(circle,rgba(168,85,247,.06) 0%,transparent 70%); pointer-events:none; }
.hero-badge { display:inline-flex; align-items:center; gap:6px; background:rgba(56,189,248,.08); border:1px solid rgba(56,189,248,.2); border-radius:100px; padding:4px 12px; font-family:'DM Mono',monospace; font-size:11px; color:#38BDF8; letter-spacing:.08em; margin-bottom:16px; }
.hero-title { font-family:'Syne',sans-serif; font-size:32px; font-weight:800; color:#F1F5F9; margin:0 0 8px; line-height:1.2; letter-spacing:-.02em; }
.hero-title span { background:linear-gradient(90deg,#38BDF8,#818CF8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero-sub { font-family:'Noto Sans KR',sans-serif; font-size:14px; color:#64748B; margin:0; font-weight:300; }

.section-label { font-family:'DM Mono',monospace; font-size:10px; letter-spacing:.15em; color:#38BDF8; text-transform:uppercase; margin-bottom:12px; display:flex; align-items:center; gap:8px; }
.section-label::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,#1E2533,transparent); }
.section-title { font-family:'Syne',sans-serif; font-size:18px; font-weight:700; color:#E2E8F0; margin:0 0 20px; }

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
.stSelectbox > div > div { background-color:#111827 !important; border:1px solid #1E2533 !important; border-radius:8px !important; color:#E2E8F0 !important; font-family:'Noto Sans KR',sans-serif !important; font-size:13px !important; }
[data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus { border-color:#38BDF8 !important; box-shadow:0 0 0 3px rgba(56,189,248,.08) !important; }
label, .stSelectbox label, .stSlider label { font-family:'Noto Sans KR',sans-serif !important; font-size:12px !important; color:#94A3B8 !important; font-weight:500 !important; }

[data-testid="stButton"] > button { background:linear-gradient(135deg,#0EA5E9,#6366F1) !important; color:#fff !important; border:none !important; border-radius:8px !important; font-family:'Syne',sans-serif !important; font-weight:700 !important; font-size:14px !important; height:48px !important; transition:opacity .2s,transform .1s !important; box-shadow:0 4px 24px rgba(14,165,233,.25) !important; }
[data-testid="stButton"] > button:hover { opacity:.9 !important; transform:translateY(-1px) !important; }
[data-testid="stDownloadButton"] > button { background:#111827 !important; color:#38BDF8 !important; border:1px solid #1E2533 !important; border-radius:8px !important; font-family:'Syne',sans-serif !important; font-weight:600 !important; font-size:13px !important; height:44px !important; }
[data-testid="stDownloadButton"] > button:hover { border-color:#38BDF8 !important; background:rgba(56,189,248,.06) !important; }

[data-testid="stTabs"] [data-baseweb="tab-list"] { background:#0D1117 !important; border-bottom:1px solid #1E2533 !important; gap:0 !important; padding:0 !important; }
[data-testid="stTabs"] [data-baseweb="tab"] { background:transparent !important; color:#64748B !important; font-family:'Syne',sans-serif !important; font-size:13px !important; font-weight:600 !important; padding:12px 20px !important; border-bottom:2px solid transparent !important; border-radius:0 !important; }
[data-testid="stTabs"] [aria-selected="true"] { color:#38BDF8 !important; border-bottom-color:#38BDF8 !important; background:transparent !important; }
[data-testid="stTabs"] [data-baseweb="tab-panel"] { background:#0D1117 !important; padding:24px !important; border:1px solid #1E2533 !important; border-top:none !important; border-radius:0 0 12px 12px !important; }

[data-testid="stDataFrame"], .stTable { background:#0D1117 !important; border:1px solid #1E2533 !important; border-radius:8px !important; overflow:hidden !important; }
[data-testid="stDataFrame"] th { background:#111827 !important; color:#94A3B8 !important; font-family:'DM Mono',monospace !important; font-size:11px !important; }
[data-testid="stDataFrame"] td { color:#CBD5E1 !important; font-size:13px !important; }

[data-testid="stProgress"] > div > div { background:linear-gradient(90deg,#0EA5E9,#6366F1) !important; border-radius:100px !important; }
[data-testid="stProgress"] > div { background:#1E2533 !important; border-radius:100px !important; }
[data-testid="stStatusWidget"] { background:#0D1117 !important; border:1px solid #1E2533 !important; border-radius:10px !important; }
[data-testid="stExpander"] { background:#0D1117 !important; border:1px solid #1E2533 !important; border-radius:10px !important; }
[data-testid="stExpander"] summary { font-family:'Syne',sans-serif !important; font-size:13px !important; font-weight:600 !important; color:#94A3B8 !important; }
[data-testid="stAlert"] { background:rgba(56,189,248,.05) !important; border:1px solid rgba(56,189,248,.15) !important; border-radius:8px !important; color:#94A3B8 !important; }
[data-testid="stAlert"][data-type="success"] { background:rgba(34,197,94,.05) !important; border-color:rgba(34,197,94,.2) !important; }
[data-testid="stAlert"][data-type="error"] { background:rgba(239,68,68,.05) !important; border-color:rgba(239,68,68,.2) !important; }

[data-testid="stMultiSelect"] span[data-baseweb="tag"] { background:rgba(56,189,248,.1) !important; border:1px solid rgba(56,189,248,.2) !important; border-radius:4px !important; color:#38BDF8 !important; font-size:11px !important; }
[data-testid="stSlider"] [role="slider"] { background:#38BDF8 !important; }
[data-testid="stCheckbox"] label { color:#94A3B8 !important; font-size:12px !important; }

.stat-chip { display:inline-flex; align-items:center; gap:6px; background:rgba(56,189,248,.06); border:1px solid rgba(56,189,248,.12); border-radius:8px; padding:8px 14px; font-family:'DM Mono',monospace; font-size:12px; color:#38BDF8; margin:4px 4px 4px 0; }
.stat-chip strong { font-size:16px; font-weight:600; color:#F1F5F9; }

.q-label { font-family:'Syne',sans-serif; font-size:14px; font-weight:700; color:#E2E8F0; margin:24px 0 12px; padding:12px 16px; background:#111827; border-left:3px solid #38BDF8; border-radius:0 8px 8px 0; }
.q-label.open { border-left-color:#34D399; }
.q-label.branch { border-left-color:#A78BFA; }

/* 문항 카드 */
.q-card { background:#0D1117; border:1px solid #1E2533; border-radius:10px; padding:16px 20px; margin-bottom:10px; }
.q-card .q-type-badge { display:inline-block; font-family:'DM Mono',monospace; font-size:9px; letter-spacing:.1em; padding:2px 8px; border-radius:4px; margin-bottom:8px; }
.q-card .q-type-badge.single  { background:rgba(56,189,248,.1);  color:#38BDF8;  border:1px solid rgba(56,189,248,.2); }
.q-card .q-type-badge.open    { background:rgba(52,211,153,.1);  color:#34D399;  border:1px solid rgba(52,211,153,.2); }
.q-card .q-type-badge.branch  { background:rgba(167,139,250,.1); color:#A78BFA;  border:1px solid rgba(167,139,250,.2); }

.ai-badge { display:inline-flex; align-items:center; gap:4px; background:rgba(56,189,248,.1); border:1px solid rgba(56,189,248,.25); border-radius:4px; padding:2px 8px; font-family:'DM Mono',monospace; font-size:9px; color:#38BDF8; letter-spacing:.08em; }
hr { border-color:#1E2533 !important; }
p, li, span, div, td, th, input, textarea, select, button { font-family:'Noto Sans KR',sans-serif; }

#copy-notice { position:fixed; top:24px; right:24px; z-index:99999; background:linear-gradient(135deg,#0EA5E9,#6366F1); color:white; padding:12px 20px; border-radius:10px; font-family:'Syne',sans-serif; font-size:13px; font-weight:600; box-shadow:0 8px 32px rgba(14,165,233,.3); display:none; }
</style>
<div id="copy-notice">✓ 클립보드에 복사되었습니다</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 헬퍼
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
    <button onclick="copyClip_{key}()" style="width:100%;height:36px;background:#111827;border:1px solid #1E2533;border-radius:6px;cursor:pointer;font-family:'Syne',sans-serif;font-weight:600;font-size:12px;color:#94A3B8;"
    onmouseover="this.style.borderColor='#38BDF8';this.style.color='#38BDF8';"
    onmouseout="this.style.borderColor='#1E2533';this.style.color='#94A3B8';">
        {button_label}
    </button>""", height=42)


# ═══════════════════════════════════════════════════════════════════════════════
# [Phase 1] PDF → 문항 구조화 JSON
# ═══════════════════════════════════════════════════════════════════════════════

def extract_questions_from_pdf(client, model: str, pdf_bytes: bytes) -> list[dict]:
    """
    PDF를 Gemini에 직접 전달해 문항을 구조화된 JSON으로 추출합니다.
    반환: [
      {
        "id": "문1",
        "text": "질문 내용",
        "type": "single" | "open" | "multi",
        "options": {"1": "보기1", "2": "보기2", "9": "모름/응답거절"},
        "branch": {"3": "문3", "9": "문3"}   # 분기 조건 (없으면 {})
      }, ...
    ]
    """
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode()

    prompt = """이 설문지 PDF에서 모든 조사 문항(SQ 포함, D문항 제외)을 추출하여
아래 JSON 배열 형식으로만 출력하세요. 다른 설명은 절대 하지 마세요.

[출력 형식]
[
  {
    "id": "문1",
    "text": "질문 전체 텍스트",
    "type": "single",
    "options": {"1": "보기1", "2": "보기2", "9": "모름/응답거절"},
    "branch": {"3": "문3", "9": "문3"}
  }
]

[type 규칙]
- "single": 보기 중 하나 선택
- "open": 자유응답 (주관식)
- "multi": 복수 선택 가능

[branch 규칙]
- 특정 응답 시 다른 문항으로 건너뛰는 경우만 기록
- 분기 없으면 빈 객체 {} 로 표시
- 값은 이동할 문항 id (예: "문3")

[주의 — 반드시 준수]
- options의 key는 반드시 문자열 숫자 ("1", "9", "97" 등)
- 자유응답(open)은 options를 {} 로
- JSON만 출력, 마크다운 코드블록 금지
- text 및 options 값 안에 큰따옴표(")를 절대 사용하지 말 것.
  큰따옴표가 필요한 경우 작은따옴표(')로 대체하거나 해당 부분을 생략하세요.
  예) "굳이 말씀하신다면 '잘하고 있다'와 '잘못하고 있다' 중 어느 쪽입니까?" → 작은따옴표 사용 유지"""

    def _call_and_parse(temperature=0.0):
        resp = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                types.Part.from_text(text=prompt),
            ],
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=8192,
            )
        )
        raw = resp.text.strip()
        # 마크다운 펜스 제거
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
        raw = raw.strip()
        return raw

    # 1차 시도
    raw = _call_and_parse(temperature=0.0)

    def _safe_parse(text):
        """JSON 파싱 시도 - 실패 시 문자열 값 내 큰따옴표를 작은따옴표로 치환 후 재시도"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 각 JSON 문자열 값을 개별 파싱해서 내부 " 를 ' 로 치환
        try:
            # "key": "value" 패턴에서 value 내부의 " 를 ' 로 변환
            def replace_inner_quotes(m):
                inner = m.group(1)
                # 이미 이스케이프된 \" 는 건드리지 않고, 날것의 " 만 ' 로 변환
                inner = re.sub(r'(?<!\\)"', "'", inner)
                return f'"{inner}"'
            sanitized = re.sub(r'"((?:[^"\\]|\\.)*)"', replace_inner_quotes, text)
            return json.loads(sanitized)
        except Exception:
            return None

    result = _safe_parse(raw)
    if result is not None:
        return [q for q in result if not any(q.get('id','').upper().startswith(p) for p in ('SQ', 'D'))]

    # 2차 시도: Gemini에게 JSON만 정제해달라고 요청
    fix_prompt = f"""아래 텍스트는 JSON 배열인데 파싱 오류가 있습니다.
파싱 오류의 주요 원인은 JSON 문자열 값 안에 이스케이프 처리되지 않은 큰따옴표(")가 포함된 것입니다.

수정 규칙:
1. 문자열 값 내부의 큰따옴표는 반드시 \" 로 이스케이프하세요.
2. JSON 구조(키, 배열, 객체)는 절대 변경하지 마세요.
3. 마크다운 코드블록 없이 JSON만 출력하세요.

{raw}"""
    fix_resp = client.models.generate_content(
        model=model,
        contents=fix_prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=8192,
        )
    )
    fixed = fix_resp.text.strip()
    fixed = re.sub(r"^```(?:json)?\s*", "", fixed, flags=re.MULTILINE)
    fixed = re.sub(r"\s*```\s*$", "", fixed, flags=re.MULTILINE)
    fixed = fixed.strip()

    try:
        qs = json.loads(fixed)
        return [q for q in qs if not any(q.get('id','').upper().startswith(p) for p in ('SQ', 'D'))]
    except json.JSONDecodeError:
        pass

    # 3차 시도: 정규식으로 각 객체 개별 추출
    objects = re.findall(r'\{[^{}]+\}', fixed, re.DOTALL)
    questions = []
    for obj_str in objects:
        try:
            q = json.loads(obj_str)
            if "id" in q and "text" in q:
                q.setdefault("type", "single")
                q.setdefault("options", {})
                q.setdefault("branch", {})
                questions.append(q)
        except json.JSONDecodeError:
            continue

    questions = [q for q in questions if not any(q.get('id','').upper().startswith(p) for p in ('SQ', 'D'))]
    if questions:
        return questions

    raise ValueError(f"JSON 파싱 실패. Gemini 원본 응답:\n{raw}")


# ═══════════════════════════════════════════════════════════════════════════════
# [Phase 2] 페르소나 응답 생성 (분기 + 자유응답 처리)
# ═══════════════════════════════════════════════════════════════════════════════

def build_question_block(questions: list[dict]) -> str:
    """문항 리스트를 프롬프트용 텍스트로 변환"""
    lines = []
    for q in questions:
        lines.append(f"[{q['id']}] ({q['type']}) {q['text']}")
        if q.get("options"):
            for k, v in q["options"].items():
                lines.append(f"  {k}. {v}")
        if q.get("branch"):
            branch_desc = ", ".join([f"응답{k}→{v}" for k, v in q["branch"].items()])
            lines.append(f"  ※ 분기: {branch_desc}")
        lines.append("")
    return "\n".join(lines)


def process_persona_structured(client, model_choice, row_data, scenario,
                                persona_tmpl, questions, temp):
    """
    구조화된 문항 리스트로 페르소나 응답을 생성합니다.
    분기 로직을 Gemini가 직접 처리하며, 자유응답은 텍스트로 반환합니다.
    반환: {"문1": "2", "문2": "경제정책이 마음에 들지 않아서", "문3": "1", ...}
    """
    result = {q["id"]: "NA" for q in questions}
    try:
        profile_ctx = "\n".join([f"- {k}: {v}" for k, v in row_data.items()])
        q_block = build_question_block(questions)
        q_ids = [q["id"] for q in questions]

        sys_inst = f"""{persona_tmpl}

[현재 상황]
{scenario}

[상세 프로필]
{profile_ctx}"""

        user_prompt = f"""아래 설문 문항에 당신의 프로필과 가치관에 따라 응답하세요.

[수행 규칙]
1. 각 문항을 순서대로 읽고 응답하세요.
2. 분기(※ 분기) 표시가 있으면 해당 응답에 따라 지정된 문항으로 건너뛰세요.
   건너뛴 문항은 "SKIP"으로 표시하세요.
3. 선택형(single/multi)은 보기 번호(숫자)만 응답하세요.
4. 자유응답(open)은 당신의 프로필에 맞는 짧고 구체적인 한 문장으로 응답하세요.
5. 출력 형식을 반드시 준수하세요. 다른 설명 금지.

[출력 형식 — 반드시 이 형식만]
{chr(10).join([f'{q_id}: <응답>' for q_id in q_ids])}

[문항]
{q_block}"""

        resp = client.models.generate_content(
            model=model_choice,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_inst, temperature=temp)
        )

        # 응답 파싱: "문1: 2" 형태
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            for q_id in q_ids:
                if line.startswith(f"{q_id}:"):
                    val = line[len(q_id)+1:].strip()
                    result[q_id] = val
                    break

    except Exception:
        pass
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 사회적 맥락 생성 (syn6 그대로 유지)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_scenario_with_grounding(api_key, model, categories, ref_date, extra_context=""):
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
    client = genai.Client(api_key=api_key)
    search_instructions = "\n".join([
        f'- [{cat}] 검색어: "{CATEGORY_SEARCH.get(cat, cat + " 최신 뉴스")}"'
        for cat in categories
    ])
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

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(tools=[grounding_tool], temperature=0.3)
    )
    raw_text = resp.text.strip()
    cleaned, removed = filter_poll_references_ai(client, model, raw_text)
    return cleaned, removed


def filter_poll_references_ai(client, model, text):
    import re as _re
    HARD_PATTERNS = [
        r'여론\s*조사',
        r'오차\s*범위',
        r'지지\s*율',
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
    if not after_hard_text:
        return after_hard_text, removed_lines

    review_prompt = f"""아래 텍스트에서 다음 기준에 해당하는 문장을 찾아내세요.

[제거 기준]
- 특정 후보·정당이 경쟁 상대보다 우세하거나 열세라고 서술하는 문장
- 선거 결과나 당락을 예측·암시하는 문장
- 여론·민심의 방향·추세를 평가하는 문장

[유지 기준]
- 사건·발언·정책·행동 등 객관적 사실 문장
- 정치적 활동 묘사 문장
- 경제·사회·외교 사실 정보

[출력 형식]
제거 대상이 있으면:
REMOVE:
<원문 그대로 한 줄씩>

없으면:
REMOVE: NONE

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
        pass
    return after_hard_text, removed_lines


# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">⬡ &nbsp;AI-POWERED SURVEY SIMULATION</div>
    <h1 class="hero-title">Persona<span>Lab</span> AI</h1>
    <p class="hero-sub">고도화된 언어 모델 기반의 페르소나 합성 데이터 생성 및 사회과학 분석 솔루션 &nbsp;·&nbsp; v6.0 Professional</p>
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
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:#334155;
                    letter-spacing:.1em;margin-top:4px;">PERSONALAB AI v6.0</div>
    </div>""", unsafe_allow_html=True)

    with st.expander("🔑  API & Model Engine", expanded=True):
        api_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
        model_choice = st.selectbox("모델 엔진", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
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

    with st.expander("🌐  사회적 맥락 (Scenario)", expanded=True):
        sc_tab_ai, sc_tab_manual = st.tabs(["🤖  AI 자동 생성", "✏️  수동 입력"])

        with sc_tab_ai:
            ref_date = st.date_input("기준 날짜", value=date.today(), key="sc_date")
            st.markdown("<div style='font-size:12px;color:#94A3B8;margin-bottom:6px;'>수집 카테고리</div>", unsafe_allow_html=True)

            ca, cb = st.columns(2)
            with ca:
                cat_election = st.checkbox("선거·공천",  value=True,  key="cat_election")
                cat_party    = st.checkbox("정당·이념",  value=True,  key="cat_party")
                cat_policy   = st.checkbox("정책·입법",  value=False, key="cat_policy")
                cat_govt     = st.checkbox("행정·사법",  value=False, key="cat_govt")
                cat_realestate=st.checkbox("부동산",     value=False, key="cat_real")
                cat_finance  = st.checkbox("주식·금융",  value=False, key="cat_fin")
                cat_labor    = st.checkbox("고용·임금",  value=False, key="cat_labor")
                cat_price    = st.checkbox("물가·소비",  value=False, key="cat_price")
                cat_edu      = st.checkbox("교육",       value=False, key="cat_edu")
                cat_welfare  = st.checkbox("복지·의료",  value=False, key="cat_welf")
            with cb:
                cat_gender   = st.checkbox("젠더·세대",  value=False, key="cat_gender")
                cat_crime    = st.checkbox("범죄·안전",  value=False, key="cat_crime")
                cat_metro    = st.checkbox("수도권",     value=False, key="cat_metro")
                cat_yeongnam = st.checkbox("영남",       value=False, key="cat_yeong")
                cat_honam    = st.checkbox("호남",       value=False, key="cat_honam")
                cat_chung    = st.checkbox("충청",       value=False, key="cat_chung")
                cat_us       = st.checkbox("한미관계",   value=False, key="cat_us")
                cat_cn       = st.checkbox("한중관계",   value=False, key="cat_cn")
                cat_jp       = st.checkbox("한일관계",   value=False, key="cat_jp")
                cat_global   = st.checkbox("글로벌",     value=False, key="cat_global")

            extra_kw = st.text_input("추가 키워드", placeholder="예: 이재명, 반도체", key="sc_extra")
            gen_btn = st.button("⬡  맥락 자동 생성", key="gen_scenario", use_container_width=True)

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
                        "수도권 이슈": cat_metro,  "영남 이슈": cat_yeongnam,
                        "호남 이슈":  cat_honam,   "충청 이슈": cat_chung,
                        "한미관계":  cat_us,       "한중관계":  cat_cn,
                        "한일관계":  cat_jp,       "글로벌 이슈": cat_global,
                    }
                    selected_cats = [k for k, v in cat_map.items() if v]
                    if not selected_cats:
                        st.warning("카테고리를 하나 이상 선택해 주세요.")
                    else:
                        with st.spinner("🔍 최신 뉴스 수집 중…"):
                            try:
                                generated, removed = generate_scenario_with_grounding(
                                    api_key, model_choice, selected_cats,
                                    str(ref_date), extra_kw)
                                st.session_state['auto_scenario'] = generated
                                st.session_state['scenario_date'] = str(ref_date)
                                if removed:
                                    st.warning(f"⚠️ 여론조사 관련 표현 {len(removed)}줄 자동 제거")
                                    with st.expander("제거된 내용"):
                                        for r in removed:
                                            st.markdown(f"<div style='font-family:\"DM Mono\",monospace;font-size:11px;color:#64748B;text-decoration:line-through;'>✕ {r}</div>", unsafe_allow_html=True)
                                else:
                                    st.success("맥락 생성 완료 ✓")
                            except Exception as e:
                                st.error(f"생성 오류: {e}")

            if 'auto_scenario' in st.session_state:
                edited_scenario = st.text_area(
                    "생성된 맥락 (편집 가능)",
                    value=st.session_state['auto_scenario'],
                    height=180, key="sc_ai_edit")
                st.session_state['auto_scenario'] = edited_scenario
                if st.button("✓  적용", key="apply_scenario", use_container_width=True):
                    st.session_state['active_scenario'] = edited_scenario
                    st.success("적용 완료!")

        with sc_tab_manual:
            manual_scenario = st.text_area(
                "현재 시점의 배경/사건",
                value=st.session_state.get('active_scenario',
                    f"오늘은 {date.today()}입니다.\n최근 주요 사건\n- "),
                height=200, key="sc_manual")
            if st.button("✓  수동 입력 적용", key="apply_manual", use_container_width=True):
                st.session_state['active_scenario'] = manual_scenario
                st.success("적용 완료!")

    # 활성 시나리오 상태
    active_scenario = st.session_state.get('active_scenario', None)
    if active_scenario:
        preview = [l for l in active_scenario.split('\n') if l.strip()]
        st.markdown(f"""
        <div style="background:#0D1117;border:1px solid rgba(56,189,248,.2);border-radius:8px;padding:10px 14px;margin-top:8px;">
            <div style="font-family:'DM Mono',monospace;font-size:9px;color:#38BDF8;letter-spacing:.12em;margin-bottom:4px;">● ACTIVE SCENARIO</div>
            <div style="font-family:'Noto Sans KR',sans-serif;font-size:11px;color:#64748B;">{(preview[0] if preview else '')[:60]}{'…' if len(preview[0] if preview else '')>60 else ''}</div>
        </div>""", unsafe_allow_html=True)

    if 'results_df' in st.session_state:
        n = len(st.session_state['results_df'])
        q = len(st.session_state.get('questions_structured', []))
        st.markdown(f"""
        <div style="background:#0D1117;border:1px solid #1E2533;border-radius:10px;padding:14px 16px;text-align:center;margin-top:12px;">
            <div style="font-family:'DM Mono',monospace;font-size:9px;color:#334155;letter-spacing:.15em;margin-bottom:8px;">LAST RUN</div>
            <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:#38BDF8;">{n:,}</div>
            <div style="font-family:'Noto Sans KR',sans-serif;font-size:11px;color:#64748B;">케이스 · {q}개 문항</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — 프로필 데이터
# ═══════════════════════════════════════════════════════════════════════════════
col_in1, col_in2 = st.columns([1, 1], gap="large")

with col_in1:
    st.markdown('<div class="section-label">01 &nbsp; Data Asset</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">응답자 프로필 로드</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("CSV 또는 XLSX", type=["xlsx", "csv"],
                                     key="uploader", label_visibility="collapsed")
    if uploaded_file:
        df = (pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv')
              else pd.read_excel(uploaded_file))
        df.columns = [str(c).lower().strip() for c in df.columns]

        for col in df.columns:
            if any(kw in col for kw in ['조사일', '날짜', 'date']):
                def safe_date(x):
                    if pd.isna(x): return x
                    try:
                        v = float(x)
                        if v > 1000:
                            return pd.to_datetime(v, unit='D', origin='1899-12-30').strftime('%Y-%m-%d')
                        return str(int(v))
                    except:
                        try: return pd.to_datetime(x).strftime('%Y-%m-%d')
                        except: return str(x)
                df[col] = df[col].apply(safe_date)

        st.markdown(f"""
        <div style="margin:16px 0 12px;">
            <span class="stat-chip"><strong>{len(df):,}</strong> Cases</span>
            <span class="stat-chip"><strong>{len(df.columns)}</strong> Variables</span>
        </div>""", unsafe_allow_html=True)

        selected_features = st.multiselect(
            "핵심 분석 변수 선택", options=df.columns.tolist(),
            default=[c for c in df.columns if not any(kw in c for kw in ['id', 'no', 'uuid', '가중치'])],
            key="features")
        with st.expander("데이터셋 미리보기 (상위 5행)"):
            st.dataframe(df[selected_features].head(5), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — 설문 문항 입력 (PDF / 텍스트 선택)
# ═══════════════════════════════════════════════════════════════════════════════
with col_in2:
    st.markdown('<div class="section-label">02 &nbsp; Survey Design</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">설문 문항 입력</div>', unsafe_allow_html=True)

    tab_pdf, tab_text = st.tabs(["📄  PDF 업로드", "✏️  텍스트 직접 입력"])

    # ── PDF 탭 ────────────────────────────────────────────────────────────────
    with tab_pdf:
        pdf_file = st.file_uploader("설문지 PDF 업로드", type=["pdf"],
                                    key="pdf_uploader", label_visibility="collapsed")
        if pdf_file:
            # 리런 후에도 유지되도록 session_state에 저장
            st.session_state['pdf_bytes'] = pdf_file.read()
            st.session_state['pdf_name'] = pdf_file.name

        if 'pdf_bytes' in st.session_state:
            pdf_bytes = st.session_state['pdf_bytes']
            st.markdown(f"""
            <div style="margin:12px 0;">
                <span class="stat-chip"><strong>{st.session_state.get('pdf_name','')}</strong></span>
                <span class="stat-chip"><strong>{len(pdf_bytes)//1024} KB</strong></span>
            </div>""", unsafe_allow_html=True)

            if st.button("⬡  문항 자동 추출", key="extract_q", use_container_width=True):
                if not api_key:
                    st.error("API Key를 먼저 입력해 주세요.")
                else:
                    with st.spinner("📄 PDF에서 문항 구조 분석 중…"):
                        try:
                            qs_extracted = extract_questions_from_pdf(
                                genai.Client(api_key=api_key), model_choice, pdf_bytes)
                            st.session_state['questions_structured'] = qs_extracted
                            st.session_state['q_input_mode'] = 'pdf'
                            st.success(f"✓ {len(qs_extracted)}개 문항 추출 완료")
                        except Exception as e:
                            st.error(f"추출 오류: {e}")

    # ── 텍스트 탭 ─────────────────────────────────────────────────────────────
    with tab_text:
        st.markdown("""
        <div style="font-family:'Noto Sans KR',sans-serif;font-size:12px;color:#64748B;
                    line-height:1.7;margin-bottom:10px;">
            한 줄에 하나의 질문. 보기는 괄호 안에 포함하거나 생략 가능.<br>
            분기 없는 단순 선택형으로 처리됩니다.
        </div>""", unsafe_allow_html=True)

        text_q_input = st.text_area(
            "질문 리스트 (줄바꿈으로 구분)",
            value=(
                "이번 지방선거에 투표하실건가요? "
                "(1. 반드시 할 것이다 2. 아마 할 것 같다 "
                "3. 아마 하지 않을 것 같다 4. 투표하지 않겠다 5. 모름)\n"
                "대구시장 선거에서 누가 당선되는 것이 조금이라도 좋다고 보십니까? "
                "(1. 더불어민주당 김부겸 2. 국민의힘 추경호 3. 모름)"
            ),
            height=180,
            key="text_q_in",
        )

        if st.button("✓  문항 적용", key="apply_text_q", use_container_width=True):
            lines = [l.strip() for l in text_q_input.split('\n') if l.strip()]
            qs_from_text = []
            for i, line in enumerate(lines):
                # 보기 파싱: "1. xxx 2. yyy" 패턴 추출
                options = {}
                opt_matches = re.findall(r'(\d+)\.\s*([^0-9\(\)]+?)(?=\s+\d+\.|$|\))', line)
                for num, label in opt_matches:
                    options[num] = label.strip()
                # 질문 텍스트: 괄호 이전까지
                q_text = re.sub(r'\s*\(.*\)\s*$', '', line).strip()
                qs_from_text.append({
                    "id": f"문{i+1}",
                    "text": q_text,
                    "type": "single" if options else "open",
                    "options": options,
                    "branch": {},
                })
            st.session_state['questions_structured'] = qs_from_text
            st.session_state['q_input_mode'] = 'text'
            st.success(f"✓ {len(qs_from_text)}개 문항 적용 완료")

    # ── 공통: 추출/적용된 문항 미리보기 ──────────────────────────────────────
    if 'questions_structured' in st.session_state:
        qs = st.session_state['questions_structured']
        mode_label = "PDF 추출" if st.session_state.get('q_input_mode') == 'pdf' else "텍스트 입력"
        with st.expander(f"📋  문항 확인 — {mode_label} ({len(qs)}개)", expanded=True):
            for q in qs:
                TYPE_COLOR = {"single": "#38BDF8", "open": "#34D399", "multi": "#FB923C"}
                type_color = TYPE_COLOR.get(q['type'], "#94A3B8")

                # 헤더 행: [문id] type 분기
                header_parts = [
                    f"<span style='font-family:\"DM Mono\",monospace;font-size:11px;"
                    f"font-weight:600;color:{type_color};'>[{q['id']}]</span>",
                    f"<span style='font-family:\"DM Mono\",monospace;font-size:9px;"
                    f"padding:1px 6px;border-radius:3px;"
                    f"background:rgba(255,255,255,.05);color:#64748B;'>{q['type']}</span>",
                ]
                if q.get("branch"):
                    branch_desc = " | ".join([f"{k}→{v}" for k, v in q["branch"].items()])
                    header_parts.append(
                        f"<span style='font-size:10px;color:#A78BFA;'>{_html.escape(branch_desc)}</span>"
                    )
                header_html = "<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>" \
                              + "".join(header_parts) + "</div>"

                # 질문 텍스트
                q_text_safe = _html.escape(q['text'])
                text_html = f"<div style='font-size:13px;color:#CBD5E1;line-height:1.6;'>{q_text_safe}</div>"

                # 보기
                opts_html = ""
                if q.get("options"):
                    opts_str = " &nbsp;/&nbsp; ".join(
                        [f"{k}. {_html.escape(str(v))}" for k, v in q["options"].items()]
                    )
                    opts_html = f"<div style='font-size:11px;color:#64748B;margin-top:6px;line-height:1.8;'>{opts_str}</div>"

                card_html = (
                    "<div style='background:#111827;border:1px solid #1E2533;"
                    "border-radius:8px;padding:12px 14px;margin-bottom:8px;'>"
                    + header_html + text_html + opts_html
                    + "</div>"
                )
                st.markdown(card_html, unsafe_allow_html=True)

            # 엑셀 다운로드
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            q_meta_df = pd.DataFrame([{
                "문항ID":  q['id'],
                "유형":    q['type'],
                "질문":    q['text'],
                "보기":    " / ".join([f"{k}.{v}" for k, v in q.get('options', {}).items()]),
                "분기":    " | ".join([f"{k}→{v}" for k, v in q.get('branch', {}).items()]),
            } for q in qs])
            q_out = BytesIO()
            with pd.ExcelWriter(q_out, engine='openpyxl') as wr:
                q_meta_df.to_excel(wr, index=False, sheet_name="문항구조")
            st.download_button(
                "📥  문항 구조 Excel 다운로드",
                q_out.getvalue(),
                "questions_structure.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_q_structure",
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 활성 시나리오 표시
# ═══════════════════════════════════════════════════════════════════════════════
active_scenario = st.session_state.get('active_scenario', None)
if active_scenario:
    with st.expander("📌  현재 적용된 사회적 맥락", expanded=False):
        st.markdown(f"""
        <div style="background:#0D1117;border-left:3px solid #38BDF8;padding:14px 18px;
                    border-radius:0 8px 8px 0;font-family:'Noto Sans KR',sans-serif;
                    font-size:13px;color:#CBD5E1;line-height:1.8;white-space:pre-wrap;">
{active_scenario}
        </div>""", unsafe_allow_html=True)
else:
    st.info("💡 사이드바 [사회적 맥락] 에서 AI 자동 생성 또는 수동 입력 후 적용해 주세요.")


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
        st.error("API Key와 프로필 데이터를 확인해 주세요.")
    elif 'questions_structured' not in st.session_state:
        st.error("설문지 PDF를 업로드하고 문항을 추출해 주세요.")
    else:
        run_scenario = st.session_state.get('active_scenario', f"오늘은 {date.today()}입니다.")
        questions_structured = st.session_state['questions_structured']

        try:
            client = genai.Client(api_key=api_key)

            with st.status("⬡  AI 페르소나가 응답을 생성하고 있습니다…", expanded=True) as status:
                results_df = df.copy()
                all_responses: dict = {}

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            process_persona_structured,
                            client, model_choice,
                            row[selected_features].to_dict(),
                            run_scenario, persona_tmpl,
                            questions_structured, temp
                        ): idx
                        for idx, row in df.iterrows()
                    }
                    pbar = st.progress(0, text="응답 생성 중…")
                    for i, future in enumerate(as_completed(futures)):
                        idx = futures[future]
                        all_responses[idx] = future.result()
                        pbar.progress((i+1)/len(df), text=f"응답 생성 중… {i+1}/{len(df)}")

                # 결과 컬럼 추가
                for q in questions_structured:
                    results_df[q['id']] = [
                        all_responses[idx].get(q['id'], 'NA')
                        for idx in range(len(df))
                    ]

                st.session_state['results_df'] = results_df
                st.session_state['used_scenario'] = run_scenario
                status.update(
                    label=f"✓  완료  ·  {len(df)}개 케이스  ·  {len(questions_structured)}개 문항",
                    state="complete")

        except Exception as e:
            st.error(f"실행 오류: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if 'results_df' in st.session_state and 'questions_structured' in st.session_state:
    res_df = st.session_state['results_df']
    qs     = st.session_state['questions_structured']

    # 문항 타입 분류
    single_qs = [q for q in qs if q['type'] in ('single', 'multi')]
    open_qs   = [q for q in qs if q['type'] == 'open']
    demo_cols = [c for c in res_df.columns if c not in [q['id'] for q in qs]]
    q_cols    = [q['id'] for q in single_qs]

    # NA 제외 완성률
    na_total = sum((res_df[q['id']].isin(['NA','SKIP'])).sum() for q in qs)
    total_cells = len(res_df) * len(qs)
    completion_rate = round((1 - na_total / max(total_cells, 1)) * 100, 1)
    skip_count = sum((res_df[q['id']] == 'SKIP').sum() for q in qs)

    st.divider()
    st.markdown('<div class="section-label">03 &nbsp; Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">분석 결과</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, val, label in [
        (m1, f"{len(res_df):,}", "총 케이스"),
        (m2, str(len(qs)), "전체 문항"),
        (m3, str(len(single_qs)), "선택형"),
        (m4, str(len(open_qs)), "자유응답"),
        (m5, f"{completion_rate}%", "응답 완성률"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#0D1117;border:1px solid #1E2533;border-radius:10px;
                        padding:16px 12px;text-align:center;">
                <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:#38BDF8;">{val}</div>
                <div style="font-family:'Noto Sans KR',sans-serif;font-size:11px;color:#64748B;margin-top:4px;">{label}</div>
            </div>""", unsafe_allow_html=True)

    if skip_count > 0:
        st.markdown(f"""
        <div style="margin:12px 0;font-family:'DM Mono',monospace;font-size:11px;color:#A78BFA;">
            ⬡ 분기 처리로 SKIP된 응답: {skip_count:,}건
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["📈  빈도 분석 (선택형)", "💬  자유응답", "🔀  교차 분석", "💾  내보내기"])

    # ── 빈도 분석 ──────────────────────────────────────────────────────────────
    with t1:
        if not single_qs:
            st.info("선택형 문항이 없습니다.")
        for q in single_qs:
            # SKIP 제외 후 분석
            valid = res_df[q['id']][~res_df[q['id']].isin(['NA', 'SKIP'])]
            if valid.empty:
                continue

            # 보기 레이블 매핑
            opt_map = q.get('options', {})
            labeled = valid.map(lambda x: f"{x}. {opt_map.get(str(x), x)}" if opt_map else x)

            counts   = labeled.value_counts().sort_index()
            percents = (labeled.value_counts(normalize=True).sort_index() * 100).round(1)
            stats_df = pd.DataFrame({'사례수 (N)': counts, '비율 (%)': percents.apply(lambda x: f"{x:.1f}%")})

            branch_tag = ""
            if q.get("branch"):
                branch_tag = f" <span style='font-size:11px;color:#A78BFA;'>분기 있음</span>"

            skip_n = (res_df[q['id']] == 'SKIP').sum()
            skip_tag = f" <span style='font-size:11px;color:#64748B;'>SKIP {skip_n}건</span>" if skip_n else ""

            st.markdown(
                f'<div class="q-label">[{q["id"]}] {_html.escape(q["text"])[:60]}{"…" if len(_html.escape(q["text"]))>60 else ""}'
                f'{branch_tag}{skip_tag}</div>',
                unsafe_allow_html=True)

            c1, c2 = st.columns([1, 2])
            with c1:
                st.dataframe(stats_df, use_container_width=True)
                st_copy_to_clipboard(stats_df.to_csv(sep='\t'), "📋  Excel로 복사", f"copy_f_{q['id']}")
            with c2:
                st.bar_chart(counts, color=["#38BDF8"])
            st.divider()

    # ── 자유응답 ────────────────────────────────────────────────────────────────
    with t2:
        if not open_qs:
            st.info("자유응답 문항이 없습니다.")
        for q in open_qs:
            st.markdown(
                f'<div class="q-label open">[{q["id"]}] {_html.escape(q["text"])[:80]}{"…" if len(_html.escape(q["text"]))>80 else ""}</div>',
                unsafe_allow_html=True)

            open_df = res_df[[q['id']] + [c for c in demo_cols[:3]]].copy()
            open_df = open_df[~open_df[q['id']].isin(['NA', 'SKIP'])]
            open_df.columns = ['응답'] + list(open_df.columns[1:])
            st.dataframe(open_df.head(50), use_container_width=True)
            st_copy_to_clipboard(open_df.to_csv(sep='\t', index=False),
                                 "📋  자유응답 복사", f"copy_open_{q['id']}")
            st.divider()

    # ── 교차 분석 ────────────────────────────────────────────────────────────────
    with t3:
        c1, c2 = st.columns(2)
        with c1:
            r_var = st.selectbox("⬇️  행 변수 (인구통계)", options=demo_cols,
                                 index=None, placeholder="행 변수 선택", key="cross_row")
        with c2:
            col_var = st.selectbox("➡️  열 변수 (선택형 문항)", options=q_cols,
                                   placeholder="문항 선택", key="cross_col")
        if r_var and col_var:
            valid_mask = ~res_df[col_var].isin(['NA', 'SKIP'])
            cross_df = res_df[valid_mask].copy()
            tn, tp = st.tabs(["사례수 (N)", "비율 (%)"])
            with tn:
                ctab_n = pd.crosstab(cross_df[r_var], cross_df[col_var],
                                     margins=True, margins_name="전체")
                st.dataframe(ctab_n, use_container_width=True)
                st_copy_to_clipboard(ctab_n.to_csv(sep='\t'), "📋  사례수 복사", "copy_cn")
            with tp:
                ctab_pct = (pd.crosstab(cross_df[r_var], cross_df[col_var],
                                        normalize='index') * 100).round(1)
                st.dataframe(ctab_pct.map(lambda x: f"{x:.1f}%"), use_container_width=True)
                st_copy_to_clipboard(ctab_pct.to_csv(sep='\t'), "📋  비율 복사", "copy_cp")
                palette = ["#818CF8","#38BDF8","#34D399","#FB923C","#F472B6","#A78BFA","#60A5FA","#4ADE80"]
                n_cols = len(ctab_pct.columns)
                st.bar_chart(ctab_pct, color=(palette * ((n_cols // len(palette)) + 1))[:n_cols])
        else:
            st.info("변수를 선택하면 교차표가 생성됩니다.")

    # ── 내보내기 ────────────────────────────────────────────────────────────────
    with t4:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1E2533;border-radius:10px;padding:24px;margin-bottom:20px;">
            <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;color:#E2E8F0;margin-bottom:8px;">📦  결과 데이터셋 다운로드</div>
            <div style="font-family:'Noto Sans KR',sans-serif;font-size:13px;color:#64748B;line-height:1.6;">
                원본 프로필 + 선택형 응답 + 자유응답 + 문항 구조 + 사용된 사회적 맥락
            </div>
        </div>""", unsafe_allow_html=True)

        out = BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as wr:
            res_df.to_excel(wr, index=False, sheet_name="Results")

            # 자유응답 별도 시트
            if open_qs:
                open_cols = [q['id'] for q in open_qs]
                open_result = res_df[demo_cols[:5] + open_cols].copy()
                open_result.to_excel(wr, index=False, sheet_name="자유응답")

            # 문항 구조 시트
            q_meta = pd.DataFrame([{
                "문항ID": q['id'],
                "유형": q['type'],
                "질문": q['text'],
                "보기": str(q.get('options', {})),
                "분기": str(q.get('branch', {})),
            } for q in qs])
            q_meta.to_excel(wr, index=False, sheet_name="문항구조")

            # 시나리오 메타
            meta_df = pd.DataFrame({
                "항목": ["생성 일시", "사용된 사회적 맥락"],
                "내용": [str(date.today()), st.session_state.get('used_scenario', '')],
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