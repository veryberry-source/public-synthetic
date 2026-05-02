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

# --- 페이지 설정 ---
st.set_page_config(
    page_title="PersonaLab AI v4.0",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 커스텀 CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .stTextArea textarea { font-family: 'Pretendard', sans-serif; }
    .step-header {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* 복사 성공 알림 스타일 */
    #copy-notice {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        background-color: #28a745;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        display: none;
    }
    </style>
    <div id="copy-notice">클립보드에 복사되었습니다!</div>
    """, unsafe_allow_html=True)

# --- 클립보드 복사 자바스크립트 함수 ---
def st_copy_to_clipboard(text, button_label, key):
    """자바스크립트를 이용한 원클릭 클립보드 복사 버튼"""
    # 탭 구분자로 변환하여 엑셀 호환성 확보
    escaped_text = text.replace("`", "\\`").replace("$", "\\$")
    
    copy_js = f"""
    <script>
    function copyToClipboard() {{
        const text = `{escaped_text}`;
        navigator.clipboard.writeText(text).then(() => {{
            const notice = window.parent.document.getElementById('copy-notice');
            notice.style.display = 'block';
            setTimeout(() => {{ notice.style.display = 'none'; }}, 2000);
        }});
    }}
    </script>
    <button onclick="copyToClipboard()" style="
        width: 100%;
        height: 40px;
        background-color: #f0f2f6;
        border: 1px solid #d1d3d8;
        border-radius: 5px;
        cursor: pointer;
        font-weight: 600;
        color: #31333f;
        margin-top: 5px;
    ">
        {button_label}
    </button>
    """
    components.html(copy_js, height=50)

# --- 타이틀 및 헤더 ---
with st.container():
    st.markdown("""
    <div class="step-header">
        <h1 style='margin:0; color:#1a1a1a;'>🧪 PersonaLab AI <span style='font-size:16px; color:#666;'>v4.3 Professional</span></h1>
        <p style='margin:5px 0 0 0; color:#444;'>고도화된 언어 모델 기반의 페르소나 합성 데이터 생성 및 사회과학 분석 솔루션</p>
    </div>
    """, unsafe_allow_html=True)

# --- 사이드바: 설정 센터 ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("Control Center")
    
    with st.expander("🔑 API & Model Engine", expanded=True):
        api_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
        model_choice = st.selectbox("엔진 선택", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
        temp = st.slider("응답 창의성 (Temp)", 0.0, 1.0, 0.7)
        max_workers = st.slider("병렬 스레드 (Speed)", 1, 40, 20)

    with st.expander("🎭 페르소나 튜닝", expanded=True):
        persona_tmpl = st.text_area("행동 지침 (System Instruction)", value="당신은 아래 제공된 프로필 정보를 가진 실존 인물입니다.\n당신의 사회적 위치, 경제적 상황에 완벽히 빙의하여 설문에 임하세요.\n추상적인 정답이 아니라, 당신의 삶의 배경에서 나올 법한 '솔직하고 주관적인' 의견을 선호합니다.", height=120)
        
    with st.expander("🌐 사회적 맥락 (Scenario)", expanded=False):
        common_scenario = st.text_area("현재 시점의 배경/사건", value="오늘은 2026년 4월 27일입니다.\n최근 주요 사건\n-지방선거 선거구 획정;광역의회 비례대표 의석 확대;- 광주 4개 선거구 광역의원 중대 선거구제 최초 도입- 이재명, 인도·베트남 국빈 방문- 한-인도 정상회담- 장동혁, ‘방미 성과’ 기자회견- 한국은행 총재 신현송 취임- 한-베트남 정상회담 ‘10만 다문화가정, 사돈의 나라‘- 코스피 사흘 연속 상승, 최고 장중 6,557.76, 종가 6,475.81- 삼성전자 노조 ‘성과급 투쟁’- ‘정동영 구성 핵시설 발언’ 공방- 이란 전쟁 휴전 연장", height=120)

# --- 핵심 로직 함수 ---
def process_persona(client, model_choice, row_data, scenario, persona_tmpl, question_tmpl, num_questions, questions_block, temp):
    persona_ans = ["NA"] * num_questions
    try:
        profile_context = "\n".join([f"- {k}: {v}" for k, v in row_data.items()])
        sys_inst = f"{persona_tmpl}\n\n[현재 상황]\n{scenario}\n\n[상세 프로필]\n{profile_context}"
        user_prompt = question_tmpl.format(num_questions=num_questions, questions_block=questions_block)

        resp = client.models.generate_content(
            model=model_choice,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=sys_inst, temperature=temp)
        )
        nums = re.findall(r'\d+', resp.text)
        if len(nums) >= num_questions:
            persona_ans = nums[:num_questions]
        elif len(nums) > 0:
            for i, n in enumerate(nums): persona_ans[i] = n
    except: pass 
    return persona_ans

# --- 메인 워크플로우 ---
col_in1, col_in2 = st.columns([1, 1], gap="large")

with col_in1:
    st.markdown("### 📁 Step 1. 데이터 에셋 로드")
    uploaded_file = st.file_uploader("응답자 프로필 데이터 (CSV/XLSX)", type=["xlsx", "csv"], key="uploader")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df.columns = [str(c).lower().strip() for c in df.columns]

        for col in df.columns:
            if any(keyword in col for keyword in ['조사일', '날짜', 'date']):
                def safe_date_convert(x):
                    if pd.isna(x): return x
                    try:
                        if isinstance(x, (int, float)):
                            num_val = float(x)
                            if num_val > 1000:
                                return pd.to_datetime(num_val, unit='D', origin='1899-12-30').strftime('%Y-%m-%d')
                            return str(int(num_val))
                        return pd.to_datetime(x).strftime('%Y-%m-%d')
                    except: return str(x)
                df[col] = df[col].apply(safe_date_convert)
        st.success(f"데이터 로드 완료 ({len(df)} Cases)")
        
        selected_features = st.multiselect("🎯 핵심 분석 변수 선택", options=df.columns.tolist(), default=[c for c in df.columns if not any(kw in c for kw in ['id', 'no', 'uuid', '가중치'])], key="features")
        
        with st.expander("👀 데이터셋 미리보기"):
            st.dataframe(df[selected_features].head(5), use_container_width=True)

with col_in2:
    st.markdown("### 📝 Step 2. 설문 문항 설계")
    question_input = st.text_area("질문 리스트 입력", value="이번 지방선거에 투표하실건가요? (1. 반드시 할 것이다 2. 아마 할 것 같다 3. 아마 하지 않을 것 같다 4. 투표하지 않겠다 5. 모름)\n대구시장 선거에서 누가 당선되는 것이 조금이라도 좋다고 보십니까? (1. 더불어민주당 김부겸 2. 국민의힘 추경호 3. 모름)", height=200, key="q_in")
    
    question_tmpl_ui = st.text_area("답변 생성 포맷", value="""아래 {num_questions}개의 설문 문항에 대해 당신의 가치관에 근거하여 답변하세요.\n\n[수행 지침]\n1. 각 문항을 읽고, 당신의 프로필이라면 어떤 선택을 할지 스스로 짧게 추론하세요.\n2. 최종 결과는 오직 '숫자'만 순서대로 콤마(,)로 구분하여 한 줄로 출력하세요.\n3. 다른 설명은 절대 하지 마세요.\n\n문항 리스트:\n{questions_block}""", height=200, key="q_form")

# --- 시뮬레이션 버튼 ---
st.divider()
if st.button("🚀 지능형 시뮬레이션 시작", type="primary", key="run_sim"):
    if not api_key or 'df' not in locals():
        st.error("설정 오류: API Key와 데이터를 확인해 주세요.")
    else:
        try:
            client = genai.Client(api_key=api_key)
            questions = [q.strip() for q in question_input.split('\n') if q.strip()]
            q_block = "\n".join([f"Q{i+1}. {q}" for i, q in enumerate(questions)])
            
            with st.status("⚡ AI 페르소나가 응답을 생성하고 있습니다...", expanded=True) as status:
                results_df = df.copy()
                all_responses = {}
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(process_persona, client, model_choice, row[selected_features].to_dict(), common_scenario, persona_tmpl, question_tmpl_ui, len(questions), q_block, temp): idx for idx, row in df.iterrows()}
                    pbar = st.progress(0)
                    for i, future in enumerate(as_completed(futures)):
                        idx = futures[future]
                        all_responses[idx] = future.result()
                        pbar.progress((i + 1) / len(df))
                
                for i in range(len(questions)):
                    results_df[f'q_{i+1}'] = [all_responses[idx][i] for idx in range(len(df))]
                
                st.session_state['results_df'] = results_df
                st.session_state['questions'] = questions
                status.update(label="✅ 시뮬레이션 완료!", state="complete")
        except Exception as e:
            st.error(f"실행 오류: {e}")

# --- 분석 섹션 ---
if 'results_df' in st.session_state:
    res_df = st.session_state['results_df']
    q_list = st.session_state['questions']
    
    st.divider()
    st.markdown("### 📊 Analysis Dashboard")
    t1, t2, t3 = st.tabs(["📈 빈도 분석 (Freq)", "🔀 교차 분석 (Crosstabs)", "💾 내보내기"])
    
    with t1:
        st.subheader("단일 문항 빈도 분석")
        for i, q in enumerate(q_list):
            q_col = f'q_{i+1}'
            st.markdown(f"**Q{i+1}. {q}**")
            
            counts = res_df[q_col].value_counts().sort_index()
            percents = (res_df[q_col].value_counts(normalize=True).sort_index() * 100).round(1)
            
            stats_df = pd.DataFrame({
                '사례수(N)': counts,
                '비율(%)': percents.apply(lambda x: f"{x:.1f}%") # 소수점 한자리 + %
            })
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.table(stats_df)
                # 클립보드 복사 버튼 (TSV 형식)
                st_copy_to_clipboard(stats_df.to_csv(sep='\t'), "📋 테이블 복사 (Excel용)", f"copy_f_{i}")
            with c2:
                st.bar_chart(counts)
            st.divider()
                
    with t2:
        st.subheader("집단별 교차 분석")
        demo_cols = [c for c in res_df.columns if not c.startswith('q_')]
        q_cols = [c for c in res_df.columns if c.startswith('q_')]
        
        c1, c2 = st.columns(2)
        with c1: r_var = st.selectbox("⬇️ 행 변수 (인구통계 등)", options=demo_cols, index=None, placeholder="행 변수 선택", key="cross_row")
        with c2: col_var = st.selectbox("➡️ 열 변수 (설문 문항)", options=q_cols, placeholder="문항 선택", key="cross_col")
        
        if r_var and col_var:
            tn, tp = st.tabs(["사례수 (N)", "비율 (%)"])
            with tn:
                ctab_n = pd.crosstab(res_df[r_var], res_df[col_var], margins=True, margins_name="전체")
                st.dataframe(ctab_n, use_container_width=True)
                st_copy_to_clipboard(ctab_n.to_csv(sep='\t'), "📋 사례수 표 복사", "copy_cn")
            with tp:
                ctab_pct = (pd.crosstab(res_df[r_var], res_df[col_var], normalize='index') * 100).round(1)
                st.dataframe(ctab_pct.map(lambda x: f"{x:.1f}%"), use_container_width=True)
                st_copy_to_clipboard(ctab_pct.to_csv(sep='\t'), "📋 비율 표 복사", "copy_cp")
                st.bar_chart(ctab_pct)
        else:
            st.info("변수를 선택하면 교차표가 생성됩니다.")

    with t3:
        st.info("시뮬레이션 완료 데이터셋 다운로드")
        out = BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as wr:
            res_df.to_excel(wr, index=False)
        st.download_button("📥 Excel 결과 리포트 다운로드", out.getvalue(), "persona_lab_report.xlsx", key="dl_final")