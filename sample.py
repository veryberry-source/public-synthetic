import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# --- 페이지 설정 ---
st.set_page_config(page_title="간편 표본 추출 도구 v2", layout="wide")
st.title("🎯 간편 표본 추출 및 할당 도구 v2")
st.markdown("분류형/수치형 변수의 **다중 리코드(누적)**를 지원하며, 입력 지연 및 적용 버그를 완벽히 해결했습니다.")

# --- 0. 세션 상태 초기화 ---
if 'master_df' not in st.session_state: st.session_state['master_df'] = None
if 'processed_df' not in st.session_state: st.session_state['processed_df'] = None
if 'final_sample' not in st.session_state: st.session_state['final_sample'] = None
if 'excel_data' not in st.session_state: st.session_state['excel_data'] = None

# 엑셀 변환 캐싱 함수
@st.cache_data
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Filtered_Data")
    return output.getvalue()

# --- Step 1: 데이터 업로드 ---
st.header("1️⃣ 데이터 업로드")
uploaded_file = st.file_uploader("원본 마스터 데이터 (CSV/Excel)", type=["xlsx", "csv"])

if uploaded_file:
    if st.session_state.get('last_uploaded') != uploaded_file.name:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df.columns = [str(c).strip() for c in df.columns]
        # 초기 로드 시 문자열 변환 및 결측치 표준화
        st.session_state['master_df'] = df.replace(['nan', 'None', '<NA>', ''], np.nan)
        st.session_state['processed_df'] = st.session_state['master_df'].copy()
        st.session_state['last_uploaded'] = uploaded_file.name
        st.session_state['final_sample'] = None
        st.session_state['excel_data'] = None
        st.rerun()

proc_df = st.session_state['processed_df']

if proc_df is not None:
    # --- Step 2: 데이터 가공 ---
    st.divider()
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.header("2️⃣ 데이터 가공 (리코드 및 필터링)")
        st.info("💡 탭별로 [적용] 버튼을 누를 때마다 데이터에 계속 추가/반영됩니다.")
    with col_h2:
        if st.button("🔄 가공 내역 초기화 (원본 복구)", use_container_width=True):
            st.session_state['processed_df'] = st.session_state['master_df'].copy()
            st.rerun()

    tab1, tab2, tab3 = st.tabs(["🔠 분류형 리코드", "🔢 수치형 리코드", "🎯 필터링 (보완됨)"])

    # [탭 1] 분류형 리코드
    with tab1:
        cat_col = st.selectbox("가공할 분류형 변수 선택", proc_df.columns, key="cat_col", index=None, placeholder="변수를 선택하세요...")
        if cat_col:
            unique_vals = proc_df[cat_col].dropna().unique()
            map_df = pd.DataFrame({"기존_값": unique_vals, "변경할_값": unique_vals})
            st.write("표의 '변경할_값'을 수정하여 그룹핑하세요.")
            edited_map = st.data_editor(map_df, use_container_width=True, hide_index=True, key="map_editor")
            new_cat_name = st.text_input("생성될 새 변수명", value=f"{cat_col}_분류", key="new_cat_name")
            
            if st.button("✅ 분류형 리코드 추가", type="primary"):
                mapping_dict = dict(zip(edited_map['기존_값'], edited_map['변경할_값']))
                st.session_state['processed_df'][new_cat_name] = proc_df[cat_col].map(mapping_dict)
                st.success(f"'{new_cat_name}' 변수가 추가되었습니다!")
                st.rerun()

    # [탭 2] 수치형 리코드
    with tab2:
        num_col = st.selectbox("가공할 수치형 변수 선택", proc_df.columns, key="num_col", index=None, placeholder="변수를 선택하세요...")
        if num_col:
            c1, c2 = st.columns(2)
            with c1: bins_str = st.text_input("구분점 (예: 20, 30, 40)", "20, 30, 40, 50, 60")
            with c2: labels_str = st.text_input("라벨 (예: 20대, 30대)", "20대, 30대, 40대, 50대, 60대이상")
            new_num_name = st.text_input("생성될 새 변수명", value=f"{num_col}_구간", key="new_num_name")
            
            if st.button("✅ 수치형 구간화 추가", type="primary"):
                try:
                    bins = [-np.inf] + [float(x.strip()) for x in bins_str.split(',')] + [np.inf]
                    labels = [x.strip() for x in labels_str.split(',')]
                    numeric_series = pd.to_numeric(proc_df[num_col], errors='coerce')
                    st.session_state['processed_df'][new_num_name] = pd.cut(numeric_series, bins=bins, labels=labels)
                    st.success(f"'{new_num_name}' 변수가 추가되었습니다!")
                    st.rerun()
                except Exception as e:
                    st.error(f"오류: {e}")

    # [탭 3] 필터링 (보완된 버전)
    with tab3:
        filt_col = st.selectbox("필터링 기준 변수", proc_df.columns, key="filt_col", index=None, placeholder="변수를 선택하세요...")
        if filt_col:
            # 1. 고유값 추출 및 결측치 처리
            raw_unique = proc_df[filt_col].unique()
            has_nan = pd.isna(raw_unique).any()
            
            # 2. 리스트 정렬 및 결측치 문자열 표시 추가
            clean_unique = sorted([x for x in raw_unique if pd.notna(x)])
            options = clean_unique + (["결측치(NaN)"] if has_nan else [])
            
            # 3. multiselect (초기 세팅: 빈 리스트)
            selected_vals = st.multiselect(
                "보존할(남길) 값 선택 (미선택 시 모두 제거됨)", 
                options, 
                default=[], # 초기 세팅을 비워둠
                placeholder="남길 값을 선택하세요..."
            )
            
            if st.button("✅ 필터링 적용", type="primary"):
                if not selected_vals:
                    st.warning("선택된 값이 없어 데이터가 모두 삭제될 수 있습니다.")
                
                # 결측치 선택 여부에 따른 로직 분기
                if "결측치(NaN)" in selected_vals:
                    # '결측치(NaN)'를 제외한 나머지 실제 값들
                    actual_selected = [v for v in selected_vals if v != "결측치(NaN)"]
                    # 실제 값 포함 OR 결측치인 행 보존
                    mask = proc_df[filt_col].isin(actual_selected) | proc_df[filt_col].isna()
                else:
                    # 결측치 미선택 시 실제 선택된 값만 보존
                    mask = proc_df[filt_col].isin(selected_vals)
                
                st.session_state['processed_df'] = proc_df[mask]
                st.success(f"필터링 적용 완료! (현재 {len(st.session_state['processed_df'])}명)")
                st.rerun()

    # 현재 상태 출력
    st.write(f"📊 **현재 데이터 상태** (총 {len(proc_df)}명)")
    st.dataframe(proc_df.head(3), use_container_width=True)
   
    filtered_excel = convert_df_to_excel(proc_df)
    st.download_button(
        label=f"📥 1차 가공 완료 데이터 다운로드 ({len(proc_df)}명)",
        data=filtered_excel,
        file_name="Survey_Filtered_Data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- 3. 층화 추출 ---
    st.divider()
    st.header("3️⃣ 층화 추출 설정")
    s_vars = st.multiselect("층화 기준 변수 선택", st.session_state['processed_df'].columns)

    if s_vars:
        quota_base = st.session_state['processed_df'].groupby(s_vars, dropna=False).size().reset_index(name='모집단_N')
        quota_base['목표_추출수'] = 0
        
        # 데이터 에디터 출력
        final_quota = st.data_editor(quota_base, use_container_width=True, hide_index=True)

        if st.button("🚀 표본 추출 실행", type="primary", use_container_width=True):
            samples = []
            for _, row in final_quota.iterrows():
                target = int(row['목표_추출수'])
                if target > 0:
                    cond = True
                    for v in s_vars:
                        if pd.isna(row[v]): cond &= st.session_state['processed_df'][v].isna()
                        else: cond &= (st.session_state['processed_df'][v] == row[v])
                    cell_data = st.session_state['processed_df'][cond]
                    samples.append(cell_data.sample(n=min(len(cell_data), target), random_state=42))
            
            if samples:
                final_df = pd.concat(samples)
                st.session_state['final_sample'] = final_df
                out = BytesIO()
                with pd.ExcelWriter(out, engine='openpyxl') as wr:
                    final_df.to_excel(wr, index=False, sheet_name="Sampled")
                    final_quota.to_excel(wr, index=False, sheet_name="Quota_Summary")
                st.session_state['excel_data'] = out.getvalue()
                st.success("✅ 표본 추출 및 파일 생성 완료!")

    if st.session_state['excel_data'] is not None:
        st.divider()
        st.header("4️⃣ 결과 다운로드")
        st.download_button(
            label="📥 최종 표본 데이터 다운로드",
            data=st.session_state['excel_data'],
            file_name="Survey_Sample_Final.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )