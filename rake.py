import streamlit as st
import pandas as pd
from ipfn import ipfn
import io
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="Professional Raking Tool", layout="wide")

st.title("⚖️ 전문 설문조사 가중치 보정 (Raking) 도구")

# --- 세션 상태 초기화 ---
if 'df_res' not in st.session_state:
    st.session_state.df_res = None
if 'margin_configs' not in st.session_state:
    st.session_state.margin_configs = []

# 1. 파일 업로드 및 사이드바 설정
st.sidebar.header("🛠️ 환경 설정")
uploaded_file = st.sidebar.file_uploader("엑셀 파일(.xlsx) 업로드", type=["xlsx"])

if uploaded_file:
    @st.cache_data
    def load_data(file):
        return pd.read_excel(file)

    df_original = load_data(uploaded_file).copy()
    cols = df_original.columns.tolist()

    # 1-1. 사전 가중치 설정
    st.sidebar.subheader("1. 가중치 기초 설정")
    use_base_weight = st.sidebar.checkbox("사전 가중치(Design Weight) 사용")
    base_weight_col = None
    if use_base_weight:
        base_weight_col = st.sidebar.selectbox("기존 가중치 변수 선택", cols)

    # 1-2. 가중치 상한/하한 설정 (Trimming)
    st.sidebar.subheader("2. 가중치 절삭(Trimming)")
    use_trimming = st.sidebar.checkbox("가중치 상한 적용")
    max_weight_val = st.sidebar.number_input("가중치 상한 (Max)", value=5.0, min_value=1.0, step=0.1)

    # 2. 데이터 미리보기
    st.subheader("📋 데이터 확인")
    st.dataframe(df_original.head(5), use_container_width=True)

    # 3. 마진 그룹 설정
    st.divider()
    st.subheader("🎯 보정 기준(Margin) 설정")
    
    c1, c2 = st.columns([1, 5])
    if c1.button("➕ 마진 그룹 추가"):
        st.session_state.margin_configs.append({"id": len(st.session_state.margin_configs)})
    if c2.button("Sweep 초기화"):
        st.session_state.margin_configs = []
        st.session_state.df_res = None # 데이터도 함께 초기화
        st.rerun()

    dimensions = []
    aggregates = []
    margin_sums = [] 

    for i, config in enumerate(st.session_state.margin_configs):
        with st.expander(f"마진 그룹 {i+1}", expanded=True):
            selected_vars = st.multiselect(f"보정 변수 선택 (그룹 {i+1})", cols, key=f"vars_{i}")
            
            if selected_vars:
                has_nan = df_original[selected_vars].isnull().any().any()
                temp_df = df_original.copy()
                
                if has_nan:
                    st.warning(f"⚠️ 선택한 변수에 결측치가 포함되어 있습니다. '결측치' 카테고리로 변환됩니다.")
                    temp_df[selected_vars] = temp_df[selected_vars].fillna("결측치")
                
                temp_df[selected_vars] = temp_df[selected_vars].astype(str)
                current_counts = temp_df.groupby(selected_vars).size().reset_index(name='현재빈도')
                current_counts['목표빈도'] = current_counts['현재빈도'].astype(float)
                
                edited_df = st.data_editor(current_counts, key=f"editor_{i}", hide_index=True, use_container_width=True)
                
                current_sum = edited_df['현재빈도'].sum()
                target_sum = edited_df['목표빈도'].sum()
                
                sum_col1, sum_col2 = st.columns(2)
                sum_col1.info(f"현재 데이터 합계: **{current_sum:,.0f}**")
                if abs(current_sum - target_sum) > 0.1:
                    sum_col2.warning(f"입력된 목표 합계: **{target_sum:,.0f}**")
                else:
                    sum_col2.success(f"입력된 목표 합계: **{target_sum:,.0f}**")
                
                target_series = edited_df.set_index(selected_vars)['목표빈도']
                target_series.index = target_series.index.map(str)
                
                dimensions.append(selected_vars)
                aggregates.append(target_series)
                margin_sums.append(target_sum)

    # 4. 가중치 계산 실행
    st.divider()
    all_sums_match = len(set(margin_sums)) <= 1 if margin_sums else True

    if not all_sums_match:
        st.error("❌ 각 마진 그룹의 '목표빈도 합계'가 다릅니다.")
    
    if st.button("🚀 가중치 산출 시작", type="primary", use_container_width=True, disabled=not (dimensions and all_sums_match)):
        try:
            with st.spinner("알고리즘 실행 중..."):
                df_run = df_original.copy()
                for dim in dimensions:
                    for col in dim:
                        df_run[col] = df_run[col].fillna("결측치").astype(str)
                
                weight_col = 'final_weight'
                if use_base_weight and base_weight_col:
                    df_run[weight_col] = df_run[base_weight_col].astype(float)
                else:
                    df_run[weight_col] = 1.0

                IPF = ipfn.ipfn(df_run, aggregates, dimensions, weight_col=weight_col)
                df_res = IPF.iteration()

                if use_trimming:
                    df_res[weight_col] = df_res[weight_col].clip(lower=min_weight_val, upper=max_weight_val)
                    total_target = aggregates[0].sum()
                    df_res[weight_col] = df_res[weight_col] * (total_target / df_res[weight_col].sum())

                # 결과 세션에 저장
                st.session_state.df_res = df_res
                st.session_state.last_dimensions = dimensions
                st.session_state.last_aggregates = aggregates
                st.success("✅ 가중치 보정이 완료되었습니다!")

        except Exception as e:
            st.error(f"계산 중 오류 발생: {e}")

    # --- 5. 분석 리포트 섹션 (버튼 외부로 독립) ---
    # 세션 상태에 결과가 있을 때만 렌더링
    if st.session_state.df_res is not None:
        st.divider()
        st.subheader("📊 가중치 정밀 분석 리포트")
        
        df_res = st.session_state.df_res
        dims = st.session_state.last_dimensions
        aggs = st.session_state.last_aggregates
        weight_col = 'final_weight'
        
        n = len(df_res)
        w = df_res[weight_col]
        deff = n * np.sum(w**2) / (np.sum(w)**2)
        ess = n / deff

        m1, m2, m3 = st.columns(3)
        m1.metric("설계 효과 (Deff)", f"{deff:.3f}")
        m2.metric("유효 표본 수 (ESS)", f"{ess:.1f}")
        m3.metric("표본 효율성", f"{(ess/n)*100:.1f}%")

        st.write("#### 보정 정합성 확인")
        # 이제 이 selectbox를 바꿔도 앱이 rerun될 때 st.session_state.df_res가 존재하므로 리포트가 유지됩니다.
        sel_idx = st.selectbox("정합성 그래프 그룹 선택", range(len(dims)), 
                               format_func=lambda x: f"그룹 {x+1}: {', '.join(dims[x])}")

        if sel_idx is not None:
            t_vars = dims[sel_idx]
            t_data = aggs[sel_idx].reset_index()
            t_data.columns = t_vars + ['Target_N']
            
            res_agg = df_res.copy()
            for v in t_vars: res_agg[v] = res_agg[v].astype(str)
            
            post_c = res_agg.groupby(t_vars)[weight_col].sum().reset_index(name='Post_N')
            comp = pd.merge(t_data, post_c, on=t_vars, how='left')
            
            comp['Target_%'] = (comp['Target_N'] / comp['Target_N'].sum()) * 100
            comp['Post_%'] = (comp['Post_N'] / w.sum()) * 100

            labels = comp[t_vars].astype(str).agg('-'.join, axis=1)
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(x=labels, y=comp['Post_%'], name='보정 후(%)', marker_color='royalblue'))
            fig_comp.add_trace(go.Scatter(x=labels, y=comp['Target_%'], name='목표치(%)', mode='markers', marker=dict(color='red', size=12, symbol='circle-open')))
            fig_comp.update_layout(title="목표 대비 보정 결과(%)", barmode='group', height=400)
            st.plotly_chart(fig_comp, use_container_width=True)

        # 다운로드
        st.subheader("💾 결과 다운로드")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_res.to_excel(writer, index=False)
        st.download_button("📥 보정 데이터(.xlsx) 다운로드", output.getvalue(), "weighted_result.xlsx", use_container_width=True)

else:
    st.info("왼쪽 사이드바에서 엑셀 파일을 업로드해 주세요.")