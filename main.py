# JML
# "/Users/jinmyounglee/Desktop/SASA_Sophomore/Sophomore2NDSemester/Interactive Physics/2. Data Science"
import streamlit as st
import pyreadstat
from time import sleep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr, chi2_contingency
import plotly.express as px
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (
    LabelEncoder, OrdinalEncoder,
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

st.set_page_config(
    page_title="주관적 행복 인지에 미치는 요인 분석",
    page_icon="🍀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    /* 선택된 태그의 배경색 및 텍스트 색상 변경 */
    .stMultiSelect div[data-baseweb="tag"] {
        background-color: #f1f1f1 !important; /* 연한 회색 배경 */
        color: #000000 !important; /* 검정 텍스트 */
        border: 1px solid #cccccc !important; /* 회색 테두리 */
        border-radius: 0.5em; /* 둥근 모서리 */
        padding: 0.2em 0.5em;
    }
    
    /* 선택된 태그의 삭제 버튼(X 아이콘) 색상 변경 */
    .stMultiSelect div[data-baseweb="tag"] svg {
        fill: #000000 !important; /* 검정색 아이콘 */
    }

    /* 선택된 태그 호버 시 스타일 */
    .stMultiSelect div[data-baseweb="tag"]:hover {
        background-color: #e6e6e6 !important; /* 조금 더 어두운 회색 */
        border-color: #aaaaaa !important; /* 호버 시 테두리 색상 */
    }

    /* 다크 모 지원 */
    @media (prefers-color-scheme: dark) {
        .stMultiSelect div[data-baseweb="tag"] {
            background-color: #333333 !important; /* 다크 모드 배경 */
            color: #ffffff !important; /* 다크 모드 텍스트 */
            border: 1px solid #555555 !important; /* 다크 모드 테두리 */
        }
        .stMultiSelect div[data-baseweb="tag"]:hover {
            background-color: #444444 !important; /* 다크 모드 호버 배경 */
            border-color: #777777 !important;
        }
        .stMultiSelect div[data-baseweb="tag"] svg {
            fill: #ffffff !important; /* 다크 모드 아이콘 흰색 */
        }
    }
    </style>
    """, unsafe_allow_html=True)

plt.rcParams['font.family'] = 'Nanum Gothic'
plt.rcParams['axes.unicode_minus'] = False

components.html("""
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap" rel="stylesheet">
    <style>
    .gradient-text {
        font-family: 'Montserrat', sans-serif;
        font-size: 36px;
        font-weight: bold;
        background: linear-gradient(270deg, #ff7e5f, #feb47b, #86a8e7, #91eae4);
        background-size: 500% 500%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientAnimation 10s ease infinite;
    }
    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .developer-text {
        font-family: 'Montserrat', sans-serif;
        font-size: 16px;
        font-weight: normal;
        color: #555555;
        margin-top: 10px;
    }            
    </style>
    <div style='text-align: center;'>
        <h1 class="gradient-text">주관적 행복 인지에 미치는 요인 분석</h1>
        <p class="developer-text">JML</p>
    </div>
    """, height=120)

@st.cache_data
def load_data(file_path):
    df, meta = pyreadstat.read_sav(file_path)
    return df, meta

file_path = "kyrbs2023.sav"

try:
    df, meta = load_data(file_path)    
except FileNotFoundError:
    st.error(f"File Not Found: {file_path}")
    st.stop()
except Exception as e:
    st.error(f"Error Loading Data: {e}")
    st.stop()    

columns_X = ['PR_HT', 'M_SUI_CON', 'M_STR', 'M_SLP_EN', 'M_SAD', 'M_LON', 'M_GAD_1', 'M_GAD_2', 'M_GAD_3',
             'M_GAD_4', 'M_GAD_5', 'M_GAD_6', 'M_GAD_7', 'PA_MSC', 'PA_TOT', 'PA_VIG_D']
columns_Y = ['PR_HD']

column_mapping = {
    'PR_HT' : '주관적 건강 인지',
    'PR_HD' : '주관적 행복 인지',
    'M_STR': '평상시 스트레스 인지',  # 1 많이 ~ 5 전혀
    'M_SLP_EN': '잠으로 피로회복 정도',  # 1 충분 ~ 5 부족
    'M_SAD': '슬픔 절망감 경험',  # 1: 없음, 2: 있음
    'M_SUI_CON': '자살 생각',
    'M_LON': '외로움 경험',  # 1: 없음 ~ 5: 항상
    'M_GAD_1': '초조함 및 불안함',  # 1: 전혀 ~ 5: 일
    'M_GAD_2': '걱정 멈출 수 없음',
    'M_GAD_3': '걱정 너무 많음',
    'M_GAD_4': '편하게 있기 어려움',
    'M_GAD_5': '너무 안절부절 못함',
    'M_GAD_6': '쉽게 짜증이 남',
    'M_GAD_7': '끔찍한 일이 생길 것 같음',
    'PA_MSC': '근력 강화 운동 일수',
    'PA_TOT': '하루 60분 이상 신체활동 일수',
    'PA_VIG_D' : '고강도 신체활동 일수'
}

variable_type_mapping = {
    'PR_HT' : 'continuous',
    'PR_HD' : 'categorical', 
    'M_STR' : 'continuous', 
    'M_SLP_EN' : 'continuous', 
    'M_SAD' : 'categorical', 
    'M_SUI_CON' : 'categorical', 
    'M_LON' : 'continuous', 
    'M_GAD_1' : 'continuous', 
    'M_GAD_2' : 'continuous',
    'M_GAD_3' : 'continuous', 
    'M_GAD_4' : 'continuous', 
    'M_GAD_5' : 'continuous', 
    'M_GAD_6' : 'continuous', 
    'M_GAD_7' : 'continuous',
    'PA_TOT' : 'continuous',
    'PA_VIG_D' : 'continuous',
    'PA_MSC' : 'continuous'
}

tab1, tab2 = st.tabs(["📚 EDA", "🍸 ML"])

with tab1:
    # 결측치 및 이상치 처리 설정
    missing_value_options = [
        "No handling",
        "Drop rows with missing values",
        "Impute with mean",
        "Impute with median",
        "Impute with mode"
    ]

    outlier_handling_options = [
        "No outlier handling",
        "Remove outliers using Z-score",
        "Remove outliers using IQR"
    ]

    selected_missing_value_method = st.selectbox(
        "Select method for handling missing values:",
        missing_value_options,
        key="missing_value_method"
    )

    selected_outlier_method = st.selectbox(
        "Select method for handling outliers:",
        outlier_handling_options,
        key="outlier_handling_method"
    )

    # If Z-score method is selected, allow user to set Z-score threshold
    if selected_outlier_method == "Remove outliers using Z-score":
        z_threshold = st.number_input(
            "Set Z-score threshold:",
            min_value=2.0,
            value=3.0,
            step=0.1,
            key="z_score_threshold"
        )

    # If IQR method is selected, allow user to set IQR multiplier
    if selected_outlier_method == "Remove outliers using IQR":
        iqr_multiplier = st.number_input(
            "Set IQR multiplier:",
            min_value=0.0,
            value=1.5,  # 기본값 유지
            step=0.1,
            key="iqr_multiplier"
        )

    if st.button("Apply"):
        # df_selected는 이미 선택된 columns_X, columns_Y로 구성한 데이터프레임이라고 가정
        df_selected_cleaned = df[columns_X + columns_Y].copy()

        # 1. Handling Missing Values
        if selected_missing_value_method == "Drop rows with missing values":
            df_selected_cleaned = df_selected_cleaned.dropna()
            st.write("Dropped rows with missing values.")
        elif selected_missing_value_method == "Impute with mean":
            numeric_cols = df_selected_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            df_selected_cleaned[numeric_cols] = df_selected_cleaned[numeric_cols].fillna(df_selected_cleaned[numeric_cols].mean())
            st.write("Imputed missing values with mean.")
        elif selected_missing_value_method == "Impute with median":
            numeric_cols = df_selected_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            df_selected_cleaned[numeric_cols] = df_selected_cleaned[numeric_cols].fillna(df_selected_cleaned[numeric_cols].median())
            st.write("Imputed missing values with median.")
        elif selected_missing_value_method == "Impute with mode":
            # 범주형 변수에 대한 대치
            for col in df_selected_cleaned.columns:
                if df_selected_cleaned[col].isnull().sum() > 0:
                    mode_val = df_selected_cleaned[col].mode()
                    if len(mode_val) > 0:
                        df_selected_cleaned[col].fillna(mode_val[0], inplace=True)
            st.write("Imputed missing values with mode.")
        else:
            st.write("No missing value handling applied.")

        # 2. Handling Outliers
        if selected_outlier_method == "Remove outliers using Z-score":
            numeric_cols = df_selected_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                z_scores = np.abs(stats.zscore(df_selected_cleaned[numeric_cols]))
                outliers = (z_scores > z_threshold).any(axis=1)
                removed_rows = outliers.sum()
                df_selected_cleaned = df_selected_cleaned[~outliers]
                st.write(f"Removed {removed_rows} rows using Z-score method with threshold {z_threshold}.")
            else:
                st.write("No numeric columns available for Z-score outlier removal.")
        elif selected_outlier_method == "Remove outliers using IQR":
            numeric_cols = df_selected_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                Q1 = df_selected_cleaned[numeric_cols].quantile(0.25)
                Q3 = df_selected_cleaned[numeric_cols].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df_selected_cleaned[numeric_cols] < (Q1 - iqr_multiplier * IQR)) | 
                            (df_selected_cleaned[numeric_cols] > (Q3 + iqr_multiplier * IQR))).any(axis=1)
                removed_rows = outliers.sum()
                df_selected_cleaned = df_selected_cleaned[~outliers]
                st.write(f"Removed {removed_rows} rows using IQR method with multiplier {iqr_multiplier}.")
            else:
                st.write("No numeric columns available for IQR outlier removal.")
        else:
            st.write("No outlier handling applied.")

        st.write("**결측치 및 이상치 처리 완료**")
        st.write("Data Preview after cleaning:", df_selected_cleaned.head())

        # 히트맵 그리기
        numeric_cols = df_selected_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()

        if len(numeric_cols) < 2:
            st.write("상관관계를 계산할 충분한 수치형 변수가 없습니다.")
        else:
            df_numeric = df_selected_cleaned[numeric_cols]
            corr = df_numeric.corr(method='spearman').fillna(0)
            labels = [column_mapping.get(col, col) for col in corr.columns]

            fig_corr = px.imshow(
                corr.values,
                x=labels,
                y=labels,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1,
                width=600,
                height=600,
                text_auto=True
            )

            fig_corr.update_layout(
                xaxis=dict(tickfont=dict(size=10)),
                yaxis=dict(tickfont=dict(size=10)),
                title=dict(text='Overall HEATMAP (Ignoring the Variable Type)', font=dict(size=20)),
                autosize=False,
                margin=dict(l=100, r=100, t=100, b=100)
            )

            st.plotly_chart(fig_corr)

    # 선택 상자에 라벨 표시
    col1, col2 = st.columns(2)

    # 만약 columns_X, columns_Y가 이미 정의되어 있다면
    selected_X = columns_X
    selected_Y = columns_Y

    # 선택된 변수들(selected_X, selected_Y) 이후에 다음과 같이 추가:
    columns_X_labels = [column_mapping.get(col, col) for col in selected_X]
    columns_Y_labels = [column_mapping.get(col, col) for col in selected_Y]

    label_to_column = {label: col for col, label in zip(selected_X + selected_Y, columns_X_labels + columns_Y_labels)}

    # 이후에 x_axis_label, y_axis_label 선택박스 코드
    col1, col2 = st.columns(2)

    with col1:
        x_axis_label = st.selectbox("X축 변수 선택", columns_X_labels)
    with col2:
        y_axis_label = st.selectbox("Y축 변수 선택", columns_Y_labels)

    x_axis = label_to_column[x_axis_label]
    y_axis = label_to_column[y_axis_label]

    # Determine the types of the selected variables
    x_type = variable_type_mapping.get(x_axis, 'categorical')
    y_type = variable_type_mapping.get(y_axis, 'categorical')

    st.markdown(f"**선택한 변수 유형:**")
    st.markdown(f"- **X축 ({x_axis_label})**: **{x_type}**")
    st.markdown(f"- **Y축 ({y_axis_label})**: **{y_type}**")

    def cramers_v(confusion_matrix):
        """
        Calculate Cramér's V for a confusion matrix.

        Parameters:
        - confusion_matrix (pd.DataFrame): Cross-tabulation table.

        Returns:
        - float: Cramér's V value.
        """
        chi2, p, dof, expected = chi2_contingency(confusion_matrix)
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        # Adjust for bias
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

    # Visualization based on variable types
    if x_type == 'categorical' and y_type == 'categorical':
        st.subheader("Cross-Tabulation")
        # 가운데 정렬을 위한 컬럼 생성
        col_left, col_center, col_right = st.columns([1, 2, 1])

        with col_center:
            cross_tab = pd.crosstab(df[x_axis], df[y_axis])
            cross_tab = cross_tab.rename(index=column_mapping, columns=column_mapping)
            st.dataframe(cross_tab)

        # 카이제곱 검정 및 크래머의 V 계산
        cross_tab = pd.crosstab(df[x_axis], df[y_axis])
        chi2, p, dof, expected = chi2_contingency(cross_tab)
        cramers_v_value = cramers_v(cross_tab)

        # 결과 표시 with gradient styling in 2x2 grid
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div style="text-align: center;">
                <p style="
                    font-size: 24px;
                    font-weight: bold;
                    background: linear-gradient(90deg, #ff7f50, #1e90ff);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin: 0;
                ">
                    Chi-Squared Test: {chi2:.3f}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="text-align: center;">
                <p style="
                    font-size: 24px;
                    font-weight: bold;
                    background: linear-gradient(90deg, #1e90ff, #ff7f50);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin: 0;
                ">
                    Degree of Freedom: {dof}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="text-align: center;">
                <p style="
                    font-size: 24px;
                    font-weight: bold;
                    background: linear-gradient(90deg, #ff7f50, #1e90ff);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin: 0;
                ">
                    p-value: {p:.3e}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div style="text-align: center;">
                <p style="
                    font-size: 24px;
                    font-weight: bold;
                    background: linear-gradient(90deg, #1e90ff, #ff7f50);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin: 0;
                ">
                    Cramér's V: {cramers_v_value:.3f}
                </p>
            </div>
            """, unsafe_allow_html=True)

    elif x_type == 'continuous' and y_type == 'continuous':
        st.subheader("Heatmap")
        # Calculate correlation matrix
        corr_matrix = df[[x_axis, y_axis]].corr(method='spearman')
        # Plot heatmap
        fig = px.imshow(
            corr_matrix,
            x=[column_mapping.get(x_axis, x_axis), column_mapping.get(y_axis, y_axis)],
            y=[column_mapping.get(x_axis, x_axis), column_mapping.get(y_axis, y_axis)],
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            text_auto=True,
            width=600,
            height=600
        )
        st.plotly_chart(fig)

        # 계산된 Spearman 상관계 및 p-value
        x_data = df[x_axis]
        y_data = df[y_axis]

        # 데이터 타입이 수치형인지 확인
        if x_data.dtype.kind in 'biufc' and y_data.dtype.kind in 'biufc':
            corr_coef, p_value = spearmanr(x_data, y_data)

            st.markdown(f"""
             <div style="text-align: center;">
                <p style="
                    font-size: 24px;
                    font-weight: bold;
                    background: linear-gradient(90deg, #ff7f50, #1e90ff);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin: 0;
                ">
                    Correlation Coefficient: {corr_coef:.3f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.write("선택한 변수는 수치형 데이터가 아니어서 상관계수를 계산할 수 없습니다.")

    elif (x_type == 'categorical' and y_type == 'continuous') or (x_type == 'continuous' and y_type == 'categorical'):
        st.subheader("Bar Graph")

        # Determine which is categorical and which is continuous
        if x_type == 'categorical':
            cat_var = x_axis
            cat_label = x_axis_label
            cont_var = y_axis
            cont_label = y_axis_label
        else:
            cat_var = y_axis
            cat_label = y_axis_label
            cont_var = x_axis
            cont_label = x_axis_label

        # Aggregate data
        agg_df = df.groupby(cat_var)[cont_var].mean().reset_index()
        agg_df = agg_df.rename(columns={
            cat_var: cat_label,
            cont_var: f'평균 {cont_label}'
        })

        # Plot bar chart
        fig = px.bar(
            agg_df,
            x=cat_label,
            y=f'평균 {cont_label}',
            labels={f'평균 {cont_label}': f'평균 {cont_label}'},
            title=f'{cat_label}에 따른 평균 {cont_label}'
        )
        st.plotly_chart(fig)

    else:
        st.error("유효한 변수 유형 조합이 아닙니다.")

    st.write("")
    st.write("")

    st.write("## Overall")

    st.markdown("""
        <style>
        /* 슬라이더 트랙의 두께 조절 */
        div[data-baseweb="slider"] > div > div {
            height: 20px;
        }

        /* 슬라이더 값의 위치 조정 */
        div[data-baseweb="slider"] > div > div > div > div {
            top: -30px !important;
            font-size: 15px !important;
            color: #fff !important;
        }
        </style>
        """, unsafe_allow_html=True)

    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    if len(numeric_df.columns) < 2:
        st.write("상관관계 히트맵을 그릴 충분한 숫자형 변수가 없습니다.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            # 종속 변수 선택 (라벨로 표시)
            dependent_var_label = st.selectbox(
            "종속 변수 선택",
            [column_mapping.get(col, col) for col in numeric_df.columns if col != '주관적 행복 인지'],
            index=[column_mapping.get(col, col) for col in numeric_df.columns if col != '주관적 행복 인지'].index('주관적 행복 인지')
            )

        with col2:
            threshold = st.slider("상관 계수 임계값 설정", min_value=0.0, max_value=1.0, value=0.3, step=0.01)

        # 라벨을 컬럼 이름으로 변환
        dependent_var = label_to_column.get(dependent_var_label, dependent_var_label)

        correlations = numeric_df.corr()

        filtered_correlations = correlations[[dependent_var]][(correlations[dependent_var].abs() >= threshold)
                                                               & (correlations.index != dependent_var)]
    
        if filtered_correlations.empty:
            st.write(f"{dependent_var_label}와 상관 계수 {threshold} 이상인 변수가 없습니다.")
        else:
            # 필터링된 상 계수 데이터프레임 표시 (라벨로 표시)
            filtered_correlations['abs_corr'] = filtered_correlations[dependent_var].abs()
            filtered_correlations = filtered_correlations.sort_values(by='abs_corr', ascending=False).drop(columns=['abs_corr'])
    
            filtered_correlations.index = [column_mapping.get(col, col) for col in filtered_correlations.index]

            st.write(f"**{dependent_var_label}**와 상관 계수 **{threshold}** 이상인 변수:")
            st.dataframe(filtered_correlations, use_container_width=True)

    st.write("Data Preview", df.head())

    # 모든 수치형 변수 선택
    numeric_cols = numeric_df.columns.tolist()

    if len(numeric_cols) < 2:
        st.write("상관관계를 계산할 충분한 수치형 변수가 없습니다.")
    else:
        # 상관계수 계
        corr_matrix = numeric_df.corr().abs()  # 절댓값 용

        # 중복 및 자기 자신을 제외한 상각 행렬의 인덱 추출
        corr_pairs = (
            corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            .stack()
            .reset_index()
        )
        corr_pairs.columns = ['변수1', '변수2', '상관계수']

        # 상관계수가 0.4 이상인 변수 쌍 필터링
        threshold = 0.4
        strong_corr_pairs = corr_pairs[corr_pairs['상관계수'] >= threshold]

        if strong_corr_pairs.empty:
            st.write(f"상관계수가 {threshold} 이상인 변수 쌍이 없습니다.")
        else:
            # 변수 이름을 라벨로 변환
            strong_corr_pairs['변수1'] = strong_corr_pairs['변수1'].apply(lambda x: column_mapping.get(x, x))
            strong_corr_pairs['변수2'] = strong_corr_pairs['변수2'].apply(lambda x: column_mapping.get(x, x))

            # 상관계수로 내림차순 정렬
            strong_corr_pairs = strong_corr_pairs.sort_values(by='상관계수', ascending=False)

            st.write(f"상관계수가 {threshold} 이상인 변수 쌍")
            st.dataframe(strong_corr_pairs.reset_index(drop=True), use_container_width=True)

with tab2:
    st.write("⚠️ EDA에서 작업하였다면, 새로고침 이후 다시 접속하세요")
    # Initialize session state keys if they do not exist
    if 'selected_Y' not in st.session_state:
        st.session_state['selected_Y'] = []  # or set it to a default value if needed

    # 단계 상태를 추적하기 위한 초기 설정
    if 'step' not in st.session_state:
        st.session_state['step'] = 0

    # 스크롤 함수를 정의합니다.
    def scroll_to(section_id):
        st.components.v1.html(f"""
            <script>
                var element = document.getElementById('{section_id}');
                if (element) {{
                    element.scrollIntoView({{behavior: 'smooth'}});
                }}
            </script>
            """, height=0)

    # 0계: 변수 택
    st.markdown('<div id="variable_selection"></div>', unsafe_allow_html=True)
    st.subheader("📂 Variable Selection")

    # Combine columns_X and columns_Y to form the initial list of variables
    available_variables = [column_mapping.get(col, col) for col in list(column_mapping.keys())]

    group_1 = columns_X[:6]
    group_2 = columns_X[6:12]
    group_3 = columns_X[12:]

    colX1, colX2, colX3 = st.columns(3)

    with colX1:
        selected_X_labels_1 = st.multiselect(
            "**M 정신건강**",
            options=[column_mapping.get(col, col) for col in group_1],
            default=[column_mapping.get(col, col) for col in group_1]
        )

    with colX2:
        selected_X_labels_2 = st.multiselect(
            "**GAD-7 범불안장애 경험 조사도구**",
            options=[column_mapping.get(col, col) for col in group_2],
            default=[column_mapping.get(col, col) for col in group_2]
        )

    with colX3:
        selected_X_labels_3 = st.multiselect(
            "**PA 신체활동**",
            options=[column_mapping.get(col, col) for col in group_3],
            default=[column_mapping.get(col, col) for col in group_3]
        )

    # 위 세 그룹에서 선택된 X 변수 라벨을 하나의 리스트로 합친다.
    selected_X_labels = selected_X_labels_1 + selected_X_labels_2 + selected_X_labels_3
    st.markdown("""
        > **주관적 행복 인지?** \\
        > ⚠️ 주관적 행복 인지는 1(1, 2, 3)과 2(4, 5)로 이진 분류합니다\\
        > \\
        > 평상시 얼마나 행복하다고 생각합니까? 
        > > 1 매우 행복한 편이다\\
        > > 2 약간 행복한 편이다\\
        > > 3 보통이다\\
        > > 4 약간 불행한 편이다\\
        > > 5 매우 불행한 편이다
        """)
    selected_Y_labels = st.multiselect(
        "Select target variables (Y):",
        options=available_variables,
        default=[column_mapping.get(col, col) for col in columns_Y],
        disabled=True
    )

    # 역매핑: 한글 이름 원래 변수 이름으로 변환
    selected_X = [
        next((key for key, value in column_mapping.items() if value == label), label)
        for label in selected_X_labels
    ]
    selected_Y = [
        next((key for key, value in column_mapping.items() if value == label), label)
        for label in selected_Y_labels
    ]

    df[selected_Y[0]] = df[selected_Y[0]].apply(lambda x: 2 if x in [4, 5] else 1)        

    # 데이터프레임 서브셋 생성
    columns_to_use = list(set(selected_X + selected_Y))

    missing_columns = [col for col in columns_to_use if col not in df.columns]
    if missing_columns:
        st.error(f"The following columns are not in the dataset: {missing_columns}")
        st.stop()

    # 실제 데이터프레임 작업
    df_subset = df[columns_to_use]

    # 변수 선택 후 'Next' 버튼
    if st.button("Next", key="next_variable_selection"):
        if not selected_X or not selected_Y:
            st.error("Please select at least one feature and one target variable.")
        else:
            st.session_state['step'] = 1
            st.session_state['df'] = df_subset.copy()
            st.session_state['selected_X'] = selected_X
            st.session_state['selected_Y'] = selected_Y
            st.session_state['column_mapping'] = column_mapping
            st.session_state['variable_type_mapping'] = variable_type_mapping
            scroll_to('dataset_overview')

    if st.session_state.get('step', 0) >= 1:
        st.markdown('<div id="dataset_overview"></div>', unsafe_allow_html=True)
        st.subheader("🍿 Dataset Overview")
        df = st.session_state['df']
        st.write("Shape of the dataset:", df.shape)
        st.write(df.head())
        st.write(df.describe())

        if st.button("Next", key="next_dataset_overview"):
            st.session_state['step'] = 2
            scroll_to('handling_missing_values')

    if st.session_state.get('step', 0) >= 2:
        st.markdown('<div id="handling_missing_values"></div>', unsafe_allow_html=True)
        st.subheader("😶 Handling Missing Values and Outliers")

        st.markdown("""
        > **Z-Score?** \\
        > 데이터 값이 평균으로부터 얼마나 떨어져 있는지를 표준편차 단위로 나타낸 값입니다. \\
        > ⚠️ Z-Score Treshold는 3.00을 추천합니다. 너무 작을 경우 에러가 나거나 학습이 불안정합니다. \\
        > **IQR?** (Interquartile Range) \\
        > 데이터의 중앙 50%를 포함하는 범위로, Q3(3분위수)에서 Q1(1분위수)를 뺀 값입니다.
        """)

        missing_value_options = [
            "Drop rows with missing values",
            "Impute with mean",
            "Impute with median",
            "Impute with mode"
        ]

        outlier_handling_options = [
            "Remove outliers using Z-score",
            "No outlier handling",
            "Remove outliers using IQR"
        ]

        selected_missing_value_method = st.selectbox(
            "Select method for handling missing values:",
            missing_value_options
        )

        selected_outlier_method = st.selectbox(
            "Select method for handling outliers:",
            outlier_handling_options
        )

        if selected_outlier_method == "Remove outliers using Z-score":
            z_threshold = st.number_input(
                "Set Z-score threshold:",
                min_value=3.0,
                value=3.0,
                step=0.1
            )

        if selected_outlier_method == "Remove outliers using IQR":
            iqr_multiplier = st.number_input(
                "Set IQR multiplier:",
                min_value=0.0,
                value=1.5,
                step=0.1
            )

        if st.button("Apply", key="apply_missing_outliers"):
            df = st.session_state['df'].copy()  # 본 데이터 복사

            initial_shape = df.shape  # 초기 데이터 형태 저장

            # ### 1. Handling Missing Values ###
            if selected_missing_value_method == "Drop rows with missing values":
                df = df.dropna()
                st.write("Dropped rows with missing values.")
            elif selected_missing_value_method == "Impute with mean":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                st.write("Imputed missing values with mean.")
            elif selected_missing_value_method == "Impute with median":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                st.write("Imputed missing values with median.")
            elif selected_missing_value_method == "Impute with mode":
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                for col in categorical_cols:
                    if df[col].isnull().sum() > 0:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                st.write("Imputed missing values with mode.")

            after_missing_shape = df.shape  # 결측치 처리 후 데이터 형태 저장
            st.write(f"Rows before handling missing values: {initial_shape[0]}, after: {after_missing_shape[0]}")

            # ### 2. Handling Outliers ###
            rows_before_outliers = df.shape[0]

            if selected_outlier_method == "Remove outliers using Z-score":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    z_scores = np.abs(stats.zscore(df[numeric_cols]))
                    # Z-점수가 임계값을 초과하는 여부를 확인 (하나의 컬럼이라도 초과면 True)
                    outliers = (z_scores > z_threshold).any(axis=1)
                    removed_rows = outliers.sum()
                    df = df[~outliers]
                    st.write(f"Removed {removed_rows} rows using Z-score method with threshold {z_threshold}.")
                else:
                    st.write("No numeric columns available for Z-score outlier removal.")
            elif selected_outlier_method == "Remove outliers using IQR":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    Q1 = df[numeric_cols].quantile(0.25)
                    Q3 = df[numeric_cols].quantile(0.75)
                    IQR = Q3 - Q1
                    # IQR을 이용해 이상 여부 판단 (하나의 컬럼라도 이상치면 True)
                    outliers = ((df[numeric_cols] < (Q1 - iqr_multiplier * IQR)) | (df[numeric_cols] > (Q3 + iqr_multiplier * IQR))).any(axis=1)
                    removed_rows = outliers.sum()
                    df = df[~outliers]
                    st.write(f"Removed {removed_rows} rows using IQR method with multiplier {iqr_multiplier}.")
                else:
                    st.write("No numeric columns available for IQR outlier removal.")
            else:
                st.write("No outlier handling applied.")

            after_outliers_shape = df.shape  # 이상치 처리 후 데이터 형태 저장
            st.write(f"Rows before handling outliers: {rows_before_outliers}, after: {after_outliers_shape}")

            # ### 3. Summary ###
            total_removed = initial_shape[0] - after_missing_shape[0] + rows_before_outliers - after_outliers_shape[0]
            st.write(f"Total rows removed: {total_removed}")

            # Update the dataframe in session state
            st.session_state['df'] = df

            st.session_state['step'] = 3
            scroll_to('encoding_categorical_features')

    if st.session_state.get('step', 0) >= 3:
        st.markdown('<div id="encoding_categorical_features"></div>', unsafe_allow_html=True)
        st.subheader("🎲 Encoding Categorical Features")
        st.markdown('**⚠️ No Encoding Recommended**')

        st.markdown("""
        > **One-Hot Encoding?** \\
        > 범주형 변수를 고유한 값마다 이진 벡터로 변환하여 각 범주를 독립적인 열로 나타냅니다. \\
        > **Ordinal Encoding?**\\
        > 범주형 변수를 순서가 있는 정수 값으로 매핑하여 데이터의 순서 정보를 유지합니다.\\
        > **Label Encoding?** \\
        > 범주형 변수를 고유한 정수 값으로 매핑하여 간결하게 표현합니다. \\
        > **Count Frequency Encoding?** \\
        > 각 범주의 등장 횟수나 빈도를 기반으로 범주형 변수를 수치형 데이터로 변환합니다.
        """)
        # Options for encoding categorical variables, including "No Encoding"
        encoding_options = [
            "No Encoding",  # 추가된 옵션
            "One Hot Encoding",
            "Ordinal Encoding",
            "Label Encoding",
            "Count Frequency Encoding"
        ]

        # User selection
        selected_encoding_method = st.selectbox(
            "Select encoding method for categorical features:",
            encoding_options
        )

        # Apply button
        if st.button("Apply Encoding"):
            df = st.session_state['df']
            columns_to_use = list(df.columns)
            variable_type_mapping = st.session_state['variable_type_mapping']

            # Identify categorical columns among the selected variables
            categorical_cols = [col for col in columns_to_use if variable_type_mapping.get(col) == 'categorical']

            if selected_encoding_method == "No Encoding":
                st.write("Skipped encoding of categorical features.")
            elif selected_encoding_method == "One Hot Encoding":
                if categorical_cols:
                    # Apply One Hot Encoding
                    df = pd.get_dummies(df, columns=categorical_cols)
                    st.write("Applied One Hot Encoding.")

                    # Update selected_X with new column names
                    new_columns = df.columns.tolist()
                    original_selected_X = st.session_state['selected_X']
                    # Remove original categorical columns from selected_X and add new one-hot encoded columns
                    st.session_state['selected_X'] = [
                        col for col in new_columns 
                        if col in original_selected_X 
                        or any(col.startswith(f"{orig_col}_") for orig_col in categorical_cols if orig_col in original_selected_X)
                    ]
                else:
                    st.write("No categorical columns available for One Hot Encoding.")
            elif selected_encoding_method == "Ordinal Encoding":
                if categorical_cols:
                    # Apply Ordinal Encoding
                    ordinal_encoder = OrdinalEncoder()
                    df[categorical_cols] = ordinal_encoder.fit_transform(df[categorical_cols])
                    st.write("Applied Ordinal Encoding.")
                else:
                    st.write("No categorical columns available for Ordinal Encoding.")
            elif selected_encoding_method == "Label Encoding":
                if categorical_cols:
                    # Apply Label Encoding
                    label_encoders = {}
                    for col in categorical_cols:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col])
                        label_encoders[col] = le
                    st.write("Applied Label Encoding.")
                else:
                    st.write("No categorical columns available for Label Encoding.")
            elif selected_encoding_method == "Count Frequency Encoding":
                if categorical_cols:
                    # Apply Count Frequency Encoding
                    for col in categorical_cols:
                        freq_encoding = df[col].value_counts()
                        df[col] = df[col].map(freq_encoding)
                    st.write("Applied Count Frequency Encoding.")
                else:
                    st.write("No categorical columns available for Count Frequency Encoding.")
            else:
                st.error("Invalid encoding method selected.")
                st.stop()

            # Update the dataframe in session state
            st.session_state['df'] = df

            st.session_state['step'] = 4
            scroll_to('scaling_continuous_features')

    if st.session_state.get('step', 0) >= 4:
        st.markdown('<div id="scaling_continuous_features"></div>', unsafe_allow_html=True)
        st.subheader("🍫 Scaling Continuous Features")
        
        st.markdown("""
        > **Standard Scaling?** \\
        > 데이터의 평균을 0, 표준편차를 1로 조정하여 정규 분포 형태로 변환합니다. \\
        > **Min-Max Scaling?**\\
        > 데이터의 최솟값과 최댓값을 0과 1 사이로 변환하여 모든 값이 동일한 범위에 위치하도록 합니다.\\
        > **Robust Scaling?** \\
        > 데이터의 중앙값과 사분위수를 사용하여 이상치의 영향을 최소화하면서 스케일링합니다. \\
        > **Max Absolute Scaling?** \\
        > 데이터의 절댓값 중 최댓값을 기준으로 모든 값을 -1에서 1 사이로 변환합니다.
        """)

        # Options for scaling methods, including "No Scaling"
        scaling_options = [
            "Standard Scaling",
            "No Scaling",
            "Min-Max Scaling",
            "Robust Scaling",
            "Max Absolute Scaling"
        ]

        # User selection
        selected_scaling_method = st.selectbox(
            "Select scaling method for continuous features:",
            scaling_options
        )

        # Apply button
        if st.button("Apply Scaling"):
            df = st.session_state['df']
            columns_to_use = list(df.columns)
            variable_type_mapping = st.session_state['variable_type_mapping']

            # Identify continuous columns among the selected variables
            continuous_cols = [col for col in columns_to_use if variable_type_mapping.get(col) == 'continuous']

            if selected_scaling_method == "No Scaling":
                st.write("Skipped scaling of continuous features.")
            elif selected_scaling_method == "Min-Max Scaling":
                if continuous_cols:
                    # Apply Min-Max Scaling
                    scaler = MinMaxScaler()
                    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
                    st.write("Applied Min-Max Scaling.")
                else:
                    st.write("No continuous columns available for Min-Max Scaling.")
            elif selected_scaling_method == "Standard Scaling":
                if continuous_cols:
                    # Apply Standard Scaling
                    scaler = StandardScaler()
                    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
                    st.write("Applied Standard Scaling.")
                else:
                    st.write("No continuous columns available for Standard Scaling.")
            elif selected_scaling_method == "Robust Scaling":
                if continuous_cols:
                    # Apply Robust Scaling
                    scaler = RobustScaler()
                    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
                    st.write("Applied Robust Scaling.")
                else:
                    st.write("No continuous columns available for Robust Scaling.")
            elif selected_scaling_method == "Max Absolute Scaling":
                if continuous_cols:
                    # Apply Max Absolute Scaling
                    scaler = MaxAbsScaler()
                    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
                    st.write("Applied Max Absolute Scaling.")
                else:
                    st.write("No continuous columns available for Max Absolute Scaling.")
            else:
                st.error("Invalid scaling method selected.")
                st.stop()

            # Update the dataframe in session state
            st.session_state['df'] = df

            st.session_state['step'] = 5
            scroll_to('dataset_splitting')

    if st.session_state.get('step', 0) >= 5:
        st.markdown('<div id="dataset_splitting"></div>', unsafe_allow_html=True)
        st.subheader("🎰 Dataset Splitting")

        # Options for splitting
        test_size = st.slider(
            "Select test set size (as a fraction):",
            min_value=0.1, max_value=0.5, value=0.2, step=0.05
        )
        random_state = st.number_input(
            "Set random state for reproducibility:",
            min_value=0, value=42, step=1
        )

        # Apply button
        if st.button("Apply Splitting"):
            df = st.session_state['df']
            selected_X = st.session_state['selected_X']
            selected_Y = st.session_state['selected_Y']

            # Extract features and target variables
            X = df[selected_X]

            # Ensure selected_Y is valid
            if not selected_Y or selected_Y[0] not in df.columns:
                st.error("The selected target variable is not valid or does not exist in the DataFrame.")
                st.stop()  # Stop further execution if the condition is not met

            y = df[selected_Y[0]]  # Assuming single target variable

            # Debugging: Check original values in y
            st.write("Original values in target variable:")
            st.write(y.value_counts())  # Display counts of each class

            # No conversion needed for the target variable

            # Debugging: Check unique values in transformed y
            unique_classes = np.unique(y)
            st.write(f"Unique classes in target variable after transformation: {unique_classes}")  # Debugging statement

            # Check if there are at least 2 classes
            if len(unique_classes) < 2:
                st.error("The target variable must contain at least two classes for classification.")
                st.stop()  # Stop further execution if the condition is not met
            else:
                # Perform the split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

                # Check the distribution of the target variable in the training set
                st.write("Training set target variable distribution:")
                st.write(pd.Series(y_train).value_counts())  # Display counts of each class

                # Store the splits in session state for later use
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test

                st.write("Training set shape:", X_train.shape)
                st.write("Testing set shape:", X_test.shape)

                st.session_state['step'] = 6
                scroll_to('building_model')

    if st.session_state.get('step', 0) >= 6:
        st.markdown('<div id="building_model"></div>', unsafe_allow_html=True)
        st.subheader("👾 Building Model")
        
        st.markdown("""
        > **Logistic Regression?** \\
        > 데이터를 직선으로 구분할 수 있는 선형 분류 모델로, 주어진 입력에 대해 특정 클래스에 속할 확률을 계산합니다. \\
        > **Decision Tree?**\\
        > 데이터를 조건에 따라 분리하며 의사결정 경로를 생성하는 비선형 분류 모델로, 데이터 분포와 구조를 이해하기 쉽습니다.\\
        > **Random Forest?** \\
        > 여러 개의 결정 트리를 학습하여 예측을 종합하는 앙상블 모델로, 과적합을 방지하고 예측 성능을 향상시킵니다. \\
        > **Support Vector Classifier?** \\
        > 데이터를 최대 폭의 경계로 분리하는 초평면을 찾는 비선형 분류 모델로, 고차원 데이터에서도 효과적입니다.
        """)

        # 세션 상태에서 변수 가져오기
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        selected_Y = st.session_state['selected_Y']
        column_mapping = st.session_state['column_mapping']

        # 모델 유형 선택 옵션 제공 (현재 Classification만 사용)
        model_type = st.selectbox("Select Model Type:", ['Classification'])

        # 모델 선택 옵션 추가
        selected_model = st.selectbox("Select Model:", ['Random Forest Classifier', 'Logistic Regression', 'Decision Tree Classifier', 'Support Vector Classifier'])

        # 하이퍼파라미터 튜닝 선택 버튼 추가
        hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning - ⏰ 3min")

        # 하이퍼파라미터 튜닝이 활성화된 경우
        if hyperparameter_tuning:
            st.markdown("""
            > **Halving Grid Search?** \\
            > 전체 매개변수 조합 중 일부를 점진적으로 선택하여 리소스를 효율적으로 사용하면서 최적의 매개변수를 찾는 방법입니다. \\
            > **Grid Search?**\\
            > 모든 매개변수 조합을 체계적으로 탐색하여 최적의 매개변수를 찾는 방법으로, 계산 시간이 많이 소요됩니다.\\
            > **Randomized Search?** \\
            > 매개변수 공간에서 랜덤하게 일부 조합을 선택하여 효율적으로 최적의 매개변수를 탐색하는 방법입니다.
            """)
            tuning_method = st.selectbox("Select Hyperparameter Tuning Method:", ["Halving Grid Search", "Grid Search", "Randomized Search"])
            k_folds = st.number_input("Select number of folds for cross-validation (k):", min_value=2, value=5, step=1)
        else:
            # 하이퍼파라미터 튜닝 비활성화 시, 사용자가 직접 하이퍼파라미터 조정
            st.write("Manually set hyperparameters:")

            if selected_model == 'Logistic Regression':
                C_value = st.number_input("C (Inverse of regularization strength):", min_value=0.0001, value=1.0, step=0.1)
                penalty = st.selectbox("Penalty:", ['l1', 'l2'])
                solver = st.selectbox("Solver:", ['liblinear', 'saga'])
            elif selected_model == 'Decision Tree Classifier':
                max_depth = st.selectbox("Max Depth:", [None, 5, 10])
                min_samples_split = st.selectbox("Min Samples Split:", [2, 5, 10])
                min_samples_leaf = st.selectbox("Min Samples Leaf:", [1, 2, 4])
            elif selected_model == 'Random Forest Classifier':
                n_estimators = st.selectbox("Number of Estimators:", [50, 100, 200])
                max_depth = st.selectbox("Max Depth:", [None, 5, 10])
                min_samples_split = st.selectbox("Min Samples Split:", [2, 5, 10])
                min_samples_leaf = st.selectbox("Min Samples Leaf:", [1, 2, 4])
            elif selected_model == 'Support Vector Classifier':
                C_value = st.selectbox("C (Regularization parameter):", [0.1, 1, 10])
                kernel = st.selectbox("Kernel:", ['linear', 'rbf', 'poly'])
                gamma = st.selectbox("Gamma:", ['scale', 'auto'])

        # Train Model 버튼 추가
        if st.button("Train Model"):
            st.write("Training model...")  # Debugging statement

            # 튜닝용 파라미터 그리드
            if selected_model == 'Logistic Regression':
                model = LogisticRegression(max_iter=1000)
                param_grid = {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            elif selected_model == 'Decision Tree Classifier':
                model = DecisionTreeClassifier(random_state=42)
                param_grid = {
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif selected_model == 'Random Forest Classifier':
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif selected_model == 'Support Vector Classifier':
                model = SVC(probability=True)
                param_grid = {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                }

            # 진행 상황 표시 바 초기화
            progress_bar = st.progress(0)
            status_text = st.empty()  # 상태 표시 텍스트    

            if hyperparameter_tuning:
                st.write("Hyperparameter tuning in progress...")

                if tuning_method == "Grid Search":
                    search_cv = GridSearchCV(model, param_grid, cv=k_folds, n_jobs=-1, scoring='f1_macro')
                elif tuning_method == "Randomized Search":
                    search_cv = RandomizedSearchCV(model, param_grid, cv=k_folds, n_jobs=-1, scoring='f1_macro', n_iter=10, random_state=42)
                elif tuning_method == "Halving Grid Search":
                    search_cv = HalvingGridSearchCV(model, param_grid, cv=k_folds, scoring='f1_macro', factor=2, random_state=42)

                for percent_complete in range(0, 101, 10):  # 진행 상황 업데이트 (가상 시뮬레이션)
                    progress_bar.progress(percent_complete / 100)
                    status_text.write(f"Preparing... {percent_complete}%")
                    sleep(0.1)  # 시뮬레이션을 위해 딜레이

                search_cv.fit(X_train, y_train)
                best_model = search_cv.best_estimator_
                st.write("Best Parameters:", search_cv.best_params_)
                model = best_model
            else:
                # 하이퍼파라미터 튜닝 비활성화 시, 사용자가 설정한 값으로 모델 설정
                if selected_model == 'Logistic Regression':
                    model = LogisticRegression(C=C_value, penalty=penalty, solver=solver, max_iter=1000)
                elif selected_model == 'Decision Tree Classifier':
                    model = DecisionTreeClassifier(random_state=42, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                elif selected_model == 'Random Forest Classifier':
                    model = RandomForestClassifier(random_state=42, n_estimators=n_estimators, max_depth=max_depth, 
                                                   min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                elif selected_model == 'Support Vector Classifier':
                    model = SVC(probability=True, C=C_value, kernel=kernel, gamma=gamma)

                # 모델 학습 진행 상황 표시 (가상 시뮬레이션)
                for percent_complete in range(0, 101, 20):
                    progress_bar.progress(percent_complete / 100)
                    status_text.write(f"Training Progress: {percent_complete}%")
                    sleep(0.2)  # 시뮬레이션을 위해 딜레이

                model.fit(X_train, y_train)

            # 학습 완료 후 진행 상황 바와 상태 텍스트 업데이트
            progress_bar.progress(1.0)
            status_text.write("Training Complete!")

            st.session_state['model'] = model  # Store the trained model in session state
            st.success(f"{selected_model} model trained successfully.")

    if st.session_state.get('step', 0) >= 6:
        st.markdown('<div id="visualization_evaluation"></div>', unsafe_allow_html=True)
        st.subheader("🎨 Visualization & Evaluation")

        if 'model' in st.session_state:
            # 모델 예측
            y_pred = st.session_state['model'].predict(st.session_state['X_test'])

            # Check if predictions are valid
            if y_pred is not None and len(y_pred) > 0:
                # 평가 지표 계산
                accuracy = accuracy_score(st.session_state['y_test'], y_pred)
                f1_macro = f1_score(st.session_state['y_test'], y_pred, average='macro')

                # Feature Importance 계산
                model = st.session_state['model']
                X_train = st.session_state['X_train']  # X_train 불러오기
                selected_features = X_train.columns
                df_imp = None

                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    df_imp = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': importances
                    })
                elif hasattr(model, 'coef_'):
                    coefs = model.coef_
                    if coefs.ndim > 1:
                        coefs = np.mean(np.abs(coefs), axis=0)
                    else:
                        coefs = np.abs(coefs.flatten())
                    df_imp = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': coefs
                    })
                else:
                    # feature importance를 제공하지 않는 모델
                    df_imp = pd.DataFrame({
                        'Feature': [],
                        'Importance': []
                    })
                    st.write("This model does not provide feature importances. Top features not available.")

                if not df_imp.empty:
                    df_imp['Feature_Label'] = df_imp['Feature'].apply(lambda f: column_mapping.get(f, f))
                    df_imp = df_imp.sort_values('Importance', ascending=False)

                gradient_style = """
                <style>
                .gradient-text {
                    font-size: 25px;
                    font-weight: bold;
                    background: linear-gradient(90deg, #ff7e5f, #feb47b, #86a8e7, #91eae4);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    text-align: center;
                    margin: 0;
                    padding: 0;
                }
                </style>
                """

                st.markdown(gradient_style, unsafe_allow_html=True)

                # 1행: Accuracy, F1
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"<p class='gradient-text'>Model Accuracy: {accuracy:.2f}</p>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<p class='gradient-text'>Macro Average F1-Score: {f1_macro:.2f}</p>", unsafe_allow_html=True)

                st.markdown('\n')

                # 2행: Feature Importances 상위 3개 (만약 df_imp가 비어있지 않다면)
                if not df_imp.empty:
                    top_3_features = df_imp['Feature_Label'].head(3).tolist()
                    top_features_text = " & ".join(top_3_features)
                    st.markdown(f"<p class='gradient-text'>{top_features_text}</p>", unsafe_allow_html=True)

                st.markdown('\n')

                # 특성 중요도 시각화 추가
                if not df_imp.empty:
                    fig_importance = px.bar(
                        df_imp.head(20),  # 상위 20개 특성 표시 (필요에 따라 조정 가능)
                        x='Importance',
                        y='Feature_Label',
                        orientation='h',
                        title='Feature Importance',
                        labels={'Importance': 'Importance Score', 'Feature_Label': 'Feature'},
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    fig_importance.update_layout(
                        yaxis={'categoryorder': 'total ascending'},  # 중요도에 따라 정렬
                        margin=dict(l=100, r=20, t=50, b=20)
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)

                # 혼동 행렬 시각화
                unique_labels = np.unique(st.session_state['y_test'])
                conf_matrix = confusion_matrix(st.session_state['y_test'], y_pred, labels=unique_labels)
                fig = px.imshow(conf_matrix,
                                labels=dict(x="Predicted", y="True", color="Count"),
                                x=[f'Predicted {cls}' for cls in unique_labels],
                                y=[f'True {cls}' for cls in unique_labels],
                                text_auto=True,
                                color_continuous_scale='Blues')
                fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='True')
                st.plotly_chart(fig)

                # 분류 리포트
                report = classification_report(st.session_state['y_test'], y_pred, output_dict=True)
                st.write("**Classification Report:**")
                st.dataframe(pd.DataFrame(report).transpose())

                # 실제값 vs 예측값 비교 그래프
                st.write('\n')
                st.write("**Actual vs Predicted Counts**")
                actual_counts = pd.Series(st.session_state['y_test'], name='Actual').value_counts().sort_index()
                predicted_counts = pd.Series(y_pred, name='Predicted').value_counts().sort_index()

                all_labels = np.unique(np.concatenate((unique_labels, np.unique(y_pred))))
                df_compare = pd.DataFrame({
                    'Class': all_labels,
                    'Actual': [actual_counts[cls] if cls in actual_counts.index else 0 for cls in all_labels],
                    'Predicted': [predicted_counts[cls] if cls in predicted_counts.index else 0 for cls in all_labels]
                })

                fig_compare = px.bar(df_compare, 
                                     x='Class', 
                                     y=['Actual', 'Predicted'], 
                                     barmode='group', 
                                     title=False,
                                     labels={'value': 'Count', 'Class':'Class'},
                                     text_auto=True,
                                     color_discrete_sequence=px.colors.qualitative.Pastel)

                fig_compare.update_layout(
                    # title_font_size=20,
                    # xaxis_title_font_size=16,
                    # yaxis_title_font_size=16,
                    legend_title_text='',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig_compare, use_container_width=True)

            else:
                st.error("No predictions were made. Please check the model training.")
        else:
            st.error("Model has not been trained yet. Please train the model first.")