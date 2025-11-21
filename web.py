# web.py
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib, pickle
import shap
import matplotlib
import matplotlib.pyplot as plt

# 兼容 numpy 旧别名
if not hasattr(np, 'bool'):
    np.bool = bool

# ============== 字体/中文显示 ==================
def setup_chinese_font():
    """设置中文字体（优先系统字体，其次 ./fonts 目录）"""
    try:
        import matplotlib.font_manager as fm
        chinese_fonts = [
            'WenQuanYi Zen Hei','WenQuanYi Micro Hei','SimHei','Microsoft YaHei',
            'PingFang SC','Hiragino Sans GB','Noto Sans CJK SC','Source Han Sans SC'
        ]
        available = [f.name for f in fm.fontManager.ttflist]
        for f in chinese_fonts:
            if f in available:
                matplotlib.rcParams['font.sans-serif'] = [f, 'DejaVu Sans', 'Arial']
                matplotlib.rcParams['font.family'] = 'sans-serif'
                return f

        # 尝试加载 ./fonts 下自带字体
        fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        candidates = [
            'NotoSansSC-Regular.otf','NotoSansCJKsc-Regular.otf',
            'SourceHanSansSC-Regular.otf','SimHei.ttf','MicrosoftYaHei.ttf'
        ]
        if os.path.isdir(fonts_dir):
            import matplotlib.font_manager as fm
            for fname in candidates:
                fpath = os.path.join(fonts_dir, fname)
                if os.path.exists(fpath):
                    fm.fontManager.addfont(fpath)
                    fam = fm.FontProperties(fname=fpath).get_name()
                    matplotlib.rcParams['font.sans-serif'] = [fam, 'DejaVu Sans', 'Arial']
                    matplotlib.rcParams['font.family'] = 'sans-serif'
                    return fam
    except Exception:
        pass


    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    return None

chinese_font = setup_chinese_font()
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = matplotlib.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============== 页面配置 ==================
st.set_page_config(
    page_title="Depression prediction in patients with chronic digestive system diseases",
    page_icon="🏥",
    layout="wide"
)

# ============== Feature names & display labels ==================
feature_names_display = [
    'Sleep_night',
    'Grip_strength',
    'Education',
    'Residence',
    'Life_assessment',
    'Health_assessment',
    'Falldown',
    'Disability',
    'Kidney_disease',
    'Arthritis',
    'Heart_disease',
    'Eyesight',
    'IADL'
]
feature_names_cn = [
    'Night sleep duration',
    'Grip strength',
    'Education level',
    'Residence',
    'Life satisfaction',
    'Self-rated health',
    'History of falls',
    'Disability',
    'Kidney disease',
    'Arthritis',
    'Heart disease',
    'Eyesight',
    'IADL function'
]
feature_dict = dict(zip(feature_names_display, feature_names_cn))

variable_descriptions = {
    'Sleep_night': 'Sleeping time at night (hours)',
    'Grip_strength': 'Average of the maximum grip strength of left and right hands (kg)',
    'Education': 'Highest education level (1: Lower than high school; 2: High school or above)',
    'Residence': 'Residence (0: Urban; 1: Rural)',
    'Life_assessment': 'Life satisfaction (1: Not satisfied; 2: Satisfied)',
    'Health_assessment': 'Self-rated health (1: Poor; 2: Good)',
    'Falldown': 'Any previous history of falls (0: No; 1: Yes)',
    'Disability': 'Disability status (0: No; 1: Yes)',
    'Kidney_disease': 'Kidney disease (0: No; 1: Yes)',
    'Arthritis': 'Arthritis (0: No; 1: Yes)',
    'Heart_disease': 'Heart disease (0: No; 1: Yes)',
    'Eyesight': 'Eyesight (1: Poor; 2: Good)',
    'IADL': 'Independent in all IADL tasks (0: No; 1: Yes)'
}

# ============== 工具函数 ==================
def _clean_number(x):
    """把 '[3.3101046E-1]'、'3,210'、' 12. ' 等转成 float；失败返回 NaN"""
    if isinstance(x, str):
        s = x.strip().strip('[](){}').replace(',', '')
        try:
            return float(s)
        except Exception:
            return np.nan
    return x

@st.cache_resource
def load_model(model_path: str = './xgb_model.pkl'):
    """加载 xgboost 模型，兼容旧版训练产物：补 use_label_encoder / gpu_id / n_gpus / predictor 等缺失属性"""
    try:
        try:
            model = joblib.load(model_path)
        except Exception:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

        # 兼容补丁：老版本 XGBoost 训练的模型里常见的已废弃/迁移属性
        try:
            if hasattr(model, "__class__") and model.__class__.__name__.startswith("XGB"):
                # 这些属性的存在只为避免 get_params() getattr 报错；值不影响 1.7.6 推理
                defaults = {
                    "use_label_encoder": False,   # 1.x 时代参数，2.x 已废弃
                    "gpu_id": 0,                  # 老版本 GPU 选择；1.7.6 不再需要
                    "n_gpus": 1,                  # 有些旧代码保存过这个
                    "predictor": None,            # 旧参数：cpu_predictor/gpu_predictor
                    "tree_method": getattr(model, "tree_method", None),
                }
                for k, v in defaults.items():
                    if not hasattr(model, k):
                        setattr(model, k, v)
        except Exception:
            pass

        
        model_feature_names = None
        try:
            if hasattr(model, 'feature_names_in_'):
                model_feature_names = list(model.feature_names_in_)
        except Exception:
            pass
        if model_feature_names is None:
            try:
                booster = getattr(model, 'get_booster', lambda: None)()
                if booster is not None and hasattr(booster, 'feature_names'):
                    model_feature_names = list(booster.feature_names)
            except Exception:
                model_feature_names = None

        return model, model_feature_names
    except Exception as e:
        raise RuntimeError(f"无法加载模型: {e}")

def predict_proba_safe(model, X_df):
    try:
        return model.predict_proba(X_df)
    except AttributeError:
        for k, v in {"use_label_encoder": False, "gpu_id": 0, "n_gpus": 1, "predictor": None}.items():
            if not hasattr(model, k):
                setattr(model, k, v)
        return model.predict_proba(X_df)
    except Exception:
        import xgboost as xgb
        booster = getattr(model, "get_booster", lambda: None)()
        if booster is None:
            raise
        dm = xgb.DMatrix(X_df.values, feature_names=list(X_df.columns))
        pred = booster.predict(dm, output_margin=False)
        if isinstance(pred, np.ndarray):
            if pred.ndim == 1:  
                proba_pos = pred.astype(float)
                return np.vstack([1 - proba_pos, proba_pos]).T
            elif pred.ndim == 2:
                return pred.astype(float)
        raise RuntimeError("Booster fallback failed: unknown output shape")

# ============== 主逻辑 ==================
def main():
    # Sidebar
    st.sidebar.title("Depression prediction in patients with chronic digestive system diseases")
    st.sidebar.image(
        "https://img.freepik.com/free-vector/hospital-logo-design-vector-medical-cross_53876-136743.jpg",
        width=200
    )
    st.sidebar.markdown("""
    ### About
    This calculator uses an XGBoost model to predict depression risk
    in patients with chronic digestive system diseases.

    **Outputs:**
    - Predicted probability of depression vs. no depression
    - SHAP-based model explanation

    """)
    with st.sidebar.expander("Variable description"):
        for f in feature_names_display:
            st.markdown(f"**{feature_dict.get(f, f)}**: {variable_descriptions.get(f, '')}")

    st.title("Depression prediction in patients with chronic digestive system diseases")


    # 加载模型
    try:
        model, model_feature_names = load_model('./xgb_model.pkl')
        st.sidebar.success("Model loaded successfully")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        return

    # 输入区域
    st.header("Patient characteristics")
    c1, c2, c3 = st.columns(3)
    with c1:
        sleep_night = st.number_input("Night sleep duration (hours)", value=7.0, step=0.5, min_value=0.0)
        grip_strength = st.number_input("Grip strength (kg)", value=20.0, step=0.5, min_value=0.0)
        education = st.selectbox("Highest education level", options=[1, 2], format_func=lambda x: "Lower than high school" if x == 1 else "High school or above", index=1)
    with c2:
        residence = st.selectbox("Residence", options=[0, 1], format_func=lambda x: "Urban" if x == 0 else "Rural", index=0)
        life_assessment = st.selectbox("Life satisfaction", options=[1, 2], format_func=lambda x: "Not satisfied" if x == 1 else "Satisfied", index=2-1)
        health_assessment = st.selectbox("Self-rated health", options=[1, 2], format_func=lambda x: "Poor" if x == 1 else "Good", index=2-1)
    with c3:
        falldown = st.selectbox("History of falls", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
        disability = st.selectbox("Disability", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
        kidney_disease = st.selectbox("Kidney disease", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    c4, c5, c6 = st.columns(3)
    with c4:
        arthritis = c4.selectbox("Arthritis", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    with c5:
        eyesight = c5.selectbox("Eyesight", options=[1, 2], format_func=lambda x: "Poor" if x == 1 else "Good", index=2-1)
    with c6:
        iadl = c6.selectbox("IADL function", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=1)

    if st.button("Run prediction", type="primary"):
        # 组装输入
        user_inputs = {
            'Sleep_night': sleep_night,
            'Grip_strength': grip_strength,
            'Education': education,
            'Residence': residence,
            'Life_assessment': life_assessment,
            'Health_assessment': health_assessment,
            'Falldown': falldown,
            'Disability': disability,
            'Kidney_disease': kidney_disease,
            'Arthritis': arthritis,
            'Heart_disease': 0,
            'Eyesight': eyesight,
            'IADL': iadl,
        }

        # Align model feature names to UI keys (alias mapping)
        alias_to_user_key = {
            'sleep_night': 'Sleep_night',
            'grip_strength': 'Grip_strength',
            'education': 'Education',
            'residence': 'Residence',
            'life_assessment': 'Life_assessment',
            'health_assessment': 'Health_assessment',
            'falldown': 'Falldown',
            'disability': 'Disability',
            'kidney_disease': 'Kidney_disease',
            'arthritis': 'Arthritis',
            'heart_disease': 'Heart_disease',
            'eyesight': 'Eyesight',
            'iadl': 'IADL'
        }

        # 构造输入 DataFrame
        if model_feature_names:
            resolved_values, missing_features = [], []
            for c in model_feature_names:
                ui_key = alias_to_user_key.get(c, c)
                val = user_inputs.get(ui_key, None)
                if val is None:
                    missing_features.append(c)
                resolved_values.append(val)
            if missing_features:
                st.error(f"The following model features are missing from the UI or name-mismatched: {missing_features}")
                with st.expander("Debug: compare model feature names and UI keys"):
                    st.write("Model feature names:", model_feature_names)
                    st.write("UI input keys:", list(user_inputs.keys()))
                return
            input_df = pd.DataFrame([resolved_values], columns=model_feature_names)
        else:
            input_df = pd.DataFrame([[user_inputs[c] for c in feature_names_display]], columns=feature_names_display)

        # Clean & convert to numeric
        input_df = input_df.applymap(_clean_number)
        for c in input_df.columns:
            input_df[c] = pd.to_numeric(input_df[c], errors='coerce')
        if input_df.isnull().any().any():
            st.error("There are missing or unparsable input values. Please check that all numeric fields are valid numbers.")
            with st.expander("Debug: current input DataFrame"):
                st.write(input_df)
            return

        # ======== Prediction ========
        try:
            proba = predict_proba_safe(model, input_df)[0]
            if len(proba) == 2:
                no_depress_prob = float(proba[0]); depress_prob = float(proba[1])
            else:
                raise ValueError("返回的概率维度异常")

            # 展示结果
            st.header("Depression risk prediction result")
            a, b = st.columns(2)
            with a:
                st.subheader("Probability of no depression")
                st.progress(no_depress_prob)
                st.write(f"{no_depress_prob:.2%}")
            with b:
                st.subheader("Probability of depression")
                st.progress(depress_prob)
                st.write(f"{depress_prob:.2%}")

            # ======= SHAP explanation =======
            st.write("---"); st.subheader("Model explanation (SHAP)")
            try:
                # 优先通用入口
                try:
                    explainer = shap.Explainer(model)
                    sv = explainer(input_df)  # Explanation
                    shap_value = np.array(sv.values[0])
                    expected_value = sv.base_values[0] if np.ndim(sv.base_values) else sv.base_values
                except Exception:
                    # 回退 TreeExplainer
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(input_df)
                    if isinstance(shap_values, list):
                        shap_value = np.array(shap_values[1][0])
                        ev = explainer.expected_value
                        expected_value = ev[1] if isinstance(ev, (list, np.ndarray)) else ev
                    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                        shap_value = shap_values[0, :, 1]
                        ev = explainer.expected_value
                        expected_value = ev[1] if isinstance(ev, (list, np.ndarray)) else ev
                    else:
                        shap_value = np.array(shap_values[0])
                        expected_value = explainer.expected_value

                current_features = list(input_df.columns)

                # --- Waterfall plot ---
                st.subheader("SHAP waterfall plot")
                import matplotlib.font_manager as fm
                try:
                    c_fonts = [
                        'WenQuanYi Zen Hei','WenQuanYi Micro Hei','Noto Sans CJK SC',
                        'Source Han Sans SC','SimHei','Microsoft YaHei','PingFang SC','Hiragino Sans GB'
                    ]
                    avail = [f.name for f in fm.fontManager.ttflist]
                    for f in c_fonts:
                        if f in avail:
                            plt.rcParams['font.sans-serif'] = [f, 'DejaVu Sans']; break
                except Exception:
                    plt.rcParams['font.sans-serif'] = ['DejaVu Sans','Arial']
                plt.rcParams['axes.unicode_minus'] = False

                fig_waterfall = plt.figure(figsize=(12, 8))
                display_data = input_df.iloc[0].copy()
                # Map categorical variables to readable English labels
                try:
                    if 'Education' in display_data.index:
                        display_data['Education'] = {1:'Lower than high school',2:'High school or above'}.get(int(display_data['Education']), display_data['Education'])
                    if 'Residence' in display_data.index:
                        display_data['Residence'] = {0:'Urban',1:'Rural'}.get(int(display_data['Residence']), display_data['Residence'])
                    if 'Life_assessment' in display_data.index:
                        display_data['Life_assessment'] = {1:'Not satisfied',2:'Satisfied'}.get(int(display_data['Life_assessment']), display_data['Life_assessment'])
                    if 'Health_assessment' in display_data.index:
                        display_data['Health_assessment'] = {1:'Poor',2:'Good'}.get(int(display_data['Health_assessment']), display_data['Health_assessment'])
                    if 'Falldown' in display_data.index:
                        display_data['Falldown'] = {0:'No',1:'Yes'}.get(int(display_data['Falldown']), display_data['Falldown'])
                    if 'Disability' in display_data.index:
                        display_data['Disability'] = {0:'No',1:'Yes'}.get(int(display_data['Disability']), display_data['Disability'])
                    if 'Kidney_disease' in display_data.index:
                        display_data['Kidney_disease'] = {0:'No',1:'Yes'}.get(int(display_data['Kidney_disease']), display_data['Kidney_disease'])
                    if 'Arthritis' in display_data.index:
                        display_data['Arthritis'] = {0:'No',1:'Yes'}.get(int(display_data['Arthritis']), display_data['Arthritis'])
                    if 'Heart_disease' in display_data.index:
                        display_data['Heart_disease'] = {0:'No',1:'Yes'}.get(int(display_data['Heart_disease']), display_data['Heart_disease'])
                    if 'Eyesight' in display_data.index:
                        display_data['Eyesight'] = {1:'Poor',2:'Good'}.get(int(display_data['Eyesight']), display_data['Eyesight'])
                    if 'IADL' in display_data.index:
                        display_data['IADL'] = {0:'No',1:'Yes'}.get(int(display_data['IADL']), display_data['IADL'])
                except Exception:
                    pass

                try:
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_value,
                            base_values=expected_value,
                            data=display_data.values,
                            feature_names=[feature_dict.get(f, f) for f in current_features]
                        ),
                        max_display=len(current_features),
                        show=False
                    )
                except Exception:
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_value,
                            base_values=expected_value,
                            data=display_data.values,
                            feature_names=current_features
                        ),
                        max_display=len(current_features),
                        show=False
                    )

                # 修正 Unicode 负号，强制字体
                for ax in fig_waterfall.get_axes():
                    for text in ax.texts:
                        s = text.get_text()
                        if '−' in s: text.set_text(s.replace('−','-'))
                        if chinese_font: text.set_fontfamily(chinese_font)
                    for label in ax.get_yticklabels() + ax.get_xticklabels():
                        t = label.get_text()
                        if '−' in t: label.set_text(t.replace('−','-'))
                        if chinese_font: label.set_fontfamily(chinese_font)
                    if chinese_font:
                        ax.set_xlabel(ax.get_xlabel(), fontfamily=chinese_font)
                        ax.set_ylabel(ax.get_ylabel(), fontfamily=chinese_font)
                        ax.set_title(ax.get_title(), fontfamily=chinese_font)

                plt.tight_layout()
                st.pyplot(fig_waterfall); plt.close(fig_waterfall)

                # --- Force plot ---
                st.subheader("SHAP force plot")
                try:
                    import streamlit.components.v1 as components
                    force_plot = shap.force_plot(
                        expected_value,
                        shap_value,
                        display_data,
                        feature_names=[feature_dict.get(f, f) for f in current_features]
                    )
                    shap_html = f"""
                    <head>{shap.getjs()}</head>
                    <body><div class="force-plot-container">{force_plot.html()}</div></body>
                    """
                    components.html(shap_html, height=400, scrolling=False)
                except Exception as e:
                    st.warning(f"Failed to generate SHAP force plot: {e}")

            except Exception as e:
                st.error(f"Failed to generate SHAP explanation: {e}")
                import traceback; st.error(traceback.format_exc())

        except Exception as e:
            st.error(f"Prediction or result display failed: {e}")
            import traceback; st.error(traceback.format_exc())

    st.write("---")
    st.caption(" Depression Risk Calculator (XGBoost)")

if __name__ == "__main__":
    main()
