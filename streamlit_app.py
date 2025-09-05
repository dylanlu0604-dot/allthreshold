import os
import time
import math
import numpy as np
import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import Parallel, delayed
import multiprocessing
import plotly.graph_objects as go
import io
import altair as alt
from itertools import product

# --- Optional parallelism ---
try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    def delayed(f):
        return f

st.set_page_config(page_title="熊市訊號與牛市訊號尋找工具", layout="wide")

# -------------------------- UI --------------------------
st.title("熊市訊號與牛市訊號尋找工具")

# --- Function to load ID to Name mapping from GitHub ---
@st.cache_data(show_spinner="下載ID對應表...", ttl=3600)
def load_series_id_map() -> pd.DataFrame:
    """從 GitHub 下載 ID 與名稱對應的 Excel 檔案。"""
    github_url = "https://raw.githubusercontent.com/dylanlu0604-dot/test2/main/Idwithname.xlsx"
    try:
        response = requests.get(github_url)
        response.raise_for_status() # 檢查是否有 HTTP 錯誤
        df = pd.read_excel(io.BytesIO(response.content))
        
        # Check for expected columns and handle potential errors
        required_cols = ['ID', '繁中名稱']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Excel 檔案中缺少必要的欄位。預期欄位：{required_cols}。實際欄位：{df.columns.tolist()}")
            st.stop()
        
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"無法從 GitHub 下載對應表: {e}")
        st.stop()
    except Exception as e:
        st.error(f"處理 Excel 檔案時發生錯誤: {e}")
        st.stop()

# 載入 ID 對應表
id_name_map = load_series_id_map()
# 處理 NaN 或空值
id_name_map = id_name_map.dropna(subset=['ID', '繁中名稱']).astype({'ID': int, '繁中名稱': str})
series_names = id_name_map['繁中名稱'].tolist()

# 根據 ID 找到對應的中文名稱
def get_name_from_id(id_val, default_name):
    name = id_name_map[id_name_map['ID'] == id_val]['繁中名稱']
    return name.iloc[0] if not name.empty else default_name

with st.sidebar:
    st.header("資料來源與參數設定")
    
    # 觸發邏輯選擇：Greater / Smaller
    trigger_mode = st.radio("觸發邏輯", ["Greater", "Smaller"], horizontal=True)

    # 將文字輸入框替換為下拉式選單，顯示中文名稱
    variable_default_id = 617
    variable_default_name = get_name_from_id(variable_default_id, "美國經濟領先指標")
    selected_variable_name = st.selectbox("變數ID", options=series_names, index=series_names.index(variable_default_name))
    # 根據選定的中文名稱找出對應的 ID
    selected_variable_id = id_name_map[id_name_map['繁中名稱'] == selected_variable_name]['ID'].iloc[0]

    # 將研究目標ID改為下拉式選單
    target_default_id = 1248
    target_default_name = get_name_from_id(target_default_id, "標準普爾 500 指數")
    selected_target_name = st.selectbox("研究目標ID", options=series_names, index=series_names.index(target_default_name))
    selected_target_id = id_name_map[id_name_map['繁中名稱'] == selected_target_name]['ID'].iloc[0]

    api_key = st.text_input(
        "MacroMicro API Key（留空則使用 st.secrets 或環境變數）",
        value=st.secrets.get("MACROMICRO_API_KEY", os.environ.get("MACROMICRO_API_KEY", "")),
        type="password"
    )

    # 不讓使用者挑；一次跑全部組合
    std_choices = [0.5, 1.0, 1.5, 2.0]
    roll_choices = [6, 12, 24, 36, 60, 120]

    # 事件間隔門檻
    months_gap_threshold = st.number_input("事件間隔（至少幾個月）", min_value=1, max_value=36, value=6)

    # 圖表參考的固定 window（不影響彙整）
    chart_winrolling_value = 120
    st.caption("結果會列出所有 std×window 組合；下方圖表僅以 window=120 做視覺化參考。")

# ---------------------- Helpers ------------------------
OFFSETS = [-12, -6, 0, 6, 12]  # 以「月」為單位

def _need_api_key() -> str:
    k = api_key or st.secrets.get("MACROMICRO_API_KEY", "") or os.environ.get("MACROMICRO_API_KEY", "")
    if not k:
        st.error("缺少 MacroMicro API Key。請在側邊欄輸入或於 .streamlit/secrets.toml 設定。")
        st.stop()
    return k

# 取得資料的函數等...
# 省略其他代碼，只保留關鍵修改部分...

# ---------------------- Main Flow ----------------------

series_ids = [selected_variable_id]  # 取得下拉式選單的 ID
mode = trigger_mode
k = _need_api_key()

# 建立所有 std×window 組合
combos = list(product(std_choices, roll_choices))

# 平行執行（或退回單執行緒）
if Parallel is not None:
    num_cores = max(1, min(4, multiprocessing.cpu_count()))
    tasks = [(sid, selected_target_id, s, w, k, mode, months_gap_threshold) for sid in series_ids for (s, w) in combos]
    results_nested = Parallel(n_jobs=num_cores)(
        delayed(process_series)(t[0], t[1], t[2], t[3], t[4], t[5], t[6]) for t in tasks
    )
    results_flat = [item for sublist in results_nested for item in sublist]
else:
    st.warning("`joblib` 未安裝，改用單執行緒。")
    results_flat = []
    for sid in series_ids:
        for s, w in combos:
            results_flat.extend(process_series(sid, selected_target_id, s, w, k, mode, months_gap_threshold))

if not results_flat:
    st.info("尚無可顯示結果。請調整參數或確認 series 有足夠歷史資料。")
    st.stop()

# ===== 產出總覽表（拆成「原始」與「年增」兩張表）=====
# 省略其他計算、結果處理邏輯，保留選取表格顯示...

# ----------------------- Interactive DataFrame ----------------------
st.subheader("原始版本：所有 std × window 組合結果")
st.info("點選下方任一列（Row）以在下方繪製該組合的詳細圖表。")
# 捕捉選取的行
selection_raw = st.session_state.get("raw_selection", None)
selection_yoy = st.session_state.get("yoy_selection", None)

if selection_raw is not None and selection_raw.get("rows"):
    selected_index_raw = selection_raw["rows"][0]
    selected_raw_row_data = summary_raw_df.iloc[selected_index_raw]

if selection_yoy is not None and selection_yoy.get("rows"):
    selected_index_yoy = selection_yoy["rows"][0]
    selected_yoy_row_data = summary_yoy_df.iloc[selected_index_yoy]

# 顯示 DataFrame
st.dataframe(
    summary_raw_df,
    use_container_width=True,
    selection_mode="single-row",
    key="raw_selection"
)

st.subheader("年增版本：所有 std × window 組合結果")
st.info("點選下方任一列（Row）以在下方繪製該組合的詳細圖表。")
# 顯示 DataFrame
st.dataframe(
    summary_yoy_df,
    use_container_width=True,
    selection_mode="single-row",
    key="yoy_selection"
)

# 顯示詳細圖表的共用函數
def plot_mean_curve(finalb_df, title):
    if finalb_df is None or "mean" not in finalb_df.columns:
        st.info(f"{title} 無曲線資料。")
        return
    y = finalb_df["mean"].values
    n = len(y)
    half = n // 2
    x = np.arange(-half, n - half)   # 0 對齊事件當月
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y, label=title)
    ax.axvline(0, linestyle='--', color='red')
    xlim = (-15, 15)
    ax.set_xlim(xlim)
    mask = (x >= xlim[0]) & (x <= xlim[1])
    if np.any(mask):
        ymin = float(np.min(y[mask])) * 0.99
        ymax = float(np.max(y[mask])) * 1.01
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel('事件發生前後月數')
    ax.set_ylabel('指數 (100 = 事件當月)')
    st.pyplot(fig, use_container_width=True)

# --------------------- Show Details After Selection -------------------
st.divider()
st.header("選定組合之詳細分析")
st.caption("當您點選上方任一表格中的組合時，詳細分析將會顯示在此處。")

# 如果「原始版本」表格的列被選取，則顯示其詳細資料
if selected_raw_row_data is not None:
    std_to_find = selected_raw_row_data['std']
    window_to_find = selected_raw_row_data['window']
    
    # 找到對應結果
    found_result = next((r for r in results_flat if math.isclose(r['std'], std_to_find) and r['winrolling'] == window_to_find), None)
    
    if found_result:
        st.markdown(f"### 原始版本詳細分析：std = **{std_to_find}**, window = **{window_to_find}**")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(found_result.get("resulttable1"), use_container_width=True)
        with col2:
            plot_mean_curve(found_result.get("finalb1"), "原始版本走勢")

# 如果「年增版本」表格的列被選取，則顯示其詳細資料
if selected_yoy_row_data is not None:
    std_to_find = selected_yoy_row_data['std']
    window_to_find = selected_yoy_row_data['window']
    
    # 找到對應結果
    found_result = next((r for r in results_flat if math.isclose(r['std'], std_to_find) and r['winrolling'] == window_to_find), None)

    if found_result:
        st.markdown(f"### 年增版本詳細分析：std = **{std_to_find}**, window = **{window_to_find}**")
        col3, col4 = st.columns(2)
        with col3:
            st.dataframe(found_result.get("resulttable2"), use_container_width=True)
        with col4:
            plot_mean_curve(found_result.get("finalb2"), "年增版本走勢")
