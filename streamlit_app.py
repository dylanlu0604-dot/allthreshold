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
id_name_map = id_name_map.dropna(subset=['ID', '繁中名稱']).astype({'ID': int, '繁中名稱': str})
series_names = id_name_map['繁中名稱'].tolist()

def get_name_from_id(id_val, default_name):
    name = id_name_map[id_name_map['ID'] == id_val]['繁中名稱']
    return name.iloc[0] if not name.empty else default_name

with st.sidebar:
    st.header("資料來源與參數設定")
    
    trigger_mode = st.radio("觸發邏輯", ["Greater", "Smaller"], horizontal=True)

    variable_default_id = 617
    variable_default_name = get_name_from_id(variable_default_id, "美國經濟領先指標")
    selected_variable_name = st.selectbox("變數ID", options=series_names, index=series_names.index(variable_default_name))
    selected_variable_id = id_name_map[id_name_map['繁中名稱'] == selected_variable_name]['ID'].iloc[0]

    target_default_id = 1248
    target_default_name = get_name_from_id(target_default_id, "標準普爾 500 指數")
    selected_target_name = st.selectbox("研究目標ID", options=series_names, index=series_names.index(target_default_name))
    selected_target_id = id_name_map[id_name_map['繁中名稱'] == selected_target_name]['ID'].iloc[0]

    api_key = st.text_input(
        "MacroMicro API Key（留空則使用 st.secrets 或環境變數）",
        value=st.secrets.get("MACROMICRO_API_KEY", os.environ.get("MACROMICRO_API_KEY", "")),
        type="password"
    )

    std_choices = [0.5, 1.0, 1.5, 2.0]
    roll_choices = [6, 12, 24, 36, 60, 120]

    months_gap_threshold = st.number_input("事件間隔（至少幾個月）", min_value=1, max_value=36, value=6)
    chart_winrolling_value = 120
    st.caption("結果會列出所有 std×window 組合；下方圖表僅以 window=120 做視覺化參考。")

# ---------------------- Helpers ------------------------
OFFSETS = [-12, -6, 0, 6, 12]

def _need_api_key() -> str:
    k = api_key or st.secrets.get("MACROMICRO_API_KEY", "") or os.environ.get("MACROMICRO_API_KEY", "")
    if not k:
        st.error("缺少 MacroMicro API Key。請在側邊欄輸入或於 .streamlit/secrets.toml 設定。")
        st.stop()
    return k

@st.cache_data(show_spinner=False, ttl=3600)
def mm(series_id: int, frequency: str, name: str, k: str) -> pd.DataFrame | None:
    url = f"https://dev-biz-api.macromicro.me/v1/stats/series/{series_id}?history=true"
    headers = {"X-Api-Key": k}
    for attempt in range(5):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            df = pd.DataFrame(data["series"]).set_index("date")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index().resample(frequency).mean()
            df.columns = [name]
            return df
        except Exception as e:
            st.write(f"Error fetching series_id {series_id} (attempt {attempt+1}/5): {e}")
            time.sleep(1)
    return None

def find_row_number_for_date(df_obj: pd.DataFrame, specific_date: pd.Timestamp) -> int:
    return df_obj.index.get_loc(pd.Timestamp(specific_date))

def _condition(df: pd.DataFrame, std: float, winrolling: int, mode: str) -> pd.Series:
    if mode == "Greater":
        return df["breath"].rolling(6).max() > df["Rolling_mean"] + std * df["Rolling_std"]
    else:
        return df["breath"].rolling(6).min() < df["Rolling_mean"] - std * df["Rolling_std"]

def process_series(variable_id: int, target_id: int, std_value: float, winrolling_value: int, k: str, mode: str, months_threshold: int) -> list[dict]:
    results: list[dict] = []
    try:
        x1, code1 = "breath", variable_id
        x2, code2 = "index", target_id

        df1 = mm(code1, "MS", x1, k)
        df2 = mm(code2, "MS", x2, k)
        if df1 is None or df2 is None:
            st.warning(f"series_id {variable_id} 或 target_id {target_id} 取檔失敗。")
            return results

        alldf_original = pd.concat([df1, df2], axis=1).resample("MS").asfreq().ffill()
        alldf = alldf_original.copy()
        timeforward, timepast = 31, 31

        # ===== 第一段分析：原始 breath =====
        df = alldf[[x1, x2]].copy()
        df["Rolling_mean"] = df["breath"].rolling(window=winrolling_value).mean()
        df["Rolling_std"] = df["breath"].rolling(window=winrolling_value).std()
        filtered_df = df[_condition(df, std_value, winrolling_value, mode)]

        finalb_dates_1: list[pd.Timestamp] = []
        if not filtered_df.empty:
            for date in filtered_df.index:
                if not finalb_dates_1 or ((date - finalb_dates_1[-1]).days / 30) >= months_threshold:
                    finalb_dates_1.append(date)

        resulttable1 = None
        finalb1 = None
        times1 = pre1 = prewin1 = after1 = afterwin1 = 0
        
        if finalb_dates_1:
            dfs = []
            for dt in finalb_dates_1:
                try:
                    a = find_row_number_for_date(alldf, dt)
                    if a - timepast >= 0 and a + timeforward < len(alldf):
                        temp_df = alldf["index"].iloc[a - timepast : a + timeforward].to_frame(name=dt).reset_index(drop=True)
                        dfs.append(temp_df)
                except KeyError:
                    continue
            
            if dfs:
                origin = pd.concat(dfs, axis=1)
                finalb1 = origin.apply(lambda col: 100 * col / col.iloc[timepast])
                finalb1 = finalb1.iloc[:, -10:] # 只保留最近 10 次事件
                finalb1["mean"] = finalb1.mean(axis=1)

                offsets = [-12, -6, 0, 6, 12]
                table1 = pd.concat([finalb1.iloc[timepast + off] for off in offsets], axis=1).T
                table1.columns = finalb1.columns[:-1].strftime('%Y-%m-%d').tolist() + ['mean']
                table1.index = [f"{off}m" for off in offsets]
                resulttable1 = table1.copy()
                perc_df = pd.DataFrame((resulttable1.iloc[:, :-1] > 100).mean(axis=1) * 100, columns=["勝率"]).T
                resulttable1 = pd.concat([resulttable1, perc_df])

                times1 = origin.shape[1]
                pre1 = resulttable1.loc["mean", "-12m"] - 100
                prewin1 = resulttable1.loc["勝率", "-12m"]
                after1 = resulttable1.loc["mean", "12m"] - 100
                afterwin1 = resulttable1.loc["勝率", "12m"]


        # ===== 第二段分析：breath / breath.shift(12) =====
        df = alldf[[x1, x2]].copy()
        df["breath"] = df["breath"] / df["breath"].shift(12)
        df.dropna(inplace=True)
        df["Rolling_mean"] = df["breath"].rolling(window=winrolling_value).mean()
        df["Rolling_std"] = df["breath"].rolling(window=winrolling_value).std()
        filtered_df = df[_condition(df, std_value, winrolling_value, mode)]

        finalb_dates_2: list[pd.Timestamp] = []
        if not filtered_df.empty:
            for date in filtered_df.index:
                if not finalb_dates_2 or ((date - finalb_dates_2[-1]).days / 30) >= months_threshold:
                    finalb_dates_2.append(date)

        resulttable2 = None
        finalb2 = None
        times2 = pre2 = prewin2 = after2 = afterwin2 = 0
        
        if finalb_dates_2:
            dfs = []
            for dt in finalb_dates_2:
                try:
                    a = find_row_number_for_date(alldf, dt)
                    if a - timepast >= 0 and a + timeforward < len(alldf):
                        temp_df = alldf["index"].iloc[a - timepast : a + timeforward].to_frame(name=dt).reset_index(drop=True)
                        dfs.append(temp_df)
                except KeyError:
                    continue
            
            if dfs:
                origin = pd.concat(dfs, axis=1)
                finalb2 = origin.apply(lambda col: 100 * col / col.iloc[timepast])
                finalb2["mean"] = finalb2.mean(axis=1)
                
                offsets = [-12, -6, 0, 6, 12]
                table2 = pd.concat([finalb2.iloc[timepast + off] for off in offsets], axis=1).T
                table2.columns = finalb2.columns[:-1].strftime('%Y-%m-%d').tolist() + ['mean']
                table2.index = [f"{off}m" for off in offsets]
                resulttable2 = table2.copy()
                perc_df = pd.DataFrame((resulttable2.iloc[:, :-1] > 100).mean(axis=1) * 100, columns=["勝率"]).T
                resulttable2 = pd.concat([resulttable2, perc_df])

                times2 = origin.shape[1]
                pre2 = resulttable2.loc["mean", "-12m"] - 100
                prewin2 = resulttable2.loc["勝率", "-12m"]
                after2 = resulttable2.loc["mean", "12m"] - 100
                afterwin2 = resulttable2.loc["勝率", "12m"]
        
        results.append({
            "series_id": variable_id, "std": std_value, "winrolling": winrolling_value,
            "pre1": pre1, "prewin1": prewin1, "after1": after1, "afterwin1": afterwin1, "times1": times1,
            "pre2": pre2, "prewin2": prewin2, "after2": after2, "afterwin2": afterwin2, "times2": times2,
            "resulttable1": resulttable1,
            "resulttable2": resulttable2,
            "finalb1": finalb1.reset_index() if finalb1 is not None and "mean" in finalb1.columns else None,
            "finalb2": finalb2.reset_index() if finalb2 is not None and "mean" in finalb2.columns else None,
        })
    except Exception as e:
        st.write(f"Error during CALCULATION for series {variable_id}, std={std_value}, win={winrolling_value}: {e}")
    return results


# ---------------------- Main Flow ----------------------
series_ids = [selected_variable_id]
mode = trigger_mode
k = _need_api_key()

combos = list(product(std_choices, roll_choices))

if 'results_flat' not in st.session_state:
    with st.spinner("正在執行所有組合分析，請稍候..."):
        if Parallel is not None:
            num_cores = max(1, min(4, multiprocessing.cpu_count()))
            tasks = [(sid, selected_target_id, s, w, k, mode, months_gap_threshold) for sid in series_ids for (s, w) in combos]
            results_nested = Parallel(n_jobs=num_cores)(
                delayed(process_series)(*t) for t in tasks
            )
            results_flat = [item for sublist in results_nested for item in sublist if item]
        else:
            st.warning("joblib 未安裝，改用單執行緒。")
            results_flat = []
            for sid in series_ids:
                for s, w in combos:
                    results_flat.extend(process_series(sid, selected_target_id, s, w, k, mode, months_gap_threshold))
        
        st.session_state['results_flat'] = results_flat

results_flat = st.session_state['results_flat']

if not results_flat:
    st.info("尚無可顯示結果。請調整參數或確認 series 有足夠歷史資料。")
    st.stop()

# ===== 產出總覽表 =====
summary_rows_raw = []
summary_rows_yoy = []
required_keys = ["series_id", "std", "winrolling", "times1", "pre1", "prewin1", "after1", "afterwin1",
                 "times2", "pre2", "prewin2", "after2", "afterwin2"]

def _to_float(x):
    try: return float(x)
    except: return None

def _to_int(x):
    try: return int(x)
    except: return 0

def _classify(pre, after, prewin, afterwin, times):
    vals = [pre, after, prewin, afterwin, times]
    if any(v is None for v in vals): return "🚫 不是有效訊號"
    if (float(pre) < 0 and float(after) < 0) and (int(times) > 8) and (float(prewin) + float(afterwin) < 70): return "🐻 熊市訊號"
    if (float(pre) > 0 and float(after) > 0) and (int(times) > 8) and (float(prewin) + float(afterwin) > 130): return "🐮 牛市訊號"
    return "🚫 不是有效訊號"

def compute_score(pre, after, prewin, afterwin, times):
    label = _classify(pre, after, prewin, afterwin, times)
    vals = [_to_float(v) for v in [pre, after, prewin, afterwin]]
    if any(v is None for v in vals): return 0.0
    pre, after, prewin, afterwin = vals
    times = _to_int(times)
    if label.startswith("🐮"): return pre + after + (prewin - 50) + (afterwin - 50) + times
    elif label.startswith("🐻"): return -pre - after - (prewin - 50) - (afterwin - 50) + times
    else: return 0.0

for r in results_flat:
    pre1_val, after1_val, prewin1_val, afterwin1_val = r.get("pre1"), r.get("after1"), r.get("prewin1"), r.get("afterwin1")
    times1_val = r.get("times1")
    summary_rows_raw.append({
        "系列": get_name_from_id(r.get("series_id"), str(r.get("series_id"))), "ID": r.get("series_id"),
        "std": r.get("std"), "window": r.get("winrolling"), "事件數": times1_val,
        "前12m均值%": pre1_val, "後12m均值%": after1_val, "勝率前": prewin1_val, "勝率後": afterwin1_val,
        "得分": compute_score(pre1_val, after1_val, prewin1_val, afterwin1_val, times1_val),
        "有效": _classify(pre1_val, after1_val, prewin1_val, afterwin1_val, times1_val),
    })

    pre2_val, after2_val, prewin2_val, afterwin2_val = r.get("pre2"), r.get("after2"), r.get("prewin2"), r.get("afterwin2")
    times2_val = r.get("times2")
    summary_rows_yoy.append({
        "系列": get_name_from_id(r.get("series_id"), str(r.get("series_id"))), "ID": r.get("series_id"),
        "std": r.get("std"), "window": r.get("winrolling"), "事件數": times2_val,
        "前12m均值%": pre2_val, "後12m均值%": after2_val, "勝率前": prewin2_val, "勝率後": afterwin2_val,
        "得分": compute_score(pre2_val, after2_val, prewin2_val, afterwin2_val, times2_val),
        "有效": _classify(pre2_val, after2_val, prewin2_val, afterwin2_val, times2_val),
    })

summary_raw_df = pd.DataFrame(summary_rows_raw).sort_values(by=["得分", "事件數"], ascending=False).reset_index(drop=True)
summary_yoy_df = pd.DataFrame(summary_rows_yoy).sort_values(by=["得分", "事件數"], ascending=False).reset_index(drop=True)

# ===================================================================
# ==================== START: MODIFIED SECTION ====================
# ===================================================================

st.subheader("原始版本：所有 std × window 組合結果")
st.info("點選下方任一列（Row）以在下方繪製該組合的詳細圖表。")
st.dataframe(
    summary_raw_df, use_container_width=True, on_select="rerun",
    selection_mode="single-row", key="raw_selection"
)

st.subheader("年增版本：所有 std × window 組合結果")
st.info("點選下方任一列（Row）以在下方繪製該組合的詳細圖表。")
st.dataframe(
    summary_yoy_df, use_container_width=True, on_select="rerun",
    selection_mode="single-row", key="yoy_selection"
)

def plot_mean_curve(finalb_df, title):
    if finalb_df is None or "mean" not in finalb_df.columns or finalb_df.empty:
        st.info(f"{title} 無曲線資料可繪製。")
        return
    y = finalb_df["mean"].values
    n = len(y)
    half = n // 2
    x = np.arange(-half, n - half)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y, label=title)
    ax.axvline(0, linestyle='--', color='red')
    xlim = (-15, 15)
    ax.set_xlim(xlim)
    
    mask = (x >= xlim[0]) & (x <= xlim[1])
    if np.any(mask):
        valid_y = y[mask]
        if len(valid_y) > 0:
            ymin, ymax = np.min(valid_y) * 0.99, np.max(valid_y) * 1.01
            if ymin == ymax: ymin -= 1; ymax += 1
            ax.set_ylim(ymin, ymax)
            
    ax.set_xlabel('事件發生前後月數')
    ax.set_ylabel('指數 (100 = 事件當月)')
    st.pyplot(fig, use_container_width=True)

st.divider()
st.header("選定組合之詳細分析")
st.caption("當您點選上方任一表格中的組合時，詳細分析將會顯示在此處。")

selected_raw_row_data = None
selected_yoy_row_data = None

selection_raw = st.session_state.get("raw_selection", {})
if selection_raw.get("rows"):
    selected_index = selection_raw["rows"][0]
    if selected_index < len(summary_raw_df):
        selected_raw_row_data = summary_raw_df.iloc[selected_index]

selection_yoy = st.session_state.get("yoy_selection", {})
if selection_yoy.get("rows"):
    selected_index = selection_yoy["rows"][0]
    if selected_index < len(summary_yoy_df):
        selected_yoy_row_data = summary_yoy_df.iloc[selected_index]

# --- 處理「原始版本」表格的點擊 ---
if selected_raw_row_data is not None:
    std_to_find = selected_raw_row_data['std']
    window_to_find = selected_raw_row_data['window']

    if pd.notna(std_to_find) and pd.notna(window_to_find):
        # 這是關鍵的查找邏輯
        found_result = next((r for r in results_flat 
                             if r.get('std') is not None and r.get('winrolling') is not None and 
                             math.isclose(r['std'], std_to_find) and 
                             int(r['winrolling']) == int(window_to_find)), None)
        
        st.markdown(f"### 原始版本詳細分析：std = **{std_to_find}**, window = **{int(window_to_find)}**")
        
        # 【修正】在使用 found_result 前，先檢查它是否存在
        if found_result:
            col1, col2 = st.columns(2)
            with col1:
                if found_result.get("resulttable1") is not None:
                    st.dataframe(found_result["resulttable1"].style.format("{:.2f}"), use_container_width=True)
                else:
                    st.info("無原始值版本表格。")
            with col2:
                plot_mean_curve(found_result.get("finalb1"), "原始版本走勢")
        else:
            st.warning(f"無法在計算結果中找到對應的詳細資料 (std={std_to_find}, window={window_to_find})。")

# --- 處理「年增版本」表格的點擊 ---
if selected_yoy_row_data is not None:
    std_to_find = selected_yoy_row_data['std']
    window_to_find = selected_yoy_row_data['window']

    if pd.notna(std_to_find) and pd.notna(window_to_find):
        found_result = next((r for r in results_flat 
                             if r.get('std') is not None and r.get('winrolling') is not None and 
                             math.isclose(r['std'], std_to_find) and 
                             int(r['winrolling']) == int(window_to_find)), None)

        st.markdown(f"### 年增版本詳細分析：std = **{std_to_find}**, window = **{int(window_to_find)}**")
        
        # 【修正】同樣，在使用前先檢查
        if found_result:
            col3, col4 = st.columns(2)
            with col3:
                if found_result.get("resulttable2") is not None:
                    st.dataframe(found_result["resulttable2"].style.format("{:.2f}"), use_container_width=True)
                else:
                    st.info("無年增率版本表格。")
            with col4:
                plot_mean_curve(found_result.get("finalb2"), "年增版本走勢")
        else:
            st.warning(f"無法在計算結果中找到對應的詳細資料 (std={std_to_find}, window={window_to_find})。")

# 如果沒有任何點擊，顯示提示
if selected_raw_row_data is None and selected_yoy_row_data is None:
    st.info("👆 請點擊上方任一表格中的一列，以查看其詳細分析。")
    
# ===================================================================
# ===================== END: MODIFIED SECTION =====================
# ===================================================================

# ===== Plot by series_ids_text: Levels & YoY =====
st.divider()
st.subheader("可調整時間區間的序列圖 (根據上方表格點選的組合更新)")

winrolling_for_levels = chart_winrolling_value
winrolling_for_yoy = chart_winrolling_value

if selected_raw_row_data is not None: winrolling_for_levels = int(selected_raw_row_data['window'])
if selected_yoy_row_data is not None: winrolling_for_yoy = int(selected_yoy_row_data['window'])

alt.data_transformers.disable_max_rows()
sigma_levels = [0.5, 1.0, 1.5, 2.0]

def levels_chart_with_brush(s: pd.Series, sid: int, name: str, winrolling_value: int):
    # ... (此部分圖表繪製邏輯不變) ...
    roll_mean = s.rolling(winrolling_value).mean()
    roll_std = s.rolling(winrolling_value).std()
    df_levels = pd.DataFrame({"Date": s.index, "Level": s.values, "Mean": roll_mean.values})
    for m in sigma_levels:
        df_levels[f"+{m}σ"] = (roll_mean + m * roll_std).values
        df_levels[f"-{m}σ"] = (roll_mean - m * roll_std).values
    long_levels = df_levels.melt("Date", var_name="Series", value_name="Value").dropna()
    brush = alt.selection_interval(encodings=["x"])
    upper = alt.Chart(long_levels).mark_line().encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Value:Q", title="Level", scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("Date:T"), "Series:N", alt.Tooltip("Value:Q", format=".2f")],
    ).transform_filter(brush).properties(title=f"{name} ({sid}) | {winrolling_value}-period rolling mean ±σ", height=320)
    lower = alt.Chart(df_levels).mark_area(opacity=0.4).encode(
        x=alt.X("Date:T", title=""), y=alt.Y("Level:Q", title="")
    ).properties(height=60).add_params(brush)
    return alt.vconcat(upper, lower)

def yoy_chart_with_brush(s: pd.Series, sid: int, name: str, winrolling_value: int):
    # ... (此部分圖表繪製邏輯不變) ...
    yoy = s.pct_change(12) * 100.0
    yoy_mean = yoy.rolling(winrolling_value).mean()
    yoy_std = yoy.rolling(winrolling_value).std()
    df_yoy = pd.DataFrame({"Date": yoy.index, "YoY (%)": yoy.values, "Mean": yoy_mean.values})
    for m in sigma_levels:
        df_yoy[f"+{m}σ"] = (yoy_mean + m * yoy_std).values
        df_yoy[f"-{m}σ"] = (yoy_mean - m * yoy_std).values
    long_yoy = df_yoy.melt("Date", var_name="Series", value_name="Value").dropna()
    brush = alt.selection_interval(encodings=["x"])
    upper = alt.Chart(long_yoy).mark_line().encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Value:Q", title="YoY (%)"),
        color=alt.Color("Series:N", legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("Date:T"), "Series:N", alt.Tooltip("Value:Q", format=".2f")],
    ).transform_filter(brush).properties(title=f"{name} ({sid}) | YoY (%) with {winrolling_value}-period rolling mean ±σ", height=320)
    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    lower = alt.Chart(df_yoy).mark_area(opacity=0.4).encode(
        x=alt.X("Date:T", title=""), y=alt.Y("YoY (%):Q", title="")
    ).properties(height=60).add_params(brush)
    return alt.vconcat(upper + zero_line, lower)

sid = id_name_map[id_name_map['繁中名稱'] == selected_variable_name]['ID'].iloc[0]
df_target = mm(int(sid), "MS", f"series_{sid}", k)
if df_target is None or df_target.empty:
    st.info(f"No data for series {sid}, skipping.")
else:
    s = df_target.iloc[:, 0].astype(float)
    with st.expander(f"Series: {selected_variable_name} ({sid})", expanded=True):
        colA, colB = st.columns(2)
        with colA:
            st.caption(f"Levels rolling window = {winrolling_for_levels}")
            st.altair_chart(levels_chart_with_brush(s, sid, selected_variable_name, winrolling_for_levels), use_container_width=True)
        with colB:
            st.caption(f"YoY rolling window = {winrolling_for_yoy}")
            st.altair_chart(yoy_chart_with_brush(s, sid, selected_variable_name, winrolling_for_yoy), use_container_width=True)
