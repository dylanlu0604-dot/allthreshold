import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
from itertools import product
import multiprocessing
import io
import altair as alt

# Optional parallelism
try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    def delayed(f):
        return f

st.set_page_config(page_title="熊市訊號與牛市訊號尋找工具", layout="wide")
st.title("熊市訊號與牛市訊號尋找工具")

# ---------- Load ID Map ----------
@st.cache_data(show_spinner="下載ID對應表...", ttl=3600)
def load_series_id_map() -> pd.DataFrame:
    github_url = "https://raw.githubusercontent.com/dylanlu0604-dot/test2/main/Idwithname.xlsx"
    try:
        response = requests.get(github_url)
        response.raise_for_status()
        df = pd.read_excel(io.BytesIO(response.content))
        required_cols = ['ID', '繁中名稱']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Excel 檔案中缺少必要欄位：{required_cols}；實際：{df.columns.tolist()}")
            st.stop()
        return df
    except Exception as e:
        st.error(f"下載/處理對應表失敗：{e}")
        st.stop()

id_name_map = load_series_id_map().dropna(subset=['ID','繁中名稱']).astype({'ID':int,'繁中名稱':str})
series_names = id_name_map['繁中名稱'].tolist()

def get_name_from_id(id_val, default_name):
    name = id_name_map[id_name_map['ID']==id_val]['繁中名稱']
    return name.iloc[0] if not name.empty else default_name

with st.sidebar:
    st.header("資料來源與參數設定")
    modes = ["Greater", "Smaller"]
    st.caption("觸發邏輯將同時評估：Greater 與 Smaller。")

    variable_default_id = 617
    variable_default_name = get_name_from_id(variable_default_id, "美國經濟領先指標")
    selected_variable_name = st.selectbox("變數ID", options=series_names, index=series_names.index(variable_default_name))
    selected_variable_id = id_name_map[id_name_map['繁中名稱']==selected_variable_name]['ID'].iloc[0]

    target_default_id = 1248
    target_default_name = get_name_from_id(target_default_id, "標準普爾 500 指數")
    selected_target_name = st.selectbox("研究目標ID", options=series_names, index=series_names.index(target_default_name))
    selected_target_id = id_name_map[id_name_map['繁中名稱']==selected_target_name]['ID'].iloc[0]

    api_key = st.text_input(
        "MacroMicro API Key（留空則使用 st.secrets 或環境變數）",
        value=st.secrets.get("MACROMICRO_API_KEY", os.environ.get("MACROMICRO_API_KEY","")),
        type="password"
    )

    std_choices = [0.5, 1.0, 1.5, 2.0]
    roll_choices = [6, 12, 24, 36, 60, 120]
    months_gap_threshold = st.number_input("事件間隔（至少幾個月）", min_value=1, max_value=36, value=6)
    chart_winrolling_value = 120
    st.caption("圖表默認使用 window=120；若有最佳組合，將用各自最佳 window 繪製（原始/年增不同）。")

# ---------- Helper Functions ----------
def _need_api_key() -> str:
    k = api_key or st.secrets.get("MACROMICRO_API_KEY","") or os.environ.get("MACROMICRO_API_KEY","")
    if not k:
        st.error("缺少 MacroMicro API Key。")
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
        except Exception:
            time.sleep(1)
    return None

def find_row_number_for_date(df_obj: pd.DataFrame, specific_date: pd.Timestamp) -> int:
    return df_obj.index.get_loc(pd.Timestamp(specific_date))

def _condition(df: pd.DataFrame, std: float, winrolling: int, mode: str) -> pd.Series:
    if mode == "Greater":
        return df["breath"].rolling(6).max() > df["Rolling_mean"] + std * df["Rolling_std"]
    else:
        return df["breath"].rolling(6).min() < df["Rolling_mean"] - std * df["Rolling_std"]

# ---------- Core Calculation Functions ----------
@st.cache_data(show_spinner="正在計算所有組合...", ttl=3600)
def run_all_combinations(_variable_id: int, _target_id: int, _std_choices: tuple, _roll_choices: tuple, _modes: tuple, _months_gap_threshold: int, _k: str) -> list:
    """
    執行所有參數組合的計算。
    此函式被快取，只有當輸入參數改變時才會重新執行。
    """
    series_name_for_debug = get_name_from_id(_variable_id, str(_variable_id))
    st.info(f"正在為變數「{series_name_for_debug} (ID: {_variable_id})」執行計算...")

    combos = list(product(_std_choices, _roll_choices, _modes))
    results_flat = []

    if Parallel is not None:
        num_cores = max(1, min(4, multiprocessing.cpu_count()))
        tasks = [(_variable_id, _target_id, s, w, _k, m, _months_gap_threshold) for (s, w, m) in combos]
        results_nested = Parallel(n_jobs=num_cores)(
            delayed(process_series)(*task_args) for task_args in tasks
        )
        results_flat = [item for sublist in results_nested for item in sublist if item]
    else:
        st.warning("joblib 未安裝，改用單執行緒。")
        for s, w, m in combos:
            results_flat.extend(process_series(_variable_id, _target_id, s, w, _k, m, _months_gap_threshold))
            
    return results_flat

def process_series(variable_id: int, target_id: int, std_value: float, winrolling_value: int, k: str, mode: str, months_threshold: int) -> list[dict]:
    results: list[dict] = []
    try:
        x1, code1 = "breath", variable_id
        x2, code2 = "index", target_id

        df1 = mm(code1, "MS", x1, k)
        df2 = mm(code2, "MS", x2, k)
        if df1 is None or df2 is None:
            print(f"WARNING: series_id {variable_id} 或 target_id {target_id} 取檔失敗。")
            return results

        alldf = pd.concat([df1, df2], axis=1).resample("MS").asfreq().ffill()
        timeforward, timepast = 31, 31

        # --- 原始版計算 ---
        df = alldf[[x1, x2]].copy()
        df["Rolling_mean"] = df["breath"].rolling(window=winrolling_value).mean()
        df["Rolling_std"] = df["breath"].rolling(window=winrolling_value).std()
        filtered_df = df[_condition(df, std_value, winrolling_value, mode)]
        finalb_dates_1: list[pd.Timestamp] = []
        for date in filtered_df.index:
            if not finalb_dates_1 or ((date - finalb_dates_1[-1]).days / 30) >= months_threshold:
                finalb_dates_1.append(date)
        
        times1, pre1, prewin1, after1, afterwin1 = 0, 0, 0, 0, 0
        resulttable1, finalb1 = None, None
        if finalb_dates_1:
            dfs = []
            for dt in finalb_dates_1:
                try:
                    a = find_row_number_for_date(alldf, dt)
                    if a - timepast < 0 or a + timeforward >= len(alldf): continue
                    temp_df = alldf["index"].iloc[a - timepast : a + timeforward].to_frame(name=dt).reset_index()
                    dfs.append(temp_df)
                except KeyError: continue
            
            if dfs:
                df_concat = pd.concat(dfs, axis=1)
                data_cols = [col for j, col in enumerate(df_concat.columns) if j % 2 == 1]
                origin = df_concat[data_cols]
                finalb1 = origin.apply(lambda col: 100 * col / col.iloc[timepast])
                finalb1 = finalb1.iloc[:, -10:]
                finalb1["mean"] = finalb1.mean(axis=1)
                offsets = [-12, -6, 0, 6, 12]
                table1 = pd.concat([finalb1.iloc[timepast + off] for off in offsets], axis=1)
                table1.columns = [f"{off}m" for off in offsets]
                resulttable1 = table1.iloc[:-1]
                if not resulttable1.empty:
                    perc_df = pd.DataFrame([(resulttable1 > 100).mean() * 100], index=["勝率"])
                    resulttable1 = pd.concat([resulttable1, perc_df, table1.iloc[-1:]])
                    times1 = len(resulttable1.columns) - 1
                    pre1 = resulttable1.loc["mean", "-12m"] - 100
                    prewin1 = resulttable1.loc["勝率", "-12m"]
                    after1 = resulttable1.loc["mean", "12m"] - 100
                    afterwin1 = resulttable1.loc["勝率", "12m"]

        # --- 年增版計算 ---
        df = alldf[[x1, x2]].copy()
        df["breath"] = df["breath"] / df["breath"].shift(12)
        df.dropna(inplace=True)
        df["Rolling_mean"] = df["breath"].rolling(window=winrolling_value).mean()
        df["Rolling_std"] = df["breath"].rolling(window=winrolling_value).std()
        filtered_df = df[_condition(df, std_value, winrolling_value, mode)]
        finalb_dates_2: list[pd.Timestamp] = []
        for date in filtered_df.index:
            if not finalb_dates_2 or ((date - finalb_dates_2[-1]).days / 30) >= months_threshold:
                finalb_dates_2.append(date)
        
        times2, pre2, prewin2, after2, afterwin2 = 0, 0, 0, 0, 0
        resulttable2, finalb2 = None, None
        if finalb_dates_2:
            dfs = []
            for dt in finalb_dates_2:
                try:
                    a = find_row_number_for_date(alldf, dt)
                    if a - timepast < 0 or a + timeforward >= len(alldf): continue
                    temp_df = alldf["index"].iloc[a - timepast : a + timeforward].to_frame(name=dt).reset_index()
                    dfs.append(temp_df)
                except KeyError: continue
            
            if dfs:
                df_concat = pd.concat(dfs, axis=1)
                data_cols = [col for j, col in enumerate(df_concat.columns) if j % 2 == 1]
                origin = df_concat[data_cols]
                finalb2 = origin.apply(lambda col: 100 * col / col.iloc[timepast])
                finalb2 = finalb2.iloc[:, -10:]
                finalb2["mean"] = finalb2.mean(axis=1)
                offsets = [-12, -6, 0, 6, 12]
                table2 = pd.concat([finalb2.iloc[timepast + off] for off in offsets], axis=1)
                table2.columns = [f"{off}m" for off in offsets]
                resulttable2 = table2.iloc[:-1]
                if not resulttable2.empty:
                    perc_df = pd.DataFrame([(resulttable2 > 100).mean() * 100], index=["勝率"])
                    resulttable2 = pd.concat([resulttable2, perc_df, table2.iloc[-1:]])
                    times2 = len(resulttable2.columns) - 1
                    pre2 = resulttable2.loc["mean", "-12m"] - 100
                    prewin2 = resulttable2.loc["勝率", "-12m"]
                    after2 = resulttable2.loc["mean", "12m"] - 100
                    afterwin2 = resulttable2.loc["勝率", "12m"]

        results.append({
            "series_id": variable_id, "mode": mode, "std": std_value, "winrolling": winrolling_value,
            "pre1": pre1, "prewin1": prewin1, "after1": after1, "afterwin1": afterwin1, "times1": times1,
            "pre2": pre2, "prewin2": prewin2, "after2": after2, "afterwin2": afterwin2, "times2": times2,
            "resulttable1": resulttable1, "resulttable2": resulttable2,
            "finalb1": finalb1.reset_index() if (finalb1 is not None and "mean" in finalb1.columns) else None,
            "finalb2": finalb2.reset_index() if (finalb2 is not None and "mean" in finalb2.columns) else None,
        })
    except Exception as e:
        print(f"ERROR during CALCULATION for series {variable_id}, params: {std_value}, {winrolling_value}, {mode}: {e}")
    return results

# ---------- Main Flow ----------
k = _need_api_key()

results_flat = run_all_combinations(
    selected_variable_id, selected_target_id, tuple(std_choices), tuple(roll_choices),
    tuple(modes), months_gap_threshold, k
)

if not results_flat:
    st.info("尚無可顯示結果。請調整參數或確認 series 有足夠歷史資料。")
    st.stop()

# ---------- Summary Table Generation ----------
def _to_float(x):
    try: return float(x)
    except (ValueError, TypeError): return None

def _to_int(x):
    try: return int(x)
    except (ValueError, TypeError): return 0

def classify_signal(pre, after, prewin, afterwin, times):
    vals = [pre, after, prewin, afterwin, times]
    if any(v is None for v in vals): return "🚫 不是有效訊號"
    if int(times) < 8: return "🚫 不是有效訊號"
    win_sum = float(prewin) + float(afterwin)
    if (float(pre) < 0 and float(after) < 0) and (win_sum < 70): return "🐮 牛市訊號"
    if (float(pre) > 0 and float(after) > 0) and (win_sum > 130): return "🐻 熊市訊號"
    return "🚫 不是有效訊號"

def compute_score(pre, after, prewin, afterwin, times):
    label = classify_signal(pre, after, prewin, afterwin, times)
    vals = [pre, after, prewin, afterwin]
    if any(v is None for v in vals): return 0.0
    pre, after, prewin, afterwin, times = float(pre), float(after), float(prewin), float(afterwin), int(times)
    if label.startswith("🐮"): return pre + after + (prewin - 50) + (afterwin - 50) + times
    elif label.startswith("🐻"): return -pre - after - (prewin - 50) - (afterwin - 50) + times
    else: return 0.0

summary_rows_raw, summary_rows_yoy = [], []
for r in results_flat:
    pre1, after1, prewin1, afterwin1, times1 = map(_to_float, [r["pre1"], r["after1"], r["prewin1"], r["afterwin1"], r["times1"]])
    summary_rows_raw.append({
        "系列": get_name_from_id(r["series_id"], str(r["series_id"])), "ID": r["series_id"], "std": r["std"], 
        "window": r["winrolling"], "觸發": r["mode"], "事件數": _to_int(times1), "前12m均值%": pre1, 
        "後12m均值%": after1, "勝率前": prewin1, "勝率後": afterwin1, 
        "得分": compute_score(pre1, after1, prewin1, afterwin1, times1), 
        "有效": classify_signal(pre1, after1, prewin1, afterwin1, times1),
    })
    pre2, after2, prewin2, afterwin2, times2 = map(_to_float, [r["pre2"], r["after2"], r["prewin2"], r["afterwin2"], r["times2"]])
    summary_rows_yoy.append({
        "系列": get_name_from_id(r["series_id"], str(r["series_id"])), "ID": r["series_id"], "std": r["std"], 
        "window": r["winrolling"], "觸發": r["mode"], "事件數": _to_int(times2), "前12m均值%": pre2, 
        "後12m均值%": after2, "勝率前": prewin2, "勝率後": afterwin2, 
        "得分": compute_score(pre2, after2, prewin2, afterwin2, times2), 
        "有效": classify_signal(pre2, after2, prewin2, afterwin2, times2),
    })

summary_raw_df = pd.DataFrame(summary_rows_raw).sort_values(by=["得分", "事件數"], ascending=False, na_position="last")
summary_yoy_df = pd.DataFrame(summary_rows_yoy).sort_values(by=["得分", "事件數"], ascending=False, na_position="last")

st.subheader("原始版本：所有 std × window × 觸發 組合結果")
st.info("點擊下方任一橫列 (row) 即可在頁面底部查看該組合的詳細圖表。")
st.dataframe(summary_raw_df, use_container_width=True, key="raw_table", on_select="rerun", selection_mode="single-row", height=350)

st.subheader("年增版本：所有 std × window × 觸發 組合結果")
st.info("點擊下方任一橫列 (row) 即可在頁面底部查看該組合的詳細圖表。")
st.dataframe(summary_yoy_df, use_container_width=True, key="yoy_table", on_select="rerun", selection_mode="single-row", height=350)

# ---------- Interactive Plotting Section ----------
st.divider()
st.subheader("互動式圖表結果")

selected_row_info = None
if "raw_table" in st.session_state and st.session_state.raw_table.selection.rows:
    selected_index = st.session_state.raw_table.selection.rows[0]
    selected_row = summary_raw_df.iloc[selected_index]
    selected_row_info = {'type': 'raw', 'params': selected_row}
elif "yoy_table" in st.session_state and st.session_state.yoy_table.selection.rows:
    selected_index = st.session_state.yoy_table.selection.rows[0]
    selected_row = summary_yoy_df.iloc[selected_index]
    selected_row_info = {'type': 'yoy', 'params': selected_row}

def show_selected_block(selected_info, all_results):
    if not selected_info:
        st.info("請點選上方任一表格的橫列以產生對應圖表。")
        return

    sel_type = selected_info['type']; sel_params = selected_info['params']
    title = "原始版本" if sel_type == 'raw' else "年增版本"
    result_key_table = "resulttable1" if sel_type == 'raw' else "resulttable2"
    result_key_curve = "finalb1" if sel_type == 'raw' else "finalb2"
    
    st.markdown(f"### {title}選擇組合：std = **{sel_params['std']}**, window = **{int(sel_params['window'])}**, 觸發 = **{sel_params['觸發']}**")
    
    full_result = next((r for r in all_results if r['std'] == sel_params['std'] and r['winrolling'] == sel_params['window'] and r['mode'] == sel_params['觸發']), None)

    if full_result:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 事件發生前後表現")
            table_data = full_result.get(result_key_table)
            if table_data is not None and not table_data.empty:
                st.dataframe(table_data, use_container_width=True)
            else:
                st.info("此組合無有效的事件可供分析。")
        with col2:
            st.markdown("##### 平均路徑圖")
            finalb_df = full_result.get(result_key_curve)
            if finalb_df is None or "mean" not in finalb_df.columns:
                st.info("無曲線資料可繪製。")
            else:
                y = finalb_df["mean"].values
                n = len(y); half = n // 2
                x = np.arange(-half, n - half)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(x, y, label=f"{title} Mean Curve")
                ax.axvline(0, color='r', linestyle='--')
                ax.axhline(100, color='gray', linestyle=':', linewidth=0.8)
                xlim = (-15, 15); ax.set_xlim(xlim)
                mask = (x >= xlim[0]) & (x <= xlim[1])
                if np.any(mask) and len(y_subset := y[mask]) > 0:
                    ymin, ymax = float(np.min(y_subset)) * 0.99, float(np.max(y_subset)) * 1.01
                    if ymin == ymax: ymin -= 1.0; ymax += 1.0
                    ax.set_ylim(ymin, ymax)
                ax.set_xlabel('事件發生前後月數'); ax.set_ylabel('指數 (100 = 事件當月)')
                ax.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig, use_container_width=True)
    else:
        st.error("找不到對應的詳細結果，請重新整理頁面。")

show_selected_block(selected_row_info, results_flat)

# ---------- Best Picks Section ----------
st.divider()
st.subheader("自動篩選最佳組合")
THRESHOLD_EVENTS = 8
st.caption(f"＊最佳組合挑選門檻：事件數 ≥ {THRESHOLD_EVENTS}。")

def pick_best(df: pd.DataFrame) -> pd.Series | None:
    if df is None or df.empty or "事件數" not in df.columns: return None
    cand = df[df["事件數"] >= THRESHOLD_EVENTS]
    return cand.iloc[0] if not cand.empty else None

best_raw = pick_best(summary_raw_df)
best_yoy = pick_best(summary_yoy_df)

def show_best_block(best_row, results_flat, title, result_key_table, result_key_curve):
    if best_row is None:
        st.info(f"{title}：沒有達到事件數門檻（≥ {THRESHOLD_EVENTS}）的組合。")
        return
    st.markdown(f"### {title}最佳組合：std = **{best_row['std']}**, window = **{int(best_row['window'])}**, 觸發 = **{best_row['觸發']}**")
    best_r = next((r for r in results_flat if r['std']==best_row['std'] and r['winrolling']==best_row['window'] and r['mode']==best_row['觸發']), None)
    col1, col2 = st.columns(2)
    with col1:
        table_data = best_r.get(result_key_table) if best_r else None
        if table_data is not None and not table_data.empty:
            st.dataframe(table_data, use_container_width=True)
        else:
            st.info("無表格資料。")
    with col2:
        finalb_df = best_r.get(result_key_curve) if best_r else None
        if finalb_df is None or "mean" not in finalb_df.columns:
            st.info(f"{title} 無曲線資料。")
        else:
            y, n = finalb_df["mean"].values, len(finalb_df)
            half = n//2
            x = np.arange(-half, n - half)
            fig, ax = plt.subplots(figsize=(6,5))
            ax.plot(x, y, label=title)
            ax.axvline(0, color='r', linestyle='--'); ax.axhline(100, color='gray', linestyle=':', linewidth=0.8)
            xlim = (-15, 15); ax.set_xlim(xlim)
            mask = (x>=xlim[0]) & (x<=xlim[1])
            if np.any(mask) and len(y_subset := y[mask]) > 0:
                ymin, ymax = float(np.min(y_subset))*0.99, float(np.max(y_subset))*1.01
                if ymin == ymax: ymin -= 1.0; ymax += 1.0
                ax.set_ylim(ymin, ymax)
            ax.set_xlabel('事件發生前後月數'); ax.set_ylabel('指數 (100 = 事件當月)')
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig, use_container_width=True)

show_best_block(best_raw, results_flat, "原始版本", "resulttable1", "finalb1")
show_best_block(best_yoy, results_flat, "年增版本", "resulttable2", "finalb2")

# ---------- Time Series Charts Section ----------
st.divider()
st.subheader("可互動的原始指標與年增率圖")
alt.data_transformers.disable_max_rows()
sigma_levels = [0.5, 1.0, 1.5, 2.0]
best_raw_window = int(best_raw['window']) if best_raw is not None else chart_winrolling_value
best_yoy_window = int(best_yoy['window']) if best_yoy is not None else chart_winrolling_value

sid = id_name_map[id_name_map['繁中名稱']==selected_variable_name]['ID'].iloc[0]
df_chart_source = mm(int(sid), "MS", f"series_{sid}", k)

if df_chart_source is None or df_chart_source.empty:
    st.warning(f"無法獲取指標 {selected_variable_name} ({sid}) 的圖表資料。")
else:
    s = df_chart_source.iloc[:,0].astype(float)
    
    def levels_chart_with_brush(s: pd.Series, name: str, winrolling_value: int):
        roll_mean, roll_std = s.rolling(winrolling_value).mean(), s.rolling(winrolling_value).std()
        df_levels = pd.DataFrame({"Date": s.index, "Level": s.values, "Mean": roll_mean.values})
        for m in sigma_levels:
            df_levels[f"Mean + {m}σ"] = (roll_mean + m * roll_std).values
            df_levels[f"Mean - {m}σ"] = (roll_mean - m * roll_std).values
        long_levels = df_levels.melt("Date", var_name="Series", value_name="Value").dropna()
        brush = alt.selection_interval(encodings=["x"])
        upper = alt.Chart(long_levels).mark_line().encode(
            x=alt.X("Date:T", title="Date"), y=alt.Y("Value:Q", title="Level"),
            color=alt.Color("Series:N", legend=alt.Legend(orient="top", title=None)),
            tooltip=["Date:T", "Series:N", alt.Tooltip("Value:Q", format=".2f")],
        ).transform_filter(brush).properties(title=f"{name} | {winrolling_value}期滾動平均", height=320)
        lower = alt.Chart(df_levels).mark_area(opacity=0.4).encode(x="Date:T", y="Level:Q").properties(height=60).add_selection(brush)
        return alt.vconcat(upper, lower).resolve_scale(y="independent")

    def yoy_chart_with_brush(s: pd.Series, name: str, winrolling_value: int):
        yoy = s.pct_change(12) * 100.0
        yoy_mean, yoy_std = yoy.rolling(winrolling_value).mean(), yoy.rolling(winrolling_value).std()
        df_yoy = pd.DataFrame({"Date": yoy.index, "YoY (%)": yoy.values, "Mean": yoy_mean.values})
        for m in sigma_levels:
            df_yoy[f"Mean + {m}σ"] = (yoy_mean + m * yoy_std).values
            df_yoy[f"Mean - {m}σ"] = (yoy_mean - m * yoy_std).values
        long_yoy = df_yoy.melt("Date", var_name="Series", value_name="Value").dropna()
        brush = alt.selection_interval(encodings=["x"])
        upper = alt.Chart(long_yoy).mark_line().encode(
            x=alt.X("Date:T", title="Date"), y=alt.Y("Value:Q", title="YoY (%)"),
            color=alt.Color("Series:N", legend=alt.Legend(orient="top", title=None)),
            tooltip=["Date:T", "Series:N", alt.Tooltip("Value:Q", format=".2f")],
        ).transform_filter(brush).properties(title=f"{name} | 年增率與{winrolling_value}期滾動平均", height=320)
        zero_line = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(strokeDash=[4,4]).encode(y="y:Q")
        lower = alt.Chart(df_yoy).mark_area(opacity=0.4).encode(x="Date:T", y="YoY (%):Q").properties(height=60).add_selection(brush)
        return alt.vconcat(upper + zero_line, lower).resolve_scale(y="independent")

    with st.expander(f"指標圖表: {selected_variable_name} ({sid})", expanded=True):
        colA, colB = st.columns(2)
        with colA:
            st.caption(f"使用最佳組合 window = {best_raw_window}")
            st.altair_chart(levels_chart_with_brush(s, selected_variable_name, best_raw_window), use_container_width=True)
        with colB:
            st.caption(f"使用最佳組合 window = {best_yoy_window}")
            st.altair_chart(yoy_chart_with_brush(s, selected_variable_name, best_yoy_window), use_container_width=True)
