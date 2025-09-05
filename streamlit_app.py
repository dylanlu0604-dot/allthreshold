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
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

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
    # 固定同時跑兩種觸發邏輯
    modes = ["Greater", "Smaller"]
    st.caption("觸發邏輯將同時評估：Greater 與 Smaller。")

    # 選擇變數與目標（以中文名稱）
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

    # 同時跑全部 std×window
    std_choices = [0.5, 1.0, 1.5, 2.0]
    roll_choices = [6, 12, 24, 36, 60, 120]

    months_gap_threshold = st.number_input("事件間隔（至少幾個月）", min_value=1, max_value=36, value=6)

    chart_winrolling_value = 120
    st.caption("圖表默認使用 window=120；若有最佳組合，將用各自最佳 window 繪製（原始/年增不同）。")

# ---------- Helpers ----------
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

        alldf = pd.concat([df1, df2], axis=1).resample("MS").asfreq().ffill()
        timeforward, timepast = 31, 31

        # 原始版
        df = alldf[[x1, x2]].copy()
        df["Rolling_mean"] = df["breath"].rolling(window=winrolling_value).mean()
        df["Rolling_std"] = df["breath"].rolling(window=winrolling_value).std()
        filtered_df = df[_condition(df, std_value, winrolling_value, mode)]

        finalb_dates_1: list[pd.Timestamp] = []
        for date in filtered_df.index:
            if not finalb_dates_1 or ((date - finalb_dates_1[-1]).days / 30) >= months_threshold:
                finalb_dates_1.append(date)

        if not finalb_dates_1:
            resulttable1 = None; finalb1 = None
            times1 = pre1 = prewin1 = after1 = afterwin1 = score1 = 0
        else:
            dfs = []
            for dt in finalb_dates_1:
                a = find_row_number_for_date(alldf, dt)
                if a - timepast < 0 or a + timeforward >= len(alldf):
                    continue
                temp_df = alldf["index"].iloc[a - timepast : a + timeforward].to_frame(name=dt).reset_index()
                dfs.append(temp_df)
            if not dfs:
                resulttable1 = None; finalb1 = None
                times1 = pre1 = prewin1 = after1 = afterwin1 = score1 = 0
            else:
                df_concat = pd.concat(dfs, axis=1)
                data_cols = [col for j, col in enumerate(df_concat.columns) if j % 2 == 1]
                origin = df_concat[data_cols]
                finalb1 = origin.apply(lambda col: 100 * col / col.iloc[timepast])
                finalb1 = finalb1[finalb1.columns[-10:]]
                finalb1["mean"] = finalb1.mean(axis=1)

                offsets = [-12, -6, 0, 6, 12]
                table1 = pd.concat([finalb1.iloc[timepast + off] for off in offsets], axis=1)
                table1.columns = [f"{off}m" for off in offsets]
                resulttable1 = table1.iloc[:-1]
                perc_df = pd.DataFrame([(resulttable1 > 100).mean() * 100], index=["勝率"])
                resulttable1 = pd.concat([resulttable1, perc_df, table1.iloc[-1:]])

                times1 = len(resulttable1) - 2
                pre1 = resulttable1.loc["mean", "-12m"] - 100 if "mean" in resulttable1.index else 0
                prewin1 = resulttable1.loc["勝率", "-12m"] if "勝率" in resulttable1.index else 0
                after1 = resulttable1.loc["mean", "12m"] - 100 if "mean" in resulttable1.index else 0
                afterwin1 = resulttable1.loc["勝率", "12m"] if "勝率" in resulttable1.index else 0

        # 年增版
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

        if not finalb_dates_2:
            resulttable2 = None; finalb2 = None
            times2 = pre2 = prewin2 = after2 = afterwin2 = score2 = 0
        else:
            dfs = []
            for dt in finalb_dates_2:
                a = find_row_number_for_date(alldf, dt)
                if a - timepast < 0 or a + timeforward >= len(alldf):
                    continue
                temp_df = alldf["index"].iloc[a - timepast : a + timeforward].to_frame(name=dt).reset_index()
                dfs.append(temp_df)
            if not dfs:
                resulttable2 = None; finalb2 = None
                times2 = pre2 = prewin2 = after2 = afterwin2 = score2 = 0
            else:
                df_concat = pd.concat(dfs, axis=1)
                data_cols = [col for j, col in enumerate(df_concat.columns) if j % 2 == 1]
                origin = df_concat[data_cols]
                finalb2 = origin.apply(lambda col: 100 * col / col.iloc[timepast])
                finalb2["mean"] = finalb2.mean(axis=1)

                offsets = [-12, -6, 0, 6, 12]
                table2 = pd.concat([finalb2.iloc[timepast + off] for off in offsets], axis=1)
                table2.columns = [f"{off}m" for off in offsets]
                resulttable2 = table2.iloc[:-1]
                perc_df = pd.DataFrame([(resulttable2 > 100).mean() * 100], index=["勝率"])
                resulttable2 = pd.concat([resulttable2, perc_df, table2.iloc[-1:]])

                times2 = len(resulttable2) - 2
                pre2 = resulttable2.loc["mean", "-12m"] - 100 if "mean" in resulttable2.index else 0
                prewin2 = resulttable2.loc["勝率", "-12m"] if "勝率" in resulttable2.index else 0
                after2 = resulttable2.loc["mean", "12m"] - 100 if "mean" in resulttable2.index else 0
                afterwin2 = resulttable2.loc["勝率", "12m"] if "勝率" in resulttable2.index else 0

        results.append({
            "series_id": variable_id,
            "mode": mode,
            "std": std_value,
            "winrolling": winrolling_value,
            "pre1": pre1, "prewin1": prewin1, "after1": after1, "afterwin1": afterwin1, "times1": times1,
            "pre2": pre2, "prewin2": prewin2, "after2": after2, "afterwin2": afterwin2, "times2": times2,
            "resulttable1": resulttable1 if resulttable1 is not None else None,
            "resulttable2": resulttable2 if resulttable2 is not None else None,
            "finalb1": finalb1.reset_index() if (finalb1 is not None and "mean" in finalb1.columns) else None,
            "finalb2": finalb2.reset_index() if (finalb2 is not None and "mean" in finalb2.columns) else None,
        })
    except Exception as e:
        st.write(f"Error during CALCULATION for series {variable_id}: {e}")
    return results

# ---------- Main Flow ----------
series_ids = [selected_variable_id]
k = _need_api_key()

combos = list(product(std_choices, roll_choices, modes))

if Parallel is not None:
    num_cores = max(1, min(4, multiprocessing.cpu_count()))
    tasks = [(sid, selected_target_id, s, w, k, m, months_gap_threshold) for sid in series_ids for (s,w,m) in combos]
    results_nested = Parallel(n_jobs=num_cores)(
        delayed(process_series)(t[0], t[1], t[2], t[3], t[4], t[5], t[6]) for t in tasks
    )
    results_flat = [item for sublist in results_nested for item in sublist]
else:
    st.warning("`joblib` 未安裝，改用單執行緒。")
    results_flat = []
    for sid in series_ids:
        for s, w, m in combos:
            results_flat.extend(process_series(sid, selected_target_id, s, w, k, m, months_gap_threshold))

if not results_flat:
    st.info("尚無可顯示結果。請調整參數或確認 series 有足夠歷史資料。")
    st.stop()

# ---------- Summary Tables (Raw & YoY) ----------
def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _to_int(x):
    try:
        return int(x)
    except Exception:
        return 0

def classify_signal(pre, after, prewin, afterwin, times):
    # 三分類
    vals = [pre, after, prewin, afterwin, times]
    if any(v is None for v in vals):
        return "🚫 不是有效訊號"
    try:
        pre = float(pre); after = float(after)
        prewin = float(prewin); afterwin = float(afterwin)
        times = int(times)
    except Exception:
        return "🚫 不是有效訊號"
    win_sum = prewin + afterwin
    if (pre < 0 and after < 0) and (times > 8) and (win_sum < 70):
        return "🐮 牛市訊號"
    if (pre > 0 and after > 0) and (times > 8) and (win_sum > 130):
        return "🐻 熊市訊號"
    return "🚫 不是有效訊號"

def compute_score(pre, after, prewin, afterwin, times):
    label = classify_signal(pre, after, prewin, afterwin, times)
    vals = [pre, after, prewin, afterwin]
    if any(v is None for v in vals):
        return 0.0
    pre = float(pre); after = float(after)
    prewin = float(prewin); afterwin = float(afterwin)
    times = int(times)
    if label.startswith("🐮"):
        return pre + after + (prewin - 50) + (afterwin - 50) + times
    elif label.startswith("🐻"):
        return -pre - after - (prewin - 50) - (afterwin - 50) + times
    else:
        return 0.0

summary_rows_raw = []
summary_rows_yoy = []

for r in results_flat:
    # 原始
    pre1 = _to_float(r.get("pre1")); after1 = _to_float(r.get("after1"))
    prewin1 = _to_float(r.get("prewin1")); afterwin1 = _to_float(r.get("afterwin1"))
    times1 = _to_int(r.get("times1"))
    label1 = classify_signal(pre1, after1, prewin1, afterwin1, times1)
    score1 = compute_score(pre1, after1, prewin1, afterwin1, times1)
    summary_rows_raw.append({
        "系列": get_name_from_id(r.get("series_id",-1), str(r.get("series_id",""))),
        "ID": r.get("series_id", None),
        "std": r.get("std", None),
        "window": r.get("winrolling", None),
        "觸發": r.get("mode", None),
        "事件數": times1,
        "前12m均值%": pre1,
        "後12m均值%": after1,
        "勝率前": prewin1,
        "勝率後": afterwin1,
        "得分": score1,
        "有效": label1,
    })

    # 年增
    pre2 = _to_float(r.get("pre2")); after2 = _to_float(r.get("after2"))
    prewin2 = _to_float(r.get("prewin2")); afterwin2 = _to_float(r.get("afterwin2"))
    times2 = _to_int(r.get("times2"))
    label2 = classify_signal(pre2, after2, prewin2, afterwin2, times2)
    score2 = compute_score(pre2, after2, prewin2, afterwin2, times2)
    summary_rows_yoy.append({
        "系列": get_name_from_id(r.get("series_id",-1), str(r.get("series_id",""))),
        "ID": r.get("series_id", None),
        "std": r.get("std", None),
        "window": r.get("winrolling", None),
        "觸發": r.get("mode", None),
        "事件數": times2,
        "前12m均值%": pre2,
        "後12m均值%": after2,
        "勝率前": prewin2,
        "勝率後": afterwin2,
        "得分": score2,
        "有效": label2,
    })

summary_raw_df = pd.DataFrame(summary_rows_raw)
summary_yoy_df = pd.DataFrame(summary_rows_yoy)

# ========== MODIFICATION START: Combine tables for interactive selection ==========

# Add a '版本' (Version) column to each dataframe
summary_raw_df['版本'] = '原始'
summary_yoy_df['版本'] = '年增'

# Concatenate the two dataframes
combined_df = pd.concat([summary_raw_df, summary_yoy_df], ignore_index=True)

# Format and sort the combined dataframe
for col in ["事件數","前12m均值%","後12m均值%","勝率前","勝率後","得分"]:
    if col in combined_df.columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

by_cols = [c for c in ["得分","事件數"] if c in combined_df.columns]
if by_cols:
    combined_df.sort_values(by=by_cols, ascending=False, na_position="last", inplace=True)

# Reorder columns to have '版本' at the beginning
first_cols = ['版本', '系列', 'ID', 'std', 'window', '觸發', '有效', '得分', '事件數']
other_cols = [col for col in combined_df.columns if col not in first_cols]
combined_df = combined_df[first_cols + other_cols]


st.subheader("所有組合結果分析")
st.caption("點選下方任一列表格，即可在下方區塊查看該組合的詳細數據與績效走勢圖。")

# Use st.dataframe with on_select to make it interactive
# Note: Using AgGrid for a better experience as it keeps the selection visually.
# If you don't have st_aggrid, you can replace this block with the st.dataframe version below.

gb = GridOptionsBuilder.from_dataframe(combined_df)
gb.configure_selection(selection_mode='single', use_checkbox=False)
gb.configure_column("前12m均值%", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=2)
gb.configure_column("後12m均值%", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=2)
gb.configure_column("勝率前", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=2)
gb.configure_column("勝率後", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=2)
gb.configure_column("得分", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=2)
gridOptions = gb.build()

grid_response = AgGrid(
    combined_df,
    gridOptions=gridOptions,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    allow_unsafe_jscode=True,
    height=400,
    width='100%',
    theme='streamlit' # or 'alpine', 'balham', 'material'
)

selected_rows = grid_response['selected_rows']

# --- Alternative using st.dataframe (less visually clear which row is selected) ---
# st.dataframe(
#     combined_df,
#     key="combined_selection",
#     on_select="rerun",
#     selection_mode="single-row",
#     use_container_width=True,
#     hide_index=True
# )
#
# selection = st.session_state.get("combined_selection")
# selected_rows = []
# if selection and selection['rows']:
#     idx = selection['rows'][0]
#     selected_rows.append(combined_df.iloc[idx].to_dict())
# ---------------------------------------------------------------------------------


st.divider()
st.subheader("選定組合的詳細結果")

if not selected_rows:
    st.info("請點選上方表格的任一列，以在此處查看詳細結果。")
else:
    # Get the data from the single selected row
    selected_row = pd.DataFrame(selected_rows).iloc[0]

    # Extract parameters from the selected row
    version = selected_row['版本']
    std_val = selected_row['std']
    win_val = selected_row['window']
    mode_val = selected_row['觸發']

    # Determine which result keys to use based on the 'version'
    if version == '原始':
        title_prefix = "原始版本"
        result_key_table = "resulttable1"
        result_key_curve = "finalb1"
    else: # '年增'
        title_prefix = "年增版本"
        result_key_table = "resulttable2"
        result_key_curve = "finalb2"

    # Find the matching full result dictionary from the original calculation
    matching_result = next((r for r in results_flat
                           if r.get('std') == std_val
                           and r.get('winrolling') == win_val
                           and r.get('mode') == mode_val), None)

    st.markdown(f"### {title_prefix}：std = **{std_val}**, window = **{int(win_val)}**, 觸發 = **{mode_val}**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### **績效摘要表**")
        if matching_result and matching_result.get(result_key_table) is not None:
            st.dataframe(matching_result[result_key_table].style.format("{:.2f}"), use_container_width=True)
        else:
            st.info("無表格資料可顯示。")

    with col2:
        st.markdown("##### **事件前後平均走勢**")
        finalb_df = matching_result.get(result_key_curve) if matching_result else None

        if finalb_df is None or "mean" not in finalb_df.columns:
            st.info("無曲線圖資料可顯示。")
        else:
            y = finalb_df["mean"].values
            n = len(y)
            half = n // 2
            x = np.arange(-half, n - half)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(x, y, label=title_prefix)
            ax.axvline(0, color='red', linestyle='--', linewidth=1)
            ax.axhline(100, color='gray', linestyle=':', linewidth=1)

            # Set a fixed view window for better comparison
            xlim = (-24, 24)
            ax.set_xlim(xlim)
            mask = (x >= xlim[0]) & (x <= xlim[1])
            if np.any(mask):
                ymin = float(np.min(y[mask])) * 0.98
                ymax = float(np.max(y[mask])) * 1.02
                if ymin == ymax: ymin -= 1.0; ymax += 1.0
                ax.set_ylim(ymin, ymax)

            ax.set_xlabel('相對於事件發生的月數')
            ax.set_ylabel('標準化指數 (事件當月 = 100)')
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig, use_container_width=True)

# ========== MODIFICATION END ==========


# Determine best windows for the charts below
# This logic can now be simplified or based on the combined table
best_raw = combined_df[(combined_df['版本'] == '原始') & (combined_df['事件數'] >= 8)].iloc[0] if not combined_df[(combined_df['版本'] == '原始') & (combined_df['事件數'] >= 8)].empty else None
best_yoy = combined_df[(combined_df['版本'] == '年增') & (combined_df['事件數'] >= 8)].iloc[0] if not combined_df[(combined_df['版本'] == '年增') & (combined_df['事件數'] >= 8)].empty else None

best_raw_window = int(best_raw['window']) if best_raw is not None else chart_winrolling_value
best_yoy_window = int(best_yoy['window']) if best_yoy is not None else chart_winrolling_value


st.divider()
st.subheader("指標原始數據與移動平均")
alt.data_transformers.disable_max_rows()

sigma_levels = [0.5, 1.0, 1.5, 2.0]

def levels_chart_with_brush(s: pd.Series, sid: int, name: str, winrolling_value: int):
    roll_mean = s.rolling(winrolling_value).mean()
    roll_std = s.rolling(winrolling_value).std()

    df_levels = pd.DataFrame({
        "Date": s.index,
        "Level": s.values,
        "Mean": roll_mean.values,
    })
    for m in sigma_levels:
        df_levels[f"+{m}σ"] = (roll_mean + m * roll_std).values
        df_levels[f"-{m}σ"] = (roll_mean - m * roll_std).values

    long_levels = df_levels.melt("Date", var_name="Series", value_name="Value").dropna()
    brush = alt.selection_interval(encodings=["x"])

    upper = (
        alt.Chart(long_levels)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Value:Q", title="Level"),
            color=alt.Color("Series:N", legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("Date:T"), "Series:N", alt.Tooltip("Value:Q", format=".2f")],
        )
        .transform_filter(brush)
        .properties(title=f"{name} ({sid}) | {winrolling_value}-period rolling mean ±σ", height=320)
    )
    lower = (
        alt.Chart(df_levels)
        .mark_area(opacity=0.4)
        .encode(x=alt.X("Date:T", title=""), y=alt.Y("Level:Q", title=""))
        .properties(height=60)
        .add_selection(brush)
    )
    return alt.vconcat(upper, lower).resolve_scale(y="independent")

def yoy_chart_with_brush(s: pd.Series, sid: int, name: str, winrolling_value: int):
    yoy = s / s.shift(12) # Use ratio for YoY calculation as in the original logic
    yoy.dropna(inplace=True)
    yoy_mean = yoy.rolling(winrolling_value).mean()
    yoy_std = yoy.rolling(winrolling_value).std()

    df_yoy = pd.DataFrame({
        "Date": yoy.index,
        "YoY": yoy.values,
        "Mean": yoy_mean.values,
    })
    for m in sigma_levels:
        df_yoy[f"+{m}σ"] = (yoy_mean + m * yoy_std).values
        df_yoy[f"-{m}σ"] = (yoy_mean - m * yoy_std).values

    long_yoy = df_yoy.melt("Date", var_name="Series", value_name="Value").dropna()
    brush = alt.selection_interval(encodings=["x"])

    upper = (
        alt.Chart(long_yoy)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Value:Q", title="YoY Ratio"),
            color=alt.Color("Series:N", legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("Date:T"), "Series:N", alt.Tooltip("Value:Q", format=".2f")],
        )
        .transform_filter(brush)
        .properties(title=f"{name} ({sid}) | YoY with {winrolling_value}-period rolling mean ±σ", height=320)
    )
    zero_line = alt.Chart(pd.DataFrame({"y":[1]})).mark_rule(strokeDash=[4,4]).encode(y="y:Q")
    lower = (
        alt.Chart(df_yoy)
        .mark_area(opacity=0.4)
        .encode(x=alt.X("Date:T", title=""), y=alt.Y("YoY:Q", title=""))
        .properties(height=60)
        .add_selection(brush)
    )
    return alt.vconcat(upper + zero_line, lower).resolve_scale(y="independent")

# Fetch series for charts
sid = id_name_map[id_name_map['繁中名稱']==selected_variable_name]['ID'].iloc[0]
df_target = mm(int(sid), "MS", f"series_{sid}", k)
if df_target is None or df_target.empty:
    st.info(f"No data for series {sid}, skipping.")
else:
    s = df_target.iloc[:,0].astype(float)
    with st.expander(f"圖表分析：{selected_variable_name} ({sid})", expanded=True):
        colA, colB = st.columns(2)
        with colA:
            st.caption(f"使用最佳原始版本 window = {best_raw_window}")
            st.altair_chart(levels_chart_with_brush(s, sid, selected_variable_name, best_raw_window), use_container_width=True)
        with colB:
            st.caption(f"使用最佳年增版本 window = {best_yoy_window}")
            st.altair_chart(yoy_chart_with_brush(s, sid, selected_variable_name, best_yoy_window), use_container_width=True)
