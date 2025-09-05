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

        def calculate_performance(df_orig, condition_df, alldf_ref):
            finalb_dates: list[pd.Timestamp] = []
            for date in condition_df.index:
                if not finalb_dates or ((date - finalb_dates[-1]).days / 30) >= months_threshold:
                    finalb_dates.append(date)

            if not finalb_dates:
                return None, None, 0, 0, 0, 0, 0

            dfs = []
            for dt in finalb_dates:
                a = find_row_number_for_date(alldf_ref, dt)
                if a - timepast < 0 or a + timeforward >= len(alldf_ref):
                    continue
                temp_df = alldf_ref["index"].iloc[a - timepast : a + timeforward].to_frame(name=dt).reset_index(drop=True)
                dfs.append(temp_df)
            
            if not dfs:
                 return None, None, 0, 0, 0, 0, 0

            origin = pd.concat(dfs, axis=1)
            finalb = origin.apply(lambda col: 100 * col / col.iloc[timepast])
            finalb["mean"] = finalb.mean(axis=1)

            offsets = [-12, -6, 0, 6, 12]
            table = pd.concat([finalb.iloc[timepast + off] for off in offsets], axis=1)
            table.columns = [f"{off}m" for off in offsets]
            resulttable = table.iloc[:-1].copy()
            perc_df = pd.DataFrame([(resulttable > 100).mean() * 100], index=["勝率"])
            resulttable = pd.concat([resulttable, perc_df, table.iloc[-1:]])

            times = len(resulttable.columns)
            pre = resulttable.loc["mean", "-12m"] - 100 if "mean" in resulttable.index else 0
            prewin = resulttable.loc["勝率", "-12m"] if "勝率" in resulttable.index else 0
            after = resulttable.loc["mean", "12m"] if "mean" in resulttable.index else 0
            afterwin = resulttable.loc["勝率", "12m"] if "勝率" in resulttable.index else 0
            return resulttable, finalb, times, pre, prewin, after, afterwin

        # 原始版
        df1_proc = alldf[[x1, x2]].copy()
        df1_proc["Rolling_mean"] = df1_proc[x1].rolling(window=winrolling_value).mean()
        df1_proc["Rolling_std"] = df1_proc[x1].rolling(window=winrolling_value).std()
        filtered_df1 = df1_proc[_condition(df1_proc, std_value, winrolling_value, mode)]
        resulttable1, finalb1, times1, pre1, prewin1, after1, afterwin1 = calculate_performance(df1_proc, filtered_df1, alldf)

        # 年增版
        df2_proc = alldf[[x1, x2]].copy()
        df2_proc[x1] = df2_proc[x1] / df2_proc[x1].shift(12)
        df2_proc.dropna(inplace=True)
        df2_proc["Rolling_mean"] = df2_proc[x1].rolling(window=winrolling_value).mean()
        df2_proc["Rolling_std"] = df2_proc[x1].rolling(window=winrolling_value).std()
        filtered_df2 = df2_proc[_condition(df2_proc, std_value, winrolling_value, mode)]
        resulttable2, finalb2, times2, pre2, prewin2, after2, afterwin2 = calculate_performance(df2_proc, filtered_df2, alldf)

        results.append({
            "series_id": variable_id, "mode": mode, "std": std_value, "winrolling": winrolling_value,
            "pre1": pre1, "prewin1": prewin1, "after1": after1, "afterwin1": afterwin1, "times1": times1,
            "pre2": pre2, "prewin2": prewin2, "after2": after2, "afterwin2": afterwin2, "times2": times2,
            "resulttable1": resulttable1, "resulttable2": resulttable2,
            "finalb1": finalb1.reset_index() if finalb1 is not None else None,
            "finalb2": finalb2.reset_index() if finalb2 is not None else None,
        })
    except Exception as e:
        st.write(f"Error for params {variable_id, mode, std_value, winrolling_value}: {e}")
    return results

k = _need_api_key()
combos = list(product([selected_variable_id], std_choices, roll_choices, modes))
tasks = [(sid, selected_target_id, s, w, k, m, months_gap_threshold) for sid, s, w, m in combos]

results_flat = []
if Parallel is not None:
    num_cores = min(4, multiprocessing.cpu_count())
    results_nested = Parallel(n_jobs=num_cores)(delayed(process_series)(*t) for t in tasks)
    results_flat = [item for sublist in results_nested for item in sublist]
else:
    st.warning("`joblib` not found. Running in single-thread mode.")
    for t in tasks:
        results_flat.extend(process_series(*t))

if not results_flat:
    st.info("No results to display. Adjust parameters or check data availability.")
    st.stop()

def classify_signal(pre, after, prewin, afterwin, times):
    if any(v is None for v in [pre, after, prewin, afterwin, times]) or times <= 8:
        return "🚫 不是有效訊號"
    win_sum = prewin + afterwin
    if pre < 0 and after < 0 and win_sum < 70: return "🐮 牛市訊號"
    if pre > 0 and after > 0 and win_sum > 130: return "🐻 熊市訊號"
    return "🚫 不是有效訊號"

def compute_score(pre, after, prewin, afterwin, times, label):
    if "不是有效訊號" in label: return 0.0
    if "🐮" in label: return pre + after + (prewin - 50) + (afterwin - 50) + times
    if "🐻" in label: return -pre - after - (prewin - 50) - (afterwin - 50) + times
    return 0.0

all_rows = []
for r in results_flat:
    for v_num, version in [(1, "原始"), (2, "年增")]:
        pre = r.get(f"pre{v_num}")
        after = r.get(f"after{v_num}")
        prewin = r.get(f"prewin{v_num}")
        afterwin = r.get(f"afterwin{v_num}")
        times = r.get(f"times{v_num}")
        label = classify_signal(pre, after, prewin, afterwin, times)
        score = compute_score(pre, after, prewin, afterwin, times, label)
        all_rows.append({
            "版本": version,
            "系列": get_name_from_id(r["series_id"], str(r["series_id"])),
            "std": r["std"], "window": r["winrolling"], "觸發": r["mode"],
            "事件數": times, "前12m均值%": pre, "後12m均值%": after,
            "勝率前": prewin, "勝率後": afterwin, "得分": score, "有效": label,
        })

combined_df = pd.DataFrame(all_rows).sort_values(by=["得分", "事件數"], ascending=False, na_position="last")
first_cols = ['版本', '系列', 'std', 'window', '觸發', '有效', '得分', '事件數']
other_cols = [c for c in combined_df.columns if c not in first_cols]
combined_df = combined_df[first_cols + other_cols]

st.subheader("所有組合結果分析")
st.caption("點選任一列，即可在下方查看該組合的詳細數據與績效走勢圖。")
st.dataframe(
    combined_df, key="selection", on_select="rerun", selection_mode="single-row",
    use_container_width=True, hide_index=True, height=400
)

selection = st.session_state.get("selection")
selected_row_data = combined_df.iloc[selection['rows'][0]] if selection and selection['rows'] else None

st.divider()
st.subheader("選定組合的詳細結果")

if selected_row_data is None:
    st.info("請點選上方表格的任一列以查看詳細結果。")
else:
    version, std_val, win_val, mode_val = selected_row_data[['版本', 'std', 'window', '觸發']]
    result_key_table = "resulttable1" if version == "原始" else "resulttable2"
    result_key_curve = "finalb1" if version == "原始" else "finalb2"
    
    matching_result = next((r for r in results_flat if r['std']==std_val and r['winrolling']==win_val and r['mode']==mode_val), None)

    st.markdown(f"### {version}版本：std = **{std_val}**, window = **{int(win_val)}**, 觸發 = **{mode_val}**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### **績效摘要表**")
        table_data = matching_result.get(result_key_table)
        if table_data is not None:
            st.dataframe(table_data.style.format("{:.2f}"), use_container_width=True)
        else:
            st.info("無表格資料。")
    with col2:
        st.markdown("##### **事件前後平均走勢**")
        curve_data = matching_result.get(result_key_curve)
        if curve_data is not None and "mean" in curve_data.columns:
            y, n = curve_data["mean"].values, len(curve_data)
            x = np.arange(-n//2, n - n//2)
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.axvline(0, color='r', linestyle='--'); ax.axhline(100, color='grey', linestyle=':')
            xlim = (-24, 24); ax.set_xlim(xlim)
            mask = (x >= xlim[0]) & (x <= xlim[1])
            if np.any(mask):
                ymin, ymax = np.min(y[mask]) * 0.98, np.max(y[mask]) * 1.02
                ax.set_ylim(ymin, ymax if ymax > ymin else ymax + 1)
            ax.set_xlabel('相對於事件的月數'); ax.set_ylabel('標準化指數 (事件月 = 100)')
            ax.grid(True, alpha=0.5)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("無曲線圖資料。")

st.divider()
st.subheader("指標原始數據與移動平均")
alt.data_transformers.disable_max_rows()
sigma_levels = [0.5, 1.0, 1.5, 2.0]

def create_altair_chart(s: pd.Series, name: str, win: int, is_yoy: bool):
    df = s.to_frame(name="value")
    if is_yoy:
        df["value"] = df["value"] / df["value"].shift(12)
        df.dropna(inplace=True)
        y_title = "YoY Ratio"
    else:
        y_title = "Level"
    
    df["mean"] = df["value"].rolling(win).mean()
    df["std"] = df["value"].rolling(win).std()
    for m in sigma_levels:
        df[f"+{m}σ"] = df["mean"] + m * df["std"]
        df[f"-{m}σ"] = df["mean"] - m * df["std"]
    
    df_long = df.reset_index().melt("date", var_name="series", value_name="val").dropna()
    brush = alt.selection_interval(encodings=["x"])
    
    upper = alt.Chart(df_long).mark_line().encode(
        x='date:T', y='val:Q', color='series:N', tooltip=['date:T', 'series:N', 'val:Q']
    ).transform_filter(brush).properties(title=f"{name} | {win}-period {'YoY' if is_yoy else 'Levels'}", height=300)
    
    lower = alt.Chart(df_long).mark_area(opacity=0.5).encode(
        x='date:T', y=alt.Y('val:Q', axis=None)
    ).add_selection(brush).properties(height=60)
    
    return alt.vconcat(upper, lower).resolve_scale(y="independent")

df_source = mm(selected_variable_id, "MS", "series", k)
if df_source is not None and not df_source.empty:
    s = df_source.iloc[:, 0].astype(float)
    best_raw = combined_df[(combined_df['版本'] == '原始') & (combined_df['事件數'] > 8)].iloc[0] if not combined_df[(combined_df['版本'] == '原始') & (combined_df['事件數'] > 8)].empty else None
    best_yoy = combined_df[(combined_df['版本'] == '年增') & (combined_df['事件數'] > 8)].iloc[0] if not combined_df[(combined_df['版本'] == '年增') & (combined_df['事件數'] > 8)].empty else None
    
    win_raw = int(best_raw['window']) if best_raw is not None else chart_winrolling_value
    win_yoy = int(best_yoy['window']) if best_yoy is not None else chart_winrolling_value

    with st.expander(f"圖表分析：{selected_variable_name} ({selected_variable_id})", expanded=True):
        colA, colB = st.columns(2)
        with colA:
            st.caption(f"使用最佳原始版本 window = {win_raw}")
            st.altair_chart(create_altair_chart(s, selected_variable_name, win_raw, is_yoy=False), use_container_width=True)
        with colB:
            st.caption(f"使用最佳年增版本 window = {win_yoy}")
            st.altair_chart(create_altair_chart(s, selected_variable_name, win_yoy, is_yoy=True), use_container_width=True)
else:
    st.info(f"No data for series {selected_variable_id}, skipping charts.")
