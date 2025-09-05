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

        df = alldf[[x1, x2]].copy()
        df["Rolling_mean"] = df["breath"].rolling(window=winrolling_value).mean()
        df["Rolling_std"] = df["breath"].rolling(window=winrolling_value).std()
        filtered_df = df[_condition(df, std_value, winrolling_value, mode)]

        finalb_dates_1: list[pd.Timestamp] = []
        for date in filtered_df.index:
            if not finalb_dates_1 or ((date - finalb_dates_1[-1]).days / 30) >= months_threshold:
                finalb_dates_1.append(date)

        if not finalb_dates_1:
            return []

        dfs = []
        for dt in finalb_dates_1:
            a = df.index.get_loc(pd.Timestamp(dt))
            if a - 31 < 0 or a + 31 >= len(alldf):
                continue
            temp_df = alldf["index"].iloc[a - 31 : a + 31].to_frame(name=dt).reset_index()
            dfs.append(temp_df)
        if not dfs:
            return []

        df_concat = pd.concat(dfs, axis=1)
        data_cols = [col for j, col in enumerate(df_concat.columns) if j % 2 == 1]
        origin = df_concat[data_cols]
        finalb1 = origin.apply(lambda col: 100 * col / col.iloc[31])
        finalb1 = finalb1[finalb1.columns[-10:]]
        finalb1["mean"] = finalb1.mean(axis=1)

        results.append({
            "series_id": variable_id,
            "mode": mode,
            "std": std_value,
            "winrolling": winrolling_value,
            "finalb1": finalb1.reset_index() if finalb1 is not None else None,
        })
    except Exception as e:
        st.write(f"Error during CALCULATION for series {variable_id}: {e}")
    return results

# ---------- Main Flow ----------
series_ids = [selected_variable_id]
k = _need_api_key()

combos = list(product(std_choices, roll_choices, modes))

results_flat = []
for sid in series_ids:
    for s, w, m in combos:
        results_flat.extend(process_series(sid, selected_target_id, s, w, k, m, months_gap_threshold))

if not results_flat:
    st.info("尚無可顯示結果。請調整參數或確認 series 有足夠歷史資料。")
    st.stop()

# ---------- Summary Tables (Raw & YoY) ----------
summary_rows_raw = []
for r in results_flat:
    pre1 = r.get("pre1", 0)
    after1 = r.get("after1", 0)
    score1 = 0.0  # Placeholder for actual score calculation logic
    summary_rows_raw.append({
        "系列": get_name_from_id(r.get("series_id", -1), str(r.get("series_id", ""))),
        "ID": r.get("series_id", None),
        "std": r.get("std", None),
        "window": r.get("winrolling", None),
        "觸發": r.get("mode", None),
        "得分": score1,
    })

summary_raw_df = pd.DataFrame(summary_rows_raw)

# ---------- Interactive Selection and Chart Generation ----------

# Create a list of row labels from your summary dataframe
row_labels = summary_raw_df['系列'].tolist()

# Let the user select a row
selected_row = st.selectbox('選擇一個 row 來顯示對應的圖表', row_labels)

# Get the row corresponding to the selected label
selected_row_data = summary_raw_df[summary_raw_df['系列'] == selected_row].iloc[0]

# Extract std, window, and mode values from the selected row
selected_std = selected_row_data['std']
selected_window = selected_row_data['window']
selected_mode = selected_row_data['觸發']

# Now use the values of selected_std, selected_window, and selected_mode to plot the chart
def plot_chart_for_selected_row(selected_std, selected_window, selected_mode, sid, k):
    results = process_series(variable_id=selected_variable_id,
                             target_id=selected_target_id,
                             std_value=selected_std,
                             winrolling_value=selected_window,
                             k=k,
                             mode=selected_mode,
                             months_threshold=months_gap_threshold)

    if results:
        best_result = results[0]
        if best_result.get("finalb1") is not None:
            st.write(best_result["finalb1"])
            plot_mean_curve(best_result['finalb1'], f"Chart for {selected_row}")

plot_chart_for_selected_row(selected_std, selected_window, selected_mode, selected_variable_id, k)

# ---------- Chart Plotting Logic ----------
def plot_mean_curve(finalb_df, title):
    if finalb_df is None or "mean" not in finalb_df.columns:
        st.info(f"{title} 無曲線資料。"); return
    y = finalb_df["mean"].values
    n = len(y); half = n//2
    x = np.arange(-half, n - half)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(x, y, label=title)
    ax.axvline(0, linestyle='--')
    xlim = (-15, 15); ax.set_xlim(xlim)
    mask = (x>=xlim[0]) & (x<=xlim[1])
    if np.any(mask):
        ymin = float(np.min(y[mask]))*0.99; ymax = float(np.max(y[mask]))*1.01
        if ymin == ymax: ymin -= 1.0; ymax += 1.0
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Months'); ax.set_ylabel('Index (100 = 事件當月)')
    st.pyplot(fig, use_container_width=True)

# Display the dataframe and the interactive chart
st.subheader("選擇合適的 std, window, 和觸發邏輯來查看圖表")
st.dataframe(summary_raw_df, use_container_width=True)
