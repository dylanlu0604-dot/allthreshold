import os
import time
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
OFFSETS = [-12, -6, 0, 6, 12] # 以「月」為單位

def _need_api_key() -> str:
    k = api_key or st.secrets.get("MACROMICRO_API_KEY", "") or os.environ.get("MACROMICRO_API_KEY", "")
    if not k:
        st.error("缺少 MacroMicro API Key。請在側邊欄輸入或於 .streamlit/secrets.toml 設定。")
        st.stop()
    return k

@st.cache_data(show_spinner=False, ttl=3600)
def mm(series_id: int, frequency: str, name: str, k: str) -> pd.DataFrame | None:
    """抓單一序列（月頻），回傳單欄 DataFrame；錯誤回 None。"""
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
    """
    根據選擇的觸發邏輯回傳布林條件：
    - Greater：過去6月最高 > 均值 + std*標準差
    - Smaller：過去6月最低 < 均值 - std*標準差
    """
    if mode == "Greater":
        return df["breath"].rolling(6).max() > df["Rolling_mean"] + std * df["Rolling_std"]
    else:
        return df["breath"].rolling(6).min() < df["Rolling_mean"] - std * df["Rolling_std"]

# 主分析（保留你的原流程，只把條件改為可切換）
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
        timeforward, timepast = 31, 31 # 定義 timepast 和 timeforward

        # ===== 第一段分析：原始 breath =====
        df = alldf[[x1, x2]].copy()
        df["Rolling_mean"] = df["breath"].rolling(window=winrolling_value).mean()
        df["Rolling_std"] = df["breath"].rolling(window=winrolling_value).std()
        filtered_df = df[_condition(df, std_value, winrolling_value, mode)]

        finalb_dates_1: list[pd.Timestamp] = []
        for date in filtered_df.index:
            if not finalb_dates_1 or ((date - finalb_dates_1[-1]).days / 30) >= months_threshold:
                finalb_dates_1.append(date)

        if not finalb_dates_1:
            resulttable1 = None
            finalb1 = None
            times1 = pre1 = prewin1 = after1 = afterwin1 = score1 = 0
            effective1 = "no"
        else:
            dfs = []
            for dt in finalb_dates_1:
                a = find_row_number_for_date(alldf, dt)
                if a - timepast < 0 or a + timeforward >= len(alldf):
                    continue
                temp_df = alldf["index"].iloc[a - timepast : a + timeforward].to_frame(name=dt).reset_index()
                dfs.append(temp_df)
            if not dfs:
                resulttable1 = None
                finalb1 = None
                times1 = pre1 = prewin1 = after1 = afterwin1 = score1 = 0
                effective1 = "no"
            else:
                df_concat = pd.concat(dfs, axis=1)
                data_cols = [col for j, col in enumerate(df_concat.columns) if j % 2 == 1]
                origin = df_concat[data_cols]
                finalb1 = origin.apply(lambda col: 100 * col / col.iloc[timepast])
                finalb1 = finalb1[finalb1.columns[-10:]] # 只保留最近 10 次事件
                finalb1["mean"] = finalb1.mean(axis=1)

                offsets = [-12, -6, 0, 6, 12]
                table1 = pd.concat([finalb1.iloc[timepast + off] for off in offsets], axis=1)
                table1.columns = [f"{off}m" for off in offsets] # 仍沿用 m 命名
                resulttable1 = table1.iloc[:-1]
                perc_df = pd.DataFrame([(resulttable1 > 100).mean() * 100], index=["勝率"])
                resulttable1 = pd.concat([resulttable1, perc_df, table1.iloc[-1:]])

                times1 = len(resulttable1) - 2
                pre1 = resulttable1.loc["mean", "-12m"] - 100 if "mean" in resulttable1.index else 0
                prewin1 = resulttable1.loc["勝率", "-12m"] if "勝率" in resulttable1.index else 0
                after1 = resulttable1.loc["mean", "12m"] - 100 if "mean" in resulttable1.index else 0
                afterwin1 = resulttable1.loc["勝率", "12m"] if "勝率" in resulttable1.index else 0
                score1 = after1 - pre1
                effective1 = "yes" if (pre1 > 0 and after1 > 0) or (pre1 < 0 and after1 < 0) and times1 > 10 else "no"

        # ===== 第二段分析：breath / breath.shift(12) =====
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
            resulttable2 = None
            finalb2 = None
            times2 = pre2 = prewin2 = after2 = afterwin2 = score2 = 0
            effective2 = "no"
        else:
            dfs = []
            for dt in finalb_dates_2:
                a = find_row_number_for_date(alldf, dt)
                if a - timepast < 0 or a + timeforward >= len(alldf):
                    continue
                temp_df = alldf["index"].iloc[a - timepast : a + timeforward].to_frame(name=dt).reset_index()
                dfs.append(temp_df)
            if not dfs:
                resulttable2 = None
                finalb2 = None
                times2 = pre2 = prewin2 = after2 = afterwin2 = score2 = 0
                effective2 = "no"
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
                score2 = after2 - pre2
                effective2 = "yes" if (pre2 > 0 and after2 > 0) or (pre2 < 0 and after2 < 0) and times2 > 10 else "no"

        results.append({
            "series_id": variable_id, "std": std_value, "winrolling": winrolling_value,
            "pre1": pre1, "prewin1": prewin1, "after1": after1, "afterwin1": afterwin1,
            "times1": times1, "effective1": effective1,
            "pre2": pre2, "prewin2": prewin2, "after2": after2, "afterwin2": afterwin2,
            "times2": times2, "effective2": effective2,
            "resulttable1": resulttable1 if resulttable1 is not None else None,
            "resulttable2": resulttable2 if resulttable2 is not None else None,
            "finalb1": finalb1.reset_index() if finalb1 is not None and "mean" in finalb1.columns else None,
            "finalb2": finalb2.reset_index() if finalb2 is not None and "mean" in finalb2.columns else None,
        })

    except Exception as e:
        st.write(f"Error during CALCULATION for series {variable_id}: {e}")
    return results


# ---------------------- Main Flow ----------------------

series_ids = [selected_variable_id] # 取得下拉式選單的 ID
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
summary_rows_raw = []
summary_rows_yoy = []
required_keys = ["series_id","std","winrolling","times1","pre1","prewin1","after1","afterwin1",
                 "times2","pre2","prewin2","after2","afterwin2"]

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

def _classify(pre, after, prewin, afterwin, times):
    # 三分類有效性邏輯（🐮/🐻/🚫）
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
        return "🐻 熊市訊號"
    if (pre > 0 and after > 0) and (times > 8) and (win_sum > 130):
        return "🐮 牛市訊號"
    return "🚫 不是有效訊號"

def compute_score(pre, after, prewin, afterwin, times):
    # 根據分類決定得分：
    # if 🐮：score = pre+after + (prewin-50) + (afterwin-50) + times
    # if 🐻：score = -pre-after - (prewin-50) - (afterwin-50) + times
    # else：0
    vals = [pre, after, prewin, afterwin]
    if any(v is None for v in vals):
        return 0.0
    try:
        pre = float(pre); after = float(after)
        prewin = float(prewin); afterwin = float(afterwin)
        times = int(times)
    except Exception:
        return 0.0
    label = _classify(pre, after, prewin, afterwin, times)
    if label.startswith("🐮"):
        return pre + after + (prewin - 50) + (afterwin - 50) + times
    elif label.startswith("🐻"):
        return -pre - after - (prewin - 50) - (afterwin - 50) + times
    else:
        return 0.0

for r in results_flat:
    missing = [k for k in required_keys if k not in r]
    if missing:
        st.warning(f"結果缺少欄位 {missing}，已以空值代替（std={r.get('std','?')}, window={r.get('winrolling','?')})")

    # 原始版
    pre1_val, after1_val = _to_float(r.get("pre1")), _to_float(r.get("after1"))
    prewin1_val, afterwin1_val = _to_float(r.get("prewin1")), _to_float(r.get("afterwin1"))
    times1_val = _to_int(r.get("times1"))
    score1_val = compute_score(pre1_val, after1_val, prewin1_val, afterwin1_val, times1_val)
    label1 = _classify(pre1_val, after1_val, prewin1_val, afterwin1_val, times1_val)

    summary_rows_raw.append({
        "系列": get_name_from_id(r.get("series_id", -1), str(r.get("series_id", ""))),
        "ID": r.get("series_id", None),
        "std": r.get("std", None),
        "window": r.get("winrolling", None),
        "事件數": times1_val,
        "前12m均值%": pre1_val,
        "後12m均值%": after1_val,
        "勝率前": prewin1_val,
        "勝率後": afterwin1_val,
        "得分": score1_val,
        "有效": label1,
    })

    # 年增版
    pre2_val, after2_val = _to_float(r.get("pre2")), _to_float(r.get("after2"))
    prewin2_val, afterwin2_val = _to_float(r.get("prewin2")), _to_float(r.get("afterwin2"))
    times2_val = _to_int(r.get("times2"))
    score2_val = compute_score(pre2_val, after2_val, prewin2_val, afterwin2_val, times2_val)
    label2 = _classify(pre2_val, after2_val, prewin2_val, afterwin2_val, times2_val)

    summary_rows_yoy.append({
        "系列": get_name_from_id(r.get("series_id", -1), str(r.get("series_id", ""))),
        "ID": r.get("series_id", None),
        "std": r.get("std", None),
        "window": r.get("winrolling", None),
        "事件數": times2_val,
        "前12m均值%": pre2_val,
        "後12m均值%": after2_val,
        "勝率前": prewin2_val,
        "勝率後": afterwin2_val,
        "得分": score2_val,
        "有效": label2,
    })

summary_raw_df = pd.DataFrame(summary_rows_raw)
summary_yoy_df = pd.DataFrame(summary_rows_yoy)

# 型別轉換（可排序）
for df in (summary_raw_df, summary_yoy_df):
    for col in ["事件數","前12m均值%","後12m均值%","勝率前","勝率後","得分"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

# 各自排序（先得分、再事件數）
if not summary_raw_df.empty:
    by_cols = [c for c in ["得分","事件數"] if c in summary_raw_df.columns]
    summary_raw_df = summary_raw_df.sort_values(by=by_cols, ascending=False, na_position="last")
if not summary_yoy_df.empty:
    by_cols = [c for c in ["得分","事件數"] if c in summary_yoy_df.columns]
    summary_yoy_df = summary_yoy_df.sort_values(by=by_cols, ascending=False, na_position="last")

st.subheader("原始版本：所有 std × window 組合結果")
st.dataframe(summary_raw_df, use_container_width=True)

st.subheader("年增版本：所有 std × window 組合結果")
st.dataframe(summary_yoy_df, use_container_width=True)









# 顯示最佳組合的細節（原始/年增 各一張）
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
    ax.axvline(0, linestyle='--')
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
    ax.set_xlabel('Months')
    ax.set_ylabel('Index (100 = 事件當月)')
    st.pyplot(fig, use_container_width=True)

THRESHOLD_EVENTS = 8
st.caption(f"＊最佳組合挑選門檻：事件數 ≥ {THRESHOLD_EVENTS}。")

# 原始最佳（只挑事件數>=8）
if 'summary_raw_df' in locals() and not summary_raw_df.empty:
    raw_candidates = summary_raw_df[summary_raw_df.get("事件數") >= THRESHOLD_EVENTS] if "事件數" in summary_raw_df.columns else summary_raw_df.iloc[0:0]
    if not raw_candidates.empty:
        best_raw = raw_candidates.iloc[0]
        st.markdown(f"### 原始版本最佳組合：std = **{best_raw['std']}**, window = **{int(best_raw['window'])}**")
        best_r_raw = next((r for r in results_flat if r.get('std')==best_raw['std'] and r.get('winrolling')==best_raw['window']), None)
        col1, col2 = st.columns(2)
        with col1:
            if best_r_raw and best_r_raw.get("resulttable1") is not None:
                st.dataframe(best_r_raw["resulttable1"], use_container_width=True)
            else:
                st.info("無原始值版本表格。")
        with col2:
            plot_mean_curve(best_r_raw.get("finalb1") if best_r_raw else None, "Final b1")
    else:
        st.info("原始版本：沒有達到事件數門檻（≥ 8）的組合可作為最佳結果。")

# 年增最佳（只挑事件數>=8）
if 'summary_yoy_df' in locals() and not summary_yoy_df.empty:
    yoy_candidates = summary_yoy_df[summary_yoy_df.get("事件數") >= THRESHOLD_EVENTS] if "事件數" in summary_yoy_df.columns else summary_yoy_df.iloc[0:0]
    if not yoy_candidates.empty:
        best_yoy = yoy_candidates.iloc[0]
        st.markdown(f"### 年增版本最佳組合：std = **{best_yoy['std']}**, window = **{int(best_yoy['window'])}**")
        best_r_yoy = next((r for r in results_flat if r.get('std')==best_yoy['std'] and r.get('winrolling')==best_yoy['window']), None)
        col3, col4 = st.columns(2)
        with col3:
            if best_r_yoy and best_r_yoy.get("resulttable2") is not None:
                st.dataframe(best_r_yoy["resulttable2"], use_container_width=True)
            else:
                st.info("無年增率版本表格。")
        with col4:
            plot_mean_curve(best_r_yoy.get("finalb2") if best_r_yoy else None, "Final b2")
    else:
        st.info("年增版本：沒有達到事件數門檻（≥ 8）的組合可作為最佳結果。")


# ===== Plot by series_ids_text: Levels & YoY

# --- 將最佳組合作為圖表的 rolling 參考（兩者可能不同） ---
best_raw_window = None
best_yoy_window = None
try:
    if 'summary_raw_df' in locals() and not summary_raw_df.empty:
        THRESHOLD_EVENTS = 8  # 與上方一致
        raw_candidates = summary_raw_df[summary_raw_df.get("事件數") >= THRESHOLD_EVENTS] if "事件數" in summary_raw_df.columns else summary_raw_df.iloc[0:0]
        if not raw_candidates.empty:
            best_raw_window = int(raw_candidates.iloc[0]['window'])
    if 'summary_yoy_df' in locals() and not summary_yoy_df.empty:
        THRESHOLD_EVENTS = 8
        yoy_candidates = summary_yoy_df[summary_yoy_df.get("事件數") >= THRESHOLD_EVENTS] if "事件數" in summary_yoy_df.columns else summary_yoy_df.iloc[0:0]
        if not yoy_candidates.empty:
            best_yoy_window = int(yoy_candidates.iloc[0]['window'])
except Exception as _:
    pass
# Fallback：若無最佳，使用原本 chart_winrolling_value
if 'chart_winrolling_value' in locals():
    if best_raw_window is None:
        best_raw_window = chart_winrolling_value
    if best_yoy_window is None:
        best_yoy_window = chart_winrolling_value
 # (brush to set x-range; y auto-rescales) =====
st.divider()
st.subheader("Each breath series: Levels (rolling mean ±σ) and YoY (brush to set time window)")

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
    # add ±σ bands
    for m in sigma_levels:
        df_levels[f"+{m}σ"] = (roll_mean + m * roll_std).values
        df_levels[f"-{m}σ"] = (roll_mean - m * roll_std).values

    # melt to long format
    long_levels = df_levels.melt("Date", var_name="Series", value_name="Value").dropna()

    # brush selection on x (time)
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
    yoy = s.pct_change(12) * 100.0
    yoy_mean = yoy.rolling(winrolling_value).mean()
    yoy_std = yoy.rolling(winrolling_value).std()

    df_yoy = pd.DataFrame({
        "Date": yoy.index,
        "YoY (%)": yoy.values,
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
            y=alt.Y("Value:Q", title="YoY (%)"),
            color=alt.Color("Series:N", legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("Date:T"), "Series:N", alt.Tooltip("Value:Q", format=".2f")],
        )
        .transform_filter(brush)
        .properties(title=f"{name} ({sid}) | YoY (%) with {winrolling_value}-period rolling mean ±σ", height=320)
    )

    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4,4]).encode(y="y:Q")

    lower = (
        alt.Chart(df_yoy)
        .mark_area(opacity=0.4)
        .encode(x=alt.X("Date:T", title=""), y=alt.Y("YoY (%):Q", title=""))
        .properties(height=60)
        .add_selection(brush)
    )

    return alt.vconcat(upper + zero_line, lower).resolve_scale(y="independent")

# 根據最佳組合決定各圖的 rolling 視窗：
# - Levels 使用原始版本最佳 window
# - YoY 使用年增版本最佳 window
winrolling_for_levels = best_raw_window if 'best_raw_window' in locals() and best_raw_window else chart_winrolling_value
winrolling_for_yoy = best_yoy_window if 'best_yoy_window' in locals() and best_yoy_window else chart_winrolling_value

# 根據名稱找到 ID
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
