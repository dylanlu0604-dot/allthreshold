
with st.sidebar:
st.header("資料來源與參數設定")
    # 固定同時跑兩種觸發邏輯
modes = ["Greater", "Smaller"]
st.caption("觸發邏輯將同時評估：Greater 與 Smaller。")

    # 選擇變數與目標（以中文名稱）
variable_default_id = 617
variable_default_name = get_name_from_id(variable_default_id, "美國經濟領先指標")
selected_variable_name = st.selectbox("變數ID", options=series_names, index=series_names.index(variable_default_name))
@@ -68,15 +66,10 @@ def get_name_from_id(id_val, default_name):
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
@@ -103,9 +96,6 @@ def mm(series_id: int, frequency: str, name: str, k: str) -> pd.DataFrame | None
time.sleep(1)
return None

def find_row_number_for_date(df_obj: pd.DataFrame, specific_date: pd.Timestamp) -> int:
    return df_obj.index.get_loc(pd.Timestamp(specific_date))

def _condition(df: pd.DataFrame, std: float, winrolling: int, mode: str) -> pd.Series:
if mode == "Greater":
return df["breath"].rolling(6).max() > df["Rolling_mean"] + std * df["Rolling_std"]
@@ -125,9 +115,7 @@ def process_series(variable_id: int, target_id: int, std_value: float, winrollin
return results

alldf = pd.concat([df1, df2], axis=1).resample("MS").asfreq().ffill()
        timeforward, timepast = 31, 31

        # 原始版
df = alldf[[x1, x2]].copy()
df["Rolling_mean"] = df["breath"].rolling(window=winrolling_value).mean()
df["Rolling_std"] = df["breath"].rolling(window=winrolling_value).std()
@@ -139,98 +127,31 @@ def process_series(variable_id: int, target_id: int, std_value: float, winrollin
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
            "pre1": pre1, "prewin1": prewin1, "after1": after1, "afterwin1": afterwin1, "times1": times1,
            "pre2": pre2, "prewin2": prewin2, "after2": after2, "afterwin2": afterwin2, "times2": times2,
            "resulttable1": resulttable1 if resulttable1 is not None else None,
            "resulttable2": resulttable2 if resulttable2 is not None else None,
            "finalb1": finalb1.reset_index() if (finalb1 is not None and "mean" in finalb1.columns) else None,
            "finalb2": finalb2.reset_index() if (finalb2 is not None and "mean" in finalb2.columns) else None,
            "finalb1": finalb1.reset_index() if finalb1 is not None else None,
})
except Exception as e:
st.write(f"Error during CALCULATION for series {variable_id}: {e}")
@@ -242,285 +163,85 @@ def process_series(variable_id: int, target_id: int, std_value: float, winrollin

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
    pre1 = r.get("pre1", 0)
    after1 = r.get("after1", 0)
    score1 = 0.0  # Placeholder for actual score calculation logic
summary_rows_raw.append({
        "系列": get_name_from_id(r.get("series_id",-1), str(r.get("series_id",""))),
        "系列": get_name_from_id(r.get("series_id", -1), str(r.get("series_id", ""))),
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

for df in (summary_raw_df, summary_yoy_df):
    for col in ["事件數","前12m均值%","後12m均值%","勝率前","勝率後","得分"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 排序（得分、事件數）
    by_cols = [c for c in ["得分","事件數"] if c in df.columns]
    if by_cols:
        df.sort_values(by=by_cols, ascending=False, na_position="last", inplace=True)

st.subheader("原始版本：所有 std × window × 觸發 組合結果")
st.dataframe(summary_raw_df, use_container_width=True)

st.subheader("年增版本：所有 std × window × 觸發 組合結果")
st.dataframe(summary_yoy_df, use_container_width=True)

# ---------- Best picks (events ≥ 8) ----------
THRESHOLD_EVENTS = 8
st.caption(f"＊最佳組合挑選門檻：事件數 ≥ {THRESHOLD_EVENTS}。")

def pick_best(df: pd.DataFrame) -> pd.Series | None:
    if df is None or df.empty: return None
    if "事件數" not in df.columns: return None
    cand = df[df["事件數"] >= THRESHOLD_EVENTS]
    if cand.empty: return None
    return cand.iloc[0]

best_raw = pick_best(summary_raw_df)
best_yoy = pick_best(summary_yoy_df)

def show_best_block(best_row, results_flat, table_key, title, result_key_table, result_key_curve):
    if best_row is None:
        st.info(f"{title}：沒有達到事件數門檻（≥ {THRESHOLD_EVENTS}）的組合。")
        return
    st.markdown(f"### {title}最佳組合：std = **{best_row['std']}**, window = **{int(best_row['window'])}**, 觸發 = **{best_row['觸發']}**")
    best_r = next((r for r in results_flat
                   if r.get('std')==best_row['std']
                   and r.get('winrolling')==best_row['window']
                   and r.get('mode')==best_row['觸發']), None)
    col1, col2 = st.columns(2)
    with col1:
        if best_r and best_r.get(result_key_table) is not None:
            st.dataframe(best_r[result_key_table], use_container_width=True)
        else:
            st.info("無表格資料。")
    with col2:
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
        plot_mean_curve(best_r.get(result_key_curve) if best_r else None, title)

show_best_block(best_raw, results_flat, "raw", "原始版本", "resulttable1", "finalb1")
show_best_block(best_yoy, results_flat, "yoy", "年增版本", "resulttable2", "finalb2")

# ---------- Charts (use best windows if available) ----------
# Determine best windows
best_raw_window = int(best_raw['window']) if best_raw is not None else chart_winrolling_value
best_yoy_window = int(best_yoy['window']) if best_yoy is not None else chart_winrolling_value

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
        .properties(title=f"{name} ({sid}) | YoY with {winrolling_value}-period rolling mean ±σ", height=320)
    )
    zero_line = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(strokeDash=[4,4]).encode(y="y:Q")
    lower = (
        alt.Chart(df_yoy)
        .mark_area(opacity=0.4)
        .encode(x=alt.X("Date:T", title=""), y=alt.Y("YoY (%):Q", title=""))
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
    with st.expander(f"Series: {selected_variable_name} ({sid})", expanded=True):
        colA, colB = st.columns(2)
        with colA:
            st.caption(f"Levels rolling window = {best_raw_window}")
            st.altair_chart(levels_chart_with_brush(s, sid, selected_variable_name, best_raw_window), use_container_width=True)
        with colB:
            st.caption(f"YoY rolling window = {best_yoy_window}")
            st.altair_chart(yoy_chart_with_brush(s, sid, selected_variable_name, best_yoy_window), use_container_width=True)
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
