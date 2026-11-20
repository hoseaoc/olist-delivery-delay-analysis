from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Pipeline"))

from ingestion import ingest
from preprocessing import TEST_END, TEST_START, preprocess, test_cohort
from scoring import add_scores, score_day


OUTPUT = ROOT / "Pipeline" / "output"
MODEL = OUTPUT / "models.joblib"
METRICS = OUTPUT / "replay_metrics.csv"
DATA = ROOT / "Data"
INPUTS = [
    MODEL,
    DATA / "olist_orders_dataset.csv.zip",
    DATA / "olist_customers_dataset.csv.zip",
]
STRATEGIES = {
    "carrier": "Model + aturan kurir",
    "approval": "Model saat approval",
    "baseline": "Riwayat wilayah",
}
SCORE_COLUMN = {"baseline": "baseline_score", "approval": "risk_score", "carrier": "risk_score"}


st.set_page_config(page_title="Antrean Pengiriman Olist", page_icon="📦", layout="wide")
st.markdown(
    """
    <style>
    .block-container { max-width: 1160px; padding-top: 2.5rem; padding-bottom: 3rem; }
    [data-testid="stMetricValue"] { font-size: 1.85rem; font-weight: 650; }
    @media (max-width: 640px) {
        .block-container { padding-top: 2rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_scores(input_versions):
    df = preprocess(ingest(DATA))
    return add_scores(test_cohort(df), joblib.load(MODEL))


@st.cache_data(show_spinner=False, max_entries=64)
def get_queue(day, strategy, capacity, input_versions):
    df = load_scores(input_versions)
    return score_day(df, pd.date_range(TEST_START, day), strategy, capacity)


def format_int(value):
    return f"{int(value):,}".replace(",", ".")


def format_decimal(value):
    return f"{value:.1f}".replace(".", ",")


def delivery_table(df):
    return pd.DataFrame({
        "Order": df["order_id"],
        "Wilayah": df["state"],
        "Estimasi tiba": df["estimate_day"].dt.strftime("%d/%m/%Y"),
        "Catatan kurir": df["carrier_known"].map({True: "Sudah tercatat", False: "Belum tercatat"}),
    })


def risk_table(queue, strategy):
    table = delivery_table(queue)
    table.insert(0, "Prioritas", range(1, len(table) + 1))
    table["Sisa hari"] = queue["days_left"]
    table["Skor"] = queue[SCORE_COLUMN[strategy]].round(3)
    if strategy == "carrier":
        table["Alasan"] = queue["urgent"].map({
            True: "Tenggat ≤ 2 hari • belum tercatat ke kurir",
            False: "Skor model",
        })
    return table


st.title("Antrean Pengiriman Olist")
st.caption("Simulasi historis Olist Brasil · Juni–Agustus 2018")

missing = [path.name for path in [*INPUTS, METRICS] if not path.is_file()]
if missing:
    st.error(f"File yang belum tersedia — {', '.join(missing)}.")
    st.markdown("Siapkan data Olist di folder `Data` dan jalankan training dari folder proyek.")
    st.code(r".\.venv\Scripts\python.exe Pipeline/run.py train", language="powershell")
    st.stop()

date_col, capacity_col, strategy_col = st.columns([1, 1, 1.5])
with date_col:
    selected_date = st.date_input(
        "Tanggal simulasi",
        value=pd.Timestamp("2018-08-15").date(),
        min_value=TEST_START.date(),
        max_value=(TEST_END - pd.Timedelta(days=1)).date(),
        format="DD/MM/YYYY",
    )
with capacity_col:
    capacity = st.selectbox("Slot review per hari", [20, 50, 100])
with strategy_col:
    strategy = st.selectbox("Strategi prioritas", STRATEGIES, format_func=STRATEGIES.get)

versions = tuple((path.stat().st_mtime_ns, path.stat().st_size) for path in INPUTS)
with st.spinner("Menyiapkan antrean…"):
    queue, overdue = get_queue(pd.Timestamp(selected_date), strategy, capacity, versions)

st.subheader("Ringkasan harian")
risk_count, overdue_count = st.columns(2)
risk_count.metric("Prioritas review", format_int(len(queue)))
overdue_count.metric("Sudah lewat estimasi", format_int(len(overdue)))

risk_tab, overdue_tab = st.tabs(["Berisiko telat", "Sudah lewat estimasi"])
with risk_tab:
    st.caption("Order yang belum tiba dan masih sebelum tenggat. Order yang sudah dipilih pada hari sebelumnya tidak memakai slot lagi.")
    if queue.empty:
        st.info("Antrean review kosong pada tanggal ini.")
    else:
        st.dataframe(risk_table(queue, strategy), hide_index=True, width="stretch", height=382)
        st.caption("Skor dipakai untuk mengurutkan prioritas review.")
with overdue_tab:
    st.caption("Tanggal estimasi sudah berakhir dan paket belum tercatat diterima pada akhir hari simulasi.")
    if overdue.empty:
        st.info("Belum ada order yang lewat estimasi pada tanggal ini.")
    else:
        table = delivery_table(overdue)
        table["Hari lewat estimasi"] = -overdue["days_left"]
        st.dataframe(table, hide_index=True, width="stretch", height=382)

st.divider()
st.subheader("Hasil selama periode uji")
report = pd.read_csv(METRICS)
result = report.loc[report["policy"].eq(strategy) & report["daily_capacity"].eq(capacity)].iloc[0]
found, missed, warning = st.columns(3)
found.metric(
    "Telat tertangkap",
    f"{format_int(result['late_found'])} / {format_int(result['eligible_late'])}",
    help="Order yang akhirnya telat dan masuk antrean sebelum tanggal estimasi berakhir.",
)
missed.metric("Telat terlewat", format_int(result["late_missed"]))
warning.metric(
    "Median waktu peringatan",
    f"{format_decimal(result['median_days_before_deadline'])} hari",
    help="Median hari sebelum tenggat untuk kasus telat yang berhasil masuk antrean.",
)
st.caption("Ringkasan ini mencakup seluruh periode uji. Waktu pembatalan tidak tercatat di dataset.")
