"""Chile GDP Explorer — Hispanic Heritage Month school project.

Deploy on Streamlit Community Cloud with the adjacent requirements.txt.
Historical source: World Bank WDI, NY.GDP.MKTP.CD, Chile (CHL).
"""

from datetime import date
from io import BytesIO
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# -----------------------------------------------------------------------------
# 1. PROJECT SETTINGS & VERIFIED PUBLIC DATA SOURCE
# -----------------------------------------------------------------------------
FIRST_YEAR = 1976
PREDICTION_YEAR = 2026
SOURCE_PAGE = "https://data.worldbank.org/indicator/NY.GDP.MKTP.CD?locations=CL"
API_ENDPOINT = "https://api.worldbank.org/v2/country/CHL/indicator/NY.GDP.MKTP.CD"
GDP_COLUMN = "GDP (billions of current US$)"
X_COLUMN = "x (years after 1976)"

st.set_page_config(
    page_title="Chile GDP Explorer | Hispanic Heritage Month",
    page_icon="🇨🇱",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {max-width: 1150px; padding-top: 1.5rem;}
        div[data-testid="stMetric"] {border: 1px solid #dce4ee; border-radius: 12px; padding: 12px;}
        h1, h2, h3 {letter-spacing: -0.015em;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_official_gdp():
    """Get actual annual GDP, not synthetic/generated values.

    The API gives USD; here we convert to billions of USD. We deliberately do
    not include year 2026 in fitting, because the assignment asks for a 2026
    forecast that is clearly separate from observed history.
    """
    response = requests.get(
        API_ENDPOINT,
        params={
            "format": "json",
            "date": f"{FIRST_YEAR}:{PREDICTION_YEAR}",
            "per_page": 100,
        },
        headers={"User-Agent": "ChileGDPExplorer/1.0 (school educational project)"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise ValueError("The World Bank API returned an unexpected data format.")

    rows = []
    for item in payload[1]:
        if not isinstance(item, dict) or item.get("value") is None:
            continue
        year = int(item["date"])
        if FIRST_YEAR <= year < PREDICTION_YEAR:
            usd = float(item["value"])
            if np.isfinite(usd) and usd >= 0:
                rows.append({"Year": year, GDP_COLUMN: usd / 1_000_000_000})

    if not rows:
        raise ValueError("No official Chile GDP observations were returned.")
    frame = (
        pd.DataFrame(rows)
        .drop_duplicates(subset="Year")
        .sort_values("Year")
        .reset_index(drop=True)
    )
    frame.insert(1, X_COLUMN, frame["Year"] - FIRST_YEAR)
    return frame


def fit_polynomial(data, degree):
    """Stable least-squares polynomial fit, returning diagnostics."""
    x = data[X_COLUMN].to_numpy(dtype=float)
    y = data[GDP_COLUMN].to_numpy(dtype=float)
    if len(x) <= degree or len(np.unique(x)) <= degree:
        raise ValueError("Not enough distinct data points for this polynomial degree.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("The working table contains a missing or invalid number.")
    # Polynomial.fit rescales x internally to improve numerical stability.
    poly = np.polynomial.Polynomial.fit(x, y, deg=degree).convert()
    predicted = np.asarray(poly(x), dtype=float)
    residuals = y - predicted
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    return poly, predicted, residuals, r2, rmse, mae


def printable_equation(poly, symbol="G", latex=False):
    """Turn numerical coefficients into readable calculus equations."""
    terms = []
    for p in range(len(poly.coef) - 1, -1, -1):
        c = float(poly.coef[p])
        # Only skip values genuinely negligible at display precision.
        if abs(c) < 1e-14:
            continue
        variable = "" if p == 0 else "x" if p == 1 else (f"x^{{{p}}}" if latex else f"x^{p}")
        magnitude = f"{abs(c):.7g}"
        term = f"{magnitude}{variable}"
        sign = "-" if c < 0 else "+"
        terms.append((sign, term))
    if not terms:
        result = "0"
    else:
        first_sign, first_term = terms[0]
        result = ("-" if first_sign == "-" else "") + first_term
        for sign, term in terms[1:]:
            result += f" {sign} {term}"
    left = f"{symbol}(x)"
    return f"{left} = {result}"


def to_csv(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")


def data_kind(year, latest_year):
    if year > latest_year:
        return "PREDICTION / EXTRAPOLATION — not observed"
    return "FITTED VALUE — not the original observation"


def make_graph(raw, working, poly, degree, chosen_year, extra_slopes, edited):
    """Create a Plotly graph with moving-model-dot and tangent (slope) lines."""
    figure = go.Figure()
    latest = int(raw["Year"].max())

    figure.add_trace(
        go.Scatter(
            x=raw["Year"], y=raw[GDP_COLUMN], mode="markers",
            name="Official World Bank observations",
            marker={"color": "#1765a3", "size": 7},
            hovertemplate="Observed %{x}: $%{y:,.2f}B<extra></extra>",
        )
    )
    if edited:
        figure.add_trace(
            go.Scatter(
                x=working["Year"], y=working[GDP_COLUMN], mode="markers",
                name="Edited scenario (not official)",
                marker={"color": "#ed9c25", "size": 8, "symbol": "diamond"},
            )
        )

    observed_years = np.linspace(FIRST_YEAR, latest, 350)
    figure.add_trace(
        go.Scatter(
            x=observed_years,
            y=poly(observed_years - FIRST_YEAR),
            mode="lines", name=f"Degree-{degree} polynomial fit",
            line={"color": "#c63c51", "width": 3},
            hovertemplate="Fitted %{x:.1f}: $%{y:,.2f}B<extra></extra>",
        )
    )
    # Distinguish forecast from the fitted historical line.
    future_years = np.linspace(latest, PREDICTION_YEAR, 60)
    figure.add_trace(
        go.Scatter(
            x=future_years,
            y=poly(future_years - FIRST_YEAR), mode="lines",
            name="Extrapolation to 2026 (not observed)",
            line={"color": "#9a43b2", "width": 3, "dash": "dash"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[PREDICTION_YEAR],
            y=[float(poly(PREDICTION_YEAR - FIRST_YEAR))], mode="markers",
            name="2026 MODEL ESTIMATE",
            marker={"color": "#9a43b2", "size": 15, "symbol": "star"},
            hovertemplate="2026 MODEL ESTIMATE: $%{y:,.2f}B<extra></extra>",
        )
    )

    tangent_centers = [chosen_year] + [int(year) for year in extra_slopes if int(year) != chosen_year]
    derivative = poly.deriv()
    for center in tangent_centers:
        center_x = center - FIRST_YEAR
        center_y = float(poly(center_x))
        slope = float(derivative(center_x))
        tangent_years = np.linspace(max(FIRST_YEAR, center - 6), min(PREDICTION_YEAR, center + 6), 35)
        tangent = center_y + slope * (tangent_years - center)
        figure.add_trace(
            go.Scatter(
                x=tangent_years, y=tangent, mode="lines",
                name=f"Tangent/slope at {center} ({slope:+.2f} B/yr)",
                line={"color": "#128770" if center == chosen_year else "#80938c", "width": 3 if center == chosen_year else 1.7, "dash": "dash"},
            )
        )

    chosen_value = float(poly(chosen_year - FIRST_YEAR))
    figure.add_trace(
        go.Scatter(
            x=[chosen_year], y=[chosen_value], mode="markers",
            name="Moving dot (use the year slider)",
            marker={"color": "#ffc247", "size": 16, "line": {"color": "#30333c", "width": 2}},
            hovertemplate="Moving dot: %{x}<br>Model: $%{y:,.2f}B<extra></extra>",
        )
    )
    figure.add_vline(x=latest, line_color="#8f969f", line_dash="dot", opacity=0.8)
    figure.update_layout(
        xaxis_title="Year (input)",
        yaxis_title="GDP (billions of current US$, output)",
        legend={"orientation": "h", "y": -0.27, "x": 0},
        margin={"l": 15, "r": 15, "t": 20, "b": 130},
        hovermode="closest", height=540,
    )
    return figure


def make_pdf_chart(raw, poly, last_year):
    """Dependency-light vector chart for the PDF; no Kaleido/Chrome required."""
    drawing = Drawing(485, 215)
    left, bottom, width, height = 48, 31, 418, 163
    ys = list(raw[GDP_COLUMN])
    curve_years = np.linspace(FIRST_YEAR, PREDICTION_YEAR, 180)
    curve_values = np.asarray(poly(curve_years - FIRST_YEAR))
    ymax = max(1.0, float(np.max(ys)), float(np.max(curve_values))) * 1.08
    ymin = min(0.0, float(np.min(curve_values)))
    if ymax - ymin < 1e-5:
        ymax += 1

    def xp(y):
        return left + (float(y) - FIRST_YEAR) / (PREDICTION_YEAR - FIRST_YEAR) * width

    def yp(v):
        return bottom + (float(v) - ymin) / (ymax - ymin) * height

    for i in range(5):
        val = ymin + i * (ymax - ymin) / 4
        yy = yp(val)
        drawing.add(Line(left, yy, left + width, yy, strokeColor=colors.HexColor("#e5e7eb")))
        drawing.add(String(left - 6, yy - 3, f"{val:.0f}", fontSize=7, textAnchor="end"))
    drawing.add(Line(left, bottom, left, bottom + height, strokeColor=colors.black))
    drawing.add(Line(left, bottom, left + width, bottom, strokeColor=colors.black))
    for year in (1976, 1986, 1996, 2006, 2016, 2026):
        drawing.add(String(xp(year), 16, str(year), fontSize=7, textAnchor="middle"))
    points = []
    for year, value in zip(curve_years, curve_values):
        points += [xp(year), yp(value)]
    drawing.add(PolyLine(points, strokeColor=colors.HexColor("#c63c51"), strokeWidth=1.8))
    for _, row in raw.iterrows():
        drawing.add(Circle(xp(row["Year"]), yp(row[GDP_COLUMN]), 1.6, fillColor=colors.HexColor("#1765a3"), strokeColor=None))
    drawing.add(Circle(xp(PREDICTION_YEAR), yp(poly(PREDICTION_YEAR - FIRST_YEAR)), 3.2, fillColor=colors.HexColor("#9a43b2"), strokeColor=None))
    drawing.add(String(left + 4, 205, "World Bank observed GDP (blue) / polynomial curve (red) / 2026 estimate (purple)", fontSize=8))
    return drawing


def pdf_report(raw, working, poly, r2, rmse, mae, degree, edited, predictions):
    """Create a downloadable PDF report: equations, chart, data, caveats."""
    out = BytesIO()
    doc = SimpleDocTemplate(
        out, pagesize=letter, leftMargin=43, rightMargin=43,
        topMargin=43, bottomMargin=43,
    )
    styles = getSampleStyleSheet()
    report = []

    def paragraph(value, style="Normal"):
        report.append(Paragraph(escape(str(value)), styles[style]))

    last_year = int(raw["Year"].max())
    paragraph("Chile GDP Explorer: 1976–2026", "Title")
    paragraph("Hispanic Heritage Month | Historical economic indicators and calculus")
    report.append(Spacer(1, 12))
    paragraph(f"Source: World Bank, WDI indicator NY.GDP.MKTP.CD. Retrieved: {date.today().isoformat()}.")
    paragraph(f"Observed records: {len(raw)}; latest observed year in model: {last_year}.")
    paragraph("Unit: billions of current US dollars (nominal GDP; not net worth).")
    paragraph("2026 IS A MODEL PREDICTION, NOT HISTORICAL OBSERVED DATA.", "Heading3")
    paragraph(f"Working table changed: {'YES - scenario values' if edited else 'No - original records'}.")
    paragraph("Polynomial and calculus", "Heading2")
    paragraph(f"x = year - {FIRST_YEAR}; polynomial degree = {degree}.")
    paragraph(printable_equation(poly))
    paragraph(printable_equation(poly.deriv(), "G'"))
    paragraph(printable_equation(poly.deriv(2), "G''"))
    paragraph(f"R-squared = {r2:.5f}; RMSE = {rmse:.3f} billion USD; MAE = {mae:.3f} billion USD.")
    paragraph(f"2026 polynomial estimate = ${float(poly(PREDICTION_YEAR - FIRST_YEAR)):,.3f} billion USD.")
    report.append(Spacer(1, 12))
    paragraph("GDP observations and the fitted curve", "Heading2")
    report.append(make_pdf_chart(raw, poly, last_year))

    paragraph("Selected-year model results", "Heading2")
    pred_rows = [["Year", "Model GDP ($B)", "Classification"]]
    for _, row in predictions.iterrows():
        pred_rows.append([
            str(int(row["Year"])), f"{row['Model GDP ($B)']:,.2f}",
            "Prediction" if int(row["Year"]) > last_year else "Fitted value",
        ])
    pred_tbl = Table(pred_rows, colWidths=[65, 125, 280], repeatRows=1)
    pred_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f4")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    report.append(pred_tbl)
    report.append(Spacer(1, 12))

    paragraph("Original official World Bank GDP data", "Heading2")
    paragraph("Input = year / x (years after 1976); output = GDP in billions of current USD.")
    rows = [["Year (input)", "x", "GDP $B (output)"]]
    for _, row in raw.iterrows():
        rows.append([str(int(row["Year"])), str(int(row[X_COLUMN])), f"{row[GDP_COLUMN]:,.3f}"])
    table = LongTable(rows, colWidths=[130, 95, 185], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cdd6df")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    report.append(table)
    report.append(Spacer(1, 12))
    paragraph("Interpretation, history, and uncertainty", "Heading2")
    paragraph("This is GDP, the annual value of goods and services produced, not the net worth of Chile. Nominal GDP in USD changes with inflation, economic activity and exchange rates. Changes in copper markets, recessions, and other historical events may affect GDP, but the regression curve cannot establish their causes.")
    paragraph("The historical modeling window begins in 1976 because the assignment selected that year, not because that is when Chile's economy began. The original World Bank values are preserved; any edits are clearly labeled as scenarios.")
    paragraph("Polynomial models can overfit and extrapolate unrealistically. R-squared, RMSE and MAE here measure fit to the data used to train the model, not tested forecast accuracy. The 2026 value is not an official forecast.")
    paragraph(f"Official dataset: {SOURCE_PAGE}")
    doc.build(report)
    return out.getvalue()


# -----------------------------------------------------------------------------
# 2. APP INTERFACE / SOURCE DATA
# -----------------------------------------------------------------------------
st.title("🇨🇱 Chile GDP Explorer")
st.caption("Hispanic Heritage Month • World Bank historical data • Cubic regression and calculus • 1976–2026")
st.info(
    "**Important:** This project studies Chile's **GDP** (gross domestic product), "
    "**not national net worth**. Real World Bank values are shown as observations; "
    "**2026 is a mathematical estimate** and is never presented as an observed value."
)

with st.sidebar:
    st.header("Model controls")
    degree = st.select_slider("Polynomial degree (3 or higher)", options=[3, 4, 5, 6], value=3)
    st.caption("Degree 3 (cubic) satisfies the assignment. Higher degrees may overfit.")
    if st.button("Reset edited working table", use_container_width=True):
        st.session_state["editor_reset_counter"] = st.session_state.get("editor_reset_counter", 0) + 1
        st.rerun()
    if st.button("Refresh historical data", use_container_width=True):
        fetch_official_gdp.clear()
        st.rerun()
    st.markdown(f"[World Bank official source]({SOURCE_PAGE})")

try:
    raw = fetch_official_gdp()
except (requests.RequestException, ValueError, KeyError, TypeError, OverflowError) as error:
    st.error("Could not load historical GDP from the World Bank. No fictional data will be substituted.")
    st.caption(f"Technical details: {error}")
    st.markdown(f"[Check the official World Bank data]({SOURCE_PAGE}). Check your internet connection and press **Refresh historical data**.")
    st.stop()

latest = int(raw["Year"].max())
if len(raw) < 7:
    st.error("The source returned too few years to fit a degree-3-or-higher polynomial.")
    st.stop()
missing = sorted(set(range(FIRST_YEAR, latest + 1)) - set(raw["Year"]))
if missing:
    st.warning("Some years have no reported GDP: " + ", ".join(map(str, missing)) + ". The app will not fill them with invented numbers.")

st.header("1 · Verified historical data")
st.markdown(
    f"The World Bank's **Chile GDP (current US$)** data uses indicator `NY.GDP.MKTP.CD`. "
    f"The app loads all available observations from **1976 through {latest}**. "
    "Amounts are divided by 1 billion to make graphs and equations easier to read. "
    f"[Open the original dataset]({SOURCE_PAGE})."
)
col1, col2, col3 = st.columns(3)
col1.metric("First year", str(int(raw["Year"].min())))
col2.metric("Latest observed year", str(latest))
col3.metric("Official observations", str(len(raw)))
st.markdown("**Raw table — inputs:** Year and `x = Year − 1976`; **output:** GDP (billions of current US$).")
st.dataframe(raw, hide_index=True, use_container_width=True, height=260)
st.download_button("Download untouched World Bank data (CSV)", to_csv(raw), "chile_gdp_raw.csv", "text/csv")

st.header("2 · Editable working data")
st.write(
    "You can change GDP values to explore 'what-if' scenarios. The original table above "
    "stays untouched, and **edited numbers are not historically verified facts**. "
    "The polynomial updates using the working table."
)
reset_count = st.session_state.get("editor_reset_counter", 0)
editor = st.data_editor(
    raw[["Year", GDP_COLUMN]],
    key=f"working_data_{reset_count}",
    num_rows="fixed", hide_index=True, use_container_width=True,
    disabled=["Year"],
    column_config={
        "Year": st.column_config.NumberColumn("Year (input)", format="%d"),
        GDP_COLUMN: st.column_config.NumberColumn("GDP ($B, output)", format="%.3f", min_value=0.0),
    },
)
working = editor.copy()
working[GDP_COLUMN] = pd.to_numeric(working[GDP_COLUMN], errors="coerce")
working[X_COLUMN] = working["Year"] - FIRST_YEAR
working = working[["Year", X_COLUMN, GDP_COLUMN]]
values = working[GDP_COLUMN].to_numpy(dtype=float)
if not np.isfinite(values).all() or (values < 0).any():
    st.error("Enter a nonnegative, finite number for each GDP value before fitting a polynomial.")
    st.stop()
edited = not np.allclose(values, raw[GDP_COLUMN].to_numpy(dtype=float), rtol=0, atol=1e-8)
if edited:
    st.warning("**Scenario mode:** the model is fitted to edited working values, not exclusively to official GDP data.")
st.download_button("Download working values (CSV)", to_csv(working), "chile_gdp_working.csv", "text/csv")

try:
    model, fitted_values, residuals, r2, rmse, mae = fit_polynomial(working, degree)
except (ValueError, np.linalg.LinAlgError) as error:
    st.error(f"Polynomial fitting could not complete: {error}")
    st.stop()

st.header("3 · Polynomial regression and calculus")
st.markdown("**Model:** `G(x) ≈ ax³ + bx² + cx + d`, where `x = year − 1976` and G(x) is GDP in billions of USD.")
st.latex(printable_equation(model, r"\widehat{G}", latex=True))
st.markdown("**First derivative:** predicted annual GDP change in billions of USD/year")
st.latex(printable_equation(model.deriv(), r"\widehat{G}'", latex=True))
st.markdown("**Second derivative:** change in the rate of GDP growth, billions of USD/year²")
st.latex(printable_equation(model.deriv(2), r"\widehat{G}''", latex=True))
st.caption("Coefficients are rounded to seven significant digits for display. Predictions use full-precision coefficients.")
with st.expander("See polynomial coefficients and their powers"):
    st.dataframe(
        pd.DataFrame({"Power of x": np.arange(len(model.coef)), "Coefficient (full precision)": model.coef}),
        hide_index=True, use_container_width=True,
    )

st.header("4 · Model quality")
rcol1, rcol2, rcol3, rcol4 = st.columns(4)
rcol1.metric("R² (in-sample)", f"{r2:.4f}" if np.isfinite(r2) else "Undefined")
rcol2.metric("RMSE", f"${rmse:,.2f}B")
rcol3.metric("MAE", f"${mae:,.2f}B")
forecast_2026 = float(model(PREDICTION_YEAR - FIRST_YEAR))
rcol4.metric("2026 model estimate", f"${forecast_2026:,.2f}B")
st.caption(
    "R² explains the proportion of variation accounted for in the training data. "
    "RMSE and MAE summarize historical fitting errors; they do not prove the 2026 prediction is accurate."
)
if forecast_2026 < 0:
    st.error("The polynomial predicts a negative GDP for 2026, which is not economically meaningful. This shows a limitation of the chosen model.")

st.header("5 · Interactive chart, moving year dot, and slope lines")
st.write("**Drag the slider's handle** to move the gold dot along the graph, change the displayed year and GDP, and rotate the green tangent (slope) line. Hover over blue dots for actual historical GDP.")
chosen_year = st.slider("Move the cursor to choose a year", FIRST_YEAR, PREDICTION_YEAR, PREDICTION_YEAR, 1)
extra_years = st.multiselect("Optional: show extra slope lines at these years", options=[1985, 1995, 2005, 2015, 2025], default=[])
selected_x = chosen_year - FIRST_YEAR
selected_model_gdp = float(model(selected_x))
selected_slope = float(model.deriv()(selected_x))
selected_acceleration = float(model.deriv(2)(selected_x))
sc1, sc2, sc3 = st.columns(3)
sc1.metric("Cursor year", str(chosen_year))
sc2.metric("Model GDP at cursor", f"${selected_model_gdp:,.2f}B")
sc3.metric("Slope at cursor", f"{selected_slope:+,.2f}B / year")
actual_at_cursor = raw.loc[raw["Year"] == chosen_year, GDP_COLUMN]
if not actual_at_cursor.empty:
    st.caption(f"**Observed value {chosen_year}:** ${float(actual_at_cursor.iloc[0]):,.2f}B. The moving dot is the polynomial's fitted value, not this raw observation.")
else:
    st.warning(f"There is no official observation in this dataset for {chosen_year}. The moving dot is a MODEL ESTIMATE, not historical data.")
chart = make_graph(raw, working, model, degree, chosen_year, extra_years, edited)
st.plotly_chart(chart, use_container_width=True, config={"displaylogo": False})
st.caption("Blue = actual observations; red = fitted polynomial; purple dashed section and star = 2026 extrapolation; gold dot = movable model year; green dashed line = tangent/slope.")
st.download_button("Download the interactive graph (HTML)", chart.to_html(include_plotlyjs=True).encode("utf-8"), "chile_gdp_interactive_graph.html", "text/html")

with st.expander("How does the slope line use calculus?"):
    st.latex(r"T(x)=G(x_0)+G'(x_0)(x-x_0)")
    st.write(
        f"At year **{chosen_year}**, x₀ = {selected_x}. "
        f"The model value is **${selected_model_gdp:,.3f} billion** and its derivative "
        f"is **{selected_slope:+,.3f} billion/year**. "
        f"Its second derivative is **{selected_acceleration:+,.3f} billion/year²**. "
        "A positive slope means that the fitted curve is rising at that point; a negative "
        "slope means it is falling. This is a mathematical estimate, not the exact real-world change."
    )

resid_graph = go.Figure()
resid_graph.add_trace(go.Scatter(
    x=working["Year"], y=residuals, mode="markers+lines", name="Residual (working − fitted)",
    marker={"color": "#168578", "size": 7},
))
resid_graph.add_hline(y=0, line_dash="dash", line_color="gray")
resid_graph.update_layout(
    title="Model error: residuals by year",
    xaxis_title="Year", yaxis_title="Residual (billions of US$)",
    height=330, margin={"l": 10, "r": 10, "t": 45, "b": 25},
)
st.plotly_chart(resid_graph, use_container_width=True)

st.header("6 · Estimate one year or several years")
predicted_year = st.number_input("Choose a year for a calculation", min_value=FIRST_YEAR, max_value=2050, value=PREDICTION_YEAR, step=1)
calculated = float(model(int(predicted_year) - FIRST_YEAR))
st.metric(f"GDP model calculation: {int(predicted_year)}", f"${calculated:,.2f}B")
st.caption(data_kind(int(predicted_year), latest))
st.write(
    f"First derivative: **{float(model.deriv()(int(predicted_year) - FIRST_YEAR)):+,.2f} B/year** · "
    f"Second derivative: **{float(model.deriv(2)(int(predicted_year) - FIRST_YEAR)):+,.2f} B/year²**"
)
if int(predicted_year) > latest:
    st.warning("This year is outside the measured historical data. Polynomial extrapolation can be unstable; this estimate is not an official forecast.")
if calculated < 0:
    st.error("Negative GDP is not economically meaningful; do not treat this extrapolation as a realistic economic prediction.")

text_years = st.text_input("Calculate multiple years (comma-separated)", value="2010, 2020, 2024, 2025, 2026")
try:
    years = [int(piece.strip()) for piece in text_years.split(",") if piece.strip()]
    if not years or len(years) > 30 or any(y < FIRST_YEAR or y > 2050 for y in years):
        raise ValueError("Enter 1 to 30 years, each between 1976 and 2050.")
    prediction_rows = []
    for year in years:
        observed = raw.loc[raw["Year"] == year, GDP_COLUMN]
        prediction_rows.append({
            "Year": year,
            "Observed GDP ($B), if available": float(observed.iloc[0]) if not observed.empty else np.nan,
            "Model GDP ($B)": float(model(year - FIRST_YEAR)),
            "Classification": data_kind(year, latest),
        })
    prediction_table = pd.DataFrame(prediction_rows)
    st.dataframe(prediction_table, hide_index=True, use_container_width=True)
    st.download_button("Download predictions (CSV)", to_csv(prediction_table), "chile_gdp_predictions.csv", "text/csv")
except ValueError as error:
    st.warning(f"Please correct the years list: {error}")
    prediction_table = pd.DataFrame([
        {"Year": PREDICTION_YEAR, "Model GDP ($B)": forecast_2026}
    ])

st.header("7 · Download results and complete PDF report")
results = working.copy()
results["Fitted GDP ($B)"] = fitted_values
results["Residual ($B)"] = residuals
st.download_button("Download calculated model and residuals (CSV)", to_csv(results), "chile_gdp_model_results.csv", "text/csv")
try:
    pdf = pdf_report(raw, working, model, r2, rmse, mae, degree, edited, prediction_table)
    st.download_button("Download complete project report (PDF)", pdf, "chile_gdp_hispanic_heritage_report.pdf", "application/pdf")
except Exception as error:
    st.error(f"The PDF could not be produced: {error}")
    st.info("You can still download the original data, predictions, and model results as CSV files.")

st.header("8 · What the economic trends mean")
st.write(
    "Chile is a Spanish-speaking country in South America. Its economy includes "
    "copper mining, farming, fishing, commerce, manufacturing and services. "
    "The 1982 debt crisis, changes in international copper demand, the 2008–2009 "
    "financial crisis, and the COVID-19 pandemic provide historical context for "
    "economic changes. A regression curve by itself cannot prove what caused them."
)
st.write(
    "**Limits of the indicator:** GDP in current US dollars is *nominal* GDP. "
    "Inflation and currency conversion can change the numbers, even when the "
    "quantity produced changes differently. GDP is not a country's net worth, "
    "nor does it tell us how evenly income is distributed or measure cultural value."
)
st.write(
    "**Limits of the model:** The data start in 1976 because that is the assigned "
    "investigation period, not the beginning of Chile's economy. Degree-3-or-higher "
    "polynomials are descriptive models; forecast reliability declines beyond "
    "the years actually observed."
)

st.header("9 · 🇨🇱 Chile fun fact")
fun_facts = [
    "Chile is among the world's leading producers of copper. That makes global copper prices an important part of its economic story.",
    "The Atacama Desert in northern Chile is among the driest places on Earth, and its clear skies are ideal for observatories.",
    "Chile stretches over 4,000 kilometers from north to south, from the Atacama region toward Patagonia.",
    "Chile uses the Chilean peso. The dollar value of GDP can change when the peso–dollar exchange rate changes.",
    "Chile celebrates Fiestas Patrias in September, during the United States' Hispanic Heritage Month season.",
]
fun_fact_id = st.selectbox("Select a fun fact", options=list(range(len(fun_facts))), format_func=lambda i: f"Fun fact #{i+1}")
st.info(fun_facts[fun_fact_id])
st.caption(
    f"Historical source: World Bank, World Development Indicators, Chile GDP (current US$), "
    f"indicator NY.GDP.MKTP.CD. Accessed {date.today().isoformat()}. "
    f"Original values are preserved; 2026 is a MODEL ESTIMATE."
)
