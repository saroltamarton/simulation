import os
import random
from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# SETTINGS
# ============================================================

TOTAL_LIMIT = 100000
years = 5
SCOREBOARD_PATH = "esg_simulation_scoreboard.csv"

plt.rcParams['font.family'] = ['Nimbus Roman', 'Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

plt.close("all")


def tprint(text):
    st.markdown(
        f"<p style='font-family: \"Nimbus Roman\", serif; font-size: 15px; "
        f"text-align: center;'>{text}</p>",
        unsafe_allow_html=True,
    )


# ============================================================
# ESG UNIVERSE
# ============================================================

green_emitters = [["Consumer","BTI","British American Tobacco",68],
                  ["Consumer","CL","Colgate-Palmolive Company",58],
                  ["Consumer","DEO","Diageo",67],
                  ["Consumer","KHC","Kraft Heinz",59],
                  ["Consumer","MDLZ","Mondelez",62],
                  ["Financials","BCS","Barclays",58],
                  ["Financials","MA","Mastercard",59],
                  ["Financials","NDAQ","Nasdaq",66],
                  ["Financials","NWG","NatWest",64],
                  ["Financials","V","Visa",57],
                  ["Healthcare","ABBV","AbbVie",69],
                  ["Healthcare","ABT","Abbott Laboratories",62],
                  ["Healthcare","GSK","GSK",76],
                  ["Healthcare","HLN","Haleon",74],
                  ["Healthcare","TEVA","Teva Pharma",58],
                  ["Industrials","ASML","ASML Holding",57],
                  ["Industrials","LMT","Lockheed Martin Corporation",54],
                  ["Industrials","TT","Trane Technologies",74],
                  ["Industrials","UNP","Union Pacific Corporation",60],
                  ["Industrials","WM","Waste Management",60],
                  ["Technology","ADBE","Adobe",57],
                  ["Technology","CRM","Salesforce",61],
                  ["Technology","CSCO","Cisco",65],
                  ["Technology","INTC","Intel",60],
                  ["Technology","NVDA","NVIDIA",61],
                  ["Utilities","DUK","Duke Energy Corporation",56],
                  ["Utilities","ELE.MC","Endesa",87],
                  ["Utilities","ENGIY","Engie SA",81],
                  ["Utilities","EXC","Exelon Corporation",58],
                  ["Utilities","SRE","Sempra",57],]

heavy_emitters = [["Consumer","ABNB","Airbnb",17],
                  ["Consumer","AMZN","Amazon",26],
                  ["Consumer","MAR","Marriott International",35],
                  ["Consumer","SBUX","Starbucks",33],
                  ["Consumer","TSLA","Tesla",30],
                  ["Financials","C","Citigroup",37],
                  ["Financials","GS","The Goldman Sachs Group",39],
                  ["Financials","JPM","JPMorgan Chase & Co",33],
                  ["Financials","MS","Morgan Stanley",41],
                  ["Financials","WFC","Wells Fargo & Company",38],
                  ["Healthcare","ISRG","Intuitive Surgical",37],
                  ["Healthcare","JNJ","Johnson & Johnson",29],
                  ["Healthcare","MCK","McKesson Corporation",36],
                  ["Healthcare","PFE","Pfizer Inc",40],
                  ["Healthcare","VRTX","Vertex Pharmaceuticals Incorporated",37],
                  ["Industrials","BA","The Boeing Company",40],
                  ["Industrials","CAT","Caterpillar",37],
                  ["Industrials","DE","Deere & Company",47],
                  ["Industrials","PH","Parker-Hannifin Corporation",37],
                  ["Industrials","RTX","RTX Corporation",27],
                  ["Technology","AAPL","Apple",34],
                  ["Technology","AVGO","Broadcom Inc",39],
                  ["Technology","INTU","Intuit",45],
                  ["Technology","ORCL","Oracle",38],
                  ["Technology","PLTR","Palantir Technologies",30],
                  ["Utilities","CEG","Constellation Energy",37],
                  ["Utilities","D","Dominion Energy",38],
                  ["Utilities","NGG","National Grid",44],
                  ["Utilities","SO","The Southern Company",39],
                  ["Utilities","XEL","Xcel Energy",46],]

rows = []
for row in green_emitters:
    rows.append(row + ["green"])
for row in heavy_emitters:
    rows.append(row + ["heavy"])

df_esg = (
    pd.DataFrame(rows, columns=["Sector","Ticker","Company","ESG","Group"])
    .sort_values(["Group","Sector","Ticker"])
    .reset_index(drop=True)
)


# ============================================================
# DOWNLOAD PRICE DATA
# ============================================================

def fetch_price_series(ticker):
    try:
        df = yf.download(
            ticker,
            period=f"{years}y",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            return None
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        s = df[col].dropna()
        if len(s) <= 50:
            return None
        return s
    except Exception:
        return None


def fetch_team_prices(team_all):
    usable = []
    series = []
    for t in team_all:
        s = fetch_price_series(t)
        if s is not None:
            usable.append(t)
            series.append(s)
    if not usable:
        return None, []
    prices = pd.concat(series, axis=1)
    prices.columns = usable
    prices = prices.sort_index().ffill().bfill()
    return prices, usable


# ============================================================
# PORTFOLIO MODELS
# ============================================================

def portfolio_no_events(df, ticks, sh, cash):
    p = df[ticks]
    vals = (sh * p.values).sum(axis=1) + cash
    return pd.Series(vals, index=p.index)


def portfolio_with_events(df, ticks, sh_start, events, cash0):
    p = df[ticks]
    sh = sh_start.copy()
    cash = cash0
    eidx = {}

    for e in events:
        eidx.setdefault(e["idx"], []).append(e)

    vals = []

    for i in range(len(p)):
        row = p.iloc[i].values

        if i in eidx:
            for ev in eidx[i]:
                j = ticks.index(ev["ticker"])
                price = row[j]
                amt = ev["cash"]

                if amt > 0:
                    buy = min(amt, cash)
                    sh[j] += buy / price
                    cash -= buy
                    ev["shares_bought"] = buy / price
                    ev["shares_sold"] = 0
                else:
                    sell = min(-amt, sh[j] * price)
                    sh[j] -= sell / price
                    cash += sell
                    ev["shares_bought"] = 0
                    ev["shares_sold"] = sell / price

                ev["remaining_shares"] = sh[j]
                ev["port_value_after"] = (sh * row).sum() + cash

        vals.append((sh * row).sum() + cash)

    last = p.iloc[-1].values
    hold = pd.DataFrame(
        {"Ticker": ticks, "Shares": sh, "Last Price": last, "Current Value": sh * last}
    )
    return pd.Series(vals, index=p.index), hold, cash


# ============================================================
# SCOREBOARD
# ============================================================

def update_scoreboard(name, team, final_portfolio, total_return, avg_esg_score):
    if os.path.exists(SCOREBOARD_PATH):
        sb = pd.read_csv(SCOREBOARD_PATH)
    else:
        sb = pd.DataFrame(
            columns=[
                "Student",
                "Team",
                "Final portfolio (£)",
                "Total return (%)",
                "Average ESG",
                "Combined Score",
            ]
        )

    combined_score = round(avg_esg_score * (1 + total_return / 100), 4)

    new_row = {
        "Student": name,
        "Team": team,
        "Final portfolio (£)": round(final_portfolio, 2),
        "Total return (%)": round(total_return, 2),
        "Average ESG": round(avg_esg_score, 2),
        "Combined Score": combined_score,
    }

    sb = pd.concat([sb, pd.DataFrame([new_row])], ignore_index=True)
    sb = sb.sort_values(by=["Combined Score"], ascending=[False]).reset_index(drop=True)
    sb.to_csv(SCOREBOARD_PATH, index=False)
    return sb


def load_scoreboard():
    if os.path.exists(SCOREBOARD_PATH):
        return pd.read_csv(SCOREBOARD_PATH)
    return pd.DataFrame(
        columns=[
            "Student",
            "Team",
            "Final portfolio (£)",
            "Total return (%)",
            "Average ESG",
            "Combined Score",
        ]
    )


# ============================================================
# STREAMLIT APP STRUCTURE
# ============================================================

def main():
    st.set_page_config(page_title="ESG Simulation", layout="wide")
    st.title("ESG Portfolio Simulation")

    page = st.sidebar.radio("Navigation", ["Play Simulation", "Scoreboard"])

    if page == "Play Simulation":
        play_page()
    else:
        scoreboard_page()


def play_page():
    tprint("ESG Portfolio Simulation")

    student_name = st.text_input("Enter your name or group name")

    team_mode = st.radio("Team selection", ["Random", "Choose team"])

    if "team_choice" not in st.session_state:
        st.session_state.team_choice = None

    if team_mode == "Choose team":
        team_raw = st.selectbox("Choose your team", ["green", "heavy"])
    else:
        if st.session_state.team_choice is None:
            st.session_state.team_choice = random.choice(["green", "heavy"])
        team_raw = st.session_state.team_choice

    if team_raw == "green":
        team_label = "team green"
        team_color = "green"
        team_all = df_esg[df_esg["Group"] == "green"]["Ticker"].tolist()
    else:
        team_label = "team heavy emitters"
        team_color = "black"
        team_all = df_esg[df_esg["Group"] == "heavy"]["Ticker"].tolist()

    tprint(f"You are in {team_label}")

    with st.expander("Show team tickers"):
        st.dataframe(df_esg[df_esg["Ticker"].isin(team_all)][["Ticker","Company","Sector","ESG"]])

    st.markdown("---")
    st.write("Step 1: Download price data")

    if st.button("Download data"):
        with st.spinner("Downloading..."):
            prices, usable = fetch_team_prices(team_all)
            if prices is None:
                st.error("No usable tickers.")
            else:
                st.session_state.prices = prices
                st.session_state.active_tickers = usable

    if "prices" not in st.session_state:
        return

    prices = st.session_state.prices
    active_tickers = st.session_state.active_tickers

    with st.expander("Show tickers with price data"):
    df_view = df_esg[df_esg["Ticker"].isin(active_tickers)][["Ticker","Company","Sector","ESG"]].copy()
    last = prices[active_tickers].iloc[-1].rename("Last Price")

    # convert the index (tickers) into a proper column so we can merge on "Ticker"
    last_df = last.to_frame().reset_index().rename(columns={"index": "Ticker"})

    df_view = df_view.merge(last_df, on="Ticker")
    st.dataframe(df_view)


    st.markdown("---")
    st.write("Step 2: Choose initial investments")

    invest_tickers = st.multiselect("Select tickers", active_tickers, default=active_tickers)

    custom_choice = st.radio("Initial allocation", ["Equal split", "Custom amounts"])

    initial_amounts = {}
    if custom_choice == "Equal split":
        eq = TOTAL_LIMIT / len(invest_tickers)
        for t in invest_tickers:
            initial_amounts[t] = eq
        total_invest = TOTAL_LIMIT
    else:
        total_invest = 0
        for t in invest_tickers:
            amt = st.number_input(f"Amount for {t}", 0.0, TOTAL_LIMIT)
            initial_amounts[t] = amt
            total_invest += amt

        if total_invest > TOTAL_LIMIT:
            st.error("Total exceeds £100,000")
            return

    initial_cash = TOTAL_LIMIT - total_invest

    initial_prices = prices[active_tickers].iloc[0]
    shares_0 = []
    for t in active_tickers:
        shares_0.append(initial_amounts.get(t, 0.0) / initial_prices[t])
    shares_0 = np.array(shares_0)

    sellable_tickers = [t for t in invest_tickers if initial_amounts.get(t, 0) > 0]

    st.markdown("---")
    st.write("Step 3: Optional events")

    num_events = st.selectbox("Number of buy/sell events", [0, 1, 2, 3])

    events = []
    dates = prices.index

    for i in range(num_events):
        st.subheader(f"Event {i+1}")
        date_str = st.text_input(f"Date for event {i+1} (YYYY-MM-DD)", key=f"date_{i}")
        ticker_ev = st.selectbox(f"Ticker for event {i+1}", active_tickers, key=f"evtick{i}")
        cash_ev = st.number_input(f"Cash for event {i+1}", key=f"evcash{i}")

        if cash_ev != 0:
            try:
                dt = pd.to_datetime(date_str)
                idx = (abs(dates - dt)).argmin()
                dt_final = dates[idx]
            except Exception:
                st.error("Invalid date")
                return

            if cash_ev < 0 and ticker_ev not in sellable_tickers:
                st.error(f"Cannot sell {ticker_ev}; not in initial investments.")
                return

            events.append(
                {"idx": idx, "date": dt_final, "cash": cash_ev, "ticker": ticker_ev}
            )

    st.markdown("---")

    if st.button("Run simulation and submit results"):
        if not student_name:
            st.error("Enter your name")
            return

        base_port = portfolio_no_events(prices, active_tickers, shares_0, initial_cash)

        port_series, holdings, final_cash = portfolio_with_events(
            prices, active_tickers, shares_0, events, initial_cash
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(base_port.index, base_port.values, label="Initial")
        ax.plot(port_series.index, port_series.values, label="With events", linestyle="--")
        ax.set_title(f"{student_name} — {team_label}")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        hold_df = holdings.merge(df_esg, on="Ticker").sort_values(["Sector","Ticker"])
        st.dataframe(hold_df)

        stock_value = hold_df["Current Value"].sum()
        final_val = port_series.iloc[-1]
        ret = (final_val / base_port.iloc[0] - 1) * 100

        avg_esg = df_esg[df_esg["Ticker"].isin(invest_tickers)]["ESG"].mean()

        sb = update_scoreboard(student_name, team_label, final_val, ret, avg_esg)

        st.subheader("Scoreboard")
        st.dataframe(sb)


def scoreboard_page():
    st.header("Live Scoreboard")
    st.dataframe(load_scoreboard())


if __name__ == "__main__":
    main()
