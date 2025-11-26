import os
import random
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(page_title="ESG Investment Portfolio Simulation",
                   page_icon="favicon.png",
                   layout="wide")

st.markdown("""<style>/* Main app and sidebar containers */
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] *,
        [data-testid="stSidebarNav"],
        [data-testid="stSidebarNav"] * {
            font-family: "Times New Roman", serif !important;}

        /* Markdown/text blocks */
        [data-testid="stMarkdownContainer"] * {
            font-family: "Times New Roman", serif !important;
        }

        /* Inputs and buttons */
        button, input, textarea, select,
        .stButton > button,
        .stTextInput input,
        .stNumberInput input {
            font-family: "Times New Roman", serif !important;
        }

        /* Tables and dataframes */
        .stTable, .stDataFrame {
            font-family: "Times New Roman", serif !important;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("sidebar_banner.png", use_column_width=True)
st.sidebar.markdown("---")




TOTAL_LIMIT = 100000
years = 5
SCOREBOARD_PATH = "esg_simulation_scoreboard.csv"

plt.rcParams["font.family"] = ["Times New Roman", "Serif"]
plt.rcParams["font.size"] = 12
plt.close("all")


def tprint(text):
    st.markdown(
        f"<p style='font-family: \"Times New Roman\", serif; font-size:16px; text-align:center;'>{text}</p>",
        unsafe_allow_html=True,
    )




highESG = [
    ["Consumer", "BTI", "British American Tobacco", 68],
    ["Consumer", "CL", "Colgate-Palmolive Company", 58],
    ["Consumer", "DEO", "Diageo", 67],
    ["Consumer", "KHC", "Kraft Heinz", 59],
    ["Consumer", "MDLZ", "Mondelez", 62],
    ["Financials", "BCS", "Barclays", 58],
    ["Financials", "MA", "Mastercard", 59],
    ["Financials", "NDAQ", "Nasdaq", 66],
    ["Financials", "NWG", "NatWest", 64],
    ["Financials", "V", "Visa", 57],
    ["Healthcare", "ABBV", "AbbVie", 69],
    ["Healthcare", "ABT", "Abbott Laboratories", 62],
    ["Healthcare", "GSK", "GSK", 76],
    ["Healthcare", "HLN", "Haleon", 74],
    ["Healthcare", "TEVA", "Teva Pharma", 58],
    ["Industrials", "ASML", "ASML Holding", 57],
    ["Industrials", "LMT", "Lockheed Martin", 54],
    ["Industrials", "TT", "Trane Technologies", 74],
    ["Industrials", "UNP", "Union Pacific", 60],
    ["Industrials", "WM", "Waste Management", 60],
    ["Technology", "ADBE", "Adobe", 57],
    ["Technology", "CRM", "Salesforce", 61],
    ["Technology", "CSCO", "Cisco", 65],
    ["Technology", "INTC", "Intel", 60],
    ["Technology", "NVDA", "NVIDIA", 61],
    ["Utilities", "DUK", "Duke Energy", 56],
    ["Utilities", "ELE.MC", "Endesa", 87],
    ["Utilities", "ENGIY", "Engie SA", 81],
    ["Utilities", "EXC", "Exelon", 58],
    ["Utilities", "SRE", "Sempra", 57],
]

lowESG = [
    ["Consumer", "ABNB", "Airbnb", 17],
    ["Consumer", "AMZN", "Amazon", 26],
    ["Consumer", "MAR", "Marriott International", 35],
    ["Consumer", "SBUX", "Starbucks", 33],
    ["Consumer", "TSLA", "Tesla", 30],
    ["Financials", "C", "Citigroup", 37],
    ["Financials", "GS", "Goldman Sachs", 39],
    ["Financials", "JPM", "JPMorgan Chase", 33],
    ["Financials", "MS", "Morgan Stanley", 41],
    ["Financials", "WFC", "Wells Fargo", 38],
    ["Healthcare", "ISRG", "Intuitive Surgical", 37],
    ["Healthcare", "JNJ", "Johnson & Johnson", 29],
    ["Healthcare", "MCK", "McKesson", 36],
    ["Healthcare", "PFE", "Pfizer", 40],
    ["Healthcare", "VRTX", "Vertex Pharma", 37],
    ["Industrials", "BA", "Boeing", 40],
    ["Industrials", "CAT", "Caterpillar", 37],
    ["Industrials", "DE", "Deere & Co", 47],
    ["Industrials", "PH", "Parker-Hannifin", 37],
    ["Industrials", "RTX", "RTX Corp", 27],
    ["Technology", "AAPL", "Apple", 34],
    ["Technology", "AVGO", "Broadcom", 39],
    ["Technology", "INTU", "Intuit", 45],
    ["Technology", "ORCL", "Oracle", 38],
    ["Technology", "PLTR", "Palantir", 30],
    ["Utilities", "CEG", "Constellation Energy", 37],
    ["Utilities", "D", "Dominion Energy", 38],
    ["Utilities", "NGG", "National Grid", 44],
    ["Utilities", "SO", "Southern Co", 39],
    ["Utilities", "XEL", "Xcel Energy", 46],
]

rows = []
for r in highESG:
    rows.append(r + ["highESG"])
for r in lowESGs:
    rows.append(r + ["lowESG"])

df_esg = (
    pd.DataFrame(rows, columns=["Sector", "Ticker", "Company", "ESG", "Group"])
    .sort_values(["Group", "Sector", "Ticker"])
    .reset_index(drop=True)
)




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
        s = df["Adj Close"] if "Adj Close" in df else df["Close"]
        s = s.dropna()
        if len(s) < 50:
            return None
        return s
    except Exception:
        return None


def fetch_team_prices(team):
    usable = []
    series = []
    for t in team:
        s = fetch_price_series(t)
        if s is not None:
            usable.append(t)
            series.append(s)

    if not usable:
        return None, []

    prices = pd.concat(series, axis=1)
    prices.columns = usable
    prices = prices.ffill().bfill()
    return prices, usable




def portfolio_no_events(df, ticks, sh, cash):
    p = df[ticks]
    return pd.Series((sh * p.values).sum(axis=1) + cash, index=p.index)


def portfolio_with_events(df, ticks, sh_start, events, cash0):
    p = df[ticks]
    sh = sh_start.copy()
    cash = cash0

    event_map = {}
    for e in events:
        event_map.setdefault(e["idx"], []).append(e)

    values = []
    for i in range(len(p)):
        row = p.iloc[i].values

        if i in event_map:
            for ev in event_map[i]:
                j = ticks.index(ev["ticker"])
                price = row[j]
                amt = ev["cash"]

                if amt > 0:
                    buy = min(amt, cash)
                    sh[j] += buy / price
                    cash -= buy
                    ev["shares_bought"] = buy / price
                    ev["shares_sold"] = 0.0
                else:
                    sell = min(-amt, sh[j] * price)
                    sh[j] -= sell / price
                    cash += sell
                    ev["shares_bought"] = 0.0
                    ev["shares_sold"] = sell / price

                ev["remaining_shares"] = sh[j]
                ev["cash_after"] = cash

        port_val = (sh * row).sum() + cash
        if i in event_map:
            for ev in event_map[i]:
                ev["port_value_after"] = port_val

        values.append(port_val)

    last_prices = p.iloc[-1].values
    holdings = pd.DataFrame(
        {
            "Ticker": ticks,
            "Shares": sh,
            "Last Price": last_prices,
            "Current Value": sh * last_prices,
        }
    )

    return pd.Series(values, index=p.index), holdings, cash




def update_scoreboard(name, team, final_val, ret, avg_esg):
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

    combined = round(avg_esg * (1 + ret / 100), 4)

    new_row = {
        "Student": name,
        "Team": team,
        "Final portfolio (£)": round(final_val, 2),
        "Total return (%)": round(ret, 2),
        "Average ESG": round(avg_esg, 2),
        "Combined Score": combined,
    }

    sb = pd.concat([sb, pd.DataFrame([new_row])], ignore_index=True)
    sb = sb.sort_values("Combined Score", ascending=False).reset_index(drop=True)
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




def main():
    st.title("ESG Investment Portfolio Simulation")

    page = st.sidebar.radio("Navigation", ["Play Simulation", "Scoreboard"])

    if page == "Play Simulation":
        play_page()
    else:
        scoreboard_page()


def play_page():
    st.image("banner.png", use_column_width=True)
    st.markdown("---")

    student_name = st.text_input("Enter your name")

    team_mode = st.radio("Team assignment", ["Random", "Choose team"])

    if "team_choice" not in st.session_state:
        st.session_state.team_choice = None

    if team_mode == "Choose team":
        team_raw = st.selectbox("Choose team", ["highESG", "lowESG"])
    else:
        if st.session_state.team_choice is None:
            st.session_state.team_choice = random.choice(["highESG", "lowESG"])
        team_raw = st.session_state.team_choice

    if team_raw == "highESG":
        team_label = "highESG team"
        team_all = df_esg[df_esg["Group"] == "highESG"]["Ticker"].tolist()
    else:
        team_label = "lowESG team"
        team_all = df_esg[df_esg["Group"] == "lowESG"]["Ticker"].tolist()

    tprint(f"You are in {team_label}")

    st.markdown("---")
    st.write("Step 1: Download price data")

    if st.button("Download data"):
        with st.spinner("Downloading..."):
            prices, usable = fetch_team_prices(team_all)
        if prices is not None:
            st.session_state.prices = prices
            st.session_state.active_tickers = usable
            tprint("Data downloaded.")

    if "prices" not in st.session_state:
        return

    prices = st.session_state.prices
    active_tickers = st.session_state.active_tickers

    with st.expander("Show tickers with price data"):
        df_view = df_esg[df_esg["Ticker"].isin(active_tickers)][
            ["Ticker", "Company", "Sector", "ESG"]
        ].copy()
        last = prices[active_tickers].iloc[-1].rename("Last Price")
        last_df = last.to_frame().reset_index().rename(columns={"index": "Ticker"})
        df_full = df_view.merge(last_df, on="Ticker")
        st.dataframe(df_full)

    st.markdown("---")
    st.write("Step 2: Choose initial investments")

    invest_tickers = st.multiselect(
        "Select tickers", active_tickers, default=active_tickers
    )

    allocation_mode = st.radio("Initial allocation", ["Equal split", "Custom"])

    initial_amounts = {}

    if allocation_mode == "Equal split":
        eq = TOTAL_LIMIT / len(invest_tickers)
        for t in invest_tickers:
            initial_amounts[t] = eq
    else:
        for t in invest_tickers:
            amt = st.number_input(
                f"Amount for {t}",
                min_value=0.0,
                max_value=float(TOTAL_LIMIT),
                step=100.0,
                key=f"init_{t}",
            )
            initial_amounts[t] = float(amt)

    total_invest = sum(initial_amounts.values())
    initial_cash = TOTAL_LIMIT - total_invest

    st.write(f"Total initial investment: £{total_invest:,.2f}")
    st.write(f"Remaining cash: £{initial_cash:,.2f}")

    if total_invest > TOTAL_LIMIT:
        st.error("Total exceeds £100,000")
        return

    initial_prices = prices[active_tickers].iloc[0]
    shares_0 = []
    for t in active_tickers:
        shares_0.append(initial_amounts.get(t, 0.0) / initial_prices[t])
    shares_0 = np.array(shares_0)

    sellable = [t for t in invest_tickers if initial_amounts.get(t, 0) > 0]

    st.markdown("---")
    st.write("Initial portfolio (no events)")

    base_port = portfolio_no_events(prices, active_tickers, shares_0, initial_cash)
    fig0, ax0 = plt.subplots(figsize=(10, 5))
    ax0.plot(base_port.index, base_port.values, label="Initial portfolio")
    ax0.grid(True)
    ax0.legend()
    st.pyplot(fig0)

    st.markdown("---")
    st.write("Step 3: Optional buy/sell events")

    sim_cash = initial_cash
    st.write(f"Initial free cash available for events: £{sim_cash:,.2f}")

    num_events = st.selectbox("Number of events", [0, 1, 2, 3])
    events = []
    dates = prices.index

    for i in range(num_events):
        st.subheader(f"Event {i+1}")
        st.write(f"Cash before this event: £{sim_cash:,.2f}")

        date_str = st.text_input(
            f"Event {i+1} date (YYYY-MM-DD)", key=f"dt_{i}"
        )
        ticker_ev = st.selectbox(
            f"Ticker for event {i+1}", active_tickers, key=f"tk_{i}"
        )
        cash_ev = st.number_input(
            f"Cash (+buy, -sell) for {ticker_ev}",
            step=100.0,
            key=f"cs_{i}",
        )

        if cash_ev != 0:
            if cash_ev < 0 and ticker_ev not in sellable:
                st.error(f"You cannot sell {ticker_ev} (not in initial investments).")
                return

            if cash_ev > 0 and sim_cash <= 0:
                st.error(
                    "You do not have any cash available for buys at this point. "
                    "You need to sell first before you can buy."
                )
                return

            if cash_ev > 0 and cash_ev > sim_cash:
                st.error(
                    f"You only have £{sim_cash:,.2f} available at this point, "
                    "so you cannot invest more than that in a single event."
                )
                return

            try:
                dt = pd.to_datetime(date_str)
                idx = (abs(dates - dt)).argmin()
                dt_final = dates[idx]
            except Exception:
                st.error("Invalid date")
                return

            if cash_ev > 0:
                sim_cash -= cash_ev
            else:
                sim_cash += -cash_ev

            events.append(
                {"idx": idx, "date": dt_final, "cash": cash_ev, "ticker": ticker_ev}
            )

    st.markdown("---")

    if st.button("Run simulation with events & submit"):
        if not student_name:
            st.error("Enter your name.")
            return

        port_series, holdings, final_cash = portfolio_with_events(
            prices, active_tickers, shares_0, events, initial_cash
        )

        initial_val = base_port.iloc[0]
        final_val = port_series.iloc[-1]
        ret = (final_val / initial_val - 1) * 100

        hold_df = holdings.merge(df_esg, on="Ticker").sort_values(
            ["Sector", "Ticker"]
        )

        tprint(f"Initial portfolio value: £{initial_val:,.2f}")
        tprint(f"Final portfolio (stocks only) value: £{hold_df['Current Value'].sum():,.2f}")
        tprint(f"Final cash (uninvested): £{final_cash:,.2f}")
        total_portfolio = hold_df["Current Value"].sum() + final_cash
        tprint(f"Total portfolio (stocks + cash): £{total_portfolio:,.2f}")
        tprint(f"Total portfolio return: {ret:.2f}%")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(base_port.index, base_port.values, label="Initial")
        ax.plot(
            port_series.index,
            port_series.values,
            label="With events",
            linestyle="--",
        )
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        st.subheader("Holdings at the end")
        st.dataframe(hold_df)

        transactions = []
        initial_date = prices.index[0]

        for t in active_tickers:
            amt = initial_amounts.get(t, 0.0)
            if amt > 0:
                j = active_tickers.index(t)
                transactions.append(
                    {
                        "Date": initial_date,
                        "Ticker": t,
                        "Cash": amt,
                        "Shares Bought": shares_0[j],
                        "Shares Sold": 0.0,
                        "Remaining Shares": shares_0[j],
                        "Cash After": initial_cash,
                        "Portfolio After": initial_val,
                    }
                )

        for e in events:
            transactions.append(
                {
                    "Date": e["date"],
                    "Ticker": e["ticker"],
                    "Cash": e["cash"],
                    "Shares Bought": e.get("shares_bought", 0.0),
                    "Shares Sold": e.get("shares_sold", 0.0),
                    "Remaining Shares": e.get("remaining_shares", np.nan),
                    "Cash After": e.get("cash_after", np.nan),
                    "Portfolio After": e.get("port_value_after", np.nan),
                }
            )

        if transactions:
            event_log = pd.DataFrame(transactions).sort_values(["Date", "Ticker"])
            st.subheader("Transaction log")
            st.dataframe(event_log)

        per_stock = []
        for t in invest_tickers:
            p0 = prices[t].iloc[0]
            p1 = prices[t].iloc[-1]
            r = (p1 / p0 - 1) * 100
            per_stock.append({"Ticker": t, "Price return (%)": r})

        if per_stock:
            st.subheader("Per-stock price returns (for invested tickers)")
            st.dataframe(pd.DataFrame(per_stock))

        avg_esg = df_esg[df_esg["Ticker"].isin(invest_tickers)]["ESG"].mean()
        sb = update_scoreboard(student_name, team_label, final_val, ret, avg_esg)

        st.subheader("Updated Scoreboard")
        st.dataframe(sb)


def scoreboard_page():
    st.header("Scoreboard")
    st.dataframe(load_scoreboard())

    if st.button("Clear scoreboard"):
        empty = pd.DataFrame(
            columns=[
                "Student",
                "Team",
                "Final portfolio (£)",
                "Total return (%)",
                "Average ESG",
                "Combined Score",
            ]
        )
        empty.to_csv(SCOREBOARD, index=False)
        st.success("Scoreboard cleared.")


if __name__ == "__main__":
    main()
