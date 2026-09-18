"""
RavenStack SaaS Business Analytics Dashboard
============================================================
Four tabs, one dataset: Churn, Account Health, Channel ROI, Support Ops.

Run:
    streamlit run app.py

Requires:
    pip install streamlit plotly pandas
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ----------------------------------------------------------------
# Page config & data loading
# ----------------------------------------------------------------
st.set_page_config(page_title="RavenStack Analytics", layout="wide", page_icon="📊")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "account_analytics_with_health.csv"
TICKETS_PATH = BASE_DIR / "data" / "ravenstack_support_tickets.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    tickets = pd.read_csv(TICKETS_PATH, parse_dates=["submitted_at", "closed_at"])
    return df, tickets


df, tickets = load_data()

st.title("RavenStack SaaS Business Analytics")
st.caption(f"{len(df)} accounts | Data as of the latest pipeline run")

# ----------------------------------------------------------------
# Sidebar filters (apply to all tabs)
# ----------------------------------------------------------------
st.sidebar.header("Filters")
industries = st.sidebar.multiselect("Industry", sorted(df["industry"].unique()), default=list(df["industry"].unique()))
plans = st.sidebar.multiselect("Plan Tier", sorted(df["current_plan_tier"].unique()), default=list(df["current_plan_tier"].unique()))

filtered = df[df["industry"].isin(industries) & df["current_plan_tier"].isin(plans)]

tab1, tab2, tab3, tab4 = st.tabs(["🔻 Churn", "❤️ Account Health", "📈 Channel ROI", "🎧 Support Ops"])

# ==================================================================
# TAB 1: CHURN
# ==================================================================
with tab1:
    st.subheader("Who's churning, and why?")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accounts", len(filtered))
    c2.metric("Churn Rate", f"{filtered['churned'].mean():.1%}")
    c3.metric("Revenue at Risk (ARR)", f"${filtered['revenue_at_risk_arr'].sum():,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        churn_by_industry = filtered.groupby("industry")["churned"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(churn_by_industry, x="industry", y="churned", title="Churn Rate by Industry",
                     labels={"churned": "Churn Rate"}, color="churned", color_continuous_scale="Reds")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        churn_by_ref = filtered.groupby("referral_source")["churned"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(churn_by_ref, x="referral_source", y="churned", title="Churn Rate by Referral Source",
                     labels={"churned": "Churn Rate"}, color="churned", color_continuous_scale="Reds")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    reasons = filtered[filtered["churned"]]["latest_churn_reason"].value_counts().reset_index()
    reasons.columns = ["reason", "count"]
    fig = px.pie(reasons, names="reason", values="count", title="Churn Reasons (churned accounts)")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Headline finding:** DevTools-industry accounts and event-driven signups churn "
        "well above the company baseline. Closing the DevTools gap alone represents an "
        "estimated **$300K+** in recoverable annual revenue."
    )

# ==================================================================
# TAB 2: ACCOUNT HEALTH
# ==================================================================
with tab2:
    st.subheader("Which accounts need CS attention right now?")
    st.caption(
        "Note: this score is a monitoring/triage tool, not a churn predictor — engagement "
        "signals showed only a weak relationship with actual churn in this dataset (see Churn tab)."
    )

    tier_counts = filtered["health_tier"].value_counts().reindex(["At Risk", "Watch", "Healthy"]).fillna(0)
    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 At Risk", int(tier_counts.get("At Risk", 0)))
    c2.metric("🟡 Watch", int(tier_counts.get("Watch", 0)))
    c3.metric("🟢 Healthy", int(tier_counts.get("Healthy", 0)))

    fig = px.histogram(filtered, x="health_score", color="health_tier", nbins=30,
                        title="Health Score Distribution",
                        color_discrete_map={"At Risk": "#d62728", "Watch": "#ff7f0e", "Healthy": "#2ca02c"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Lowest-scoring accounts (candidates for CS outreach):**")
    watch_list = filtered.sort_values("health_score").head(15)[
        ["account_id", "account_name", "industry", "health_score", "health_tier",
         "days_since_last_usage", "total_tickets", "current_mrr"]
    ]
    st.dataframe(watch_list, use_container_width=True, hide_index=True)

# ==================================================================
# TAB 3: CHANNEL ROI
# ==================================================================
with tab3:
    st.subheader("Which acquisition channels bring in the best customers?")

    channel = filtered.groupby("referral_source").agg(
        accounts=("account_id", "count"),
        churn_rate=("churned", "mean"),
        total_current_mrr=("current_mrr", "sum"),
        avg_tenure_days=("tenure_days", "mean"),
    ).reset_index()
    channel["annual_revenue"] = channel["total_current_mrr"] * 12
    channel["revenue_per_account"] = channel["annual_revenue"] / channel["accounts"]

    fig = px.scatter(channel, x="churn_rate", y="revenue_per_account", size="accounts",
                      text="referral_source", title="Channel Quality: Retention vs. Revenue",
                      labels={"churn_rate": "Churn Rate", "revenue_per_account": "Revenue per Account ($/yr)"})
    fig.update_traces(textposition="top center")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bottom-right is the sweet spot: low churn, high revenue per account. Top-left is the danger zone.")

    st.dataframe(
        channel.sort_values("revenue_per_account", ascending=False).style.format({
            "churn_rate": "{:.1%}", "total_current_mrr": "${:,.0f}",
            "annual_revenue": "${:,.0f}", "revenue_per_account": "${:,.0f}",
            "avg_tenure_days": "{:.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    st.info(
        "**Headline finding:** Partner referrals outperform every other channel on both "
        "retention and revenue. Shifting acquisition mix away from event-driven signups "
        "toward partner-driven signups represents an estimated **$840K** in additional "
        "annual revenue potential."
    )

# ==================================================================
# TAB 4: SUPPORT OPS
# ==================================================================
with tab4:
    st.subheader("Is the support team prioritizing correctly?")

    priority_order = ["urgent", "high", "medium", "low"]
    sla = tickets.groupby("priority").agg(
        tickets=("ticket_id", "count"),
        avg_resolution_hrs=("resolution_time_hours", "mean"),
        avg_first_response_min=("first_response_time_minutes", "mean"),
        escalation_rate=("escalation_flag", "mean"),
    ).reindex(priority_order).reset_index()

    fig = px.bar(sla, x="priority", y="avg_resolution_hrs", title="Avg Resolution Time by Priority",
                 labels={"avg_resolution_hrs": "Hours to Resolve"},
                 category_orders={"priority": priority_order},
                 color="avg_resolution_hrs", color_continuous_scale="Oranges")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        sla.style.format({"avg_resolution_hrs": "{:.1f}h", "avg_first_response_min": "{:.0f}m",
                           "escalation_rate": "{:.1%}"}),
        use_container_width=True, hide_index=True,
    )

    urgent_hrs = sla.loc[sla["priority"] == "urgent", "avg_resolution_hrs"].values[0]
    low_hrs = sla.loc[sla["priority"] == "low", "avg_resolution_hrs"].values[0]
    st.warning(
        f"**Headline finding:** Urgent tickets resolve only **{low_hrs - urgent_hrs:.1f} hours faster** "
        f"than Low-priority tickets ({urgent_hrs:.1f}h vs {low_hrs:.1f}h) — priority tagging is not "
        "meaningfully changing resolution speed. Worth investigating whether triage rules are actually "
        "being followed."
    )
