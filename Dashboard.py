import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib


# =========================================================
# Load Data and Model
# =========================================================

df = pd.read_csv("data/Churn_Modelling.csv")

model = joblib.load("models/bank_churn_model.pkl")

rf_model = model.named_steps["model"]
preprocessor = model.named_steps["preprocessor"]


# ---------------------------------------------------------
# Derived columns used across the whole dashboard
# (these mirror the segments discussed in the project report)
# ---------------------------------------------------------

df["BalancePerProduct"] = df.apply(
    lambda r: r["Balance"] / r["NumOfProducts"] if r["NumOfProducts"] != 0 else 0,
    axis=1
)

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[17, 30, 40, 50, 60, 100],
    labels=["18-30", "31-40", "41-50", "51-60", "60+"]
)

df["BalanceSegment"] = pd.cut(
    df["Balance"],
    bins=[-1, 0, 50000, 120000, df["Balance"].max() + 1],
    labels=["Zero Balance", "Low (0-50K)", "Medium (50K-120K)", "High (120K+)"]
)

df["CreditScoreBand"] = pd.cut(
    df["CreditScore"],
    bins=[300, 579, 669, 739, 799, 851],
    labels=["Poor (<580)", "Fair (580-669)", "Good (670-739)",
            "Very Good (740-799)", "Excellent (800+)"]
)

df["TenureGroup"] = pd.cut(
    df["Tenure"],
    bins=[-1, 1, 4, 7, 20],
    labels=["New (0-1y)", "Early (2-4y)", "Established (5-7y)", "Loyal (8y+)"]
)

df["CardLabel"] = df["HasCrCard"].map({1: "Has Card", 0: "No Card"})
df["ActiveLabel"] = df["IsActiveMember"].map({1: "Active", 0: "Inactive"})
df["ExitedLabel"] = df["Exited"].map({1: "Churned", 0: "Retained"})

# Exact feature columns the trained pipeline expects (must match training)
MODEL_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary", "BalancePerProduct",
    "Geography", "Gender"
]

# Score every customer once, reused by the Segmentation tab
df["ChurnProbability"] = model.predict_proba(df[MODEL_FEATURES])[:, 1]
df["RiskTier"] = pd.cut(
    df["ChurnProbability"],
    bins=[-0.01, 0.30, 0.60, 1.01],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

OVERALL_CHURN_RATE = df["Exited"].mean()


# =========================================================
# App Configuration
# =========================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

app.title = "Bank Churn Analytics"
server = app.server

# Business color palette
COLOR_RETAINED = "#2E86AB"
COLOR_CHURNED = "#E63946"
COLOR_ACCENT = "#F4A261"
COLOR_NEUTRAL = "#6C757D"
PLOTLY_TEMPLATE = "plotly_white"


# =========================================================
# Sidebar
# =========================================================

sidebar = dbc.Card(
    [
        html.Div(
            [
                html.I(className="bi bi-bank2", style={"fontSize": "2rem", "color": "#2E86AB"}),
                html.H4("Bank Churn", className="mt-2 mb-0"),
                html.Small("Analytics Console", className="text-muted"),
            ],
            className="text-center py-3"
        ),

        html.Hr(),

        dbc.Nav(
            [
                dbc.NavLink([html.I(className="bi bi-speedometer2 me-2"), "Overview"],
                            href="/", active="exact"),

                html.Small("DEEP DIVES", className="text-muted ps-2 d-block mt-2 mb-1",
                           style={"fontSize": "0.7rem", "letterSpacing": "1px"}),
                dbc.NavLink([html.I(className="bi bi-wallet2 me-2"), "Balance"],
                            href="/balance", active="exact"),
                dbc.NavLink([html.I(className="bi bi-credit-card me-2"), "Credit Card"],
                            href="/creditcard", active="exact"),
                dbc.NavLink([html.I(className="bi bi-globe-americas me-2"), "Geography"],
                            href="/geography", active="exact"),
                dbc.NavLink([html.I(className="bi bi-people me-2"), "Demographics"],
                            href="/demographics", active="exact"),
                dbc.NavLink([html.I(className="bi bi-clock-history me-2"), "Tenure & Loyalty"],
                            href="/tenure", active="exact"),

                html.Small("DECISIONS", className="text-muted ps-2 d-block mt-2 mb-1",
                           style={"fontSize": "0.7rem", "letterSpacing": "1px"}),
                dbc.NavLink([html.I(className="bi bi-diagram-3 me-2"), "Risk Segmentation"],
                            href="/segmentation", active="exact"),
                dbc.NavLink([html.I(className="bi bi-person-check me-2"), "Prediction"],
                            href="/prediction", active="exact"),
                dbc.NavLink([html.I(className="bi bi-bar-chart-line me-2"), "Model Insights"],
                            href="/insights", active="exact"),
                dbc.NavLink([html.I(className="bi bi-clipboard-check me-2"), "Executive Summary"],
                            href="/summary", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),

        html.Hr(),

        html.Div(
            [
                html.Small("Dataset", className="text-muted d-block"),
                html.Small(f"{len(df):,} customers", className="fw-bold d-block"),
                html.Small(f"Overall churn: {OVERALL_CHURN_RATE:.1%}",
                            className="text-danger fw-bold d-block"),
            ],
            className="px-2"
        ),
    ],
    body=True,
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "height": "100vh",
        "width": "16%",
        "overflowY": "auto",
        "zIndex": 1000,
    }
)


# =========================================================
# Reusable helpers
# =========================================================

def kpi_card(title, value, icon, color="primary", sub=None, card_id=None):
    """A polished KPI card with icon + optional subtitle (used for deltas)."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.I(className=f"bi {icon}", style={"fontSize": "1.6rem"}),
                        html.Span(title, className="ms-2 text-muted", style={"fontSize": "0.85rem"}),
                    ],
                    className="d-flex align-items-center mb-2"
                ),
                html.H3(value, className="mb-0", id=card_id) if card_id else html.H3(value, className="mb-0"),
                html.Small(sub, className="text-muted") if sub else None,
            ]
        ),
        className=f"shadow-sm border-start border-{color} border-4 h-100"
    )


def insight_box(text, icon="bi-lightbulb"):
    return dbc.Alert(
        [html.I(className=f"bi {icon} me-2"), text],
        color="light",
        className="border shadow-sm small"
    )


# =========================================================
# Page 1 - Overview Dashboard (with filters)
# =========================================================

filter_panel = dbc.Card(
    dbc.CardBody(
        [
            html.H6([html.I(className="bi bi-funnel me-2"), "Filters"], className="mb-3"),

            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Geography", className="fw-bold small"),
                            dcc.Checklist(
                                id="f-geo",
                                options=[{"label": " " + g, "value": g} for g in sorted(df["Geography"].unique())],
                                value=list(sorted(df["Geography"].unique())),
                                inline=True,
                                inputStyle={"marginRight": "5px", "marginLeft": "10px"}
                            ),
                        ],
                        md=3
                    ),

                    dbc.Col(
                        [
                            dbc.Label("Gender", className="fw-bold small"),
                            dcc.Checklist(
                                id="f-gender",
                                options=[{"label": " " + g, "value": g} for g in sorted(df["Gender"].unique())],
                                value=list(sorted(df["Gender"].unique())),
                                inline=True,
                                inputStyle={"marginRight": "5px", "marginLeft": "10px"}
                            ),
                        ],
                        md=2
                    ),

                    dbc.Col(
                        [
                            dbc.Label("Membership Status", className="fw-bold small"),
                            dcc.Checklist(
                                id="f-active",
                                options=[
                                    {"label": " Active", "value": 1},
                                    {"label": " Inactive", "value": 0},
                                ],
                                value=[0, 1],
                                inline=True,
                                inputStyle={"marginRight": "5px", "marginLeft": "10px"}
                            ),
                        ],
                        md=2
                    ),

                    dbc.Col(
                        [
                            dbc.Label("Number of Products", className="fw-bold small"),
                            dcc.Checklist(
                                id="f-products",
                                options=[{"label": f" {p}", "value": p} for p in sorted(df["NumOfProducts"].unique())],
                                value=list(sorted(df["NumOfProducts"].unique())),
                                inline=True,
                                inputStyle={"marginRight": "5px", "marginLeft": "10px"}
                            ),
                        ],
                        md=3
                    ),

                    dbc.Col(
                        dbc.Button(
                            [html.I(className="bi bi-arrow-counterclockwise me-1"), "Reset"],
                            id="f-reset", color="secondary", outline=True, size="sm", className="mt-4"
                        ),
                        md=2
                    ),
                ],
                className="mb-3"
            ),

            dbc.Row(
                dbc.Col(
                    [
                        dbc.Label("Age Range", className="fw-bold small"),
                        dcc.RangeSlider(
                            id="f-age",
                            min=int(df["Age"].min()), max=int(df["Age"].max()),
                            value=[int(df["Age"].min()), int(df["Age"].max())],
                            marks={a: str(a) for a in range(int(df["Age"].min()), int(df["Age"].max()) + 1, 10)},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                    md=12
                )
            ),
        ]
    ),
    className="shadow-sm mb-4"
)


dashboard_page = dbc.Container(
    [
        html.H1("Customer Churn Overview", className="my-4"),
        html.P("Filter the customer base below to explore how churn behavior shifts across segments.",
               className="text-muted"),

        filter_panel,

        html.Div(id="dash-insight"),

        dbc.Row(
            [
                dbc.Col(kpi_card("Total Customers", "-", "bi-people", "primary", card_id="kpi-total"), md=3),
                dbc.Col(kpi_card("Churned Customers", "-", "bi-person-dash", "danger", card_id="kpi-churned"), md=3),
                dbc.Col(kpi_card("Churn Rate", "-", "bi-graph-up-arrow", "warning", card_id="kpi-rate"), md=3),
                dbc.Col(kpi_card("Active Members", "-", "bi-person-check", "success", card_id="kpi-active"), md=3),
            ],
            className="mb-4 g-3"
        ),

        dbc.Row(
            [
                dbc.Col(dcc.Loading(dcc.Graph(id="graph-geo")), md=6),
                dbc.Col(dcc.Loading(dcc.Graph(id="graph-age")), md=6),
            ],
            className="mb-3"
        ),

        dbc.Row(
            [
                dbc.Col(dcc.Loading(dcc.Graph(id="graph-balance")), md=6),
                dbc.Col(dcc.Loading(dcc.Graph(id="graph-products")), md=6),
            ],
            className="mb-3"
        ),

        dbc.Row(
            dbc.Col(dcc.Loading(dcc.Graph(id="graph-tenure-active")), md=12),
            className="mb-4"
        ),
    ],
    fluid=True
)


# =========================================================
# Page 2 - Prediction
# =========================================================

prediction_form = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col([dbc.Label("Credit Score"), dbc.Input(id="credit", value=650, type="number")], md=6),
                    dbc.Col([dbc.Label("Age"), dbc.Input(id="age", value=40, type="number")], md=6),
                ], className="mb-2"
            ),
            dbc.Row(
                [
                    dbc.Col([dbc.Label("Tenure (years)"), dbc.Input(id="tenure", value=5, type="number")], md=6),
                    dbc.Col([dbc.Label("Balance"), dbc.Input(id="balance", value=50000, type="number")], md=6),
                ], className="mb-2"
            ),
            dbc.Row(
                [
                    dbc.Col([dbc.Label("Number of Products"), dbc.Input(id="products", value=2, type="number")], md=6),
                    dbc.Col([dbc.Label("Estimated Salary"), dbc.Input(id="salary", value=60000, type="number")], md=6),
                ], className="mb-2"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Geography"),
                            dcc.Dropdown(
                                id="geo",
                                options=[{"label": g, "value": g} for g in ["France", "Germany", "Spain"]],
                                value="France", clearable=False
                            ),
                        ], md=6
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Gender"),
                            dcc.Dropdown(
                                id="gender",
                                options=[{"label": g, "value": g} for g in ["Male", "Female"]],
                                value="Male", clearable=False
                            ),
                        ], md=6
                    ),
                ], className="mb-2"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Has Credit Card?"),
                            dcc.Dropdown(
                                id="card",
                                options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
                                value=1, clearable=False
                            ),
                        ], md=6
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Active Member?"),
                            dcc.Dropdown(
                                id="active",
                                options=[{"label": "Yes", "value": 1}, {"label": "No", "value": 0}],
                                value=1, clearable=False
                            ),
                        ], md=6
                    ),
                ], className="mb-3"
            ),

            dbc.Button(
                [html.I(className="bi bi-lightning-charge me-2"), "Predict Churn Risk"],
                id="predict", color="primary", className="w-100"
            ),
        ]
    ),
    className="shadow-sm"
)

prediction_page = dbc.Container(
    [
        html.H1("Customer Risk Prediction", className="my-4"),
        html.P("Enter a customer's profile to estimate their probability of churn.",
               className="text-muted"),

        dbc.Row(
            [
                dbc.Col(prediction_form, md=5),
                dbc.Col(
                    dcc.Loading(html.Div(id="prediction_result")),
                    md=7
                ),
            ],
            className="g-4"
        ),
    ],
    fluid=True
)


# =========================================================
# Page 3 - Balance Analysis (deep dive)
# =========================================================

def build_balance_page():
    avg_retained = df.loc[df["Exited"] == 0, "Balance"].mean()
    avg_exited = df.loc[df["Exited"] == 1, "Balance"].mean()

    zero_balance_pct = (df["Balance"] == 0).mean()
    zero_balance_churn = df.loc[df["Balance"] == 0, "Exited"].mean()
    nonzero_balance_churn = df.loc[df["Balance"] > 0, "Exited"].mean()

    fig_dist = px.histogram(
        df, x="Balance", color="ExitedLabel", barmode="overlay", nbins=40,
        color_discrete_map={"Retained": COLOR_RETAINED, "Churned": COLOR_CHURNED},
        template=PLOTLY_TEMPLATE, title="Balance Distribution: Retained vs Churned"
    )
    fig_dist.update_layout(legend_title_text="", bargap=0.05)

    seg = df.groupby("BalanceSegment", observed=True)["Exited"].mean().reset_index()
    fig_segment = px.bar(
        seg, x="BalanceSegment", y="Exited", text_auto=".1%",
        color="Exited", color_continuous_scale=["#2E86AB", "#E63946"],
        template=PLOTLY_TEMPLATE, title="Churn Rate by Balance Segment"
    )
    fig_segment.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False,
                               yaxis_title="Churn Rate", xaxis_title="Balance Segment")

    fig_perproduct = px.box(
        df, x="ExitedLabel", y="BalancePerProduct", color="ExitedLabel",
        color_discrete_map={"Retained": COLOR_RETAINED, "Churned": COLOR_CHURNED},
        template=PLOTLY_TEMPLATE, title="Balance-per-Product: Retained vs Churned",
        points=False
    )
    fig_perproduct.update_layout(showlegend=False, xaxis_title="", yaxis_title="Balance per Product")

    sample = df.sample(min(3000, len(df)), random_state=42)
    fig_scatter = px.scatter(
        sample, x="Age", y="Balance", color="ExitedLabel", opacity=0.55,
        color_discrete_map={"Retained": COLOR_RETAINED, "Churned": COLOR_CHURNED},
        template=PLOTLY_TEMPLATE, title="Balance vs Age (colored by churn)"
    )
    fig_scatter.update_layout(legend_title_text="")

    return dbc.Container(
        [
            html.H1("Balance Deep-Dive", className="my-4"),
            html.P("A focused look at how account balance relates to churn behavior, "
                   "since the report highlighted balance as a meaningful churn signal.",
                   className="text-muted"),

            dbc.Row(
                [
                    dbc.Col(kpi_card("Avg Balance (Retained)", f"{avg_retained:,.0f}",
                                     "bi-piggy-bank", "primary"), md=3),
                    dbc.Col(kpi_card("Avg Balance (Churned)", f"{avg_exited:,.0f}",
                                     "bi-piggy-bank", "danger"), md=3),
                    dbc.Col(kpi_card("Zero-Balance Customers", f"{zero_balance_pct:.1%}",
                                     "bi-wallet", "secondary"), md=3),
                    dbc.Col(kpi_card("Churn: Zero vs Non-Zero", f"{zero_balance_churn:.1%} / {nonzero_balance_churn:.1%}",
                                     "bi-exclamation-triangle", "warning"), md=3),
                ],
                className="mb-4 g-3"
            ),

            insight_box(
                f"Customers who churned carry a higher average balance ({avg_exited:,.0f}) than those "
                f"who stayed ({avg_retained:,.0f}). Counter-intuitively, zero-balance customers churn "
                f"{'less' if zero_balance_churn < nonzero_balance_churn else 'more'} often "
                f"({zero_balance_churn:.1%}) than customers holding a balance ({nonzero_balance_churn:.1%}) — "
                "high-balance customers may be shopping for better offers elsewhere and deserve "
                "targeted retention outreach."
            ),

            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=fig_dist), md=6),
                    dbc.Col(dcc.Graph(figure=fig_segment), md=6),
                ],
                className="mb-3"
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=fig_perproduct), md=6),
                    dbc.Col(dcc.Graph(figure=fig_scatter), md=6),
                ],
                className="mb-4"
            ),
        ],
        fluid=True
    )


# =========================================================
# Page 4 - Credit Card Analysis (deep dive)
# =========================================================

def build_creditcard_page():
    churn_with_card = df.loc[df["HasCrCard"] == 1, "Exited"].mean()
    churn_without_card = df.loc[df["HasCrCard"] == 0, "Exited"].mean()

    avg_score_retained = df.loc[df["Exited"] == 0, "CreditScore"].mean()
    avg_score_exited = df.loc[df["Exited"] == 1, "CreditScore"].mean()

    card_rate = df.groupby("CardLabel")["Exited"].mean().reset_index()
    fig_bar = px.bar(
        card_rate, x="CardLabel", y="Exited", text_auto=".1%",
        color="CardLabel", color_discrete_sequence=[COLOR_RETAINED, COLOR_ACCENT],
        template=PLOTLY_TEMPLATE, title="Churn Rate: Credit Card Ownership"
    )
    fig_bar.update_layout(yaxis_tickformat=".0%", showlegend=False,
                           yaxis_title="Churn Rate", xaxis_title="")

    fig_box = px.box(
        df, x="ExitedLabel", y="CreditScore", color="ExitedLabel",
        color_discrete_map={"Retained": COLOR_RETAINED, "Churned": COLOR_CHURNED},
        template=PLOTLY_TEMPLATE, title="Credit Score: Retained vs Churned", points=False
    )
    fig_box.update_layout(showlegend=False, xaxis_title="", yaxis_title="Credit Score")

    band_card = df.groupby(["CreditScoreBand", "CardLabel"], observed=True)["Exited"].mean().reset_index()
    fig_bins = px.bar(
        band_card, x="CreditScoreBand", y="Exited", color="CardLabel", barmode="group",
        color_discrete_sequence=[COLOR_RETAINED, COLOR_ACCENT],
        template=PLOTLY_TEMPLATE, title="Churn Rate by Credit Score Band & Card Ownership"
    )
    fig_bins.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn Rate",
                            xaxis_title="Credit Score Band", legend_title_text="")

    sample = df.sample(min(3000, len(df)), random_state=42)
    fig_scatter = px.scatter(
        sample, x="CreditScore", y="Balance", color="ExitedLabel", opacity=0.5,
        color_discrete_map={"Retained": COLOR_RETAINED, "Churned": COLOR_CHURNED},
        template=PLOTLY_TEMPLATE, title="Credit Score vs Balance (colored by churn)"
    )
    fig_scatter.update_layout(legend_title_text="")

    return dbc.Container(
        [
            html.H1("Credit Card Deep-Dive", className="my-4"),
            html.P("Credit card ownership showed a weak standalone effect in the report — "
                   "this page checks whether it interacts with credit score or balance.",
                   className="text-muted"),

            dbc.Row(
                [
                    dbc.Col(kpi_card("Churn - With Card", f"{churn_with_card:.1%}",
                                     "bi-credit-card", "primary"), md=3),
                    dbc.Col(kpi_card("Churn - No Card", f"{churn_without_card:.1%}",
                                     "bi-credit-card-2-front", "secondary"), md=3),
                    dbc.Col(kpi_card("Avg Score (Retained)", f"{avg_score_retained:,.0f}",
                                     "bi-star", "success"), md=3),
                    dbc.Col(kpi_card("Avg Score (Churned)", f"{avg_score_exited:,.0f}",
                                     "bi-star-half", "danger"), md=3),
                ],
                className="mb-4 g-3"
            ),

            insight_box(
                f"Card ownership alone barely separates churners ({churn_with_card:.1%}) from "
                f"non-owners ({churn_without_card:.1%}), confirming the report's finding that it's a "
                "weak standalone predictor. It's more useful combined with other segments — e.g. "
                "low credit-score customers without a card may warrant a retention check-in."
            ),

            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=fig_bar), md=6),
                    dbc.Col(dcc.Graph(figure=fig_box), md=6),
                ],
                className="mb-3"
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=fig_bins), md=6),
                    dbc.Col(dcc.Graph(figure=fig_scatter), md=6),
                ],
                className="mb-4"
            ),
        ],
        fluid=True
    )


# =========================================================
# Page 5 - Model Insights
# =========================================================

feature_names = preprocessor.get_feature_names_out()

clean_names = []
for feature in feature_names:
    feature = feature.replace("num__", "")
    feature = feature.replace("cat__", "")
    feature = feature.replace("_", " ")
    clean_names.append(feature)

importance_df = pd.DataFrame({
    "Feature": clean_names,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=True)

top_features_text = ", ".join(
    importance_df.sort_values("Importance", ascending=False)["Feature"].head(3).tolist()
)

fig_importance = px.bar(
    importance_df, x="Importance", y="Feature", orientation="h",
    template=PLOTLY_TEMPLATE, title="Factors Affecting Customer Churn",
    color="Importance", color_continuous_scale="Blues"
)
fig_importance.update_layout(coloraxis_showscale=False, height=650)

insights_page = dbc.Container(
    [
        html.H1("Model Insights", className="my-4"),
        html.P("Understanding which factors the Tuned Random Forest model relies on most when "
               "flagging customers as high-risk.", className="text-muted"),

        dbc.Row(
            [
                dbc.Col(kpi_card("Model", "Tuned Random Forest", "bi-cpu", "primary"), md=4),
                dbc.Col(kpi_card("ROC-AUC", "0.864", "bi-graph-up", "success"), md=4),
                dbc.Col(kpi_card("Churn Recall", "66%", "bi-bullseye", "warning"), md=4),
            ],
            className="mb-4 g-3"
        ),

        insight_box(
            f"The top drivers of churn risk in this model are: {top_features_text}. "
            "These align with the EDA findings — retention efforts should prioritize older, "
            "single-product customers with high balances."
        ),

        dcc.Graph(figure=fig_importance),
    ],
    fluid=True
)


# =========================================================
# Page 6 - Geography Deep-Dive
# =========================================================

def build_geography_page():
    geo_stats = df.groupby("Geography").agg(
        Customers=("Exited", "size"),
        ChurnRate=("Exited", "mean")
    ).reset_index()

    top_geo = geo_stats.sort_values("ChurnRate", ascending=False).iloc[0]

    kpi_row = dbc.Row(
        [
            dbc.Col(
                kpi_card(f"{row.Geography}", f"{row.ChurnRate:.1%}", "bi-flag",
                         "danger" if row.Geography == top_geo["Geography"] else "primary",
                         sub=f"{row.Customers:,} customers"),
                md=4
            )
            for row in geo_stats.itertuples()
        ],
        className="mb-4 g-3"
    )

    fig_rate = px.bar(
        geo_stats, x="Geography", y="ChurnRate", text_auto=".1%", color="Geography",
        color_discrete_sequence=px.colors.qualitative.Set2,
        template=PLOTLY_TEMPLATE, title="Churn Rate by Country"
    )
    fig_rate.update_layout(yaxis_tickformat=".0%", showlegend=False, yaxis_title="Churn Rate")

    geo_gender = df.groupby(["Geography", "Gender"])["Exited"].mean().reset_index()
    fig_gender = px.bar(
        geo_gender, x="Geography", y="Exited", color="Gender", barmode="group", text_auto=".1%",
        color_discrete_sequence=[COLOR_ACCENT, COLOR_RETAINED],
        template=PLOTLY_TEMPLATE, title="Churn Rate by Country & Gender"
    )
    fig_gender.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn Rate")

    fig_balance = px.box(
        df, x="Geography", y="Balance", color="ExitedLabel",
        color_discrete_map={"Retained": COLOR_RETAINED, "Churned": COLOR_CHURNED},
        template=PLOTLY_TEMPLATE, title="Balance by Country, Split by Churn"
    )
    fig_balance.update_layout(legend_title_text="")

    geo_products = df.groupby(["Geography", "NumOfProducts"])["Exited"].mean().reset_index()
    fig_products = px.bar(
        geo_products, x="Geography", y="Exited", color="NumOfProducts", barmode="group", text_auto=".0%",
        template=PLOTLY_TEMPLATE, title="Churn Rate by Country & Number of Products"
    )
    fig_products.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn Rate")

    return dbc.Container(
        [
            html.H1("Geography Deep-Dive", className="my-4"),
            html.P("Where the bank is losing customers, and what makes that country's "
                   "churners different.", className="text-muted"),

            kpi_row,

            insight_box(
                f"{top_geo['Geography']} has the highest churn rate ({top_geo['ChurnRate']:.1%}), "
                f"noticeably above the portfolio average ({OVERALL_CHURN_RATE:.1%}). Cross-referencing "
                "with gender and product count below helps decide whether this needs a country-specific "
                "retention campaign or is explained by a specific customer sub-segment."
            ),

            dbc.Row(
                [dbc.Col(dcc.Graph(figure=fig_rate), md=6), dbc.Col(dcc.Graph(figure=fig_gender), md=6)],
                className="mb-3"
            ),
            dbc.Row(
                [dbc.Col(dcc.Graph(figure=fig_balance), md=6), dbc.Col(dcc.Graph(figure=fig_products), md=6)],
                className="mb-4"
            ),
        ],
        fluid=True
    )


# =========================================================
# Page 7 - Demographics (Age & Gender)
# =========================================================

def build_demographics_page():
    churn_male = df.loc[df["Gender"] == "Male", "Exited"].mean()
    churn_female = df.loc[df["Gender"] == "Female", "Exited"].mean()
    avg_age_retained = df.loc[df["Exited"] == 0, "Age"].mean()
    avg_age_exited = df.loc[df["Exited"] == 1, "Age"].mean()

    fig_age_rate = px.bar(
        df.groupby("AgeGroup", observed=True)["Exited"].mean().reset_index(),
        x="AgeGroup", y="Exited", text_auto=".1%",
        color="Exited", color_continuous_scale=["#2E86AB", "#E63946"],
        template=PLOTLY_TEMPLATE, title="Churn Rate by Age Group"
    )
    fig_age_rate.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False, yaxis_title="Churn Rate")

    age_gender = df.groupby(["AgeGroup", "Gender"], observed=True)["Exited"].mean().reset_index()
    fig_age_gender = px.bar(
        age_gender, x="AgeGroup", y="Exited", color="Gender", barmode="group", text_auto=".1%",
        color_discrete_sequence=[COLOR_ACCENT, COLOR_RETAINED],
        template=PLOTLY_TEMPLATE, title="Churn Rate by Age Group & Gender"
    )
    fig_age_gender.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn Rate")

    fig_box = px.box(
        df, x="Gender", y="Age", color="ExitedLabel",
        color_discrete_map={"Retained": COLOR_RETAINED, "Churned": COLOR_CHURNED},
        template=PLOTLY_TEMPLATE, title="Age Distribution by Gender & Churn"
    )
    fig_box.update_layout(legend_title_text="")

    fig_gender_pie = px.pie(
        df[df["Exited"] == 1], names="Gender", hole=0.5,
        color_discrete_sequence=[COLOR_ACCENT, COLOR_RETAINED],
        template=PLOTLY_TEMPLATE, title="Gender Mix Among Churned Customers"
    )

    return dbc.Container(
        [
            html.H1("Demographics: Age & Gender", className="my-4"),
            html.P("Age is the single strongest churn driver in the model — this page shows exactly "
                   "which age/gender combination to prioritize.", className="text-muted"),

            dbc.Row(
                [
                    dbc.Col(kpi_card("Churn - Female", f"{churn_female:.1%}", "bi-gender-female", "danger"), md=3),
                    dbc.Col(kpi_card("Churn - Male", f"{churn_male:.1%}", "bi-gender-male", "primary"), md=3),
                    dbc.Col(kpi_card("Avg Age (Retained)", f"{avg_age_retained:.1f}", "bi-person", "success"), md=3),
                    dbc.Col(kpi_card("Avg Age (Churned)", f"{avg_age_exited:.1f}", "bi-person-x", "warning"), md=3),
                ],
                className="mb-4 g-3"
            ),

            insight_box(
                f"Female customers churn at {churn_female:.1%} vs {churn_male:.1%} for male customers. "
                f"Churned customers are on average {avg_age_exited - avg_age_retained:.1f} years older "
                "than retained ones. The 41-60 age band combined with female gender is the highest-risk "
                "demographic slice — a strong candidate for a dedicated retention offer."
            ),

            dbc.Row(
                [dbc.Col(dcc.Graph(figure=fig_age_rate), md=6), dbc.Col(dcc.Graph(figure=fig_age_gender), md=6)],
                className="mb-3"
            ),
            dbc.Row(
                [dbc.Col(dcc.Graph(figure=fig_box), md=6), dbc.Col(dcc.Graph(figure=fig_gender_pie), md=6)],
                className="mb-4"
            ),
        ],
        fluid=True
    )


# =========================================================
# Page 8 - Tenure & Loyalty
# =========================================================

def build_tenure_page():
    new_churn = df.loc[df["Tenure"] <= 1, "Exited"].mean()
    loyal_churn = df.loc[df["Tenure"] >= 8, "Exited"].mean()
    avg_tenure_retained = df.loc[df["Exited"] == 0, "Tenure"].mean()
    avg_tenure_exited = df.loc[df["Exited"] == 1, "Tenure"].mean()

    tenure_rate = df.groupby("Tenure")["Exited"].mean().reset_index()
    fig_line = px.line(
        tenure_rate, x="Tenure", y="Exited", markers=True,
        template=PLOTLY_TEMPLATE, title="Churn Rate by Tenure (years)",
        color_discrete_sequence=[COLOR_CHURNED]
    )
    fig_line.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn Rate")

    fig_bucket = px.bar(
        df.groupby("TenureGroup", observed=True)["Exited"].mean().reset_index(),
        x="TenureGroup", y="Exited", text_auto=".1%",
        color="Exited", color_continuous_scale=["#2E86AB", "#E63946"],
        template=PLOTLY_TEMPLATE, title="Churn Rate by Loyalty Stage"
    )
    fig_bucket.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False, yaxis_title="Churn Rate")

    fig_balance = px.box(
        df, x="TenureGroup", y="Balance", color="ExitedLabel",
        color_discrete_map={"Retained": COLOR_RETAINED, "Churned": COLOR_CHURNED},
        template=PLOTLY_TEMPLATE, title="Balance by Loyalty Stage, Split by Churn"
    )
    fig_balance.update_layout(legend_title_text="")

    tenure_active = df.groupby(["TenureGroup", "ActiveLabel"], observed=True)["Exited"].mean().reset_index()
    fig_active = px.bar(
        tenure_active, x="TenureGroup", y="Exited", color="ActiveLabel", barmode="group", text_auto=".1%",
        color_discrete_sequence=[COLOR_CHURNED, COLOR_RETAINED],
        template=PLOTLY_TEMPLATE, title="Churn Rate by Loyalty Stage & Activity Status"
    )
    fig_active.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn Rate")

    return dbc.Container(
        [
            html.H1("Tenure & Loyalty", className="my-4"),
            html.P("Tenure alone is a weak predictor per the report, but it becomes useful once "
                   "split into loyalty stages and combined with activity status.", className="text-muted"),

            dbc.Row(
                [
                    dbc.Col(kpi_card("New Customers (0-1y) Churn", f"{new_churn:.1%}", "bi-person-plus", "warning"), md=3),
                    dbc.Col(kpi_card("Loyal Customers (8y+) Churn", f"{loyal_churn:.1%}", "bi-award", "success"), md=3),
                    dbc.Col(kpi_card("Avg Tenure (Retained)", f"{avg_tenure_retained:.1f}y", "bi-calendar-check", "primary"), md=3),
                    dbc.Col(kpi_card("Avg Tenure (Churned)", f"{avg_tenure_exited:.1f}y", "bi-calendar-x", "danger"), md=3),
                ],
                className="mb-4 g-3"
            ),

            insight_box(
                "Churn rate stays fairly flat across tenure years, confirming the report's finding — "
                "but combining tenure with activity status (right-most chart) shows inactive customers "
                "churn more regardless of how long they've been with the bank. Loyalty programs should "
                "target activity, not just tenure."
            ),

            dbc.Row(
                [dbc.Col(dcc.Graph(figure=fig_line), md=6), dbc.Col(dcc.Graph(figure=fig_bucket), md=6)],
                className="mb-3"
            ),
            dbc.Row(
                [dbc.Col(dcc.Graph(figure=fig_balance), md=6), dbc.Col(dcc.Graph(figure=fig_active), md=6)],
                className="mb-4"
            ),
        ],
        fluid=True
    )


# =========================================================
# Page 9 - Risk Segmentation (the decision-support tab)
# =========================================================

def build_segmentation_page():
    tier_stats = df.groupby("RiskTier", observed=True).agg(
        Customers=("Exited", "size"),
        ActualChurnRate=("Exited", "mean"),
        BalanceAtRisk=("Balance", "sum")
    ).reset_index()

    high_risk = df[df["RiskTier"] == "High Risk"]
    high_risk_balance = high_risk["Balance"].sum()
    high_risk_count = len(high_risk)

    fig_pie = px.pie(
        df, names="RiskTier", hole=0.5,
        color="RiskTier",
        color_discrete_map={"Low Risk": COLOR_RETAINED, "Medium Risk": COLOR_ACCENT, "High Risk": COLOR_CHURNED},
        template=PLOTLY_TEMPLATE, title="Customer Base by Predicted Risk Tier"
    )

    fig_validation = px.bar(
        tier_stats, x="RiskTier", y="ActualChurnRate", text_auto=".1%",
        color="RiskTier",
        color_discrete_map={"Low Risk": COLOR_RETAINED, "Medium Risk": COLOR_ACCENT, "High Risk": COLOR_CHURNED},
        template=PLOTLY_TEMPLATE, title="Model Validation: Actual Churn Rate per Predicted Tier"
    )
    fig_validation.update_layout(yaxis_tickformat=".0%", showlegend=False, yaxis_title="Actual Churn Rate")

    top_risk_table = high_risk.sort_values("ChurnProbability", ascending=False).head(25)[
        ["CustomerId", "Surname", "Geography", "Age", "Balance", "NumOfProducts",
         "IsActiveMember", "ChurnProbability"]
    ].copy()
    top_risk_table["ChurnProbability"] = (top_risk_table["ChurnProbability"] * 100).round(1).astype(str) + "%"
    top_risk_table["Balance"] = top_risk_table["Balance"].round(0)

    from dash import dash_table
    risk_table = dash_table.DataTable(
        data=top_risk_table.to_dict("records"),
        columns=[{"name": c, "id": c} for c in top_risk_table.columns],
        style_cell={"textAlign": "left", "fontSize": "0.85rem", "padding": "6px"},
        style_header={"backgroundColor": "#2E86AB", "color": "white", "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"}
        ],
        page_size=10,
        sort_action="native",
    )

    return dbc.Container(
        [
            html.H1("Risk Segmentation", className="my-4"),
            html.P("Every customer scored by the model and grouped into actionable risk tiers — "
                   "this is the tab to hand to the retention team.", className="text-muted"),

            dbc.Row(
                [
                    dbc.Col(kpi_card("High Risk Customers", f"{high_risk_count:,}",
                                     "bi-exclamation-octagon", "danger"), md=3),
                    dbc.Col(kpi_card("Balance at Risk", f"{high_risk_balance:,.0f}",
                                     "bi-cash-stack", "danger"), md=3),
                    dbc.Col(kpi_card("Medium Risk Customers",
                                     f"{int(tier_stats.loc[tier_stats['RiskTier']=='Medium Risk','Customers'].values[0]):,}",
                                     "bi-exclamation-triangle", "warning"), md=3),
                    dbc.Col(kpi_card("Low Risk Customers",
                                     f"{int(tier_stats.loc[tier_stats['RiskTier']=='Low Risk','Customers'].values[0]):,}",
                                     "bi-shield-check", "success"), md=3),
                ],
                className="mb-4 g-3"
            ),

            insight_box(
                f"{high_risk_count:,} customers are flagged High Risk, holding a combined balance of "
                f"{high_risk_balance:,.0f} — this is the money that's actually at risk of leaving the bank. "
                "The validation chart below confirms the tiers are meaningful: actual churn rate rises "
                "sharply from Low to High risk. Start retention outreach with the table at the bottom."
            ),

            dbc.Row(
                [dbc.Col(dcc.Graph(figure=fig_pie), md=6), dbc.Col(dcc.Graph(figure=fig_validation), md=6)],
                className="mb-4"
            ),

            html.H5("Top 25 Highest-Risk Customers", className="mb-3"),
            risk_table,
            html.Br(),
        ],
        fluid=True
    )


# =========================================================
# Page 10 - Executive Summary
# =========================================================

def build_summary_page():
    recommendations = [
        ("bi-megaphone", "Re-engage Inactive Members",
         "Launch personalized notifications and loyalty campaigns for inactive customers — "
         "activity status is one of the clearest churn signals in the data."),
        ("bi-box-seam", "Cross-Sell to Single-Product Customers",
         "Customers with only 1 product churn at ~28% vs ~8% for those with 2 — offer additional "
         "products to strengthen loyalty."),
        ("bi-people-fill", "Target Older, High-Balance Segments",
         "Age and balance are top churn drivers — build retention offers aimed at older customers "
         "carrying higher balances."),
        ("bi-cpu", "Embed the Model into the CRM",
         "Feed the Risk Segmentation tier into the CRM so relationship managers get proactive "
         "alerts on high-risk accounts."),
    ]

    rec_cards = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.I(className=f"bi {icon}", style={"fontSize": "1.8rem", "color": "#2E86AB"}),
                            html.H6(title, className="mt-2"),
                            html.P(text, className="small text-muted mb-0"),
                        ]
                    ),
                    className="shadow-sm h-100"
                ),
                md=3
            )
            for icon, title, text in recommendations
        ],
        className="g-3 mb-4"
    )

    high_risk_balance = df.loc[df["RiskTier"] == "High Risk", "Balance"].sum()

    return dbc.Container(
        [
            html.H1("Executive Summary", className="my-4"),
            html.P("One-page recap for stakeholders: where we stand, and what to do next.",
                   className="text-muted"),

            dbc.Row(
                [
                    dbc.Col(kpi_card("Portfolio Churn Rate", f"{OVERALL_CHURN_RATE:.1%}",
                                     "bi-graph-up-arrow", "danger"), md=3),
                    dbc.Col(kpi_card("Model ROC-AUC", "0.864", "bi-cpu", "primary"), md=3),
                    dbc.Col(kpi_card("Churn Recall", "66%", "bi-bullseye", "success"), md=3),
                    dbc.Col(kpi_card("Balance at High Risk", f"{high_risk_balance:,.0f}",
                                     "bi-cash-stack", "warning"), md=3),
                ],
                className="mb-4 g-3"
            ),

            insight_box(
                "The optimized Random Forest model catches about 2 out of 3 customers who will "
                "actually churn, letting the bank act before they leave rather than after. "
                "Recommended next step: operationalize the Risk Segmentation tab into weekly "
                "retention outreach lists."
            ),

            html.H5("Business Recommendations", className="mt-4 mb-3"),
            rec_cards,
        ],
        fluid=True
    )


# =========================================================
# App Layout & Routing
# =========================================================

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        sidebar,
        html.Div(
            id="page-content",
            style={"marginLeft": "18%", "marginRight": "2%", "paddingBottom": "40px"}
        ),
    ]
)


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def render_page(pathname):
    if pathname == "/prediction":
        return prediction_page
    elif pathname == "/insights":
        return insights_page
    elif pathname == "/balance":
        return build_balance_page()
    elif pathname == "/creditcard":
        return build_creditcard_page()
    elif pathname == "/geography":
        return build_geography_page()
    elif pathname == "/demographics":
        return build_demographics_page()
    elif pathname == "/tenure":
        return build_tenure_page()
    elif pathname == "/segmentation":
        return build_segmentation_page()
    elif pathname == "/summary":
        return build_summary_page()
    return dashboard_page


# =========================================================
# Overview Dashboard Callback (filters -> KPIs + charts)
# =========================================================

@app.callback(
    Output("kpi-total", "children"),
    Output("kpi-churned", "children"),
    Output("kpi-rate", "children"),
    Output("kpi-active", "children"),
    Output("graph-geo", "figure"),
    Output("graph-age", "figure"),
    Output("graph-balance", "figure"),
    Output("graph-products", "figure"),
    Output("graph-tenure-active", "figure"),
    Output("dash-insight", "children"),
    Input("f-geo", "value"),
    Input("f-gender", "value"),
    Input("f-active", "value"),
    Input("f-products", "value"),
    Input("f-age", "value"),
)
def update_dashboard(geo, gender, active, products, age_range):
    geo = geo or []
    gender = gender or []
    active = active or []
    products = products or []

    dff = df[
        df["Geography"].isin(geo)
        & df["Gender"].isin(gender)
        & df["IsActiveMember"].isin(active)
        & df["NumOfProducts"].isin(products)
        & df["Age"].between(age_range[0], age_range[1])
    ]

    if dff.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(template=PLOTLY_TEMPLATE, title="No data for this filter combination")
        msg = insight_box("No customers match the current filters — try widening your selection.",
                           "bi-info-circle")
        return "0", "0", "0.0%", "0", empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, msg

    total = len(dff)
    churned = int(dff["Exited"].sum())
    rate = dff["Exited"].mean()
    active_n = int(dff["IsActiveMember"].sum())

    fig_geo = px.bar(
        dff.groupby("Geography")["Exited"].mean().reset_index(),
        x="Geography", y="Exited", text_auto=".1%", color="Geography",
        color_discrete_sequence=px.colors.qualitative.Set2,
        template=PLOTLY_TEMPLATE, title="Churn Rate by Geography"
    )
    fig_geo.update_layout(yaxis_tickformat=".0%", showlegend=False, yaxis_title="Churn Rate")

    fig_age = px.histogram(
        dff, x="Age", color="ExitedLabel", barmode="overlay", nbins=30,
        color_discrete_map={"Retained": COLOR_RETAINED, "Churned": COLOR_CHURNED},
        template=PLOTLY_TEMPLATE, title="Age Distribution by Churn Status"
    )
    fig_age.update_layout(legend_title_text="")

    fig_balance = px.box(
        dff, x="ExitedLabel", y="Balance", color="ExitedLabel",
        color_discrete_map={"Retained": COLOR_RETAINED, "Churned": COLOR_CHURNED},
        template=PLOTLY_TEMPLATE, title="Balance Distribution by Churn Status", points=False
    )
    fig_balance.update_layout(showlegend=False, xaxis_title="")

    prod_rate = dff.groupby("NumOfProducts")["Exited"].mean().reset_index()
    fig_products = px.bar(
        prod_rate, x="NumOfProducts", y="Exited", text_auto=".1%",
        template=PLOTLY_TEMPLATE, title="Churn Rate by Number of Products",
        color="Exited", color_continuous_scale=["#2E86AB", "#E63946"]
    )
    fig_products.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False, yaxis_title="Churn Rate")

    tenure_active = dff.groupby(["Tenure", "ActiveLabel"])["Exited"].mean().reset_index()
    fig_tenure = px.line(
        tenure_active, x="Tenure", y="Exited", color="ActiveLabel", markers=True,
        color_discrete_sequence=[COLOR_RETAINED, COLOR_CHURNED],
        template=PLOTLY_TEMPLATE, title="Churn Rate by Tenure, Split by Membership Activity"
    )
    fig_tenure.update_layout(yaxis_tickformat=".0%", yaxis_title="Churn Rate", legend_title_text="")

    delta = rate - OVERALL_CHURN_RATE
    direction = "above" if delta > 0 else "below"
    msg = insight_box(
        f"This segment's churn rate is {rate:.1%}, {abs(delta):.1%} {direction} the overall "
        f"portfolio average ({OVERALL_CHURN_RATE:.1%}). Use the filters to isolate high-risk "
        "segments (e.g. single-product, inactive, older customers) for targeted retention campaigns."
    )

    return (
        f"{total:,}",
        f"{churned:,}",
        f"{rate:.1%}",
        f"{active_n:,}",
        fig_geo, fig_age, fig_balance, fig_products, fig_tenure,
        msg
    )


@app.callback(
    Output("f-geo", "value"),
    Output("f-gender", "value"),
    Output("f-active", "value"),
    Output("f-products", "value"),
    Output("f-age", "value"),
    Input("f-reset", "n_clicks"),
    prevent_initial_call=True
)
def reset_filters(n):
    return (
        list(sorted(df["Geography"].unique())),
        list(sorted(df["Gender"].unique())),
        [0, 1],
        list(sorted(df["NumOfProducts"].unique())),
        [int(df["Age"].min()), int(df["Age"].max())],
    )


# =========================================================
# Prediction Callback
# =========================================================

@app.callback(
    Output("prediction_result", "children"),
    Input("predict", "n_clicks"),
    State("credit", "value"),
    State("age", "value"),
    State("tenure", "value"),
    State("balance", "value"),
    State("products", "value"),
    State("salary", "value"),
    State("geo", "value"),
    State("gender", "value"),
    State("card", "value"),
    State("active", "value"),
)
def prediction(n, credit, age, tenure, balance, products, salary, geo, gender, card, active):
    if n is None:
        return dbc.Card(
            dbc.CardBody(
                [
                    html.I(className="bi bi-arrow-left-circle", style={"fontSize": "2rem", "color": "#6C757D"}),
                    html.P("Fill in the customer profile and click Predict to see the churn risk assessment.",
                           className="text-muted mt-2"),
                ],
                className="text-center py-5"
            ),
            className="shadow-sm"
        )

    balance_per_product = balance / products if products not in (0, None) else 0

    data = pd.DataFrame({
        "CreditScore": [credit],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [products],
        "HasCrCard": [card],
        "IsActiveMember": [active],
        "EstimatedSalary": [salary],
        "BalancePerProduct": [balance_per_product],
        "Geography": [geo],
        "Gender": [gender],
    })

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": "Churn Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": COLOR_CHURNED if pred == 1 else COLOR_RETAINED},
            "steps": [
                {"range": [0, 40], "color": "#e8f4f8"},
                {"range": [40, 70], "color": "#fdebd0"},
                {"range": [70, 100], "color": "#fadbd8"},
            ],
        }
    ))
    gauge.update_layout(height=280, margin=dict(t=50, b=10, l=30, r=30), template=PLOTLY_TEMPLATE)

    risk_factors = []
    if age is not None and age > 44:
        risk_factors.append("Age above the typical churned-customer average (~45)")
    if products is not None and products == 1:
        risk_factors.append("Single product — customers with 1 product churn far more often (~28%)")
    if active == 0:
        risk_factors.append("Inactive membership status")
    if balance is not None and balance > 91000:
        risk_factors.append("High account balance — a segment shown to churn more often")
    if not risk_factors:
        risk_factors.append("No major red flags detected among the strongest churn drivers")

    result_card = dbc.Card(
        [
            dbc.CardHeader(
                [html.I(className="bi bi-exclamation-triangle me-2" if pred == 1 else "bi bi-check-circle me-2"),
                 "High Risk Customer" if pred == 1 else "Low Risk Customer"],
                className=("bg-danger text-white" if pred == 1 else "bg-success text-white")
            ),
            dbc.CardBody(
                [
                    dcc.Graph(figure=gauge, config={"displayModeBar": False}),
                    html.H6("Contributing factors", className="mt-3"),
                    html.Ul([html.Li(r) for r in risk_factors]),
                ]
            ),
        ],
        className="shadow-sm"
    )

    return result_card


if __name__ == "__main__":
    app.run(debug=True)