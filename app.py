import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Library Book Demand Prediction",
    page_icon="📚",
    layout="wide"
)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("📚 Library Book Demand Prediction System")

st.write(
    "Predict future library book demand using "
    "Machine Learning and historical circulation data."
)

st.divider()

# ---------------------------------------------------------
# SAMPLE DATASET
# ---------------------------------------------------------

np.random.seed(42)

categories = [
    "History",
    "Literature",
    "Physics",
    "Mathematics",
    "Medical",
    "Business",
    "Engineering",
    "Computer Science"
]

data = []

for i in range(500):

    category = np.random.choice(categories)
    month = np.random.randint(1, 13)
    semester = np.random.randint(1, 9)
    students = np.random.randint(50, 1001)
    borrowing = np.random.randint(20, 250)
    previous_demand = np.random.randint(30, 300)

    demand = (
        0.35 * borrowing
        + 0.30 * previous_demand
        + 0.08 * students
        + 5 * semester
        + np.random.normal(0, 20)
    )

    data.append([
        category,
        month,
        semester,
        students,
        borrowing,
        previous_demand,
        max(10, int(demand))
    ])

df = pd.DataFrame(
    data,
    columns=[
        "Category",
        "Month",
        "Semester",
        "Students",
        "Borrowing",
        "Previous_Demand",
        "Demand"
    ]
)

# ---------------------------------------------------------
# CATEGORY ENCODING
# ---------------------------------------------------------

category_mapping = {
    category: index
    for index, category in enumerate(categories)
}

df["Category_Code"] = df["Category"].map(category_mapping)

# ---------------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------------

features = [
    "Category_Code",
    "Month",
    "Semester",
    "Students",
    "Borrowing",
    "Previous_Demand"
]

X = df[features]
y = df["Demand"]

# ---------------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ---------------------------------------------------------
# RANDOM FOREST MODEL
# ---------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("🎛️ Input Parameters")

st.sidebar.write(
    "Enter the book and circulation details:"
)

category = st.sidebar.selectbox(
    "📖 Book Category",
    categories,
    index=7
)

month_name = st.sidebar.selectbox(
    "📅 Month",
    [
        "January (1)",
        "February (2)",
        "March (3)",
        "April (4)",
        "May (5)",
        "June (6)",
        "July (7)",
        "August (8)",
        "September (9)",
        "October (10)",
        "November (11)",
        "December (12)"
    ],
    index=3
)

month = int(
    month_name.split("(")[1].replace(")", "")
)

semester = st.sidebar.number_input(
    "🎓 Academic Semester",
    min_value=1,
    max_value=8,
    value=4,
    step=1
)

students = st.sidebar.number_input(
    "👨‍🎓 Number of Enrolled Students",
    min_value=1,
    max_value=5000,
    value=280,
    step=10
)

borrowing = st.sidebar.number_input(
    "📚 Previous Borrowing Count",
    min_value=1,
    max_value=1000,
    value=80,
    step=1
)

previous_demand = st.sidebar.number_input(
    "📊 Previous Demand Count",
    min_value=1,
    max_value=1000,
    value=80,
    step=1
)

predict_button = st.sidebar.button(
    "🔮 Predict Demand",
    use_container_width=True
)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if predict_button:

    category_code = category_mapping[category]

    input_data = pd.DataFrame(
        [[
            category_code,
            month,
            semester,
            students,
            borrowing,
            previous_demand
        ]],
        columns=features
    )

    prediction = model.predict(input_data)[0]
    prediction = int(round(prediction))

    if prediction >= 150:
        demand_level = "High"

    elif prediction >= 80:
        demand_level = "Medium"

    else:
        demand_level = "Low"

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.header("📈 Prediction Result")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Predicted Demand",
            f"{prediction} books"
        )

    with col2:
        st.metric(
            "Demand Level",
            demand_level
        )

    with col3:
        st.metric(
            "Model MAE",
            f"{mae:.2f}"
        )

    with col4:
        st.metric(
            "Model R² Score",
            f"{r2:.4f}"
        )

    st.divider()

    # -----------------------------------------------------
    # RECOMMENDATION
    # -----------------------------------------------------

    if demand_level == "High":

        st.error(
            "🚨 HIGH PRIORITY – Maintain Sufficient Stock"
        )

        st.subheader("📌 Recommendation")

        st.write(
            f"**{category}** books are expected to have "
            "high demand."
        )

        st.write(
            "Increase available physical copies, "
            "place priority reorders with suppliers, "
            "and maintain sufficient reserve stock."
        )

    elif demand_level == "Medium":

        st.warning(
            "⚠️ MEDIUM PRIORITY – Monitor Stock"
        )

        st.subheader("📌 Recommendation")

        st.write(
            f"**{category}** books are expected to have "
            "moderate demand."
        )

        st.write(
            "Monitor borrowing activity and maintain "
            "adequate stock based on usage."
        )

    else:

        st.success(
            "✅ LOW PRIORITY – Current Stock May Be Sufficient"
        )

        st.subheader("📌 Recommendation")

        st.write(
            f"**{category}** books are expected to have "
            "low demand."
        )

        st.write(
            "Existing stock may be sufficient. "
            "Continue monitoring borrowing patterns."
        )

# ---------------------------------------------------------
# ANALYTICS
# ---------------------------------------------------------

st.divider()

st.header("📊 Visualizations & Analytics")

tab1, tab2, tab3 = st.tabs(
    [
        "📈 Monthly Trend",
        "📚 Category Analysis",
        "🤖 Model Details"
    ]
)

# ---------------------------------------------------------
# MONTHLY TREND
# ---------------------------------------------------------

with tab1:

    st.subheader("Historical Monthly Demand")

    monthly_demand = (
        df.groupby("Month")["Demand"]
        .mean()
    )

    month_labels = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]

    monthly_demand.index = [
        month_labels[i - 1]
        for i in monthly_demand.index
    ]

    st.line_chart(monthly_demand)

    st.caption(
        "Monthly Average Book Demand Trend"
    )

# ---------------------------------------------------------
# CATEGORY ANALYSIS
# ---------------------------------------------------------

with tab2:

    st.subheader("Category-wise Demand Analysis")

    category_demand = (
        df.groupby("Category")["Demand"]
        .mean()
        .sort_values()
    )

    st.bar_chart(category_demand)

    st.caption(
        "Average Demand by Book Category"
    )

# ---------------------------------------------------------
# MODEL DETAILS
# ---------------------------------------------------------

with tab3:

    st.subheader("🤖 Machine Learning Model")

    st.write(
        "**Algorithm:** Random Forest Regressor"
    )

    st.write(
        f"**Mean Absolute Error (MAE):** {mae:.2f}"
    )

    st.write(
        f"**R² Score:** {r2:.4f}"
    )

    st.write(
        "**Purpose:** Predict future library book "
        "demand using historical circulation data."
    )

# ---------------------------------------------------------
# DATASET EXPLORER
# ---------------------------------------------------------

st.divider()

st.header("📋 Dataset Explorer")

st.dataframe(
    df.head(20),
    use_container_width=True
)

st.caption(
    "📚 Library Book Demand Prediction System | "
    "Random Forest Machine Learning"
)
