import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Hyderabad House Predictor", layout="wide")

# ---------------- LOGIN SYSTEM ----------------
users = {"admin": "1234", "user": "abcd"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- LOAD DATA ----------------
data = pd.read_csv("data.csv")

# Encode location
data_encoded = pd.get_dummies(data, columns=["location"])

X = data_encoded.drop("price", axis=1)
y = data_encoded["price"]

# ---------------- TRAIN MODELS ----------------
lr = LinearRegression()
dt = DecisionTreeRegressor()

lr.fit(X, y)
dt.fit(X, y)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Prediction", "📊 Analytics", "📁 Dataset"])

# ---------------- PREDICTION PAGE ----------------
if page == "🏠 Prediction":
    st.title("🏠 Hyderabad House Price Predictor")

    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.slider("Area (sq ft)", 500, 3000, 1000)

    with col2:
        bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])

    with col3:
        location = st.selectbox("Location", data["location"].unique())

        # Get training columns
    model_columns = X.columns

    # Create empty dataframe
    input_df = pd.DataFrame(columns=model_columns)

    # Fill base values
    input_df.loc[0, 'area'] = area
    input_df.loc[0, 'bedrooms'] = bedrooms

    # Set all locations = 0
    for col in model_columns:
        if 'location_' in col:
            input_df.loc[0, col] = 0

    # Set selected location = 1
    input_df.loc[0, f'location_{location}'] = 1

    # Fill missing values
    input_df = input_df.fillna(0)

    if st.button("🔍 Predict Price"):
        with st.spinner("Calculating price..."):
            pred_lr = lr.predict(input_df)[0]
            pred_dt = dt.predict(input_df)[0]

        st.success(f"💰 Linear Regression: ₹ {int(pred_lr):,}")
        st.info(f"🌳 Decision Tree: ₹ {int(pred_dt):,}")

        # AI Suggestions
        st.subheader("🤖 AI Suggestions")
        if location == "Banjara Hills":
            st.write("💎 Premium location → High investment value")
        elif area > 2000:
            st.write("🏡 Large house → Best for big families")
        elif bedrooms >= 4:
            st.write("👨‍👩‍👧‍👦 Suitable for joint families")
        else:
            st.write("👍 Budget-friendly choice")

        # Save prediction
        result = input_df.copy()
        result["prediction"] = int(pred_lr)

        try:
            result.to_csv("predictions.csv", mode="a", header=False, index=False)
        except:
            result.to_csv("predictions.csv", index=False)

# ---------------- ANALYTICS PAGE ----------------
elif page == "📊 Analytics":
    st.title("📊 Advanced Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.scatter(
            data,
            x="area",
            y="price",
            color="location",
            title="Area vs Price"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        avg_price = data.groupby("location")["price"].mean().reset_index()

        fig2 = px.bar(
            avg_price,
            x="location",
            y="price",
            color="location",
            title="Average Price by Location"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Model metrics
    st.subheader("📈 Model Performance")

    y_pred = lr.predict(X)
    mae = mean_absolute_error(y, y_pred)

    st.write("✔️ Accuracy (R²):", round(lr.score(X, y), 3))
    st.write("📉 Mean Absolute Error:", int(mae))

 # ---------------- DATASET PAGE ----------------
elif page == "📁 Dataset":
    st.title("📁 Dataset Viewer")

    st.dataframe(data)

    if st.checkbox("Show Raw Data Info"):
        st.write(data.describe())

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("🚀 Built with Python, Machine Learning & Streamlit")