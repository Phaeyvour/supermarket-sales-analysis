import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    model = joblib.load('sales_model.pkl')
    le_category = joblib.load('le_category.pkl')
    le_segment = joblib.load('le_segment.pkl')
    le_region = joblib.load('le_region.pkl')
    le_ship_mode = joblib.load('le_ship_mode.pkl')
    return model, le_category, le_segment, le_region, le_ship_mode

model, le_category, le_segment, le_region, le_ship_mode = load_model()

st.title("Superstore Sales Predictor")
st.markdown("Predict sales based on product and order details")

st.sidebar.header("Product Details")

quantity = st.sidebar.number_input("Quantity", min_value=1, max_value=100, value=5)
unit_price = st.sidebar.number_input("Unit Price ($)", min_value=0.0, max_value=10000.0, value=100.0, step=10.0)
discount = st.sidebar.slider("Discount (%)", min_value=0, max_value=80, value=15) / 100

category = st.sidebar.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])
segment = st.sidebar.selectbox("Customer Segment", ["Consumer", "Corporate", "Home Office"])
region = st.sidebar.selectbox("Region", ["Central", "East", "South", "West"])
ship_mode = st.sidebar.selectbox("Ship Mode", ["First Class", "Same Day", "Second Class", "Standard Class"])

year = st.sidebar.number_input("Year", min_value=2020, max_value=2030, value=2024)
month = st.sidebar.selectbox("Month", list(range(1, 13)), index=10)
quarter = (month - 1) // 3 + 1
day_of_week = st.sidebar.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)

day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
day_num = day_map[day_of_week]

if st.sidebar.button("Predict Sales", type="primary"):
    cat_enc = le_category.transform([category])[0]
    seg_enc = le_segment.transform([segment])[0]
    reg_enc = le_region.transform([region])[0]
    ship_enc = le_ship_mode.transform([ship_mode])[0]

    input_df = pd.DataFrame({
        'Quantity': [quantity],
        'Discount': [discount],
        'Unit_Price': [unit_price],
        'Category_encoded': [cat_enc],
        'Segment_encoded': [seg_enc],
        'Region_encoded': [reg_enc],
        'Ship_Mode_encoded': [ship_enc],
        'Year': [year],
        'Month': [month],
        'Quarter': [quarter],
        'DayOfWeek': [day_num]
    })

    prediction = model.predict(input_df)[0]
    expected = quantity * unit_price * (1 - discount)

    st.success("Prediction Complete!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Sales", f"${prediction:.2f}")
    col2.metric("Expected (Formula)", f"${expected:.2f}")
    col3.metric("Difference", f"${abs(prediction - expected):.2f}")

    st.divider()
    st.subheader("Details")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Quantity:** {quantity}")
        st.write(f"**Unit Price:** ${unit_price:.2f}")
        st.write(f"**Discount:** {discount*100:.0f}%")
    with col2:
        st.write(f"**Category:** {category}")
        st.write(f"**Region:** {region}")
        st.write(f"**Segment:** {segment}")

else:
    st.info("Enter product details in the sidebar and click Predict Sales")

    st.divider()
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("R2 Score", "0.83")
    col2.metric("MAE", "$30")
    col3.metric("RMSE", "$316")

st.caption("Built with Random Forest | Accuracy: 83%")
