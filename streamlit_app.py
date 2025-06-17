import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

st.set_page_config(page_title="Outstanding Debt Prediction",layout="centered")
st.image("CreditScore.png", use_container_width=True)
page = st.sidebar.selectbox("Select Page",["Introduction 📘","Visualization 📊","Prediction 🔮"])
st.sidebar.title("Outstanding Debt Prediction")
df = pd.read_csv("./Credit.csv")

if page == "Introduction 📘":
    st.title("💰Outstanding Debt Prediction")
    st.markdown("This app is used to explore the influencing factors of finance in society to predict the outstanding debt of customers. We use datasets spanning a large range and eventually establish a regression model to estimate the relationship among delay from due date, change in credit and outstanding debt.")
    # df = pd.read_csv("Credit.csv")
    st.subheader(" Data Preview")
    st.dataframe(df.head(5))
    st.subheader(" Summary Statistics")
    st.dataframe(df.describe())


elif page == "Visualization 📊":
    st.title("Data Visualization")
    st.subheader("Explore how different factors impact credit score! 🏦")
    st.write("")
    st.write("")
    st.write("")

    st.subheader("Looker dashboard:")
    st.components.v1.iframe("https://lookerstudio.google.com/embed/reporting/1862255d-c299-4b3b-8ad4-f3294ab171d7/page/etzNF", height=480, width=800)



elif page == "Prediction 🔮":
    st.title("04 Prediction with Linear Regression")

    # ---------------------------------------------
    # Data Preprocessing
    # ---------------------------------------------
    st.subheader("Data Preprocessing")

    # --------------------------------------------------------------------------------
    # Clean Data
    # --------------------------------------------------------------------------------
    def convert_id_to_decimal(id_value):
        """Convert hex IDs like '0x1602' to decimal integers."""
        if isinstance(id_value, str) and id_value.startswith('0x'):
            return int(id_value, 16)
        return id_value

    def convert_customer_id_to_decimal(customer_id):
        """Convert 'CUS_0x...' strings to decimal integers."""
        if isinstance(customer_id, str) and customer_id.startswith('CUS_0x'):
            hex_part = customer_id[6:]
            return int(hex_part, 16)
        return customer_id

    def credit_history_to_float(val):
        """Convert 'x Years y Months' strings to total months."""
        if isinstance(val, str):
            match = re.match(r'(\d+)\s+Years.*?(\d+)\s+Months', val)
            if match:
                years, months = map(int, match.groups())
                return years + months / 12
        return np.nan

    def clean_num_of_loan(column):
        """Clean Num_of_Loan column by converting valid numbers and handling anomalies."""
        def convert_value(val):
            val = val.strip()
            if val == '-100' or val.endswith('_') or not val.isdigit():
                return np.nan
            return int(val)
        return column.astype(str).apply(convert_value)

    # -----------------------------
    # 2. Initial Fixes
    # -----------------------------

    # Convert IDs
    df['ID'] = df['ID'].apply(convert_id_to_decimal)
    df['Customer_ID'] = df['Customer_ID'].apply(convert_customer_id_to_decimal)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # -----------------------------
    # 3. Column-specific Cleaning
    # -----------------------------

    # Age: convert to numeric and filter invalid ranges
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Age'] = df['Age'].apply(lambda x: np.nan if x < 0 or x > 120 else x)

    # SSN: extract valid SSN pattern
    df['SSN'] = df['SSN'].astype(str).str.extract(r'(\d{3}-\d{2}-\d{4})', expand=False)

    # Occupation: replace unwanted placeholders
    df['Occupation'] = df['Occupation'].replace(['_______', '_', 'nan', 'NaN', np.nan], np.nan)

    # Numeric columns: extract valid numeric values
    numeric_cols = [
        'Annual_Income', 'Monthly_Inhand_Salary', 'Interest_Rate', 'Changed_Credit_Limit',
        'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
        'Amount_invested_monthly', 'Monthly_Balance'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.extract(r'([-\d.]+)')[0], errors='coerce')

    # Num of Delayed Payment & Credit Inquiries
    df['Num_of_Delayed_Payment'] = pd.to_numeric(
        df['Num_of_Delayed_Payment'].astype(str).str.extract(r'(\d+)')[0], errors='coerce'
    )
    df['Num_Credit_Inquiries'] = pd.to_numeric(
        df['Num_Credit_Inquiries'].astype(str).str.extract(r'(\d+)')[0], errors='coerce'
    )

    # Payment Behaviour: keep only valid categories
    valid_payment_behaviours = [
        'Low_spent_Small_value_payments', 'Low_spent_Large_value_payments',
        'High_spent_Small_value_payments', 'High_spent_Large_value_payments',
        'High_spent_Medium_value_payments', 'Low_spent_Medium_value_payments'
    ]
    df['Payment_Behaviour'] = df['Payment_Behaviour'].where(
        df['Payment_Behaviour'].isin(valid_payment_behaviours), np.nan
    )

    # Credit History Age: convert to months
    df['Credit_History_Age'] = df['Credit_History_Age'].apply(credit_history_to_float)

    # Type of Loan: remove 'Not Specified' and clean
    df['Type_of_Loan'] = df['Type_of_Loan'].astype(str).str.replace('Not Specified,?', '', regex=True).str.strip()
    df['Type_of_Loan'] = df['Type_of_Loan'].replace(['', 'nan', 'NaN'], np.nan)

    # Credit Mix: keep only valid values
    df['Credit_Mix'] = df['Credit_Mix'].where(df['Credit_Mix'].isin(['Bad', 'Standard', 'Good']), np.nan)

    # Payment of Min Amount: allow only Yes/No/NM
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace(['NM', 'No', 'Yes'], ['NM', 'No', 'Yes'])
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].where(
        df['Payment_of_Min_Amount'].isin(['Yes', 'No', 'NM']), np.nan
    )

    # Num_of_Loan: clean and convert
    df['Num_of_Loan'] = clean_num_of_loan(df['Num_of_Loan'])

    # -----------------------------
    # 4. Final Touches
    # -----------------------------

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Drop rows with any missing values
    df.dropna(inplace=True)

    # Label Encoding for categorical columns
    categorical_cols = [
        'ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Occupation',
        'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount',
        'Payment_Behaviour', 'Credit_Score'
    ]

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])



    # Sidebar - Feature & Target selection
    st.sidebar.header("Feature & Target Selection")
    all_vars = list(df.columns)
    features_selection = st.sidebar.multiselect("Select Features (X)", all_vars, default='ID')
    target_selection = st.sidebar.selectbox("Select Target Variable (Y)", all_vars)
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics to Display",
        ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R2 Score"]
    )

    if features_selection and target_selection:
        X = df[features_selection]
        y = df[target_selection]

        st.subheader("Selected Features and Target")
        st.write("**X (Features):**")
        st.dataframe(X.head())
        st.write("**y (Target):**")
        st.dataframe(y.head())

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

        # Evaluation
        st.subheader("Model Evaluation")
        if "Mean Squared Error (MSE)" in selected_metrics:
            mse = metrics.mean_squared_error(y_test, predictions)
            st.write(f"- **MSE:** {mse:,.2f}")

        if "Mean Absolute Error (MAE)" in selected_metrics:
            mae = metrics.mean_absolute_error(y_test, predictions)
            st.write(f"- **MAE:** {mae:,.2f}")
        else:
            mae = None

        if "R2 Score" in selected_metrics:
            r2 = metrics.r2_score(y_test, predictions)
            st.write(f"- **R² Score:** {r2:.3f}")

        if mae is not None:
            st.success(f"'My model performance is of {np.round(mae, 2)}")

        # Plot Actual vs Predicted
        st.subheader("📉 Actual vs Predicted")
        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions, alpha=0.5, s=3)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

    else:
        st.warning("Please select at least one feature and a target variable from the sidebar.")
    # 🔮 Simple Prediction Interface
    st.markdown("---")
    st.subheader("📩 Make a Prediction with Your Input")

    if st.button("Predict Outstanding Debt"):
      # 自动构建输入数据，顺序和features_selection一致
      input_values = []
      for feature in features_selection:
           if feature == "Changed_Credit_Limit":
            input_values.append(input_limit)
          elif feature == "Delay_from_due_date":
            input_values.append(input_delay)
          else:
            input_values.append(0)  # 或者提示错误

      input_df = pd.DataFrame([input_values], columns=features_selection)

      try:
          prediction = model.predict(input_df)
          st.success(f"📊 Predicted Outstanding Debt: **{prediction[0]:.2f}**")
      except Exception as e:
          st.error(f"Prediction failed: {e}")


    else:  
        st.info("To use the prediction input, please include both 'Changed_Credit_Limit' and 'Delay_from_due_date' in the feature selection.")





