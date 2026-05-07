Financial Risk Analytics Lab using Synthetic Banking Data
Industry Use Case
Banks, NBFCs, credit card companies, and FinTech firms use financial risk analytics to evaluate whether a customer is likely to repay a loan on time. Since real banking data is confidential, this lab uses synthetic customer data to simulate a real-world lending risk assessment workflow.

Lab Objective
In this lab, students will learn how to:

Create a synthetic banking dataset with realistic customer information.
Understand key financial risk input parameters such as income, EMI, credit score, employment stability, and missed payments.
Calculate Debt-to-Income Ratio (DTI).
Classify customers into Low, Medium, and High financial risk categories.
Generate bank decisions such as Approve, Approve with Conditions, or Reject / Manual Review.
Visualize financial risk patterns using charts.
Interpret risk strategy and outcome from a business decision-making perspective.
Real-World Relevance
This workflow is similar to how lending institutions use data analytics and AI to support:

Loan pre-screening
Credit risk assessment
Customer segmentation
Risk-based pricing
Fraud and default prevention
Responsible lending decisions
Dataset Note
This dataset is fully synthetic and created only for educational purposes. It does not contain real customer data.


[ ]
# CELL 1: Import required libraries
# pandas is used for creating and analyzing tabular data.
# numpy is used for generating synthetic numerical data.
# matplotlib and seaborn are used for data visualization.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display all columns clearly in Colab output
pd.set_option('display.max_columns', None)

print("Libraries imported successfully.")

[ ]
# CELL 2: Create synthetic customer names and professional categories
# These names and professions make the dataset easier to understand as a real-world banking scenario.

np.random.seed(42)

customer_names = [
    "Aarav Sharma", "Priya Mehta", "Rohan Verma", "Neha Gupta", "Karan Malhotra",
    "Ananya Singh", "Vikram Rao", "Sneha Iyer", "Rahul Nair", "Meera Joshi",
    "Arjun Kapoor", "Pooja Sinha", "Nitin Bansal", "Ritika Jain", "Sahil Khan",
    "Divya Menon", "Manish Reddy", "Kavita Das", "Harsh Vardhan", "Isha Agarwal"
]

professions = [
    "Salaried Employee",
    "Business Owner",
    "Freelancer",
    "Self-Employed",
    "Government Employee",
    "Consultant",
    "Teacher",
    "Doctor",
    "IT Professional",
    "Small Trader"
]

print("Customer names and professions created successfully.")

[ ]
# CELL 3: Generate synthetic banking dataset
# We are creating 100 customer records.
# Each record includes realistic financial and professional input parameters.

n = 100

df = pd.DataFrame({
    "Customer_ID": range(1001, 1001 + n),
    "Customer_Name": np.random.choice(customer_names, n),
    "Age": np.random.randint(22, 60, n),
    "Profession": np.random.choice(professions, n),
    "Monthly_Income": np.random.randint(25000, 200000, n),
    "Existing_EMI": np.random.randint(0, 80000, n),
    "Loan_Amount_Requested": np.random.randint(100000, 3000000, n),
    "Credit_Score": np.random.randint(300, 900, n),
    "Employment_Years": np.random.randint(0, 25, n),
    "Missed_Payments_Last_12_Months": np.random.randint(0, 8, n)
})

# Display first 10 records
df.head(10)

[ ]
# CELL 4: Calculate Debt-to-Income Ratio
# DTI = Existing EMI / Monthly Income
# It shows what percentage of income is already going toward debt repayment.
# Example: If income is 50,000 and EMI is 20,000, DTI = 0.40 or 40%.

df["DTI"] = (df["Existing_EMI"] / df["Monthly_Income"]).round(2)

df[["Customer_Name", "Monthly_Income", "Existing_EMI", "DTI"]].head(10)

[ ]
# CELL 5: Create financial risk classification logic
# This is a simplified rule-based risk model.
# Real banks use more complex AI/ML models, but the logic below is beginner-friendly.

risk_level = []
bank_decision = []
strategy = []
outcome = []

for index, row in df.iterrows():
    dti = row["DTI"]
    credit_score = row["Credit_Score"]
    missed_payments = row["Missed_Payments_Last_12_Months"]
    employment_years = row["Employment_Years"]

    # High Risk:
    # Customer already has high debt burden OR poor credit score OR many missed payments.
    if dti > 0.50 or credit_score < 500 or missed_payments >= 5:
        risk_level.append("High")
        bank_decision.append("Reject / Manual Review")
        strategy.append("Reduce loan amount, request guarantor, or ask customer to improve credit history.")
        outcome.append("High default probability. Lending should be avoided or reviewed carefully.")

    # Medium Risk:
    # Customer has moderate debt burden, average credit score, or some missed payments.
    elif dti > 0.35 or credit_score < 650 or missed_payments >= 2 or employment_years < 2:
        risk_level.append("Medium")
        bank_decision.append("Approve with Conditions")
        strategy.append("Offer lower loan amount, higher interest rate, or shorter tenure.")
        outcome.append("Customer can be served with controlled lending terms.")

    # Low Risk:
    # Customer has low debt burden, good credit score, and stable repayment behavior.
    else:
        risk_level.append("Low")
        bank_decision.append("Approve")
        strategy.append("Offer standard loan with competitive interest rate.")
        outcome.append("Strong repayment capacity. Customer is suitable for loan approval.")

df["Risk_Level"] = risk_level
df["Bank_Decision"] = bank_decision
df["Strategy"] = strategy
df["Outcome"] = outcome

df.head(10)

[ ]
# CELL 6: View complete sample customer risk profile
# This gives a business-friendly view of customer risk, bank strategy, and expected outcome.

selected_columns = [
    "Customer_ID", "Customer_Name", "Age", "Profession",
    "Monthly_Income", "Existing_EMI", "Loan_Amount_Requested",
    "Credit_Score", "Employment_Years", "Missed_Payments_Last_12_Months",
    "DTI", "Risk_Level", "Bank_Decision", "Strategy", "Outcome"
]

df[selected_columns].head(10)

[ ]
# CELL 7: Risk level distribution
# This tells us how many customers fall into Low, Medium, and High risk categories.

risk_counts = df["Risk_Level"].value_counts()

print("Risk Level Distribution:")
print(risk_counts)

[ ]
# CELL 8: Visualize risk level distribution
# This chart helps managers quickly understand overall portfolio risk.

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Risk_Level", order=["Low", "Medium", "High"])
plt.title("Distribution of Customer Financial Risk Levels")
plt.xlabel("Risk Level")
plt.ylabel("Number of Customers")
plt.show()

[ ]
# CELL 9: Analyze DTI by risk level
# Higher DTI usually means higher financial stress.
# This boxplot shows how DTI differs across risk groups.

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Risk_Level", y="DTI", order=["Low", "Medium", "High"])
plt.title("Debt-to-Income Ratio by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Debt-to-Income Ratio")
plt.show()

[ ]
# CELL 10: Analyze credit score by risk level
# Lower credit score generally indicates higher repayment risk.
# This chart helps connect credit history with risk classification.

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Risk_Level", y="Credit_Score", order=["Low", "Medium", "High"])
plt.title("Credit Score by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Credit Score")
plt.show()

[ ]
# CELL 11: Analyze profession-wise average risk indicators
# This helps banks understand which professional groups may need more careful assessment.
# Note: This is synthetic data, so students should not treat it as real-world truth.

profession_summary = df.groupby("Profession").agg(
    Average_Income=("Monthly_Income", "mean"),
    Average_DTI=("DTI", "mean"),
    Average_Credit_Score=("Credit_Score", "mean"),
    Average_Missed_Payments=("Missed_Payments_Last_12_Months", "mean")
).round(2)

profession_summary.sort_values(by="Average_DTI", ascending=False)

[ ]
# CELL 12: Identify high-risk customers
# These customers need manual review, rejection, or stricter loan terms.

high_risk_customers = df[df["Risk_Level"] == "High"]

high_risk_customers[selected_columns].head(10)

[ ]
# CELL 13: Identify customers suitable for loan approval
# These customers have stronger repayment capacity and lower financial stress.

low_risk_customers = df[df["Risk_Level"] == "Low"]

low_risk_customers[selected_columns].head(10)

[ ]
# CELL 14: Create a business decision summary
# This summary helps management understand how many applications can be approved, conditionally approved, or rejected.

decision_summary = df["Bank_Decision"].value_counts()

print("Bank Decision Summary:")
print(decision_summary)

[ ]
# CELL 15: Visualize bank decisions
# This is useful for presentation and reporting.

plt.figure(figsize=(8, 5))
sns.countplot(data=df, y="Bank_Decision")
plt.title("Bank Decision Summary")
plt.xlabel("Number of Customers")
plt.ylabel("Bank Decision")
plt.show()

[ ]
# CELL 16: Create a scenario-based customer explanation function
# This function allows students to enter a customer ID and get a simple business interpretation.

def explain_customer_risk(customer_id):
    customer = df[df["Customer_ID"] == customer_id]

    if customer.empty:
        print("Customer ID not found.")
        return

    row = customer.iloc[0]

    print("Customer Risk Profile")
    print("----------------------")
    print(f"Customer Name: {row['Customer_Name']}")
    print(f"Age: {row['Age']}")
    print(f"Profession: {row['Profession']}")
    print(f"Monthly Income: ₹{row['Monthly_Income']}")
    print(f"Existing EMI: ₹{row['Existing_EMI']}")
    print(f"Loan Amount Requested: ₹{row['Loan_Amount_Requested']}")
    print(f"Credit Score: {row['Credit_Score']}")
    print(f"Employment Years: {row['Employment_Years']}")
    print(f"Missed Payments: {row['Missed_Payments_Last_12_Months']}")
    print(f"DTI: {row['DTI']}")
    print(f"Risk Level: {row['Risk_Level']}")
    print(f"Bank Decision: {row['Bank_Decision']}")
    print(f"Strategy: {row['Strategy']}")
    print(f"Outcome: {row['Outcome']}")

# Example: Change the customer ID to test different customers
explain_customer_risk(1001)

[ ]
# CELL 17: Export dataset as CSV
# Students can download this file and use it in Excel, Power BI, or GitHub submission.

df.to_csv("synthetic_financial_risk_dataset.csv", index=False)

print("Dataset exported successfully as synthetic_financial_risk_dataset.csv")

[ ]
# CELL 18: Download the dataset in Google Colab
# This cell works only in Google Colab.
# If running locally, the file will be saved in the current working directory.

try:
    from google.colab import files
    files.download("synthetic_financial_risk_dataset.csv")
except ImportError:
    print("Download function is available only in Google Colab.")
    print("File has been saved locally as synthetic_financial_risk_dataset.csv")

[ ]
# CELL 19: Save risk summary report as CSV
# This summary can be used in student reports or dashboards.

summary_report = df.groupby("Risk_Level").agg(
    Total_Customers=("Customer_ID", "count"),
    Average_Income=("Monthly_Income", "mean"),
    Average_DTI=("DTI", "mean"),
    Average_Credit_Score=("Credit_Score", "mean"),
    Average_Missed_Payments=("Missed_Payments_Last_12_Months", "mean")
).round(2)

summary_report.to_csv("risk_summary_report.csv")

summary_report

[ ]
# CELL 20: Final outcomes and real-world industry relevance
# This final cell prints the learning outcomes and business relevance of the lab.

print("LAB OUTCOMES")
print("------------")
print("1. Students created a synthetic banking dataset with realistic customer features.")
print("2. Students calculated Debt-to-Income Ratio, a key financial risk indicator.")
print("3. Students classified customers into Low, Medium, and High risk groups.")
print("4. Students connected credit score, EMI burden, missed payments, and employment stability with lending risk.")
print("5. Students generated bank decisions and risk-based strategies.")
print("6. Students visualized financial risk patterns using charts.")
print("7. Students understood how AI and analytics support loan decision-making.")

print("\nREAL-WORLD INDUSTRY RELEVANCE")
print("-----------------------------")
print("Banks and FinTech companies use similar risk analytics workflows for loan approval, credit card approval,")
print("risk-based pricing, customer profiling, fraud prevention, and portfolio risk monitoring.")
print("This lab simulates a simplified version of real-world credit risk analytics used in financial institutions.")
Student Mini Assignment
Students should answer the following questions after completing the lab:

Identify any 5 high-risk customers and explain why they are risky.
Identify any 5 low-risk customers and explain why they are safer for loan approval.
Explain the role of DTI in financial risk assessment.
Explain how credit score affects lending decisions.
Suggest one improvement that can make this risk model more realistic.
Explain how banks can use this type of analysis before approving loans.
Suggested GitHub Submission
Students can upload the following files to GitHub:

Financial-Risk-Analytics-Lab/
│
├── README.md
├── notebook/
│   └── Financial_Risk_Analytics_Lab.ipynb
├── dataset/
│   └── synthetic_financial_risk_dataset.csv
├── screenshots/
│   ├── risk_distribution.png
│   ├── dti_by_risk.png
│   └── credit_score_by_risk.png
└── report/
    └── financial_risk_analysis_report.pdf
Industry Interpretation
This project can be described as:

"An AI-driven financial risk analytics project that uses synthetic banking data to classify customers into risk levels and recommend lending decisions based on income, EMI burden, credit score, employment stability, and repayment history."# Financial_Risk_Analytics_Industry_Use_Case_Lab.ipynb
