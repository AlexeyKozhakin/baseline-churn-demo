import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

# Title
st.title("🎮 Player Churn Prediction (Analytical Model)")

# Layout: Image + QR Code side by side
col1, col2 = st.columns([3, 1])

with col1:
    # Display main image
    image_path = os.path.join("img", "churn_img.png")
    st.image(image_path, use_container_width=True, caption="📊 Player Churn Analysis")

with col2:
    # Display QR Code with caption
    qr_path = os.path.join("img", "qr_code.png")
    st.image(qr_path, use_container_width=True)
    st.markdown("📱 **Try it yourself!**\nScan the QR code to test the app.")

# Input: Number of inactive days
N = st.slider("📅 How many days has the player been inactive?", 1, 30, 7)

# Selection: Activity frequency
rate_options = {
    "Once per week": 1/7,
    "Twice per week": 2/7,
    "Once per month": 1/30,  # Approximately once every 4 weeks
    "On weekends": 2/7,
    "Three times per week": 3/7,
}
activity_rate = st.selectbox("🎮 Player's activity frequency:", list(rate_options.keys()))
R = rate_options[activity_rate]

# Logistic regression formula coefficients
a, b = 1, 1  # Can be adjusted based on data

# Analytical logistic regression function
def logistic_regression(N, R):
    return (1 - R/(1+N)**(2*(R-1/30))) ** (30 - N)

# Calculate churn probability
prob_churn = logistic_regression(N, R)

# Display result
st.markdown(f"### 🔥 Churn Probability: **{prob_churn:.0%}**")

# Visualization of churn probability over N
N_values = np.linspace(1, 30, 100)
prob_values = [logistic_regression(n, R) for n in N_values]

# Create a styled figure
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(N_values, prob_values, label="Churn Probability", color='red', linewidth=2)
ax.scatter([N], [prob_churn], color='blue', s=100, label="Selected Value", edgecolors='black', zorder=3)
ax.set_ylim(0,1.1)

# Styling the graph
ax.set_xlabel("Inactive Days", fontsize=12, fontweight='bold')
ax.set_ylabel("Churn Probability", fontsize=12, fontweight='bold')
ax.set_title("Churn Probability vs. Inactive Days", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)

st.pyplot(fig)