import streamlit as st
import numpy as np
import plotly.graph_objects as go



def write(state):
    
    st.markdown("# Fuel Calculator (IN PROGRESS)")

    st.warning("""
    This is a very new feature for both the MXC project and this calculator. The
    calculations provided may be wrong. With time we'll be able to double-check
    the results and ensure they are indeed correct. Use at your own risk.
    """)

    st.markdown("""
    If you choose to withdraw MXC from your miner wallet, you will deplete your fuel.
    This affects the rewards you will be getting. This tools helps you visualize
    your rewards over time as well as your fuel.

    """)

    st.markdown("### **How do you wish to input your data?**")
    options = ["I plan to mine MXC and withdraw in the future",
               "(Coming soon...)"]
    input_option = st.selectbox("", options, index=0)

    st.markdown("### Your data")
    if (input_option == "I plan to mine MXC and withdraw in the future"):
        pass