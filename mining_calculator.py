from operator import add
import streamlit as st
import os
from scipy.optimize import curve_fit
import datetime as dt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

INIT_DAY = dt.date(2021, 4, 17)

# Equation to predict mPower
# TODO Add flexibility to this
def exp_regression(x, a, b):
    return a * np.exp(b * x)

def write():
    st.markdown(
        """
        # DHX Mining calculator
        """
    )

    st.markdown(
        """
        ## Current prices
        
        Enter the [MXC price](https://coinmarketcap.com/currencies/mxc/) and [DHX price](https://coinmarketcap.com/currencies/datahighway/) 
        in the corresponding boxes. In a future update, the default values will be updated automatically.
        """)
    col1, col2 = st.beta_columns(2)
    mxc_price = col1.number_input("MXC Price ($)", value=0.02, format="%.4f", step=0.001)
    dhx_price = col2.number_input("DHX Price ($)", value=70.00, format="%.3f", step=1.)

    st.markdown("## Date selection")
    today = dt.date.today()
    col1, col2 = st.beta_columns(2)
    start_day = col1.date_input("Start day", value=dt.date.today(), min_value=INIT_DAY)
    end_day = col2.date_input("End day", value=today+dt.timedelta(days=60), min_value=today+dt.timedelta(days=1))
    
    delta_days = (end_day - start_day).days

    with st.beta_expander("Show advanced configuration: mPower per DHX estimation"):
        st.markdown(
            """
            ## mPower estimation

            The graph shows an exponential fit for the amount of mPower required per DHX bonded.  

            The following slider reflects the mPower growth rate. A higher gain represents a higher estimated
            growth. Advanced users may change the value, otherwise leave on default.
            
            """)

        col1, col2 = st.beta_columns(2)
        exp_gain = col1.slider("Exponential growth gain", 0.1, 2.0, value=0.6, step=0.05)
        prop_gain = col2.slider("Proportional growth gain", 0.1, 2.0, value=1.40, step=0.05)

        # Train data
        with open("total_mpower_from_apr17.txt") as f:
            mpower_data = list(map(float, f.read().split("\n")))

        n_samples = len(mpower_data)
        final_train_day = INIT_DAY + dt.timedelta(days=n_samples-1)

        # Model (TODO keep researching)
        X_train, y_train= np.arange(n_samples), mpower_data

        weights = np.linspace(1, 0.01, n_samples) # Give more importance to recent samples
        model_params, _ = curve_fit(exp_regression, X_train, y_train, p0=(1,0.02), sigma = weights)
        model_params[0] *= prop_gain
        model_params[1] *= exp_gain # apply gain

        # Predictions
        start_ix = n_samples + (start_day - final_train_day).days
        final_ix = start_ix + delta_days

        X_test = np.arange(start_ix, final_ix)

        offset = exp_regression(n_samples, *model_params) - mpower_data[-1] # fit to latest known data point
        y_test = exp_regression(X_test, *model_params) - offset
        
        # Plotting
        fig = go.Figure()
        fig.update_layout(
            title = 'mPower per DHX requirements',
            xaxis_title="Time",
            yaxis_title="mPower",
            height=400, width=700)

        # Plot train data
        x_train_dates = np.arange(INIT_DAY, final_train_day+dt.timedelta(days=1))
        fig.add_trace(go.Scatter(x=x_train_dates, y=mpower_data, mode='lines', name='Actual data'))

        # Plot predicted data
        x_test_dates = np.arange(start_day, end_day)
        fig.add_trace(go.Scatter(x=x_test_dates, y=y_test, mode='lines', name='Predicted data'))

        st.plotly_chart(fig, use_container_width=True)


    st.markdown(
        """
        # Rewards

        Please set your desired settings.
         
        """)

    col1, col2 = st.beta_columns(2)

    col1.markdown(
        """
        ### Do you have one or more M2 Pro miners?
        Having an M2 Pro miner boosts mPower by 100%
        """)
    has_miner = col1.checkbox("I have an M2 Pro miner or more")
    if has_miner:
        n_miners = col1.number_input(label="How many miners do you own?", value=1, min_value=1, step=1)

    col2.markdown(
        """
        ### For how long do you plan to lock your MXC?
        Longer locking periods provide mPower boosts
        """)
    boost_options = {"3 Months (0% mPower boost)"  : 0.00, 
                     "9 Months (10% mPower boost)" : 0.10, 
                     "12 Months (20% mPower boost)": 0.20, 
                     "24 Months (40% mPower boost)": 0.40}
    lock_time = col2.selectbox("Choose your locking period", list(boost_options.keys()), index=0)
    lock_bonus = boost_options[lock_time]

    # Results
    total_boost_rate = ((1+lock_bonus)*(has_miner+1))
    mpower_per_dhx = y_test
    discounted_mpower_per_dhx = mpower_per_dhx / total_boost_rate

    
    st.markdown("### **How do you wish to input your data?**")
    options = ["Current mPower and bonded DHX", "Amount of money to invest", "Current bonded DHX", "Current locked MXC"]
    input_option = st.selectbox("", options, index=0)

    st.markdown("### Your data")
    if (input_option == "Current mPower and bonded DHX"):
        col1, col2 = st.beta_columns(2)
        mPower = col1.number_input("Current mPower", value=50000.00, format="%.2f", step=1000.)

       
        if has_miner:
            with st.beta_expander("Show advanced settings: MXC boost limit per miner"):
                st.markdown("""
                    To factor the MXC boosted per miner limit and estimate future "re-locks" please specify how
                    much MXC you have locked.
                    """)
                locked_mxc = st.number_input("MXC currently locked", min_value=0., value=0., step=1000.)

        bonded_dhx = col2.number_input("Current bonded DHX", value=4.5000, format="%.4f", step=0.5)

        st.markdown("### Initial calculations")

        initial_mined_dhx = min(bonded_dhx/70, (mPower/mpower_per_dhx[0]) / 70 )

        additional_mxc_to_lock_placeholder = st.empty()
        additional_dhx_to_bond_placeholder = st.empty()
        st.markdown("> ## **Initial DHX mined per day: `{:.3f}` (${:.2f}) **".format(initial_mined_dhx, dhx_price*initial_mined_dhx))

    elif (input_option == "Amount of money to invest"):
        col1, col2 = st.beta_columns(2)
        investment = col1.number_input("Desired investment ($)", value=5000.00, format="%.2f", step=100.)

        #with st.beta_expander("Show advanced settings"):
        #    col1, col2 = st.beta_columns(2)
        #    current_dhx = col1.number_input("Do you already own some DHX?", value=0.00, format="%.4f", min_value=0., step=0.1)
        #    current_mxc = col2.number_input("Do you already own some MXC?", value=0.00, format="%.2f", min_value=0., step=1000.)

        st.markdown("### Initial calculations")

        #offset_dhx_to_buy = current_dhx*dhx_price + current_mxc*mxc_price + current_dhx*discounted_mpower_per_dhx[0]*mxc_price
        #dhx_to_buy = ((investment) / ((discounted_mpower_per_dhx[0]*mxc_price)+dhx_price))
        #bonded_dhx = dhx_to_buy + current_dhx
        #mxc_to_buy = dhx_to_buy * discounted_mpower_per_dhx[0]
        
        bonded_dhx = dhx_to_buy = investment / ((discounted_mpower_per_dhx[0]*mxc_price)+dhx_price)
        locked_mxc = mxc_to_buy = bonded_dhx * discounted_mpower_per_dhx[0]
        
        #if (current_dhx > 0) or (current_mxc > 0):
        #    st.markdown("> Total MXC after investment: `{:.2f}` MXC".format(mxc_to_buy+current_mxc))
        #    st.markdown("> Total DHX after investment: `{:.4f}` DHX".format(dhx_to_buy+current_dhx))

        st.markdown("> Initial MXC to buy: **`{:.2f}` MXC** (${:.2f})".format(mxc_to_buy, mxc_to_buy*mxc_price))
        st.markdown("> Initial DHX to buy: **`{:.4f}` DHX** (${:.2f})".format(dhx_to_buy, dhx_to_buy*dhx_price))

        mPower = mxc_to_buy * total_boost_rate

        # Miner limit 1 million mPower
        if (n_miners*(10**6) > mPower):
            # Nerf mPower and adjust values
            pass
            

        st.markdown("> mPower: **`{:.3f}`**".format(mPower))

        st.markdown("> ## **Initial DHX mined per day: `{:.3f}` (${:.2f}) **".format(dhx_to_buy/70, dhx_price*dhx_to_buy/70))

    elif (input_option == "Current bonded DHX"):
        col1, col2 = st.beta_columns(2)
        bonded_dhx = col1.number_input("Current bonded DHX", value=4.5000, format="%.4f", step=0.5)

        st.markdown("### Initial calculations")

        locked_mxc = mxc_to_buy = bonded_dhx * discounted_mpower_per_dhx[0]

        st.markdown("> Initial MXC to buy: **`{:.3f}` MXC** (${:.2f})".format(mxc_to_buy, mxc_to_buy*mxc_price))
        mPower = mxc_to_buy * total_boost_rate
        st.markdown("> mPower: **`{:.3f}`**".format(mPower))

        st.markdown("> ## **Initial DHX mined per day: `{:.3f}` (${:.2f}) **".format(bonded_dhx/70, dhx_price*bonded_dhx/70))

    elif (input_option == "Current locked MXC"):
        col1, col2 = st.beta_columns(2)
        locked_mxc = col1.number_input("Current locked MXC", value=50000.00, format="%.2f", step=1000.)

        st.markdown("### Initial calculations")

        mPower = locked_mxc * total_boost_rate
        bonded_dhx = dhx_to_buy = mPower / mpower_per_dhx[0]

        st.markdown("> Initial DHX to buy: **`{:.3f}` DHX** (${:.2f})".format(dhx_to_buy, dhx_to_buy*dhx_price))
        st.markdown("> mPower: **`{:.3f}`**".format(mPower))
        st.markdown("> ## **Initial DHX mined per day: `{:.3f}` (${:.2f}) **".format(dhx_to_buy/70, dhx_price*dhx_to_buy/70))

    
    st.markdown(
        """
        # Rewards over time

        In these graphs, the left vertical axis represents the amount of DHX mined, while the right vertical
        axis represents the $USD equivalent at current rates.

        """
    )


    # TODO Vectorize all this, current implementation is very suboptimal

    ideal_bonded_dhx_i = bonded_dhx
    ideal_bonded_dhx_v = []
    mined_dhx_v = []
    ideal_mined_dhx_v = []
    additional_mxc_to_lock_v = []
    additional_dhx_to_bond_v = []
    #print("-"*200)
    for i, mpower_per_dhx_i in enumerate(mpower_per_dhx):

        if (i > 6): # Mined DHX gets automatically bonded after 7 days
            ideal_bonded_dhx_i += ideal_bonded_dhx_v[i - 7] / 70 # Theoretically, only if max rewards
        ideal_bonded_dhx_v.append(ideal_bonded_dhx_i) # keep track of bonded dhx

        fueled_dhx_i = mPower / mpower_per_dhx_i

        # Calculate additional mPower required
        # Network growth produced by yourself if you were to provide the necessary mPower for compounding
        aux_mpower_per_dhx_i = mpower_per_dhx_i
        accumulated_mpower_needed = mPower
        additional_mpower_needed = sys.float_info.max

        # While the additional mpower needed is bigger than 1 percent of the network size
        #j=0
        while(additional_mpower_needed > (aux_mpower_per_dhx_i*35000 / 100) ):
            additional_mpower_needed = ideal_bonded_dhx_i*aux_mpower_per_dhx_i - accumulated_mpower_needed
            aux_mpower_per_dhx_i += (additional_mpower_needed / 35000)
            accumulated_mpower_needed += additional_mpower_needed
            #j +=1

        #print("{:d} iterations were needed to converge for self-growth".format(j))
        additional_mpower_i = max(0, accumulated_mpower_needed - mPower)
        #additional_mpower_i = ideal_bonded_dhx_i*mpower_per_dhx_i - mPower # SIMPLE WAY, NO SELF GROWTH


        # Mined DHX
        mined_dhx_i = min(fueled_dhx_i/70, ideal_bonded_dhx_i/70, 5000)
        mined_dhx_v.append(mined_dhx_i)

        # Additional MXC to lock
        if has_miner:
            boostable_mxc_left = max(0, (10**6)*n_miners - locked_mxc)
            mpower_from_boosted_mxc = boostable_mxc_left * total_boost_rate

            if (additional_mpower_i < mpower_from_boosted_mxc): # enough boostable MXC to cover mPower needs
                additional_mxc_to_lock_i = min(max(0, (additional_mpower_i) / total_boost_rate), 5000*mpower_per_dhx_i)
            else:
                # We have to account for the boosted and non-boosted portions
                additional_mxc_to_lock_non_boosted = min( (additional_mpower_i-mpower_from_boosted_mxc) / (1+lock_bonus), 5000*mpower_per_dhx_i)
                additional_mxc_to_lock_i = boostable_mxc_left + additional_mxc_to_lock_non_boosted
                pass

        else:
            additional_mxc_to_lock_i = min(max(0, (additional_mpower_i) / total_boost_rate), 5000*mpower_per_dhx_i)
        
        #additional_mxc_to_lock_i = min(max(0, (additional_mpower_i) / total_boost_rate), 5000*mpower_per_dhx_i)
        additional_mxc_to_lock_v.append(additional_mxc_to_lock_i)

        # Ideal mined DHX
        ideal_mined_dhx_i = min(ideal_bonded_dhx_i / 70, 5000)
        ideal_mined_dhx_v.append(ideal_mined_dhx_i)

        # Additional DHX to bond
        additional_dhx_to_bond_i = max(0, fueled_dhx_i - ideal_bonded_dhx_i )
        additional_dhx_to_bond_v.append(additional_dhx_to_bond_i)


    # Don't forget to add this
    if (input_option == "Current mPower and bonded DHX"):
        additional_mxc_to_lock_placeholder.markdown("> Additional MXC to lock for full 'day 0' earnings: **`{:.3f}` MXC** (${:.2f})"
                                                    .format(additional_mxc_to_lock_v[0], additional_mxc_to_lock_v[0]*mxc_price))
        additional_dhx_to_bond_placeholder.markdown("> Additional DHX to bond for full 'day 0' earnings: **`{:.4f}` DHX** (${:.2f})"
                                                    .format(additional_dhx_to_bond_v[0], additional_dhx_to_bond_v[0]*dhx_price))

    # Plotting

    # 1. DHX rewards no compounding
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        title = '<b>DHX rewards over time (No additional locking)<b>',
        xaxis_title="Time",
        hovermode='x unified',
        height=400, width=700)
    fig.update_yaxes(title_text="<b>DHX Mined per day</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>USD Equivalent</b> ", secondary_y=True)

    # Plot
    fig.add_trace(go.Scatter(x=x_test_dates, y=mined_dhx_v, hovertemplate='DHX: %{y:.4f} <extra></extra>',
                             mode='lines+markers', name='DHX Mined', marker=dict(size=2) ), secondary_y=False)
    fig.add_trace(go.Scatter(x=x_test_dates, y=np.array(mined_dhx_v)*dhx_price, hovertemplate='$%{y:.2f} <extra></extra>',
                             mode='lines', name='USD Rewards <br>@ current DHX Price'), secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # 2. Cumulative DHX rewards no compounding
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        title = '<b>Cumulative DHX rewards over time (No additional locking)<b>',
        xaxis_title="Time",
        hovermode='x unified',
        height=400, width=700)
    fig.update_yaxes(title_text="<b>Cumulative DHX Mined</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Cumulative USD Equivalent</b> ", secondary_y=True)

    # Plot
    cumulative_mined_dhx_v = np.cumsum(mined_dhx_v)
    fig.add_trace(go.Scatter(x=x_test_dates, y=cumulative_mined_dhx_v, hovertemplate='DHX: %{y:.4f} <extra></extra>',
                             mode='lines+markers', name='Cumulative DHX Mined', marker=dict(size=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=x_test_dates, y=cumulative_mined_dhx_v*dhx_price, hovertemplate='$%{y:.2f} <extra></extra>',
                             mode='lines', name='Cumulative USD Rewards <br>@ current DHX Price'), secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    with st.beta_expander("Show experimental features: Maximizing earnings to keep all your DHX fueled providing mPower"):
        st.markdown("# Potential rewards over time")
        st.error("""
            ** Note: this feature is under development and it is in experimental state. It may not be accurate.
            **
            """)
        st.info("""  
                Below are the graphs that show **potential** rewards if you keep locking MXC or bonding DHX for maximum profits.
                """)

        # 3. DHX rewards WITH compounding
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.update_layout(
            title = '<b>DHX rewards over time (Requires additional mPower)<b>',
            xaxis_title="Time",
            hovermode='x unified',
            height=400, width=700)
        fig.update_yaxes(title_text="<b>DHX Mined per day</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>USD Equivalent</b> ", secondary_y=True)

        # Plot
        fig.add_trace(go.Scatter(x=x_test_dates, y=ideal_mined_dhx_v, hovertemplate='DHX: %{y:.4f} <extra></extra>',
                                mode='lines+markers', name='DHX Mined', marker=dict(size=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=x_test_dates, y=np.array(ideal_mined_dhx_v)*dhx_price, hovertemplate='$%{y:.2f} <extra></extra>',
                                mode='lines', name='USD Rewards <br>@ current DHX Price'), secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

        # 4. Cumulative DHX rewards WITH compounding
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.update_layout(
            title = '<b>Cumulative DHX rewards over time (Requires additional mPower)<b>',
            xaxis_title="Time",
            hovermode='x unified',
            height=400, width=700)
        fig.update_yaxes(title_text="<b>Cumulative DHX Mined</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>Cumulative USD Equivalent</b> ", secondary_y=True)

        # Plot
        cumulative_ideal_mined_dhx_v = np.cumsum(ideal_mined_dhx_v)
        fig.add_trace(go.Scatter(x=x_test_dates, y=cumulative_ideal_mined_dhx_v, hovertemplate='DHX: %{y:.4f} <extra></extra>', 
                                mode='lines+markers', name='Cumulative DHX Mined', marker=dict(size=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=x_test_dates, y=cumulative_ideal_mined_dhx_v*dhx_price, hovertemplate='$%{y:.2f} <extra></extra>', 
                                mode='lines', name='Cumulative USD Rewards <br>@ current DHX Price'), secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("# How to maximize rewards")
        st.error("""
            ** Note: this feature is under development and it is in experimental state. It may not be accurate.
            **
            """)
        st.info("""
            To maximize your earnings you will need to keep accumulating mPower. This is compounded interest.
            """)

        # 5. Additional MXC to lock
        fig = go.Figure()
        fig.update_layout(
            title = 'Cumulative additional MXC to lock',
            xaxis_title="Time",
            yaxis_title="<b>MXC<b>",
            hovermode='x unified',
            height=400, width=700)

        # Plot
        
        x_test_dates = np.arange(start_day, end_day)
        fig.add_trace(go.Scatter(x=x_test_dates, y=additional_mxc_to_lock_v, 
                                mode='lines', name='<b>MXC to lock<b>', hovertemplate = 'MXC: %{y}<extra></extra>'))

        st.plotly_chart(fig, use_container_width=True)

        # 6. Additional DHX to bond
        fig = go.Figure()
        fig.update_layout(
            title = 'Cumulative additional DHX to bond',
            xaxis_title="Time",
            yaxis_title="<b>DHX<b>",
            hovermode='x unified',
            height=400, width=700)

        # Plot
        
        x_test_dates = np.arange(start_day, end_day)
        fig.add_trace(go.Scatter(x=x_test_dates, y=additional_dhx_to_bond_v, 
                                mode='lines', name='<b>MXC to lock<b>', hovertemplate = 'DHX: %{y}<extra></extra>'))

        st.plotly_chart(fig, use_container_width=True)
