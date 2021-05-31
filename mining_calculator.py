from operator import add
import streamlit as st
import os
from scipy.optimize import curve_fit
import datetime as dt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import time

import cmc_api # Custom calls to CMC API and get mxc and dhx prices

INIT_DAY = dt.date(2021, 4, 17)

# Equation to predict mPower
# TODO Add flexibility to this
def exp_regression(x, a, b):
    return a * np.exp(b * x)



def write(state):

    # Query API if enough time has passed
    print("-"*50)
    t_state, mxc_price_state, dhx_price_state = state
    t = time.time()
    if (t - t_state) > (10*60):
        debug_new_query = True
        mxc_price_default, dhx_price_default = cmc_api.get_mxc_dhx_prices()
        print("Updated prices at time " + time.ctime(int(t)) + " with a delta of {:.2f}s".format(t - t_state))
        del state[:] # Horrible, but only way to make streamlit actually store values across runs
        state.extend([t, mxc_price_default, dhx_price_default])
    else:
        print("New run, but no new prices were fetched")
        debug_new_query = False
        mxc_price_default, dhx_price_default = mxc_price_state, dhx_price_state


    st.markdown(
        """
        # DHX Mining calculator
        """
    )

    st.info("""
        Note: I try to add new features constantly. I also try to make sure there are no bugs with every new release. Still, I may have
        messed something up. If you think there is a bug or you got an error please send it to me over discord: `PHAN#3179`
        """)

    st.markdown(
        """
        ## Current prices
        
        Default MXC and DHX prices are obtained through Coinmarketcap latest rates, but feel free to change them if you want to.
        """)


    col1, col2 = st.beta_columns(2)
    mxc_price = col1.number_input("MXC Price ($)", value=mxc_price_default, format="%.4f", step=0.001)
    dhx_price = col2.number_input("DHX Price ($)", value=dhx_price_default, format="%.3f", step=1.)

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
        exp_gain = col1.slider("Exponential growth gain", 0.01, 1.0, value=0.20, step=0.01) # 0.25
        prop_gain = col2.slider("Proportional growth gain", 0.1, 20.0, value=5.30, step=0.05) # 3.10

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

        st.plotly_chart(fig)


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
        boostable_mxc = n_miners * (10**6)
    else:
        n_miners = 0
        boostable_mxc = 0

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
    options = ["Current mPower and bonded DHX",
               "Current locked MXC and bonded DHX",
               "Amount of money to invest", 
               "Current bonded DHX", 
               "Current locked MXC"]
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

            if (locked_mxc < 1):
                st.warning("""
                It seems you haven't filled the amount of locked MXC you have. In order to consider the MXC boost per miner limit you must specify it.   
                If you are not interested in this feature, then don't worry!
                """)

        bonded_dhx = col2.number_input("Current bonded DHX", value=4.5000, format="%.4f", step=0.5)

        st.markdown("### Initial calculations")

        initial_mined_dhx = min(bonded_dhx/70, (mPower/mpower_per_dhx[0]) / 70 )

        additional_mxc_to_lock_placeholder = st.empty()
        additional_dhx_to_bond_placeholder = st.empty()
        st.markdown("> ## **Initial DHX mined per day: `{:.4f}` (${:.2f}) **".format(initial_mined_dhx, dhx_price*initial_mined_dhx))

    
    elif (input_option == "Current locked MXC and bonded DHX"):
        col1, col2 = st.beta_columns(2)

        locked_mxc = col1.number_input("Current locked MXC", value=50000.00, format="%.2f", step=1000.)
        bonded_dhx = col2.number_input("Current bonded DHX", value=4.5000, format="%.4f", step=0.5)

        st.markdown("### Initial calculations")
        mPower = locked_mxc * total_boost_rate

        # Factor 1 million MXC boost limit per miner
        if has_miner:
            if (locked_mxc > boostable_mxc):
                st.info("""
                    The amount of MXC to buy exceeds the boost limit per miner(1 Million MXC per miner). Calculations have been adjusted
                    to compensate for this. This is just a warning but there is nothing to worry about as it has been considered.
                    """)

                # Recompute
                mPower = boostable_mxc*total_boost_rate + (locked_mxc-boostable_mxc)*(1+lock_bonus)


        st.markdown("> mPower: **`{:.2f}`**".format(mPower))

        additional_mxc_to_lock_placeholder = st.empty()
        additional_dhx_to_bond_placeholder = st.empty()

        initial_mined_dhx = min(bonded_dhx/70, (mPower/mpower_per_dhx[0]) / 70 )
        st.markdown("> ## **Initial DHX mined per day: `{:.4f}` (${:.2f}) **".format(initial_mined_dhx, dhx_price*initial_mined_dhx))


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
        mPower = mxc_to_buy * total_boost_rate

        # Factor 1 million MXC boost limit per miner
        if has_miner:
            if (mxc_to_buy > boostable_mxc):
                st.info("""
                    The amount of MXC to buy exceeds the boost limit per miner(1 Million MXC per miner). Calculations have been adjusted
                    to compensate for this. This is just a warning but there is nothing to worry about as it has been considered.
                    """)
                # First stage: Full bonus applies
                investment_first_part = boostable_mxc*mxc_price + (boostable_mxc/discounted_mpower_per_dhx[0])*dhx_price
                dhx_to_buy_first_part = investment_first_part / ((discounted_mpower_per_dhx[0]*mxc_price)+dhx_price)
                mxc_to_buy_first_part = dhx_to_buy_first_part * discounted_mpower_per_dhx[0]

                # Second stage: Miner bonus no longer applies
                investment_left = investment - investment_first_part
                discounted_mpower_per_dhx_no_miner_bonus = mpower_per_dhx / (1+lock_bonus)
                dhx_to_buy_second_part = investment_left / ((discounted_mpower_per_dhx_no_miner_bonus[0]*mxc_price)+dhx_price)
                mxc_to_buy_second_part = dhx_to_buy_second_part * discounted_mpower_per_dhx_no_miner_bonus[0]

                # Add together
                bonded_dhx = dhx_to_buy = dhx_to_buy_first_part + dhx_to_buy_second_part
                locked_mxc = mxc_to_buy = mxc_to_buy_first_part + mxc_to_buy_second_part

                # Recompute
                mPower = boostable_mxc*total_boost_rate + mxc_to_buy_second_part*(1+lock_bonus)
        
        #if (current_dhx > 0) or (current_mxc > 0):
        #    st.markdown("> Total MXC after investment: `{:.2f}` MXC".format(mxc_to_buy+current_mxc))
        #    st.markdown("> Total DHX after investment: `{:.4f}` DHX".format(dhx_to_buy+current_dhx))

        st.markdown("> Initial MXC to buy: **`{:.2f}` MXC** (${:.2f})".format(mxc_to_buy, mxc_to_buy*mxc_price))
        st.markdown("> Initial DHX to buy: **`{:.4f}` DHX** (${:.2f})".format(dhx_to_buy, dhx_to_buy*dhx_price))

            

        st.markdown("> mPower: **`{:.3f}`**".format(mPower))

        st.markdown("> ## **Initial DHX mined per day: `{:.4f}` (${:.2f}) **".format(dhx_to_buy/70, dhx_price*dhx_to_buy/70))

    elif (input_option == "Current bonded DHX"):
        col1, col2 = st.beta_columns(2)
        bonded_dhx = col1.number_input("Current bonded DHX", value=4.5000, format="%.4f", step=0.5)

        st.markdown("### Initial calculations")

        locked_mxc = mxc_to_buy = bonded_dhx * discounted_mpower_per_dhx[0]
        mPower = mxc_to_buy * total_boost_rate

        # Factor 1 million MXC boost limit per miner
        if has_miner:
            if (mxc_to_buy > boostable_mxc):
                st.info("""
                    The amount of MXC to buy exceeds the boost limit per miner(1 Million MXC per miner). Calculations have been adjusted
                    to compensate for this. This is just a warning but there is nothing to worry about as it has been considered.
                    """)
                # First stage: Full bonus applies
                mxc_to_buy_first_part = boostable_mxc
                fueled_dhx_first_part = mxc_to_buy_first_part / discounted_mpower_per_dhx[0]

                # Second stage: Miner bonus no longer applies
                discounted_mpower_per_dhx_no_miner_bonus = mpower_per_dhx / (1+lock_bonus)
                mxc_to_buy_second_part = (bonded_dhx - fueled_dhx_first_part) * discounted_mpower_per_dhx_no_miner_bonus[0]

                # Add together
                locked_mxc = mxc_to_buy = mxc_to_buy_first_part + mxc_to_buy_second_part

                # Recompute
                mPower = mxc_to_buy_first_part*total_boost_rate + mxc_to_buy_second_part*(1+lock_bonus)

        st.markdown("> Initial MXC to buy: **`{:.3f}` MXC** (${:.2f})".format(mxc_to_buy, mxc_to_buy*mxc_price))
        st.markdown("> mPower: **`{:.3f}`**".format(mPower))

        st.markdown("> ## **Initial DHX mined per day: `{:.4f}` (${:.2f}) **".format(bonded_dhx/70, dhx_price*bonded_dhx/70))

    elif (input_option == "Current locked MXC"):
        col1, col2 = st.beta_columns(2)

        locked_mxc = col1.number_input("Current locked MXC", value=50000.00, format="%.2f", step=1000.)

        st.markdown("### Initial calculations")
        mPower = locked_mxc * total_boost_rate

        # Factor 1 million MXC boost limit per miner
        if has_miner:
            if (locked_mxc > boostable_mxc):
                st.info("""
                    The amount of MXC to buy exceeds the boost limit per miner(1 Million MXC per miner). Calculations have been adjusted
                    to compensate for this. This is just a warning but there is nothing to worry about as it has been considered.
                    """)

                # Recompute
                mPower = boostable_mxc*total_boost_rate + (locked_mxc-boostable_mxc)*(1+lock_bonus)


        bonded_dhx = dhx_to_buy = mPower / mpower_per_dhx[0]

        st.markdown("> Initial DHX to buy: **`{:.3f}` DHX** (${:.2f})".format(dhx_to_buy, dhx_to_buy*dhx_price))
        st.markdown("> mPower: **`{:.3f}`**".format(mPower))
        st.markdown("> ## **Initial DHX mined per day: `{:.4f}` (${:.2f}) **".format(dhx_to_buy/70, dhx_price*dhx_to_buy/70))

    
    st.markdown(
        """
        # Rewards over time

        In these graphs, the left vertical axis represents the amount of DHX mined, while the right vertical
        axis represents the $USD equivalent at current rates.  

        Keep in mind that this data is computed using **predicted** network growth. That means that for longer
        periods of time the results may not be accurate.

        """
    )

    with st.beta_expander("Will you lock more MXC every day? Try this feature!"):

        if st.checkbox("Enable additional MXC locks per day", value=False):
            additional_mxc_per_day = st.number_input("How much MXC per day will you lock?", value=0.0, min_value=0.0, step=100.)
        else:
            additional_mxc_per_day = 0



    # TODO Vectorize all this, current implementation is very suboptimal

    bonded_dhx_i = bonded_dhx
    bonded_dhx_v = []
    ideal_bonded_dhx_i = bonded_dhx
    ideal_bonded_dhx_v = []
    fueled_dhx_v = []
    mined_dhx_v = []
    ideal_mined_dhx_v = []
    additional_mxc_to_lock_v = []
    additional_dhx_to_bond_v = []
    mpower_v = []
    for i, mpower_per_dhx_i in enumerate(mpower_per_dhx):

        # Ideal bonded dhx calculation
        if (i > 6): # Mined DHX gets automatically bonded after 7 days
            ideal_bonded_dhx_i += ideal_bonded_dhx_v[i - 7] / 70 # Theoretically, only if max rewards
        ideal_bonded_dhx_v.append(ideal_bonded_dhx_i) # keep track of bonded dhx
    
        # We recompute it since an additional_mxc_per_day will be locked
        if (input_option != "Current mPower and bonded DHX"):
            if (locked_mxc > boostable_mxc):
                mPower = boostable_mxc*total_boost_rate + (locked_mxc-boostable_mxc)*(1+lock_bonus)
            else:
                mPower = locked_mxc * total_boost_rate
        mpower_v.append(mPower)

        fueled_dhx_i = mPower / mpower_per_dhx_i
        fueled_dhx_v.append(fueled_dhx_i)

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

        # Bonded DHX
        if (i > 6): # Mined DHX gets automatically bonded after 7 days
            bonded_dhx_i += mined_dhx_v[i - 7]
        bonded_dhx_v.append(bonded_dhx_i) # keep track of bonded dhx
    

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

        # Additional mxc locked per day (optional, defaults to zero)
        if (input_option != "Current mPower and bonded DHX"):
            locked_mxc += additional_mxc_per_day
        else:
            # If has_miner we have to account for the limit
            if has_miner:
                locked_mxc += additional_mxc_per_day
                if (locked_mxc > boostable_mxc): 
                    boostable_mxc_portion = max(0, (boostable_mxc - (locked_mxc-additional_mxc_per_day)  ))
                    additional_mpower_first_part = boostable_mxc_portion*total_boost_rate
                    additional_mpower_second_part = (additional_mxc_per_day - boostable_mxc_portion)*(1+lock_bonus)
                    mPower += additional_mpower_first_part + additional_mpower_second_part

                else: # otherwise no problem
                    mPower += additional_mxc_per_day*total_boost_rate

            else:
                mPower += additional_mxc_per_day*total_boost_rate
    

    # Don't forget to add this
    if (input_option == "Current mPower and bonded DHX" ) or (input_option == "Current locked MXC and bonded DHX"):
        additional_mxc_to_lock_placeholder.markdown("> Additional MXC to lock for max 'day 0' earnings: **`{:.3f}` MXC** (${:.2f})"
                                                    .format(additional_mxc_to_lock_v[0], additional_mxc_to_lock_v[0]*mxc_price))
        additional_dhx_to_bond_placeholder.markdown("> Additional DHX to bond for max 'day 0' earnings: **`{:.4f}` DHX** (${:.2f})"
                                                    .format(additional_dhx_to_bond_v[0], additional_dhx_to_bond_v[0]*dhx_price))

    # Plotting

    if additional_mxc_per_day > 0:
        additional_lock_title = ' (Additional {:.2f} MXC locked per day)<b>'.format(additional_mxc_per_day)
    else:
        additional_lock_title = ' (No additional locking)<b>'

    # 1. DHX rewards no compounding
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        title = '<b>DHX rewards over time' + additional_lock_title,
        xaxis_title="Time",
        xaxis={'fixedrange':True},
        yaxis={'fixedrange':True},
        hovermode='x unified',
        height=400, width=700)

    fig.update_yaxes(title_text="<b>DHX Mined per day</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>USD Equivalent</b> ", secondary_y=True, fixedrange=True)

    # Plot
    fig.add_trace(go.Scatter(x=x_test_dates, y=mined_dhx_v, hovertemplate='DHX: %{y:.4f} <extra></extra>',
                             mode='lines+markers', name='DHX Mined', marker=dict(size=2) ), secondary_y=False)
    fig.add_trace(go.Scatter(x=x_test_dates, y=np.array(mined_dhx_v)*dhx_price, hovertemplate='$%{y:.2f}<extra></extra>',
                             mode='lines', name='USD Rewards <br>@ current DHX Price'), secondary_y=True)

    config = {'displayModeBar': False}
    st.plotly_chart(fig, config=config)

    # 2. Cumulative DHX rewards no compounding
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        title = '<b>Cumulative DHX rewards over time' + additional_lock_title,
        xaxis_title="Time",
        hovermode='x unified',
        xaxis={'fixedrange':True},
        yaxis={'fixedrange':True},
        height=400, width=700)
    fig.update_yaxes(title_text="<b>Cumulative DHX Mined</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Cumulative USD Equivalent</b> ", secondary_y=True, fixedrange=True)

    # Plot
    cumulative_mined_dhx_v = np.cumsum(mined_dhx_v)
    fig.add_trace(go.Scatter(x=x_test_dates, y=cumulative_mined_dhx_v, hovertemplate='DHX: %{y:.4f} <extra></extra>',
                             mode='lines+markers', name='Cumulative DHX Mined', marker=dict(size=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=x_test_dates, y=cumulative_mined_dhx_v*dhx_price, hovertemplate='$%{y:.2f} <extra></extra>',
                             mode='lines', name='Cumulative USD Rewards <br>@ current DHX Price'), secondary_y=True)

    st.plotly_chart(fig)

    with st.beta_expander("Show additional graphs"):

        # EXTRA.1 DHX fueled vs cumulative
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1, shared_xaxes=True)#go.Figure()#make_subplots(specs=[[{"secondary_y": True}]])
        fig.update_layout(
            title = '<b>Fueled DHX and Bonded DHX' + additional_lock_title,
            xaxis_title="Time",
            xaxis={'fixedrange':True},
            yaxis={'fixedrange':True},
            hovermode='x unified',
            height=600, width=700)

        # Plot
        fig.add_trace(go.Scatter(x=x_test_dates, y=fueled_dhx_v, hovertemplate='Fueled DHX: %{y:.4f} <extra></extra>',
                                mode='lines+markers', name='Fueled DHX', marker=dict(size=2)), row=1, col=1 )
        fig.add_trace(go.Scatter(x=x_test_dates, y=bonded_dhx_v, hovertemplate='Bonded DHX: %{y:.4f} <extra></extra>',
                                mode='lines', name='Bonded DHX'), row=1, col=1)

        safe_to_withdraw_dhx = [max(0, bonded-fueled) for bonded, fueled in zip(bonded_dhx_v, fueled_dhx_v)]
        fig.add_trace(go.Scatter(x=x_test_dates, y=safe_to_withdraw_dhx, hovertemplate='Safe to withdraw DHX: %{y:.4f} <extra></extra>',
                                mode='lines', name='Safe to withdraw DHX'), row=2, col=1)
        fig.update_yaxes(row=2, col=1, fixedrange=True)

        config = {'displayModeBar': False}
        st.plotly_chart(fig, config=config)

        st.info("""
            The "Safe to withdraw DHX" indicates the difference between your bonded DHX and fueled DHX. If the result is positive and
            keeps growing, it's absolutely safe to unbond it and withdraw it, as it no longer contributes to produce earnings.
            """)


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
            xaxis={'fixedrange':True},
            yaxis={'fixedrange':True},
            height=400, width=700)
        fig.update_yaxes(title_text="<b>DHX Mined per day</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>USD Equivalent</b> ", secondary_y=True, fixedrange=True)

        # Plot
        fig.add_trace(go.Scatter(x=x_test_dates, y=ideal_mined_dhx_v, hovertemplate='DHX: %{y:.4f} <extra></extra>',
                                mode='lines+markers', name='DHX Mined', marker=dict(size=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=x_test_dates, y=np.array(ideal_mined_dhx_v)*dhx_price, hovertemplate='$%{y:.2f} <extra></extra>',
                                mode='lines', name='USD Rewards <br>@ current DHX Price'), secondary_y=True)

        st.plotly_chart(fig)

        # 4. Cumulative DHX rewards WITH compounding
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.update_layout(
            title = '<b>Cumulative DHX rewards over time (Requires additional mPower)<b>',
            xaxis_title="Time",
            hovermode='x unified',
            xaxis={'fixedrange':True},
            yaxis={'fixedrange':True},
            height=400, width=700)
        fig.update_yaxes(title_text="<b>Cumulative DHX Mined</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>Cumulative USD Equivalent</b> ", fixedrange=True, secondary_y=True)

        # Plot
        cumulative_ideal_mined_dhx_v = np.cumsum(ideal_mined_dhx_v)
        fig.add_trace(go.Scatter(x=x_test_dates, y=cumulative_ideal_mined_dhx_v, hovertemplate='DHX: %{y:.4f} <extra></extra>', 
                                mode='lines+markers', name='Cumulative DHX Mined', marker=dict(size=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=x_test_dates, y=cumulative_ideal_mined_dhx_v*dhx_price, hovertemplate='$%{y:.2f} <extra></extra>', 
                                mode='lines', name='Cumulative USD Rewards <br>@ current DHX Price'), secondary_y=True)

        st.plotly_chart(fig)

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
            xaxis={'fixedrange':True},
            yaxis={'fixedrange':True},
            height=400, width=700)

        # Plot
        
        x_test_dates = np.arange(start_day, end_day)
        fig.add_trace(go.Scatter(x=x_test_dates, y=additional_mxc_to_lock_v, 
                                mode='lines', name='<b>MXC to lock<b>', hovertemplate = 'MXC: %{y}<extra></extra>'))

        st.plotly_chart(fig)

        # 6. Additional DHX to bond
        fig = go.Figure()
        fig.update_layout(
            title = 'Cumulative additional DHX to bond',
            xaxis_title="Time",
            yaxis_title="<b>DHX<b>",
            hovermode='x unified',
            xaxis={'fixedrange':True},
            yaxis={'fixedrange':True},
            height=400, width=700)

        # Plot
        
        x_test_dates = np.arange(start_day, end_day)
        fig.add_trace(go.Scatter(x=x_test_dates, y=additional_dhx_to_bond_v, 
                                mode='lines', name='<b>MXC to lock<b>', hovertemplate = 'DHX: %{y}<extra></extra>'))

        st.plotly_chart(fig)

    with st.beta_expander("Show debug info"):
        st.text("State is {}".format(state))
        st.text("debug_new_query = {}".format(debug_new_query))
