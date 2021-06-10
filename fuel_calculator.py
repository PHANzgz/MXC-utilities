from pkg_resources import working_set
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import datetime as dt

import cmc_api


INIT_DAY = dt.date(2021, 6, 10)

def write(state):

    # Query API if enough time has passed
    #print("-"*50)
    t_state, mxc_price_state, dhx_price_state = state
    t = time.time()
    if (t - t_state) > (10*60):
        debug_new_query = True
        mxc_price_default, dhx_price_default = cmc_api.get_mxc_dhx_prices()
        #print("Updated prices at time " + time.ctime(int(t)) + " with a delta of {:.2f}s".format(t - t_state))
        del state[:] # Horrible, but only way to make streamlit actually store values across runs
        state.extend([t, mxc_price_default, dhx_price_default])
    else:
        #print("New run, but no new prices were fetched")
        debug_new_query = False
        mxc_price_default, dhx_price_default = mxc_price_state, dhx_price_state


    
    st.markdown("# Fuel Calculator (BETA)")

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


    st.markdown("### **Base reward data**")
    col1, col2 = st.beta_columns(2)
    mxc_price = col1.number_input("MXC Price ($)", value=mxc_price_default, format="%.4f", step=0.001)
    base_reward = col2.number_input("Base reward ($)", value=11.00, format="%.2f", step=1.0)
    mxc_base_reward = base_reward / mxc_price
    #mxc_base_reward = 300.

    st.markdown("""
        Assuming a **constant** MXC price of ${:.2f} and a base reward of ${:.2f} the estimated MXC reward is  

        > ### Estimated base MXC reward: `{:.2f}` MXC 
    """.format(mxc_price, base_reward, mxc_base_reward))

    st.warning("""
        Remember the MXC price will fluctuate over time but this calculator assumes a constant price
        since one cannot predict its future.
    """)

    st.markdown("### **How do you wish to input your data?**")
    options = ["I plan to mine MXC and withdraw in the future",
               "(Coming soon...)"]
    input_option = st.selectbox("", options, index=0)

    st.markdown("### Your data")
    if (input_option == "I plan to mine MXC and withdraw in the future"):
        col1, col2 = st.beta_columns(2)
        n_miners = col1.number_input(label="How many miners do you own?", value=1, min_value=1, step=1)
        mxc_base_reward *= n_miners

        st.markdown("Date selection")
        today = dt.date.today()
        col1, col2 = st.beta_columns(2)
        start_day = col1.date_input("Start day", value=dt.date.today(), min_value=INIT_DAY)
        end_day = col2.date_input("End day", value=today+dt.timedelta(days=30*4), min_value=today+dt.timedelta(days=1))

        default_date = start_day + ((end_day - start_day) // 2)
        withdraw_date = st.slider("When do you plan to withdraw(YYYY-MM-DD)?", start_day+dt.timedelta(days=1), end_day, value=default_date)
        percentage_withdrawn = st.slider("What percentage of your future balance will you withdraw?", 0., 100., step=1., value=20.)
        mxc_withdrawn = mxc_base_reward * (withdraw_date - start_day).days * percentage_withdrawn / 100

        st.markdown("""
            >**Info**: You will be withdrawing in  
            **Date**: `{}`  
            the {:.2f}% of your mined MXC up until then for a total of  
            **Withdrawn MXC**: `{:.2f}` MXC (${:.2f})
        """.format(withdraw_date.strftime("%B %d, %Y"), percentage_withdrawn, mxc_withdrawn, mxc_withdrawn*mxc_price))

        st.info("""
            Until further confirmation from the team I assume fuel has a 90% weight on miner health and
            miner health has a direct impact on base rewards(i.e. 50% miner health = 50% base rewards)
        """)


        # Simulation
        dates = np.arange(start_day, end_day + dt.timedelta(days=1))
        
        delta_days = (end_day + dt.timedelta(days=1) - start_day).days
        dates_ix = np.arange(0, delta_days, step=1.)
        withdraw_date_ix = (withdraw_date - start_day).days

        mined_mxc_v = [mxc_base_reward]
        tank_capacity_v = [0.]
        miner_fuel_v = [0.]
        miner_health_v = [1.]
        
        for date_ix in dates_ix[1:]:

            # Standard calculations
            mined_mxc_i = mxc_base_reward * miner_health_v[-1]     
            tank_capacity_i = mined_mxc_i + tank_capacity_v[-1]
            miner_fuel_i = mined_mxc_i + miner_fuel_v[-1]

            # Withdrawal event
            if date_ix == withdraw_date_ix:
                miner_fuel_i -= miner_fuel_i*percentage_withdrawn/100

            miner_health_i = 0.1 + 0.9*(miner_fuel_i/tank_capacity_i) # Uptime at full percent considered

            miner_fuel_v.append(miner_fuel_i)
            miner_health_v.append(miner_health_i)
            tank_capacity_v.append(tank_capacity_i)
            mined_mxc_v.append(mined_mxc_i)


        # Miner health and fuel
        fig = go.Figure()
        fig.update_layout(
            title = 'Miner health and miner fuel (%)',
            xaxis_title="Time",
            yaxis_title="Percentage",
            yaxis = dict(range=(0., 101)),
            height=400, width=700)

        fig.add_trace(go.Scatter(x=dates, y=np.array(miner_health_v)*100, mode='lines', name='Miner health(%)',
                    hovertemplate='%{x} - %{y:.2f}% <extra></extra>'))
        
        miner_health_pct = np.array(miner_fuel_v[1:]) / np.array(tank_capacity_v[1:])
        miner_health_pct = np.concatenate([[100.], miner_health_pct])
        fig.add_trace(go.Scatter(x=dates, y=np.array(miner_health_pct)*100, mode='lines', name='Miner fuel(%)',
                    hovertemplate='%{x} - %{y:.2f}% <extra></extra>'))
        

        st.plotly_chart(fig)

        # Mined MXC per day
        fig = go.Figure()
        fig.update_layout(
            title = 'Mined MXC per day',
            xaxis_title="Time",
            yaxis_title="MXC",
            showlegend=True,
            yaxis = dict(range=(0., mxc_base_reward+20)),
            height=400, width=700)

        # Plot train data
        fig.add_trace(go.Scatter(x=dates, y=mined_mxc_v, mode='lines', name='Mined MXC per day',
                        hovertemplate='%{x} - %{y:.2f} MXC <extra></extra>' ))

        st.plotly_chart(fig)

        fig = go.Figure()
        fig.update_layout(
            title = 'Tank capacity and absolute miner fuel',
            xaxis_title="Time",
            yaxis_title="MXC",
            height=400, width=700)

        # Plot train data
        fig.add_trace(go.Scatter(x=dates, y=miner_fuel_v, mode='lines', name='Miner fuel',
                        hovertemplate='%{x} - %{y:.2f} MXC <extra></extra>'))

        fig.add_trace(go.Scatter(x=dates, y=tank_capacity_v, mode='lines', name='Tank capacity',
                        hovertemplate='%{x} - %{y:.2f} MXC <extra></extra>'))

        st.plotly_chart(fig)

        st.info("Keep in mind that your total cumulative mined MXC is equal to your tank capacity")

    