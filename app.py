import streamlit as st
import streamlit_analytics

# Load pages
import mining_calculator
import mining_tutorial
import next_steps
import fuel_calculator

PAGE_NAMES = ["Mining calculator", "Fuel calculator", "Next steps"]
PAGE_SRCS = [mining_calculator, fuel_calculator, next_steps]

st.set_page_config(page_title="DHX Mining calculator")

# Persistent storage for API calls
@st.cache(allow_output_mutation=True)
def storage():
    # time, mxc_price, dhx_price
    return []

state = storage()

def main():

    # Init storage
    if len(state) == 0:
        state.extend([0., 0., 0.])

    with streamlit_analytics.track():

        # Application select
        st.sidebar.title("MXC/DHX Mining guide")
        page_selection = st.sidebar.radio("Menu", PAGE_NAMES)
        page_selection_ix = PAGE_NAMES.index(page_selection)
        page = PAGE_SRCS[page_selection_ix]

        # Write selected page
        page.write(state)

        st.sidebar.title("About")
        st.sidebar.info(
            """
            This calculator was created by PHAN to help the community and it is ad-free
            and open-source. If you want to support me, you can buy me some coffe here:   
            ETH, MXC (ERC20 and BEP20): `0xEEBFbb5EF279dCBeAA3eEe505d1CefBA040FFD5a`  
            DASH: `XeFNCe39ebB4vnruL1hwqsr57mHDqE8D9i`  
            Paypal:  [![Donate](https://www.paypalobjects.com/en_US/i/btn/btn_donate_SM.gif)](https://www.paypal.com/donate?hosted_button_id=TX8C42JLJ27AC)
            
            You can find the source and contribute [here](https://github.com/PHANzgz/MXC-utilities).  
            
            Special thanks to @Midir21, @TavernSideGaming, @Mlazear and @Monok who have actively or passively
            contributed to the project.
            """
            )


if __name__ == "__main__":
    main()