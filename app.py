import streamlit as st
import streamlit_analytics

# Load pages
import mining_calculator
import mining_tutorial
import next_steps

PAGE_NAMES = ["Mining calculator", "Mining guide", "Next steps"]
PAGE_SRCS = [mining_calculator, mining_tutorial, next_steps]


def main():

    with streamlit_analytics.track():

        st.set_page_config(page_title="DHX Mining calculator")

        # Application select
        st.sidebar.title("MXC/DHX Mining guide")
        page_selection = st.sidebar.radio("Menu", PAGE_NAMES)
        page_selection_ix = PAGE_NAMES.index(page_selection)
        page = PAGE_SRCS[page_selection_ix]

        # Write selected page
        page.write()

        st.sidebar.title("About")
        st.sidebar.info(
            """
            This calculator was created by PHAN to help the community and it is ad-free
            and open-source. If you want to support me, you can buy me some coffe here:   
            ETH, MXC: `0xEEBFbb5EF279dCBeAA3eEe505d1CefBA040FFD5a`  
            DASH: `XeFNCe39ebB4vnruL1hwqsr57mHDqE8D9i`
            
            You can find the source and contribute [here](https://github.com/PHANzgz/MXC-utilities).  
            
            Special thanks to @Midir21, @TavernSideGaming and @Mlazear who have actively or passively
            contributed to the project.
            """
            )


if __name__ == "__main__":
    main()