import streamlit as st

# Load pages
import mining_calculator
import mining_tutorial

PAGE_NAMES = ["Mining calculator", "Mining guide"]
PAGE_SRCS = [mining_calculator, mining_tutorial]


def main():

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
        and open-source. If you want to support me, you can send me MXC here: 
        `0xEEBFbb5EF279dCBeAA3eEe505d1CefBA040FFD5a`  
        
        You can find the source and contribute [here](https://github.com/PHANzgz/MXC-utilities).  
        
        Special thanks to @Midir21 on Telegram for the amazing spreadsheet this web-app is based.
        """
        )

if __name__ == "__main__":
    main()