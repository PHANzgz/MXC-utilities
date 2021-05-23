import streamlit as st
import os


def write():
    st.markdown(
        """
        # Next steps

        This calculator is already functional, but it is still in development to add functionalities,
        flexibility and more accurate results.  

        Here is a list of the things I want to add/improve (the project is open-source so DM me if you
        want to contribute):

        - API Calls to get the default value for MXC and DHX
        - Take into account the network growth(in terms of mPower) produced by yourself. This will solve
        the potential rewards and allow for larger investors to get more accurate results.
        - Add an option in the "mPower and DHX bonded" input method where the user can specify if he/she
        already owns some MXC or DHX.
        - Add an option where the user may specify the amount of MXC/mPower he is able to keep adding daily.
        For example if he/she owns several miners and wants to lock the profits this would be a very useful
        feature.
        - Write the mining guide
        - Take into account that each miner only boosts up to 1 million mPower.

        """
    )