import streamlit as st
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json

url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
parameters = {
  'symbol':'MXC,DHX',
  'convert':'USD'
}
headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': st.secrets["cmc_api_key"],
}

session = Session()
session.headers.update(headers)

#@st.cache
def get_mxc_dhx_prices():
    try:
        response = session.get(url, params=parameters)
        data = json.loads(response.text)
        mxc_price = data["data"]["MXC"]["quote"]["USD"]["price"]
        dhx_price = data["data"]["DHX"]["quote"]["USD"]["price"]
        return mxc_price, dhx_price
    except (ConnectionError, Timeout, TooManyRedirects) as e:
        st.warning("Unable to get MXC and DHX prices, please set them manually")
        return 0.03, 80. # arbitrary