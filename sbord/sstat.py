import os
import glob
import sys
import random
import argparse
import re
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
from natsort import natsorted

# currently assumes to be started with
# streamlit run sstat.py -- <path/to/log/folder>

DEFAULT_USER = "pesser"

def main(path):
    st.sidebar.title("sstat")

    data = dict()
    for k in ["free_data.csv", "process_data.csv", "utilization_data.csv"]:
        csv_path = os.path.join(path, k)
        df = pd.read_csv(csv_path, quotechar="'")
        data[k] = df

        st.text(k)

        if k == "process_data.csv":
            timestamp = df["timestamp"][0]
            st.sidebar.text("updated: {}".format(timestamp))

            users = df["user"].unique()
            score = [len(df[df["user"]==user]) for user in users]
            score_df = pd.DataFrame({"user": users, "score": score})
            score_df.sort_values(by=["score"], inplace=True, ascending=False)
            st.text("High Score")
            st.dataframe(score_df)

            user_options = [None] + list(df["user"].unique())
            if DEFAULT_USER in user_options:
                default_idx = user_options.index(DEFAULT_USER)
            else:
                default_idx = 0
            user = st.selectbox("User", user_options, index=default_idx)
            if user is not None:
                df = df[df["user"]==user]
            st.text("# Processes: {}".format(len(data[k])))

        table_display = {"dataframe": st.dataframe, "table": st.table}
        method = st.radio("Display method", sorted(table_display.keys()), key=k)
        table_display[method](df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display output of queue/checkgpu.py')

    parser.add_argument('path', default=".", nargs="?")
    args = parser.parse_args()

    main(args.path)
