import argparse
import glob
import os
import random
import re
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from natsort import natsorted
from PIL import Image

st.beta_set_page_config(
    page_title="sstat",
    # page_icon="🔌",
    page_icon="🔋",
    # layout="centered",
    layout="wide",
    initial_sidebar_state="expanded",
)

# allow setting of default values via query parameters
query_params = st.experimental_get_query_params()
DEFAULT_USER = query_params.get("user", ["anonymous"])[0]

table_display = {"dataframe": st.dataframe, "table": st.table}
table_display_keys = sorted(table_display.keys())
DEFAULT_METHOD = query_params.get("display", ["dataframe"])[0]
try:
    DEFAULT_METHOD = table_display_keys.index(DEFAULT_METHOD)
except ValueError:
    DEFAULT_METHOD = 0


# subset of keys to display
process_keys = [
    "hostname",
    "index",
    "command",
    "used_gpu_memory",
    "memory.total",
    "memory.free",
    "utilization.gpu",
    "user",
]


def main(path):
    headers = {
        "process_data.csv": "Processes",
        "utilization_data.csv": "GPU utilization",
        "free_data.csv": "Free GPUs",
    }

    method = st.sidebar.radio(
        "Display method", table_display_keys, index=DEFAULT_METHOD
    )
    update_info = st.sidebar.empty()
    vram_options = {
        "[0,∞)": None,
        "[0, 20)": [0, 20000],
        "[20,40)": [20000, 40000],
        "[40,∞)": [40000, np.infty],
    }
    filter_by_vram = vram_options[
        st.sidebar.radio("Filter by VRAM (GB)", list(vram_options.keys()))
    ]

    data = dict()
    for k in ["free_data.csv", "process_data.csv", "utilization_data.csv"]:
        csv_path = os.path.join(path, k)
        df = pd.read_csv(csv_path, quotechar="'")
        data[k] = df

        if k == "process_data.csv":
            # get users
            user_options = [None] + list(df["user"].unique())

            # get timestamp
            timestamp = df["timestamp"][0]
            update_info.text("updated: {}".format(timestamp))

            # add gpu indices
            index_df = pd.read_csv(
                os.path.join(path, "utilization_data.csv"), quotechar="'"
            )
            index_df = index_df[
                ["uuid", "index", "memory.total", "memory.free", "utilization.gpu"]
            ]
            df = df.join(index_df.set_index("uuid"), on="gpu_uuid")

        # convert to integers
        for col in ["memory.free", "memory.total", "memory.used", "used_gpu_memory"]:
            if col in df:
                df[col] = df[col].map(lambda x: int(x.split()[0]))

        if filter_by_vram is not None:
            if "memory.total" not in df:
                assert k == "free_data.csv"
                memkey = "memory.free"
            else:
                memkey = "memory.total"

            df = df[
                (filter_by_vram[0] <= df[memkey]) & (df[memkey] < filter_by_vram[1])
            ]

        if k == "process_data.csv":
            # add per user data
            users = df["user"].unique()
            score = list()
            memscore = list()
            for user in users:
                usrprocesses = df[df["user"] == user]
                ngpus = len(usrprocesses["gpu_uuid"].unique())
                score.append(ngpus)
                vram = usrprocesses["used_gpu_memory"].sum()
                memscore.append(vram)
            # score = [len(df[df["user"]==user]) for user in users]
            gpukey = "gpus "
            memkey = "vram "
            score_df = pd.DataFrame({"user": users, gpukey: score, memkey: memscore})
            score_df[" "] = len(score_df) * [" "]  # hack to increase width of display
            score_df.sort_values(by=[gpukey], inplace=True, ascending=False)
            st.subheader("GPU usage by user")
            st.markdown("`gpus` number of gpus running processes from user")
            st.markdown("`vram` total vram used by gpu processes from user")
            table_display[method](score_df)

            if DEFAULT_USER in user_options:
                default_idx = user_options.index(DEFAULT_USER)
            else:
                default_idx = 0
            user = st.sidebar.selectbox(
                "Filter Processes by User", user_options, index=default_idx
            )
            if user is not None:
                df = df[df["user"] == user]
            st.sidebar.text("# Processes: {}".format(len(df)))

            # filter keys
            df = df[process_keys]

        if k == "utilization_data.csv":
            # add list of hosts
            hosts = df["hostname"].unique()
            st.sidebar.text("Hosts:")
            st.sidebar.code("\n".join(hosts))

        st.subheader(headers[k])
        table_display[method](df)

    if st.checkbox("Show History"):
        history(HISTORY_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display output of queue/checkgpu.py")

    parser.add_argument("path", default=".", nargs="?")
    args = parser.parse_args()

    main(args.path)
