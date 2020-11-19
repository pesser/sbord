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

st.beta_set_page_config(
    page_title="sstat",
    #page_icon="🔌",
    page_icon="🔋",
    #layout="centered",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_USER = "pesser"

process_keys = ["hostname", "index", "command", "used_gpu_memory",
                "memory.total", "memory.free",
                "utilization.gpu", "user"]


def main(path):
    #st.sidebar.title("Comput0rZz")

    headers = {"process_data.csv": "Processes",
               "utilization_data.csv": "GPU utilization",
               "free_data.csv": "Free GPUs"}

    table_display = {"dataframe": st.dataframe, "table": st.table}
    method = st.sidebar.radio("Display method", sorted(table_display.keys()))

    data = dict()
    for k in ["free_data.csv", "process_data.csv", "utilization_data.csv"]:
    #for k in ["process_data.csv", "utilization_data.csv"]:
        csv_path = os.path.join(path, k)
        df = pd.read_csv(csv_path, quotechar="'")
        data[k] = df


        if k == "process_data.csv":
            timestamp = df["timestamp"][0]
            st.sidebar.text("updated: {}".format(timestamp))

            users = df["user"].unique()
            score = list()
            memscore = list()
            for user in users:
                usrprocesses = df[df["user"]==user]
                ngpus = len(usrprocesses["gpu_uuid"].unique())
                score.append(ngpus)
                vram = usrprocesses["used_gpu_memory"].map(lambda x: int(x.split()[0])).sum()
                memscore.append(vram)
            #score = [len(df[df["user"]==user]) for user in users]
            gpukey = "gpus "
            memkey = "vram "
            score_df = pd.DataFrame({"user": users, gpukey: score, memkey: memscore})
            score_df[" "] = len(score_df)*[" "] # hack to increase width of display
            score_df.sort_values(by=[gpukey], inplace=True, ascending=False)
            st.subheader("GPU usage by user")
            st.markdown("`gpus` number of gpus running processes from user")
            st.markdown("`vram` total vram used by gpu processes from user")
            table_display[method](score_df)

            user_options = [None] + list(df["user"].unique())
            user_options = [None] + list(df["user"].unique())
            if DEFAULT_USER in user_options:
                default_idx = user_options.index(DEFAULT_USER)
            else:
                default_idx = 0
            user = st.sidebar.selectbox("Filter Processes by User", user_options, index=default_idx)
            if user is not None:
                df = df[df["user"]==user]
            st.sidebar.text("# Processes: {}".format(len(df)))

            # add gpu indices
            index_df = pd.read_csv(os.path.join(path, "utilization_data.csv"), quotechar="'")
            index_df = index_df[["uuid", "index", "memory.total", "memory.free", "utilization.gpu"]]
            df = df.join(index_df.set_index("uuid"), on="gpu_uuid")

            df = df[process_keys]

        for col in ["memory.free", "memory.total", "memory.used",
                    "used_gpu_memory"]:
            if col in df:
                df[col] = df[col].map(lambda x: int(x.split()[0]))
                #df[col+("MiB")] = df[col].map(lambda x: int(x.split()[0]))
                #del df[col]

        if k == "utilization_data.csv":
            hosts = df["hostname"].unique()
            st.sidebar.text("Hosts:")
            st.sidebar.code("\n".join(hosts))

        st.subheader(headers[k])
        table_display[method](df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display output of queue/checkgpu.py')

    parser.add_argument('path', default=".", nargs="?")
    args = parser.parse_args()

    main(args.path)
