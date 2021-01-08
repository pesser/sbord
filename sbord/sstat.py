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
import plotly.graph_objects as go
from natsort import natsorted

# currently assumes to be started with
# streamlit run sstat.py -- <path/to/log/folder>

DEFAULT_USER = "rrombach"
HISTORY_PATH = "/home/robin/mnt/export/data2/rrombach/etc/sstatlogs"  # adopt as needed


def history(hpath):
    st.header("History")
    plots = {"num_gpu_processes_score": {},
             "accumulated_vram": {}}
    for k in ["free_data", "process_data", "utilization_data"]:
        if k == "process_data":
            initialized = False
            all_files = os.listdir(os.path.join(hpath, k))
            prog = 0.
            prog_bar = st.progress(prog)
            with st.spinner("Crunching the latest data for you. Hang tight."):
                for i, csv_file in enumerate(all_files):
                    csv_path = os.path.join(hpath, k, csv_file)
                    df = pd.read_csv(csv_path, quotechar="'")
                    users = df["user"].unique()

                    if not initialized:
                        for user in users:
                            plots["accumulated_vram"][user] = list()
                            plots["num_gpu_processes_score"][user] = [len(df[df["user"] == user])]

                        initialized = True
                    for user in users:
                        try:
                            plots["accumulated_vram"][user].append(df.loc[df["user"] == user]["used_gpu_memory"])

                            print("##########################")
                            print(df.loc[df["user"] == user]["used_gpu_memory"])
                            print("------")
                            print(df.loc[df["user"] == user]["used_gpu_memory"].sum())
                            print("##########################")

                        except:
                            plots["accumulated_vram"][user] = list()
                            plots["accumulated_vram"][user].append(df.loc[df["user"] == user]["used_gpu_memory"])
                    else:
                        for user in users:
                            try:
                                plots["num_gpu_processes_score"][user].append(len(df[df["user"] == user]))
                            except KeyError:
                                plots["num_gpu_processes_score"][user] = [len(df[df["user"] == user])]

                    prog += 1./(len(all_files))
                    prog_bar.progress(prog)

    for pk in plots:
        fig = go.Figure()
        if pk in ["num_gpu_processes_score", "accumulated_vram"]:
            for user in plots[pk]:
                timeax = np.linspace(0, 1., len(plots[pk][user]))
                fig.add_trace(go.Scatter(x=timeax, y=plots[pk][user], mode='lines', name=user))

        if pk == "accumulated_vram":
            for user in plots[pk]:
                pass
                #print(user, plots[pk])
        st.write(pk)
        st.write(fig)


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

    if st.checkbox("Show History"):
        history(HISTORY_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display output of queue/checkgpu.py')

    parser.add_argument('path', default=".", nargs="?")
    args = parser.parse_args()

    main(args.path)
