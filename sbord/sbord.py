import os
import glob
import sys
import random
import argparse
import re
import math
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from natsort import natsorted
from scipy.signal import savgol_filter
import subprocess

# currently assumes to be started with
# streamlit run sbord.py -- <path/to/log/folder>

# parses filenames of the form
# name_gs-001_e-0_b-112.png
regex = re.compile(r"(.*)_gs-([0-9]+)_e-([0-9]+)_b-([0-9]+).\b(png|mp4)\b")

# start displaying images (0) or scalars (1)
DEFAULT_MODE=0

def oddify(x):
    if x % 2 == 0:
        x = x - 1
    return x

def get_csv_root(path):
    for k in ["csvlogger", "testtube"]:
        csv_root = os.path.join(path, k)
        if os.path.exists(csv_root):
            break
    return csv_root

def main(paths):
    st.sidebar.title("sbord")
    logdir_text = st.sidebar.empty()
    mode = st.sidebar.radio("Mode", ("Images", "Scalars", "Compare", "Configs",'Videos'), index=DEFAULT_MODE)

    if mode == "Compare":
        active_paths = dict()
        for p in paths:
            active_paths[p] = st.sidebar.checkbox(p, value=True)
    else:
        if len(paths) == 1:
            path_idx = 0
        else:
            path_idx = st.sidebar.radio("logdir", list(range(len(paths))), index=0, format_func=lambda idx: paths[idx])
        path = paths[path_idx]
        logdir = os.path.realpath(path).split("/")
        logdir = logdir[-1] if logdir[-1] else logdir[-2]
        logdir_text.text(logdir)

    if mode == "Images":
        ignore_alpha = st.sidebar.checkbox("Ignore alpha", value=False)
        max_width = st.sidebar.slider("Image width", min_value=-1, max_value=1024)
        max_width = None if max_width <= 0 else max_width

        imagedirs = natsorted(os.listdir(os.path.join(path, "images")))
        imagedir = st.sidebar.radio("Image directory", imagedirs)
        imagedir_idx = imagedirs.index(imagedir)
        imagedir = imagedirs[imagedir_idx]
        imagedir = os.path.join(path, "images", imagedir)

        fpaths = natsorted(glob.glob(os.path.join(imagedir, "*.png")))
        fnames = [os.path.split(fpath)[1] for fpath in fpaths]
        matches = [regex.match(fname) for fname in fnames]
        indices = [i for i in range(len(matches)) if matches[i] is not None]

        fpaths = [fpaths[i] for i in indices]
        fnames = [fnames[i] for i in indices]
        matches = [matches[i] for i in indices]
        names = [match.group(1) for match in matches]
        gs = [int(match.group(2)) for match in matches]
        epoch = [int(match.group(3)) for match in matches]
        batch = [int(match.group(4)) for match in matches]

        df = pd.DataFrame({
            "gs": gs,
            "epoch": epoch,
            "batch": batch,
            "names": names,
            "fnames": fnames,
            "fpaths": fpaths,
        })


        steps = sorted(df["gs"].unique())
        idx_selection = st.sidebar.selectbox("Step selection", ("index input",
                                                                "index slider",
                                                                "step selection",
                                                                ))
        if idx_selection == "index input":
            idx = st.sidebar.number_input("Global step idx",
                                          min_value=0,
                                          max_value=len(steps)-1,
                                          value=len(steps)-1)
            global_step = steps[idx]
        elif idx_selection == "index slider":
            idx = st.sidebar.slider("Global step idx",
                                    min_value=0,
                                    max_value=len(steps)-1,
                                    value=len(steps)-1)
            global_step = steps[idx]
        elif idx_selection == "step selection":
            global_step = st.sidebar.selectbox("Global step", steps)

        st.sidebar.text("Global step: {}".format(global_step))
        entries = df[df["gs"]==global_step]

        #st.sidebar.text("Selected")
        #st.sidebar.dataframe(entries)
        #st.sidebar.text("All images")
        #st.sidebar.dataframe(df)

        for name, fpath in zip(entries["names"], entries["fpaths"]):

            I = Image.open(fpath)
            if ignore_alpha and np.array(I).shape[-1]==4:
                I = Image.fromarray(np.array(I)[:, :, :3])
            st.text(name)
            st.image(I, width=max_width)
            # download original
            ext = os.path.splitext(fpath)[1]
            with open(fpath, "rb") as f:
                st.download_button(
                        "Original Image",
                        data=f,
                        file_name=os.path.basename(fpath),
                        mime=f"image/{ext}",
                        )

    elif mode == 'Videos':
        imagedirs = natsorted(os.listdir(os.path.join(path, "images")))
        imagedirs = [imagedir for imagedir in imagedirs if
                     len(glob.glob(os.path.join(path, 'images', imagedir, '*.mp4'))) > 0]
        if len(imagedirs) == 0:
            st.info('No videos logged for this run')
        else:
            imagedir = st.sidebar.radio("Video directory", imagedirs)
            imagedir_idx = imagedirs.index(imagedir)
            imagedir = imagedirs[imagedir_idx]
            imagedir = os.path.join(path, "images", imagedir)

            fpaths = natsorted(glob.glob(os.path.join(imagedir, "*.mp4")))
            fnames = [os.path.split(fpath)[1] for fpath in fpaths]
            matches = [regex.match(fname) for fname in fnames]
            indices = [i for i in range(len(matches)) if matches[i] is not None]

            fpaths = [fpaths[i] for i in indices]
            fnames = [fnames[i] for i in indices]
            matches = [matches[i] for i in indices]
            names = [match.group(1) for match in matches]
            gs = [int(match.group(2)) for match in matches]
            epoch = [int(match.group(3)) for match in matches]
            batch = [int(match.group(4)) for match in matches]

            df = pd.DataFrame({
                "gs": gs,
                "epoch": epoch,
                "batch": batch,
                "names": names,
                "fnames": fnames,
                "fpaths": fpaths,
            })


            steps = sorted(df["gs"].unique())
            idx_selection = st.sidebar.selectbox("Step selection", ("index input",
                                                                    "index slider",
                                                                    "step selection",
                                                                    ))
            if idx_selection == "index input":
                idx = st.sidebar.number_input("Global step idx",
                                              min_value=0,
                                              max_value=len(steps) - 1,
                                              value=len(steps) - 1)
                global_step = steps[idx]
            elif idx_selection == "index slider":
                idx = st.sidebar.slider("Global step idx",
                                        min_value=0,
                                        max_value=len(steps) - 1,
                                        value=len(steps) - 1)
                global_step = steps[idx]
            elif idx_selection == "step selection":
                global_step = st.sidebar.selectbox("Global step", steps)

            st.sidebar.text("Global step: {}".format(global_step))
            entries = df[df["gs"] == global_step]

            reencode = st.sidebar.checkbox("Re-encode videos", value=False)
            for name, fpath in zip(entries["names"], entries["fpaths"]):
                if reencode:
                    # correct video codec for streamlit cf https://github.com/streamlit/streamlit/issues/1580
                    basename = '/'.join(fpath.split('/')[:-1])
                    f_name = fpath.split('/')[-1]
                    newpath = f'{os.path.join(basename,f_name.split(".")[0])}_.{fpath.split(".")[-1]}'
                    rc = f'ffmpeg -y -hide_banner -loglevel error  -i {fpath} -vcodec libx264 '+newpath
                    subprocess.run(rc,shell=True)
                    mvc = f'mv {newpath} {fpath}'
                    subprocess.run(mvc,shell=True)
                st.text(name)
                st.video(fpath)


    elif mode=="Scalars":
        csv_root = get_csv_root(path)
        csv_paths = glob.glob(os.path.join(csv_root, "**/metrics.csv"))
        csv_paths = natsorted(csv_paths)
        short_csv_paths = ["/".join(csv_path.split("/")[-2:]) for csv_path in csv_paths]
        csv_path = st.sidebar.radio("CSV file", short_csv_paths, index=len(short_csv_paths)-1)
        csv_idx = short_csv_paths.index(csv_path)
        csv_path = csv_paths[csv_idx]

        df = pd.read_csv(csv_path)

        keys = list(df.keys())
        xaxis_options = ["contiguous", None]+keys
        xaxis = st.sidebar.selectbox("x-axis", xaxis_options)

        def get_group(k):
            ksplit = k.split("/", 1)
            if len(ksplit) == 1:
                return "ungrouped"
            return ksplit[0]

        groups = sorted(set([get_group(k) for k in keys]))
        active_groups = dict()
        st.sidebar.text("Groups")
        for g in groups:
            active_groups[g] = st.sidebar.checkbox(g, value=True)

        filter_ = st.sidebar.text_input("Regex Filter")
        filter_ = re.compile(filter_)
        active_keys = [k for k in df if active_groups[get_group(k)]]
        active_keys = [k for k in active_keys if filter_.match(k)]
        max_plots = st.sidebar.selectbox("Maximum plots per page", (10, 25,
                                                                    50, 100))
        pages = max(1, int(math.ceil(len(active_keys) / max_plots)))
        page = st.sidebar.selectbox("Page", list(range(1, pages+1)))-1
        active_keys = active_keys[page*max_plots:(page+1)*max_plots]

        idx_selection = st.sidebar.selectbox("Step selection", ("index input",
                                                                "index slider",
                                                                "step selection",
                                                                ))

        alpha = st.sidebar.slider("Smoothing", min_value=0.0, max_value=1.0, step=0.01, value=0.4)

        for k in active_keys:
            data = df[df[k].notnull()]
            if xaxis=="contiguous":
                x = np.arange(len(data))
            else:
                x = xaxis
            vanilla = True
            if alpha > 0.0:
                vanilla = False
                try:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=data[k], x=x, mode='lines', name=k, line=dict(color="lightblue") ))

                    wss = np.arange(5, 99, 2)
                    ws = wss[int(len(wss)*alpha)]

                    ws = min(ws, oddify(len(data)-1))
                    ysm = savgol_filter(data[k], ws, 3)
                    fig.add_trace(go.Scatter(y=ysm, x=x, mode='lines', line=dict(color="midnightblue")))

                    fig.update_layout(title=k)
                    st.plotly_chart(fig)
                except Exception as e:
                    vanilla = True
                    print(e)

            if vanilla:
                fig=px.line(data, x=x, y=k)
                st.plotly_chart(fig)

    elif mode=="Compare":
        paths = [p for p in paths if active_paths[p]]
        dfs = []
        for p in paths:
            csv_root = get_csv_root(p)
            csv_paths = glob.glob(os.path.join(csv_root, "**/metrics.csv"))
            csv_paths = natsorted(csv_paths)
            if len(csv_paths) == 0:
                continue
            elif len(csv_paths) == 1:
                csv_idx = 0
            else:
                short_csv_paths = ["/".join(csv_path.split("/")[-2:]) for csv_path in csv_paths]
                csv_path = st.sidebar.radio(p, short_csv_paths, index=len(short_csv_paths)-1)
                csv_idx = short_csv_paths.index(csv_path)

            csv_path = csv_paths[csv_idx]
            df = pd.read_csv(csv_path)
            dfs.append(df)

        keys = [set(df.keys()) for df in dfs]
        xaxis_keys = list(set.intersection(*keys))
        keys = list(set.union(*keys))
        xaxis_options = ["contiguous", None]+xaxis_keys
        xaxis = st.selectbox("x-axis", xaxis_options)

        def get_group(k):
            ksplit = k.split("/", 1)
            if len(ksplit) == 1:
                return "ungrouped"
            return ksplit[0]

        groups = sorted(set([get_group(k) for k in keys]))
        active_groups = dict()
        st.sidebar.text("Groups")
        for g in groups:
            default = (g != "ungrouped")
            active_groups[g] = st.sidebar.checkbox(g, value=default)

        filter_ = st.sidebar.text_input("Regex Filter")
        filter_ = re.compile(filter_)
        active_keys = [k for k in keys if active_groups[get_group(k)]]
        active_keys = [k for k in active_keys if filter_.match(k)]

        st.header("Choose Variables from logs")
        fig = go.Figure()
        for i, df in enumerate(dfs):
            st.subheader(paths[i])
            name = st.text_input("Name for plot legend:", paths[i])
            name_leg = f"{name}: " if len(dfs) > 1 else ""
            for key in df.keys():
                if key not in active_keys: continue
                active = st.checkbox(key, value=False, key=paths[i]+key)
                if not active: continue

                data = df[df[key].notnull()]
                if xaxis == "contiguous":
                    x = np.arange(len(data))
                else:
                    x = data[xaxis]
                fig.add_trace(go.Scatter(y=data[key], x=x, mode="lines", name=f"{name_leg}{key}     "))

        st.header("Plot")
        name = st.text_input("Name plot", "")
        config = {
          'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': name if name else 'plot',
            'height': 1000,
            'width': 1400,
            'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
          }
        }
        fig.update_layout(title=name, xaxis_title=xaxis, font={"size":18})
        st.plotly_chart(fig, config=config)
    elif mode=="Configs":
        import yaml
        cfg_root = os.path.join(path, "configs")
        cfg_paths = glob.glob(os.path.join(cfg_root, "*.yaml"))
        cfg_paths = natsorted(cfg_paths)[::-1]
        cfg_names = [os.path.split(path)[1] for path in cfg_paths]
        active_names = dict()
        for name in cfg_names:
            active_names[name] = st.sidebar.checkbox(name, value=True)
        for name, path in zip(cfg_names, cfg_paths):
            if active_names[name]:
                with open(path, "r") as f:
                    try:
                        cfg = yaml.load(f, Loader=yaml.CLoader)
                    except AttributeError:
                        cfg = yaml.load(f)
                st.text(os.path.split(path)[1])
                st.json(cfg)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The streamlit based alternative to tensorboard.')

    parser.add_argument('paths', default=".", nargs="*")
    args = parser.parse_args()

    main(args.paths)
