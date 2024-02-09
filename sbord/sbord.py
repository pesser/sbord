import argparse
import glob
import math
import os
import random
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from natsort import natsorted
from PIL import Image
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

# currently assumes to be started with
# streamlit run sbord.py -- <path/to/log/folder>

# parses filenames of the form
# name_gs-001_e-0_b-112.png
regex = re.compile(r"(.*)_gs-([0-9]+)_e-([0-9]+)_b-([0-9]+).\b(png|mp4)\b")

# start displaying images (0) or scalars (1)
DEFAULT_MODE = 0


@st.cache_data()
def load_df(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


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
    mode = st.sidebar.radio(
        "Mode",
        ("Images", "Scalars", "Compare", "Configs", "Videos"),
        index=DEFAULT_MODE,
    )

    if mode != "Compare":
        if len(paths) == 1:
            path_idx = 0
        else:
            path_idx = st.sidebar.radio(
                "logdir",
                list(range(len(paths))),
                index=0,
                format_func=lambda idx: paths[idx],
            )
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

        df = pd.DataFrame(
            {
                "gs": gs,
                "epoch": epoch,
                "batch": batch,
                "names": names,
                "fnames": fnames,
                "fpaths": fpaths,
            }
        )

        steps = sorted(df["gs"].unique())
        idx_selection = st.sidebar.selectbox(
            "Step selection",
            (
                "index input",
                "index slider",
                "step selection",
            ),
        )
        if idx_selection == "index input":
            idx = st.sidebar.number_input(
                "Global step idx",
                min_value=0,
                max_value=len(steps) - 1,
                value=len(steps) - 1,
            )
            global_step = steps[idx]
        elif idx_selection == "index slider":
            idx = st.sidebar.slider(
                "Global step idx",
                min_value=0,
                max_value=len(steps) - 1,
                value=len(steps) - 1,
            )
            global_step = steps[idx]
        elif idx_selection == "step selection":
            global_step = st.sidebar.selectbox("Global step", steps)

        st.sidebar.text("Global step: {}".format(global_step))
        entries = df[df["gs"] == global_step]

        group_by = st.sidebar.radio("Group by", ["Type", "Batch id"])

        if group_by == "Type":
            entries = entries.sort_values(by=["names", "epoch", "batch"])
        elif group_by == "Batch id":
            entries = entries.sort_values(by=["epoch", "batch", "names"])

        for name, fpath in zip(entries["names"], entries["fpaths"]):
            I = Image.open(fpath)
            if ignore_alpha and np.array(I).shape[-1] == 4:
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

    elif mode == "Videos":
        imagedirs = natsorted(os.listdir(os.path.join(path, "images")))
        imagedirs = [
            imagedir
            for imagedir in imagedirs
            if len(glob.glob(os.path.join(path, "images", imagedir, "*.mp4"))) > 0
        ]
        if len(imagedirs) == 0:
            st.info("No videos logged for this run")
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

            df = pd.DataFrame(
                {
                    "gs": gs,
                    "epoch": epoch,
                    "batch": batch,
                    "names": names,
                    "fnames": fnames,
                    "fpaths": fpaths,
                }
            )

            steps = sorted(df["gs"].unique())
            idx_selection = st.sidebar.selectbox(
                "Step selection",
                (
                    "index input",
                    "index slider",
                    "step selection",
                ),
            )
            if idx_selection == "index input":
                idx = st.sidebar.number_input(
                    "Global step idx",
                    min_value=0,
                    max_value=len(steps) - 1,
                    value=len(steps) - 1,
                )
                global_step = steps[idx]
            elif idx_selection == "index slider":
                idx = st.sidebar.slider(
                    "Global step idx",
                    min_value=0,
                    max_value=len(steps) - 1,
                    value=len(steps) - 1,
                )
                global_step = steps[idx]
            elif idx_selection == "step selection":
                global_step = st.sidebar.selectbox("Global step", steps)

            st.sidebar.text("Global step: {}".format(global_step))
            entries = df[df["gs"] == global_step]

            reencode = st.sidebar.checkbox("Re-encode videos", value=False)
            for name, fpath in zip(entries["names"], entries["fpaths"]):
                if reencode:
                    # correct video codec for streamlit cf
                    # https://github.com/streamlit/streamlit/issues/1580
                    basename = "/".join(fpath.split("/")[:-1])
                    f_name = fpath.split("/")[-1]
                    newpath = (
                        f'{os.path.join(basename,f_name.split(".")[0])}'
                        f'_.{fpath.split(".")[-1]}'
                    )
                    rc = (
                        f"ffmpeg -y -hide_banner -loglevel error  "
                        f"-i {fpath} -vcodec libx264 {newpath}"
                    )
                    subprocess.run(rc, shell=True)
                    mvc = f"mv {newpath} {fpath}"
                    subprocess.run(mvc, shell=True)
                st.text(name)
                st.video(fpath)

    elif mode == "Scalars":
        csv_root = get_csv_root(path)
        csv_paths = glob.glob(os.path.join(csv_root, "**/metrics.csv"))
        csv_paths = natsorted(csv_paths)
        short_csv_paths = ["/".join(csv_path.split("/")[-2:]) for csv_path in csv_paths]
        csv_path = st.sidebar.radio(
            "CSV file", short_csv_paths, index=len(short_csv_paths) - 1
        )
        csv_idx = short_csv_paths.index(csv_path)
        csv_path = csv_paths[csv_idx]

        df = load_df(csv_path)

        keys = list(df.keys())
        xaxis_options = ["contiguous", None] + keys
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
        active_keys = [k for k in active_keys if filter_.search(k)]

        if st.sidebar.checkbox("Use Pages", value=False):
            max_plots = st.sidebar.selectbox(
                "Maximum plots per page", (10, 25, 50, 100)
            )
            pages = max(1, int(math.ceil(len(active_keys) / max_plots)))
            page = st.sidebar.selectbox("Page", list(range(1, pages + 1))) - 1
            active_keys = active_keys[page * max_plots : (page + 1) * max_plots]

        idx_selection = st.sidebar.selectbox(
            "Step selection",
            (
                "index input",
                "index slider",
                "step selection",
            ),
        )

        alpha = st.sidebar.slider(
            "Smoothing", min_value=0.0, max_value=1.0, step=0.01, value=0.4
        )

        for k in active_keys:
            data = df[df[k].notnull()]
            if xaxis == "contiguous":
                x = np.arange(len(data))
            else:
                x = xaxis
            vanilla = True
            if alpha > 0.0:
                vanilla = False
                try:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            y=data[k],
                            x=x,
                            mode="lines",
                            name=k,
                            line={"color": "lightblue"},
                        )
                    )

                    wss = np.arange(5, 99, 2)
                    ws = wss[int((len(wss) - 1) * alpha)]

                    ws = min(ws, oddify(len(data) - 1))
                    ysm = savgol_filter(data[k], ws, 3)
                    fig.add_trace(
                        go.Scatter(
                            y=ysm, x=x, mode="lines", line=dict(color="midnightblue")
                        )
                    )

                    fig.update_layout(title=k)
                    st.plotly_chart(fig)
                except Exception as e:
                    vanilla = True
                    print(e)

            if vanilla:
                fig = px.line(data, x=x, y=k)
                st.plotly_chart(fig)

    elif mode == "Compare":
        dfs = []
        dfs_extra = []
        st.header("Choose Variables from logs")
        for p in paths:
            csv_root = get_csv_root(p)
            csv_paths = glob.glob(os.path.join(csv_root, "**/metrics.csv"))
            csv_paths = natsorted(csv_paths)
            if len(csv_paths) == 0:
                continue

            st.subheader(os.path.split(p)[1])
            if not st.checkbox(
                "active?", value=(len(paths) == 1), key=p + "active" + "checkbox"
            ):
                continue

            if len(csv_paths) == 1:
                active_csv_paths = csv_paths
            else:
                # this seems like a hack, but looks nice
                _, indent = st.columns([0.02, 1])
                short_csv_paths = [
                    "/".join(csv_path.split("/")[-2:]) for csv_path in csv_paths
                ]
                active_csv_paths = []
                for k, csv_p in enumerate(short_csv_paths):
                    if indent.checkbox(csv_p, value=True, key=csv_paths[k]):
                        active_csv_paths.append(csv_paths[k])

                if len(active_csv_paths) == 0:
                    continue

            df = pd.DataFrame()
            for csv_p in active_csv_paths:
                df = pd.concat((df, load_df(csv_p)), ignore_index=True)

            dfs.append(df)
            dfs_extra.append((p, st.container()))

        if len(dfs) == 0:
            st.warning("You need to activate some logs to plot anything")
            return

        fig = go.Figure()

        st.sidebar.text("Settings")
        st.header("Plot")
        keys = [set(df.keys()) for df in dfs]
        xaxis_keys = list(set.intersection(*keys))
        keys = list(set.union(*keys))
        xaxis_options = ["contiguous", None] + xaxis_keys
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
            default = g != "ungrouped"
            active_groups[g] = st.sidebar.checkbox(g, value=default)

        active_keys = [k for k in keys if active_groups[get_group(k)]]

        def get_ending(k):
            ksplit = k.rsplit("/", 1)[-1].rsplit("_", 1)
            if k == "created_at" or len(ksplit) == 1:
                return "unspecified"
            return ksplit[-1]

        endings = sorted(set([get_ending(k) for k in active_keys]))
        active_endings = dict()
        st.sidebar.text("Frequency")
        for e in endings:
            active_endings[e] = st.sidebar.checkbox(e, value=True)

        active_keys = [k for k in active_keys if active_endings[get_ending(k)]]

        filter_ = st.sidebar.text_input("Regex Filter")
        filter_ = re.compile(filter_)
        active_keys = [k for k in active_keys if filter_.search(k)]
        check_all = st.sidebar.checkbox("Check all", value=False)
        col1, col2 = st.columns([1, 6])
        with col1:
            smooth_mode = st.radio("Smoothing mode", ["Savgol", "Running Mean"])
        with col2:
            alpha = st.slider(
                "Smoothing amount", min_value=0.0, max_value=1.0, step=0.01, value=0.0
            )

        line_mode = st.selectbox("Line style", ["markers", "lines"])

        for df, (p, container) in zip(dfs, dfs_extra):
            with container:
                name = st.text_input("Name for plot legend:", os.path.split(p)[1])
                name_leg = f"{name}: " if len(dfs) > 1 else ""
                df_keys = natsorted(list(df.keys()))

                for key in df_keys:
                    if key not in active_keys:
                        continue

                    active = st.checkbox(key, value=check_all, key=p + key)
                    if not active:
                        continue

                    data = df[df[key].notnull()]
                    if xaxis == "contiguous":
                        x = np.arange(len(data))
                    else:
                        x = data[xaxis]

                    if alpha == 0.0:
                        fig.add_trace(
                            go.Scatter(
                                y=data[key],
                                x=x,
                                mode=line_mode,
                                name=f"{name_leg}{key}",
                            )
                        )
                    else:
                        if smooth_mode == "Savgol":
                            wss = np.arange(5, 99, 2)
                            ws = wss[int((len(wss) - 1) * alpha)]
                            ws = min(ws, oddify(len(data) - 1))
                            ysm = savgol_filter(data[key], ws, 3)
                        elif smooth_mode == "Running Mean":
                            size = int(len(data[key]) * alpha)
                            ysm = uniform_filter1d(
                                data[key],
                                size=size,
                                # align filter start on left:
                                # origin=-(size//2),
                                # align filter start on right:
                                origin=size // 2 - 1,
                                mode="nearest",
                            )
                        fig.add_trace(
                            go.Scatter(
                                y=ysm, x=x, mode="lines", name=f"{name_leg}{key}"
                            )
                        )

        colors = (
            ["default"]
            + ["qualitative." + c for c in dir(px.colors.qualitative) if c[0].isupper()]
            + ["diverging." + c for c in dir(px.colors.diverging) if c[0].isupper()]
            + ["sequential." + c for c in dir(px.colors.sequential) if c[0].isupper()]
            + ["cyclical." + c for c in dir(px.colors.cyclical) if c[0].isupper()]
        )
        colors = st.selectbox("Colorscheme", colors)
        if colors is not None and colors != "default":
            module, color_name = colors.split(".")
            fig.layout["template"]["layout"]["colorway"] = getattr(
                getattr(px.colors, module), color_name
            )
        name = st.text_input("Name plot", "")
        config = {
            "toImageButtonOptions": {
                "format": "svg",  # one of png, svg, jpeg, webp
                "filename": name if name else "plot",
                "height": None,
                "width": None,
                "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
            }
        }
        fig.update_layout(title=name, xaxis_title=xaxis, font={"size": 18})
        st.plotly_chart(fig, config=config)
    elif mode == "Configs":
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
    parser = argparse.ArgumentParser(
        description="The streamlit based alternative to tensorboard."
    )

    parser.add_argument("paths", default=".", nargs="*")
    args = parser.parse_args()

    # remove trailing "/"
    paths = [p[: -len(os.sep)] if p.endswith(os.sep) else p for p in args.paths]
    paths.sort()
    main(paths)
