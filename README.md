# sbord & sstat

Some streamlit scripts that we use to display our training logs (`sbord`) and
to monitor our servers (`sstat`). Note that the latter requires monitoring data
in some quite specific and undocumented format so it is probably not that re-usable.

# sbord

Run directly as `streamlit run sbord/sbord.py -- <path/to/log/folder>` or install and
run via

```
conda env create -f environment.yaml
conda activate sbord
sbord <port> <path/to/log/folder>
```

`<port>` defaults to 8501 and `<path/to/log/folder>` defaults to `.` if not specified.

The `sbord` executable assumes you are using `miniconda3` and that it is
located at `~/miniconda3`. If you run into problems, adjust those paths in
`bin/sbord` and reinstall using `pip install -e .`.

To run from anywhere without activating the environment first, add
`~/miniconda3/envs/sbord/bin/sbord` to your path:

```
export PATH="${PATH}:~/miniconda3/envs/sbord/bin/sbord"
```

# sstat

Same as above. Point first argument to folder containing `free_data.csv`,
`process_data.csv` and `utilization.csv`. It's convenient to to

```
alias sstat="~/miniconda3/envs/sbord/bin/sstat <path>"
```

To adjust streamlit settings directly, run as

```
streamlit run sboard/sstat.py --server.port 8080 -- <path/to/queue>
```

# usrstat

Just a copy of sstat for deployment.
