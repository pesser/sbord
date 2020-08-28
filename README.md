# sbord

```
conda env create -f environment.yaml
conda activate sbord
sbord <path/to/log/folder>
```

`<path/to/log/folder>` defaults to `.` if not specified.

The `sbord` executable assumes you are using `miniconda3` and that it is
located at `~/miniconda3`. If you run into problems, adjust those paths in
`bin/sbord` and reinstall using `pip install -e .`.

To run from anywhere without activating the environment first, add
`~/miniconda3/envs/sbord/bin/sbord` to your path:

```
export PATH="${PATH}:~/miniconda3/envs/sbord/bin/sbord"
```
