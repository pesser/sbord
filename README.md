# sbord

```
conda env create -f environment.yaml
conda activate sbord
sbord <path/to/log/folder>
```

`<path/to/log/folder>` defaults to `.` if not specified.

To run from anywhere without activating the environment first, add
`~/miniconda3/envs/sbord/bin/sbord` to your path:

```
export PATH="${PATH}:~/miniconda3/envs/sbord/bin/sbord"
```
