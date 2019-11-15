Cloudy pixel recovery for rapid response flood mapping
==============================

## Purpose:
------------
. . . . 

--------

Set up
------------

Install the virtual environment with conda and activate it:

```bash
$ conda env create -f environment.yml
$ source activate tf_intel_cpu
```

Install `cpr` in the virtual environment:

```bash
$ pip install --editable .
```

Run Jupyter Notebook and open the notebooks in `notebooks/`:

```bash
$ jupyter notebook
```

Create config.py file in CPR folder to point to data folder