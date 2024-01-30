from one.api import ONE
import numpy as np
from pynwb import NWBHDF5IO
from nwbwidgets import nwb2widget
import matplotlib.pyplot as plt

fpath = "/Users/duuta/ppp/data/iblData/000017/sub-Cori/sub-Cori_ses-20161214T120000.nwb"
with NWBHDF5IO(fpath, mode='r') as io:
    nwbfile = io.read()

nwb2widget(nwbfile)