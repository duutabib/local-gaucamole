import os
import h5py
import typing 
import logging
import argparse
from scipy.io import loadmat
import numpy as np




class RobustSigs:
    def __init__(self, 
    cells_datapath, 
    cells_positionsKey: str=None, 
    cells_dholderkey: str='CellResp', 
    cells_positions_path: str = None,
    eliminateNeurons: bool =True,
    simplify_cells: bool =True,
    ):
        self.cells_datapath = cells_datapath
        self.cells_positionsKey = cells_positionsKey
        self.cells_dholderkey = cells_dholderkey
        self.cells_positions_path = cells_positions_path
        self.eliminateNeurons=eliminateNeurons
        self.eliminateNeuronsKey = None
        self.simplify_cells = simplify_cells


    def __load_cellpositions__(self, simplify_cells: bool = True):

       if not self.cells_positions_path: 
        self.cells_positions = 'CellXYZ'
       if self.eliminateNeurons:
        self.eliminateNeuronsKey = 'IX_inval_anat'

       tmpdata = loadmat(self.cells_positions_path, simplify_cells=self.simplify_cells) 
       roi_positions_to_discard = tmpdata[self.cells_positionsKey][self.eliminateNeuronsKey]
       all_roi_positions = tmpdata[self.cells_positions]

       used_rois_positions = np.array(
        [row for j, row in enumerate(all_roi_positions) if j not in roi_positions_to_discard]
       ) 

       return used_rois_positions

        
    def __load_cell_responses__(self, num_neurons, sessions):
        # returns all valid cell responses (excluding invalid cells)
        tmpdata = h5py.File(self.cells_datapath, 'r')
        responses = tmpdata[self.cells_dholderkey][:]
        return responses


    def __get_cell_positions__(self, num_neurons):
        positions = self.__load_cellpositions__()[:num_neurons, :].T
        return positions

    def __get_cells_responses_and_positions__(self, num_neurons, sessions):
        responses = self.__load_cell_responses__(num_neurons, sessions)
        positions = self.__get_cell_positions__(num_neurons)
        return responses, positions 
  