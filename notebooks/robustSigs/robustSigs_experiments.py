import os
import h5py
from typing import Tuple, Optional
import logging
import argparse
from scipy.io import loadmat
import numpy as np


# TODO: Refactor this code to be more readable and maintainable

# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class RobustSigs:
    def __init__(self, 
    cells_datapath, 
    cells_positionsKey: str=None, 
    cells_dholderkey: str='CellResp', 
    cells_positions_path: str = None,
    eliminateNeurons: bool =True,
    simplify_cells: bool =True,
    ):

        if not cells_datapath:
            raise ValueError("cells_datapath is required")

        if not os.path.exists(cells_datapath):
            raise FileNotFoundError(f"Cell data file not found: {cells_datapath}")

        self.cells_datapath = cells_datapath
        self.cells_positionsKey = cells_positionsKey or 'CellXYZ'
        self.cells_dholderkey = cells_dholderkey
        self.cells_positions_path = cells_positions_path
        self.eliminateNeurons=eliminateNeurons
        self.eliminateNeuronsKey = 'IX_inval_anat' if eliminateNeurons else None
        self.simplify_cells = simplify_cells


    def __load_cell_positions__(self, simplify_cells: bool = True) -> np.ndarray:

        try: 
            tmpdata = loadmat(self.cells_positions_path, simplify_cells=self.simplify_cells) 
        except Exception as e:
            raise IOError(f"Failed to laoad positions file {self.cells_positions_path} : {e}")
        
        if self.cells_positionsKey not in tmpdata:
            raise KeyError(f"Key '{self.cells_positionsKey}' not found in positions file") 
        
        if self.eliminateNeuronsKey and self.eliminateNeuronsKey not in tmpdata[self.cells_positionsKey]:
            raise KeyError(f"Key '{self.eliminateNeuronsKey}' not found in position data")

        roi_positions_to_discard = tmpdata[self.cells_positionsKey][self.eliminateNeuronsKey]
        all_roi_positions = tmpdata[self.cells_positions]

        if self.eliminateNeurons:
            mask = np.ones(len(all_roi_positions), dtype=bool)
            mask[roi_positions_to_discard] = False
            used_rois_positions = all_roi_positions[mask]
        else:
            used_rois_positions = all_roi_positions

        return used_rois_positions


    def __load_number_of_cell_responses__(self, number_of_cells: Optional[int] = None ) -> np.ndarray:
        # returns all valid cell responses (excluding invalid cells)
        with h5py.File(self.cells_datapath, 'r') as tmpdata:
            responses = tmpdata[self.cells_dholderkey][:number_of_cells]
        return responses


    def __get_number_of_cell_positions__(self, number_of_cells: int) -> np.ndarray:
        """Load 3D cell positions from MATLAB file"""
        logger.info(f"Loading cell positions for {self.number_of_cells} cells")

        positions = self.__load_cell_positions__()[:number_of_cells, :].T

        logger.info(f"Loaded cell positions for {self.number_of_cells} cells")
        return positions


    def __get_number_of_cell_responses_and_positions__(self, number_of_cells: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        responses = self.__load_number_of_cell_responses__(number_of_cells)
        positions = self.__get_number_of_cell_positions__(number_of_cells)
        return responses, positions 
  



if __name__ == "__main__":
   main() 
