# KSRV

## KSRV: A Kernel PCA-Based Framework for Inferring Spatial RNA Velocity at Single-Cell Resolution

### Implementation description

Python implementation can be found in the 'KSRV' folder. The ```Notebooks``` folder contains scripts showing how to reproduce the results when testing KSRV on the developing chicken heart, developing mouse brain, and mouse organogenesis datasets.

For full description, please check the ```KSRV``` function description in ```main.py```.

### Download the Repository

1. Clone the repository from GitHub:

   ```bash
   git clone https://github.com/YanYan116/KSRV.git
   cd KSRV
   ```

2. Create a virtual environment using Anaconda:

   ```bash
   conda create -n KSRV python=3.11.7
   conda activate KSRV
   ```

3. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Datasets

```Chicken_heart```, ```Mouse_brain``` and ```Mouse_organogenesis``` dataset can be downloaded as ```scanpy Anndata .h5``` files from [Zenodo](https://doi.org/10.5281/zenodo.6798659). ```U-2 OS``` can be downloaded from `https://www.pnas.org/doi/suppl/10.1073/pnas.1912459116`.