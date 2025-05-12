##### Load modules on SLURM cluster
ml Open3D/0.17.0-foss-2022a-CUDA-11.7.0
ml PyTorch3D/0.7.1-foss-2022a-CUDA-11.7.0

##### Create a virtual environment
python3 -m venv unsup3D
source unsup3D/bin/activate

##### Install dependencies
pip install torch_scatter
pip install hdbscan
pip install open3d typing_extensions requests pyarrow
pip install pyaml tqdm tensorboard nuscenes-devkit pandas transforms3d
pip install -e WaffleIron/
pip install git+https://github.com/valeoai/Alpine
