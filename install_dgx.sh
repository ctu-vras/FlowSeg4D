##### Load modules on SLURM cluster
ml Open3D/0.17.0-foss-2022a-CUDA-11.7.0

##### Create a virtual environment
python3 -m venv dgx
source dgx/bin/activate

##### Install dependencies
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch-scatter --no-index -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt1131/download.html
pip install open3d pyarrow hdbscan
pip install pyaml tqdm tensorboard nuscenes-devkit pandas transforms3d
pip install -e WaffleIron/
pip install -e Alpine/
