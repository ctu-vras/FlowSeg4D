##### Load modules on SLURM cluster
ml PyTorch3D/0.7.1-foss-2021b-CUDA-11.4.1
ml torchvision/0.13.0-foss-2021b-CUDA-11.4.1

##### Create a virtual environment
python3 -m venv unsup3D
source unsup3D/bin/activate

##### Install dependencies
pip install torch_scatter
pip install pyaml tqdm tensorboard nuscenes-devkit pandas transforms3d
pip install -e WaffleIron/



##### Download models and install ScaLR
cd ScaLR

wget https://github.com/valeoai/ScaLR/releases/download/v0.1.0/info_datasets.tar.gz
tar -xvzf info_datasets.tar.gz
rm info_datasets.tar.gz

wget https://github.com/valeoai/ScaLR/releases/download/v0.1.0/WI_768-DINOv2_ViT_L_14-NS_KI_PD-linear_probing-nuscenes.tar.gz
tar -xvzf WI_768-DINOv2_ViT_L_14-NS_KI_PD-linear_probing-nuscenes.tar.gz
rm WI_768-DINOv2_ViT_L_14-NS_KI_PD-linear_probing-nuscenes.tar.gz

wget https://github.com/valeoai/ScaLR/releases/download/v0.1.0/WI_768-DINOv2_ViT_L_14-NS_KI_PD-pretrained.tar.gz
tar -xvzf WI_768-DINOv2_ViT_L_14-NS_KI_PD-pretrained.tar.gz
rm WI_768-DINOv2_ViT_L_14-NS_KI_PD-pretrained.tar.gz

cd ../
