conda create -n MedoidFormer python=3.10
conda activate MedoidFormer
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install open3d
pip3 install timm
cd Pointnet2_Pytorch
python setup.py install
cd lib/pointops
python setup.py install
