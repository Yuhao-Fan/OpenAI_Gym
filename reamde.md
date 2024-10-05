conda create -n cuda11.8-env python=3.8 cudatoolkit=11.8
conda activate cuda11.8-env
pip install stable-baselines3[extra] gymnasium[box2d]
<!-- sudo apt-get install swig
pip install box2d-py -->
<!-- pip install gymnasium[box2d] -->
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


python LunarLander_train.py
python LunarLander_play.py