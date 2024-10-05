conda create -n cuda11.8-env python=3.8 cudatoolkit=11.8
conda activate cuda11.8-env
pip install stable-baselines3[extra] gymnasium[box2d]
<!-- sudo apt-get install swig
pip install box2d-py -->
<!-- pip install gymnasium[box2d] -->
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


python LunarLander_train.py -timesteps 500000
python LunarLander_play.py

python CarRacing_train.py -timesteps 500000
python CarRacing_play.py

Q: Import "stable_baselines3.common.evaluation" could not be resolved
vs code cannot detect stablebaselines and gymnasium

A: Set Python Interpreter in VS Code
To make sure VS Code is using the correct environment, you need to set the right Python interpreter:

Open the Command Palette in VS Code by pressing Ctrl + Shift + P (or Cmd + Shift + P on macOS).

Type Python: Select Interpreter and select it.

From the list, choose the Python interpreter corresponding to your Conda environment (it should show the environment name, e.g., your_env_name).

If you don’t see your environment, it’s possible that your environment isn’t listed, in which case you can manually add it to the workspace settings.