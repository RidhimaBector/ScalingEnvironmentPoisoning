# ScalingEnvironmentPoisoning
This repository aims to scale environment poisoning attacks to larger discrete as well as continuous victim environments.

Installation Guide:
1. Install a python IDE and package manager
    1. Download Anaconda from https://www.anaconda.com/download/success
    2. Install Anaconda by running the download .exe file.

2. Create virtual environment 
    1. Open Anaconda Prompt
    2. Run command “conda create -n envPois python=3.10” to create environment with name “envPois”.
    3. Run command “conda activate envPois” to activate the environment

3. Install required packages in the virtual environment
    1. conda install spyder
    2. conda install -c pytorch pytorch
    3. conda install -c conda-forge numpy
    4. conda install -c conda-forge gym
    5. conda install -c conda-forge pot
    6. conda install -c conda-forge tensorboardx
    7. conda install -c conda-forge yacs

4. Open spyder within envPois environment
    1. Open Anaconda Prompt
    2. Run command “conda activate envPois” to activate the environment
    3. Run command “spyder”
    4. If spyder complains about paramiko and pyls-black, run following commands in Anaconda Prompt.
        1. conda install paramiko
        2. conda install conda-forge::pyls-black
    5. Spyder still might complain about paramiko (it’s a bug). Run “conda list” in Anaconda Prompt to make sure you have paramiko and then ignore the spyder warning.

5. Run main.py to train the attacker. If you get GPU memory error (i.e. GPU has very less memory), modify the following line of code in files main.py, ./ae/ae.py and ./attack/DDPG.py to train the attacker without GPU.
    * From: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    * To: device = “cpu” # torch.device("cuda" if torch.cuda.is_available() else "CPU")
  
