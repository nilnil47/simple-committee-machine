curl -LO https://repo.anaconda.com/archive/Anaconda3-2025.12-1-Linux-x86_64.sh && \
bash Anaconda3-2025.12-1-Linux-x86_64.sh -b -p $HOME/anaconda3 && \
source $HOME/anaconda3/bin/activate && \
conda init bash && \
source ~/.bashrc
