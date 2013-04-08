# installing requirements on an empty machine

## fetching neurodebian package list
wget -O- http://neuro.debian.net/lists/quantal.de-md | sudo tee /etc/apt/sources.list.d/neurodebian.sources.list
sudo apt-key adv --recv-keys --keyserver pgp.mit.edu 2649A5A9

sudo aptitude update
sudo aptitude upgrade

# 
sudo aptitude install python-pynn
sudo aptitude install python-matplotlib
sudo aptitude install python-pip
sudo pip install NeuroTools
sudo aptitude install python-scipy
sudo aptitude install ffmpeg 

#PyNN+NeuroTools
   sudo aptitude install python-pynn
   sudo aptitude install python-pip
   sudo aptitude install python-imaging
   sudo pip install NeuroTools
#NEST dependencies
#   sudo aptitude install gsl-bin libgsl0-dev
#   sudo aptitude install python-dev
#   sudo aptitude install libncurses-dev
#   sudo aptitude install libreadline-dev
#NEST
#   wget http://www.nest-initiative.org/download/yHTCUpeiTsCr/nest-2.0.0.tar.gz
#   tar zxvf nest-2.0.0.tar.gz
#   cd nest-2.0.0
#   ./configure
#   make
#   sudo make install
#   cd ..

# BRIAN
# sudo aptitude install python-brian python-brian-lib

