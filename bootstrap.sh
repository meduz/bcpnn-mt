# installing requirements on an empty machine
#sudo aptitude update
#sudo aptitude upgrade
#sudo aptitude install python-pynn
#sudo aptitude install python-matplotlib
#sudo aptitude install python-pip
#sudo pip install NeuroTools
#sudo aptitude install python-scipy

#   sudo aptitude install ffmpeg 

#PyNN+NeuroTools
#   sudo aptitude install python-pynn
#   sudo aptitude install python-pip
#   sudo aptitude install python-imaging
#   sudo pip install NeuroTools
#NEST dependencies
#   sudo aptitude install gsl-bin libgsl0-dev
#   sudo aptitude install python-dev
#   sudo aptitude install libncurses-dev
#   sudo aptitude install libreadline-dev
# sudo apt-get install build-essential autoconf automake libtool libltdl7-dev \
#				libreadline5-dev libncurses5-dev libgsl0-dev python-all-dev python-numpy python-scipy \
#				python-matplotlib ipython 
#NEST
   wget http://www.nest-initiative.org/download/gplreleases/nest-2.2.1.tar.gz
   tar zxvf nest-2.2.1.tar.gz
   cd nest-2.2.1
   ./configure --prefix=$HOME/opt/nest  --without-readline
   make
   sudo make install
   cd ..

# BRIAN
# sudo aptitude install python-brian python-brian-lib

