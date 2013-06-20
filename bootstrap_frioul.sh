# installing requirements on FRIOUL 

## fetching neurodebian package list

# 
pip install --user -U pynn
pip install --user -U NeuroTools

#NEST
wget http://www.nest-initiative.org/download/gplreleases/nest-2.2.2.tar.gz
tar zxvf nest-2.2.2.tar.gz
cd nest-2.2.2
./configure --prefix=$HOME/opt/nest  --without-readline
make
make install
cd ..

# add this to your ~/.bash_personal file:
#
# export PATH=$HOME/.local/bin:$PATH
# # NEST
# export PATH=$HOME/opt/nest/bin:$PATH
# export PYTHONPATH=$HOME/opt/nest/lib/python2.6/site-packages:$PYTHONPATH
# 

