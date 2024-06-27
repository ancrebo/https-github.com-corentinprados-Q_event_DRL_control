##############################
# RUN TO BE READY FOR DRL in DARDEL
##############################

# module load in DARDEL 
ml PDC
ml METIS/5.1.0-cpeCray-21.11
ml CMake/3.22.3

# Activate tensorflow environment
source /cfs/klemming/home/p/polsm/my-tensorflow/bin/activate

# install python packages (specific versions)
python3 -m pip install tensorflow==2.6.0
python3 -m pip install tensorforce==0.6.5
python3 -m pip install keras==2.6.0

