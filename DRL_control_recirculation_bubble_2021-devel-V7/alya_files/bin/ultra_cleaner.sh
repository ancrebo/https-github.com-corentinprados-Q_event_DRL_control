#!/bin/bash
#
# Cleaner for the files of the DRL
#
# Pol Suarez, Arnau Miro

# Delete automatically generated files
rm -rf saved_models best_model saver_data actions rewards probes_comms final_rewards
rm -rf alya_files/case alya_files/environment* alya_files/deterministic py_toplot/*.png

# Clean csv
rm -f *.csv

# Clean parameters
rm -f parameters.py

# Clean logs
# rm -f alya_files/case/logs/*.log

# Clean alya case
# cd alya_files/case/ && ./alya-clean

# Clean hostlist
#rm -f nodelist

# Delete python caches
rm -rf *pycache*
