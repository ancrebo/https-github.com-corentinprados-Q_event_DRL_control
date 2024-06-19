###################################################################
# Copyright 2005 - 2021 Barcelona Supercomputing Center.          #
# Distributed under the ALYA AVAILABLE SOURCE ("ALYA AS") LICENSE #
# for nonprofit scientific purposes only.                         #
# See companion file LICENSE.txt.                                 #
###################################################################



import sys
import math
import numpy as np


meshName=sys.argv[1] #hay que adjuntarle el nombre de archivo


coordfile       = '{}.coord'.format(meshName) # ya existe se crea con el getCoordinates.py 
velofile        = 'VELOC.alya' #crear archivo nuevo de velocidades (velo.dat es un dummie)
boundfile				= '{}.fix.dat'.format(meshName) #archivo con las boundaries
#tempfile        = 'TEMPE.alya'
#enthfile        = 'ENTHA.alya'
#concePatt       = 'CON{:02d}.alya'

fCoord          =open(coordfile,'r') # abrir en modo read
fBound          =open(boundfile, 'r') # abrimos las coordenadas de boundaries cond
fVel            =open(velofile,'w') # abrir en modo write
#fTemp           =open(tempfile,'w')

c = 0
pi = np.pi #libreria de numPy es para science
tol = 0.5

print('---| Start writing initial condition')
for line in fCoord:  # read file line by line
    data=line.split() # lo va separando por los espacios 

    pid = int(data[0]) #number ID 
    dims = len(data)-1 #dimensiones de la malla  
    x   = float(data[1]) #primer numero
    y   = float(data[2]) #segundo numero
	#    z   = float(data[3])

    vx = 0.0
    vy = 0.0
    #vz = 0.0
    #T = np.sin(2*pi*x/1)

    #if x<= c+tol and y<=1.7 and y>=0.3:
    #	vx = 1

    fVel.write('{} {} {}\n'.format(pid,vx,vy))
    #fTemp.write('{} {}\n'.format(pid,T))
        
fCoord.close()
fVel.close()
#fTemp.close()

print('---| End writing initial condition')


