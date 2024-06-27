###################################################################
# Copyright 2005 - 2021 Barcelona Supercomputing Center.          #
# Distributed under the ALYA AVAILABLE SOURCE ("ALYA AS") LICENSE #
# for nonprofit scientific purposes only.                         #
# See companion file LICENSE.txt.                                 #
###################################################################



import sys
meshName=sys.argv[1]
direction=sys.argv[2]
minmax=sys.argv[3]
coordName='{}.coord'.format(meshName)

f=open(coordName,'r')

boundMax=[float('-Inf'),float('-Inf'),float('-Inf')]
boundMin=[float('Inf' ),float('Inf' ),float('Inf' )]


for line in f:
	data = line.split()
	for i in range(len(data)-1):
		boundMax[i] = max(boundMax[i],float(data[i+1])) 
		boundMin[i] = min(boundMin[i],float(data[i+1])) 

if boundMin[2] == float('Inf' ):
	boundMin[2] = 0.0
if boundMax[2] == float('-Inf'):
	boundMax[2] = 0.0


if minmax == 'min':
	res = boundMin
else:
	res = boundMax

if direction == 'x':
	print('{}'.format(res[0]))

if direction == 'y':
	print('{}'.format(res[1]))

if direction == 'z':
	print('{}'.format(res[2]))

f.close()

