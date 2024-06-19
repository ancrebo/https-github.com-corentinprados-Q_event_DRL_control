#|/bin/env python
#
# Create initial condition for ChannelFlow case
from __future__ import print_function, division

import sys, random, numpy as np
import pyAlya

## Flow parameters
CASENAME = sys.argv[1]
UINF     = float(sys.argv[2])
AOA      = float(sys.argv[3])
print('UINF = ',UINF)
print('AOA  = ',AOA)


## Initial field for VELOC
# assume turbulent parabolic profile
coordfile = pyAlya.io.MPIO_AUXFILE_S_FMT % (CASENAME,'COORD')
header    = pyAlya.io.AlyaMPIO_header.read(coordfile)
# Read the node coordinates in serial
xyz,_ = pyAlya.io.AlyaMPIO_readByChunk_serial(coordfile,header.npoints,0)
# Generate the velocity array
vel = np.zeros_like(xyz) # Same dimensions as xyz
vel[:,0] = UINF*np.cos(np.deg2rad(AOA))
vel[:,1] = UINF*np.sin(np.deg2rad(AOA))


## Store velocity as initial condition
outname = pyAlya.io.MPIO_XFLFILE_S_FMT % (CASENAME,1,1) 
h       = pyAlya.io.AlyaMPIO_header(
	fieldname   = 'XFIEL',
	dimension   = 'VECTO',
	association = 'NPOIN',
	dtype       = 'REAL',
	size        = '8BYTE',
	npoints     = header.npoints,
	nsub        = header.nsubd,
	sequence    = header.header['Sequence'],
	ndims       = xyz.shape[1],
	itime       = 0,
	time        = 0.,
	tag1        = 1,
	tag2        = 1,
	ignore_err  = True
)
pyAlya.io.AlyaMPIO_writeByChunk_serial(outname,vel,h,h.npoints,0)

pyAlya.cr_info()
