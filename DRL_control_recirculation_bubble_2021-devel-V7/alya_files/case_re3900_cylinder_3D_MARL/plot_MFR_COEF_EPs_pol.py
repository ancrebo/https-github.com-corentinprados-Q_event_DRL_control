#!/usr/bFopeFoin/python
import os
import numpy as np
import math
import matplotlib.pyplot as plt

### Read input data from file
def _readInputdata():
	bound = []
	bound_mass = []
	variabound = []
	porous = False

	d = 0
	D_exp = 0
	S_exp = 0
	L_exp = 0
	R_exp = 0
	P_exp = 0
	Y_exp = 0
	u = [1,1,1]
	phi = 0
	sin_phi = 0
	cos_phi = 0

	input_folder = os.getcwd()
	input_file = input_folder + '/input_data.dat'

	f = open(input_file,'r')

	lines = f.readline()
	while lines:
		line = lines.split()
		if len(line) != 0:
			if line[0] == 'filename':
				filename = str(line[2])
			if line[0] == 'bound':
				bound = []
				line = line[2].split(',')
				for i in range(len(line)):
					bound.append(int(line[i]))
			if line[0] == 'bound_mass':
				bound_mass = []
				line = line[2].split(',')
				for i in range(len(line)):
					bound_mass.append(int(line[i]))
			if line[0] == 'variabound':
				variabound = []
				line = line[2].split(',')
				for i in range(len(line)):
					variabound.append(int(line[i]))
			if line[0] == 'porous':
				if (line[2]) == 'True':
					porous = True
			if line[0] == 'density':
				density = float(line[2])
			if line[0] == 'veloc':
				veloc = float(line[2])
			if line[0] == 'scale_area':
				scale_area = float(line[2])
			if line[0] == 'd':
				d = float(line[2])
			if line[0] == 'time':
				initial_time = 0
				time = float(line[2])
				if time > 0:
					initial_time = float(input('Initial time to postprocess:'));
					superTitle = 'Averaged from ' + str(abs(initial_time)) + ' to ' + str(abs(initial_time + time))
				elif time < 0:
					superTitle = 'Averaged over the last ' + str(abs(time))
				elif time == 0:
					superTitle = 'Averaged over all the runtime'
			if line[0] == 'rotation_vector':
				rotation_vector = []
				line = line[2].split(',')
				for i in range(len(line)):
					rotation_vector.append(int(line[i]))
				vecro = np.asarray(rotation_vector)
				npvro = vecro.astype(np.float)
				u = npvro/(math.sqrt(((npvro[0])**2)+((npvro[1])**2)+((npvro[2])**2)))
			if line[0] == 'phi':
				phi = float(line[2])
				sin_phi = math.sin(math.radians(phi))
				cos_phi = math.cos(math.radians(phi))
			if line[0] == 'D_exp':
				D_exp = float(line[2])
			if line[0] == 'S_exp':
				S_exp = float(line[2])
			if line[0] == 'L_exp':
				L_exp = float(line[2])
			if line[0] == 'R_exp':
				R_exp = float(line[2])
			if line[0] == 'P_exp':
				P_exp = float(line[2])
			if line[0] == 'Y_exp':
				Y_exp = float(line[2])
		lines = f.readline()
	return (filename, bound, bound_mass, variabound, porous, density, veloc, scale_area, d, initial_time, time, superTitle, u, phi, sin_phi, cos_phi, D_exp, S_exp, L_exp, R_exp, P_exp, Y_exp)

# Read header
def _readHeader(entrada):
	F_visc_columns = []
	F_pres_columns = []
	F_vari_columns = []
	F_poro_columns = []
	M_visc_columns = []
	M_pres_columns = []
	Mass_columns   = []

	for i in range(3):
		line = entrada.readline().strip().split()

	header = 0
	while line[1] != 'START':
		line = entrada.readline().strip().split()
		if line[1] == 'FORCE':
			F_visc_columns.append(int(line[5])-1)
			F_visc_columns.append(int(line[5]))
			F_visc_columns.append(int(line[5])+1)
		if line[1] == 'F_p_x':
			F_pres_columns.append(int(line[5])-1)
			F_pres_columns.append(int(line[5]))
			F_pres_columns.append(int(line[5])+1)
		if line[1] == 'INTFX':
			F_vari_columns.append(int(line[5])-1)
			F_vari_columns.append(int(line[5]))
			F_vari_columns.append(int(line[5])+1)
		if line[1] == 'FPORX':
			F_poro_columns.append(int(line[5])-1)
			F_poro_columns.append(int(line[5]))
			F_poro_columns.append(int(line[5])+1)
		if line[1] == 'TORQU':
			M_visc_columns.append(int(line[5])-1)
			M_visc_columns.append(int(line[5]))
			M_visc_columns.append(int(line[5])+1)
		if line[1] == 'T_p_x':
			M_pres_columns.append(int(line[5])-1)
			M_pres_columns.append(int(line[5]))
			M_pres_columns.append(int(line[5])+1)
		if line[1] == 'MASS':
			Mass_columns.append(int(line[5])-1)
		if line[1] == 'NUMSETS':
			totalBound = int(line[3])
		header+=1
	return (F_visc_columns, F_pres_columns, F_vari_columns, F_poro_columns, M_visc_columns, M_pres_columns, Mass_columns, header, totalBound)

# Read File 
def _readFile(entrada, header):
	lines=entrada.readlines()
	entrada.close()
	nline=len(lines)
	fileLines=[]
	j=0
	for i in range(0,nline):
		line = lines[i]
		line = line.split()
		fileLines.append(line)
	return (fileLines)

# Ṛead unique lines
def _uniqueLines(fileLines):
	steps = []
	index = []
	for i in range (0,len(fileLines)):
		if fileLines[i][1] == 'Iterations':
			if fileLines[i][3] != '0':
				step = int(fileLines[i][3])
				steps.append(step)
				index.append(i+1)
	npStep=np.asarray(steps)
	uniqueStep , uniqueIndex=np.unique(npStep, return_index=True)
	return (uniqueIndex, index)

# Time Arrays
def _timeArrays(uniqueIndex, index):
	time_steps = []
	accumulated_array =[]
	for i in range(len(uniqueIndex)):
		line = index[uniqueIndex[i]]
		time_step = float(fileLines[line][3]) - float(fileLines[line-2-totalBound][3])
		time_steps.append(time_step)
		accumulated_time = float(fileLines[line][3])
		accumulated_array.append(accumulated_time)
	return (time_steps, accumulated_array)

# Rotation fileLines
def _rotationMatrix(u, sin_phi, cos_phi):
	R = np.zeros((3,3))

	R[0,0] = cos_phi + (1 - cos_phi)*(u[0])**2
	R[1,1] = cos_phi + (1 - cos_phi)*(u[1])**2
	R[2,2] = cos_phi + (1 - cos_phi)*(u[2])**2

	R[0,1] = (1 - cos_phi)*u[0]*u[1] - sin_phi*u[2]
	R[1,2] = (1 - cos_phi)*u[1]*u[2] - sin_phi*u[0]
	R[2,0] = (1 - cos_phi)*u[2]*u[0] - sin_phi*u[1]

	R[0,2] = (1 - cos_phi)*u[0]*u[2] + sin_phi*u[1]
	R[1,0] = (1 - cos_phi)*u[1]*u[0] + sin_phi*u[2]
	R[2,1] = (1 - cos_phi)*u[2]*u[1] + sin_phi*u[0]

	return (R)

# Forces types
def _forcesTypes(bound, variabound, porous):
	typfo = 0
	if len(bound) != 0:
		typfo+=2
		# typfo[0] = True
		# typfo[1] = True
	if len(variabound)!= 0:
		typfo+=1
		# typfo[2] = True
	if porous:
		# typfo[3] = True
		typfo+=1
	return (typfo)

# Calcule forces and momentum coefficients
def _calculateFnM(uniqueIndex, fileLines, bound, variabound, F_visc_columns, F_pres_columns, F_vari_columns,
	F_poro_columns, M_visc_columns, M_pres_columns, phi, density):
	array_drag = []
	array_lift = []
	array_side = []

	array_roll  = []
	array_pitch = []
	array_yaw   = []

	for i in range (len(uniqueIndex)):
		line = index[uniqueIndex[i]]
		Forces = []
		Momentum =[]
		if len(bound) != 0:
			ViFor = [0,0,0]
			PrFor = [0,0,0]
			for j in range(len(bound)):
				jb = bound[j]
				ViFor[0] += (float(fileLines[line+jb][F_visc_columns[0]]))
				ViFor[1] += (float(fileLines[line+jb][F_visc_columns[1]]))
				ViFor[2] += (float(fileLines[line+jb][F_visc_columns[2]]))
					
				PrFor[0] += (float(fileLines[line+jb][F_pres_columns[0]]))
				PrFor[1] += (float(fileLines[line+jb][F_pres_columns[1]]))
				PrFor[2] += (float(fileLines[line+jb][F_pres_columns[2]]))
			Forces.append(ViFor)
			Forces.append(PrFor)

		if len(variabound) != 0:
			VaFor = [0,0,0]
			for j in range(len(variabound)):
				jb = variabound[j]
				VaFor[0] += -(float(fileLines[line+jb][F_vari_columns[0]]))
				VaFor[1] += -(float(fileLines[line+jb][F_vari_columns[1]]))
				VaFor[2] += -(float(fileLines[line+jb][F_vari_columns[2]]))
			Forces.append(VaFor)
		
		if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
			if len(variabound) != 0 or len(bound) != 0:
				ViMom = [0,0,0]
				PrMom = [0,0,0]
				M_bound = bound + variabound
				for j in range(len(M_bound)):
					jb = M_bound[j]
					ViMom[0] += (float(fileLines[line+jb][M_visc_columns[0]]))
					ViMom[1] += (float(fileLines[line+jb][M_visc_columns[1]]))
					ViMom[2] += (float(fileLines[line+jb][M_visc_columns[2]]))
						
					PrMom[0] += (float(fileLines[line+jb][M_pres_columns[0]]))
					PrMom[1] += (float(fileLines[line+jb][M_pres_columns[1]]))
					PrMom[2] += (float(fileLines[line+jb][M_pres_columns[2]]))
				Momentum.append(ViMom)
				Momentum.append(PrMom)

		if porous:
			PoFor = [0,0,0]
			PoFor[0] += (float(fileLines[line+jb][F_poro_columns[0]]))
			PoFor[1] += (float(fileLines[line+jb][F_poro_columns[1]]))
			PoFor[2] += (float(fileLines[line+jb][F_poro_columns[2]]))

		FD = 0
		FS = 0
		FL = 0
		RM = 0
		PM = 0
		YM = 0

		for i in range(typfo):
			FD += Forces[i][0]
			FS += Forces[i][1]
			FL += Forces[i][2]

		drag = -FD*2/(float(density)*float(scale_area)*float(veloc)**2)
		side = -FS*2/(float(density)*float(scale_area)*float(veloc)**2)
		lift = -FL*2/(float(density)*float(scale_area)*float(veloc)**2)

		if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
			for i in range(2):
				RM += Momentum[i][0]
				PM += Momentum[i][1]
				YM += Momentum[i][2]

			roll  = -RM*2/(float(density)*float(scale_area)*float(veloc)**2)*d
			pitch = -PM*2/(float(density)*float(scale_area)*float(veloc)**2)*d
			yaw   = -YM*2/(float(density)*float(scale_area)*float(veloc)**2)*d

		if phi != 0:
			sumForces = np.array([drag, lift, side])
			drag, lift, side = R.dot(sumForces)
			if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
				sumMomentum = np.array([roll, pitch, yaw])
				roll, pitch, yaw = R.dot(sumMomentum)

		array_drag.append(drag)
		array_side.append(side)
		array_lift.append(lift)

		if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
			array_roll.append(roll)
			array_pitch.append(pitch)
			array_yaw.append(yaw)
	return(array_drag, array_lift, array_side, array_roll, array_pitch, array_yaw)

# Calcule mass flow rates
def _calculateMFR(uniqueIndex, fileLines, bound_mass, Mass_columns):
	array_mass = []

	for i in range (len(uniqueIndex)):
		line = index[uniqueIndex[i]]
		if len(bound_mass) != 0:
			MFR = []
			for j in range(len(bound_mass)):
				jb = bound_mass[j]
				MFR.append(float(fileLines[line+jb][Mass_columns[0]]))
			array_mass.append(MFR)

	return(array_mass)

# Summation of coefficients depending on the time
def _timeSummation(initial_time, time, time_steps, array_drag, array_lift, array_side, array_roll, array_pitch, array_yaw,
	M_visc_columns, M_pres_columns):
	sum_drag = 0
	sum_side = 0
	sum_lift = 0

	sum_roll  = 0
	sum_pitch = 0
	sum_yaw   = 0

	if time < 0:
		tu = 0
		k = 1
		while tu <= abs(time):
			tu += time_steps[len(time_steps)-k]
			k += 1
		for i in range(len(time_steps)-k, len(time_steps)):
			sum_drag  += (array_drag[i]*time_steps[i])
			sum_side  += (array_side[i]*time_steps[i])
			sum_lift  += (array_lift[i]*time_steps[i])
			if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
				sum_roll  += (array_roll[i]*time_steps[i])
				sum_pitch += (array_pitch[i]*time_steps[i])
				sum_yaw   += (array_yaw[i]*time_steps[i])

		initial_step = len(time_steps)-k
		final_step = len(time_steps)

	elif time == 0:
		for i in range(len(time_steps)):
			sum_drag  += (array_drag[i]*time_steps[i])
			sum_side  += (array_side[i]*time_steps[i])
			sum_lift  += (array_lift[i]*time_steps[i])
			if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
				sum_roll  += (array_roll[i]*time_steps[i])
				sum_pitch += (array_pitch[i]*time_steps[i])
				sum_yaw   += (array_yaw[i]*time_steps[i])
		initial_step = 0
		final_step = len(time_steps)
		tu = accumulated_time

	else:
		tu = 0
		k = 0
		ti = 0 
		j = 0

		while ti < initial_time:
			ti += time_steps[j]
			j += 1

		while tu <= time:
			tu += time_steps[j+k]
			k += 1

		for i in range(j,j+k):
			sum_drag  += (array_drag[i]*time_steps[i])
			sum_side  += (array_side[i]*time_steps[i])
			sum_lift  += (array_lift[i]*time_steps[i])
			if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
				sum_roll  += (array_roll[i]*time_steps[i])
				sum_pitch += (array_pitch[i]*time_steps[i])
				sum_yaw   += (array_yaw[i]*time_steps[i])
		initial_step = j
		final_step = j+k
	return (sum_drag, sum_side, sum_lift, sum_roll, sum_pitch, sum_yaw, tu, initial_step, final_step)

# Average coefficients
def _avNrms(sum_drag, sum_side, sum_lift, sum_roll, sum_pitch, sum_yaw, tu, M_visc_columns, M_pres_columns):
	average_drag = sum_drag/tu
	average_side = sum_side/tu
	average_lift = sum_lift/tu
	if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
		average_roll  = sum_roll/tu
		average_pitch = sum_pitch/tu
		average_yaw   = sum_yaw/tu

	sum_drag_rms = 0
	sum_side_rms = 0
	sum_lift_rms = 0

	if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
		sum_roll_rms  = 0
		sum_pitch_rms = 0
		sum_yaw_rms   = 0

	# RMS coefficients

	for i in range(initial_step, final_step):
		sum_drag_rms  += (array_drag[i]-average_drag)**2
		sum_side_rms  += (array_side[i]-average_side)**2
		sum_lift_rms  += (array_lift[i]-average_lift)**2
		if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
			sum_roll_rms  += (array_roll[i]-average_roll)**2
			sum_pitch_rms += (array_pitch[i]-average_pitch)**2
			sum_yaw_rms   += (array_yaw[i]-average_yaw)**2

	n = final_step - initial_step

	rms_drag = (sum_drag_rms/n)**0.5
	rms_side = (sum_side_rms/n)**0.5
	rms_lift = (sum_lift_rms/n)**0.5
	if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
		rms_roll  = (sum_roll_rms/n)**0.5
		rms_pitch = (sum_pitch_rms/n)**0.5
		rms_yaw   = (sum_yaw_rms/n)**0.5
	if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
		return(average_drag, average_side, average_lift, average_roll, average_pitch, average_yaw,
			rms_drag, rms_side, rms_lift, rms_roll, rms_pitch, rms_yaw)
	else:
		return(average_drag, average_side, average_lift, False, False, False,
			rms_drag, rms_side, rms_lift, False, False, False)

# Print coefficients
def _printCoeff(average_drag, average_side, average_lift, average_roll, average_pitch, average_yaw, rms_drag, rms_side,
	rms_lift, rms_roll, rms_pitch, rms_yaw, M_visc_columns, M_pres_columns):
	print('Average_drag: ' + str(average_drag))
	print('RMS_drag: ' + str(rms_drag))
	print('Average_side: ' + str(average_side))
	print('RMS_side: ' + str(rms_side))
	print('Average_lift: ' + str(average_lift))
	print('RMS_lift: ' + str(rms_lift))
	if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
		print('Average_roll: ' + str(average_roll))
		print('RMS_roll: ' + str(rms_roll))
		print('Average_pitch: ' + str(average_pitch))
		print('RMS_pitch: ' + str(rms_pitch))
		print('Average_yaw: ' + str(average_yaw))
		print('RMS_yaw: ' + str(rms_yaw))

# Arrays for plotting
def _plotArrays(accumulated_array, array_drag, array_side, array_lift, initial_step, final_step, M_visc_columns, M_pres_columns):
	plot_drag  = []
	plot_side  = []
	plot_lift  = []
	plot_roll = []
	plot_pitch = []
	plot_yaw = []
	plot_time  = []

	av_initial_time = accumulated_array[initial_step]
	av_final_time = accumulated_array[final_step-1]

	initial_step = 0
	final_step = len(time_steps)
	j = 0
	for i in range(initial_step, final_step):
		j+=1
		plot_drag.append(array_drag[i])
		plot_side.append(array_side[i])
		plot_lift.append(array_lift[i])
		plot_time.append(accumulated_array[i])
		if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
			plot_roll.append(array_roll[i])
			plot_pitch.append(array_pitch[i])
			plot_yaw.append(array_yaw[i])
	if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
		return (plot_drag, plot_lift, plot_side, plot_time, plot_roll, plot_pitch, plot_yaw, av_initial_time, av_final_time)
	else:
		return (plot_drag, plot_lift, plot_side, plot_time, False, False, False, av_initial_time, av_final_time)

# Plot
def _plots(superTitle, plot_time, av_initial_time, av_final_time, plot_drag, average_drag, rms_drag,
	plot_side, average_side, rms_side, plot_lift, average_lift, rms_lift, plot_roll, average_roll, rms_roll,
	plot_pitch, average_pitch, rms_pitch, plot_yaw, average_yaw, rms_yaw, M_visc_columns, M_pres_columns):
	
	np.savetxt("time_ep.csv", plot_time, fmt="%1.9f", delimiter=";")
	plt.figure(1)

	plt.suptitle(superTitle, fontsize=20)
	np.savetxt("drag_ep.csv", plot_drag,  fmt="%1.9f", delimiter=';')
	plt.plot(plot_time, plot_drag,'c-', label=r"$C_d$" )
	plt.axhline(y=average_drag, xmin=0, xmax=1, linestyle='-.', color = 'k', linewidth=2, label=r"$\overline{C_d}$")
	plt.axhline(y=average_drag + rms_drag, xmin=0, xmax=1, linestyle='--', color = 'k', label=r'$\overline{C_d} \pm \sigma$')
	plt.axhline(y=average_drag - rms_drag, xmin=0, xmax=1, linestyle='--', color = 'k')
	plt.axhline(y=D_exp, xmin=0, xmax=1, linestyle='-', color = 'r', linewidth=2, label=r"$C_d,exp$")
	plt.axvspan(av_initial_time, av_final_time, facecolor='0.5', alpha=0.2)
	plt.legend(loc='lower left', fancybox=True)
	plt.title('Drag', fontsize=15)
	plt.xlabel('Time (s)', fontsize=12)
	plt.ylabel(r"$C_d$", fontsize=12)
	plt.grid(True, which='major', axis='both', linestyle=':')
	plt.ylim([average_drag - 15*rms_drag, average_drag + 15*rms_drag])
	plt.savefig('plot_drag.png')

	plt.figure(2)
	plt.suptitle(superTitle, fontsize=20)

	plt.plot(plot_time, plot_side,'lime', label=r"$C_s$" )
	np.savetxt("side_ep.csv", plot_side,  fmt="%1.9f", delimiter=";")
	plt.axhline(y=average_side, xmin=0, xmax=1, linestyle='-.', color = 'k', linewidth=2, label=r"$\overline{C_s}$")
	plt.axhline(y=average_side + rms_side, xmin=0, xmax=1, linestyle='--', color = 'k', label=r'$\overline{C_s} \pm \sigma$')
	plt.axhline(y=average_side - rms_side, xmin=0, xmax=1, linestyle='--', color = 'k')
	plt.axhline(y=S_exp, xmin=0, xmax=1, linestyle='-', color = 'r', linewidth=2, label=r"$C_s,exp$")
	plt.axvspan(av_initial_time, av_final_time, facecolor='0.5', alpha=0.2)
	plt.legend(loc='lower left', fancybox=True)
	plt.title('Side', fontsize=15)
	plt.xlabel('Time (s)', fontsize=12)
	plt.ylabel(r"$C_s$", fontsize=12)
	plt.grid(True, which='major', axis='both', linestyle=':')
	plt.ylim([average_side - 15*rms_side, average_side + 15*rms_side])
	plt.savefig('plot_side.png')

	plt.figure(3)
	plt.suptitle(superTitle, fontsize=20)

	plt.plot(plot_time, plot_lift,'m-', label=r"$C_l$" )
	np.savetxt("lift_ep.csv", plot_lift,  fmt="%1.9f", delimiter=";")
	plt.axhline(y=average_lift, xmin=0, xmax=1, linestyle='-.', color = 'k', linewidth=2, label=r"$\overline{C_l}$")
	plt.axhline(y=average_lift + rms_lift, xmin=0, xmax=1, linestyle='--', color = 'k', label=r"$\overline{C_l} \pm \sigma$")
	plt.axhline(y=average_lift - rms_lift, xmin=0, xmax=1, linestyle='--', color = 'k')
	plt.axhline(y=L_exp, xmin=0, xmax=1, linestyle='-', color = 'r', linewidth=2, label=r"$C_l,exp$")
	plt.axvspan(av_initial_time, av_final_time, facecolor='0.5', alpha=0.2)
	plt.legend(loc='lower left', fancybox=True)
	plt.title('Lift', fontsize=15)
	plt.xlabel('Time (s)', fontsize=12)
	plt.ylabel(r"$C_l$", fontsize=12)
	plt.grid(True, which='major', axis='both', linestyle=':')
	plt.ylim([average_lift - 15*rms_lift, average_lift + 15*rms_lift])
	plt.savefig('plot_lift.png')

	if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
		plt.figure(4)
		plt.suptitle(superTitle, fontsize=20)

		plt.plot(plot_time, plot_roll,'blue', label=r"$C_rm$" )
		plt.legend(loc='best', fancybox=True)
		plt.axhline(y=average_roll, xmin=0, xmax=1, linestyle='-.', color = 'k', linewidth=2, label=r"$\overline{C_rm}$")
		plt.axhline(y=average_roll + rms_roll, xmin=0, xmax=1, linestyle='--', color = 'k', label=r'$\overline{C_rm} \pm \sigma$')
		plt.axhline(y=average_roll - rms_roll, xmin=0, xmax=1, linestyle='--', color = 'k')
		plt.axhline(y=R_exp, xmin=0, xmax=1, linestyle='-', color = 'r', linewidth=2, label=r"$C_rm,exp$")
		plt.axvspan(av_initial_time, av_final_time, facecolor='0.5', alpha=0.2)
		plt.legend(loc='best', fancybox=True)
		plt.title('Roll', fontsize=15)
		plt.xlabel('Time (s)', fontsize=12)
		plt.ylabel(r"$C_rm$", fontsize=12)
		plt.grid(True, which='major', axis='both', linestyle=':')
		plt.ylim([average_roll - 6*rms_roll, average_roll + 6*rms_roll])
		plt.savefig('plot_roll.png')
		
		plt.figure(5)
		plt.suptitle(superTitle, fontsize=20)

		plt.plot(plot_time, plot_pitch,'green', label=r"$C_pm$" )
		plt.legend(loc='best', fancybox=True)
		plt.axhline(y=average_pitch, xmin=0, xmax=1, linestyle='-.', color = 'k', linewidth=2, label=r"$\overline{C_pm}$")
		plt.axhline(y=average_pitch + rms_pitch, xmin=0, xmax=1, linestyle='--', color = 'k', label=r'$\overline{C_pm} \pm \sigma$')
		plt.axhline(y=average_pitch - rms_pitch, xmin=0, xmax=1, linestyle='--', color = 'k')
		plt.axhline(y=S_exp, xmin=0, xmax=1, linestyle='-', color = 'r', linewidth=2, label=r"$C_pm,exp$")
		plt.axvspan(av_initial_time, av_final_time, facecolor='0.5', alpha=0.2)
		plt.legend(loc='best', fancybox=True)
		plt.title('Pitch', fontsize=15)
		plt.xlabel('Time (s)', fontsize=12)
		plt.ylabel(r"$C_pm$", fontsize=12)
		plt.grid(True, which='major', axis='both', linestyle=':')
		plt.ylim([average_pitch - 6*rms_pitch, average_pitch + 6*rms_pitch])
		plt.savefig('plot_pitch.png')

		plt.figure(6)
		plt.suptitle(superTitle, fontsize=20)

		plt.plot(plot_time, plot_yaw,'purple', label=r"$C_ym$" )
		plt.legend(loc='best', fancybox=True)
		plt.axhline(y=average_yaw, xmin=0, xmax=1, linestyle='-.', color = 'k', linewidth=2, label=r"$\overline{C_ym}$")
		plt.axhline(y=average_yaw + rms_yaw, xmin=0, xmax=1, linestyle='--', color = 'k', label=r'$\overline{C_ym} \pm \sigma$')
		plt.axhline(y=average_yaw - rms_yaw, xmin=0, xmax=1, linestyle='--', color = 'k')
		plt.axhline(y=Y_exp, xmin=0, xmax=1, linestyle='-', color = 'r', linewidth=2, label=r"$C_ym,exp$")
		plt.axvspan(av_initial_time, av_final_time, facecolor='0.5', alpha=0.2)
		plt.legend(loc='best', fancybox=True)
		plt.title('Yaw', fontsize=15)
		plt.xlabel('Time (s)', fontsize=12)
		plt.ylabel(r"$C_ym$", fontsize=12)
		plt.grid(True, which='major', axis='both', linestyle=':')
		plt.ylim([average_yaw - 6*rms_yaw, average_yaw + 6*rms_yaw])
		plt.savefig('plot_yaw.png')

# Plot of the mass flow rates. THIS IS HARCODED FOR ONLY TWO JETS
def _plotMass(accumulated_array, array_mass, plot_time):
	plot_mass_bot = []
	plot_mass_top = []

	initial_step = 0
	final_step = len(time_steps)
	for i in range(initial_step, final_step):
		plot_mass_bot.append(array_mass[i][0])
		plot_mass_top.append(array_mass[i][1])
	np.savetxt("mfr_top_ep.csv", plot_mass_top,  fmt="%1.9f", delimiter=";")
	np.savetxt("mfr_bot_ep.csv", plot_mass_bot,  fmt="%1.9f", delimiter=";")
	plt.figure(4)
	plt.suptitle('Mass flow rate', fontsize=20)
	plt.plot(plot_time, plot_mass_bot,'c-', label=r"$Bottom$" )
	plt.plot(plot_time, plot_mass_top,'r-', label=r"$Top$" )
	plt.legend(loc='lower left', fancybox=True)
	plt.xlabel('Time (s)', fontsize=12)
	plt.ylabel(r"$Q_i$", fontsize=12)
	plt.grid(True, which='major', axis='both', linestyle=':')
	plt.savefig('plot_mfr.png')


#### MAIN ####

filename, bound, bound_mass, variabound, porous, density, veloc, scale_area, d, initial_time, time, superTitle, u, phi, sin_phi, cos_phi, D_exp, S_exp, L_exp, R_exp, P_exp, Y_exp = _readInputdata()

print(filename, bound, bound_mass, variabound, porous, density, veloc, scale_area, d, initial_time, time, superTitle, u, phi, sin_phi, cos_phi, D_exp, S_exp, L_exp, R_exp, P_exp, Y_exp)

file = '-boundary.nsi.set'
entrada = open(filename + file,'r')

F_visc_columns, F_pres_columns, F_vari_columns, F_poro_columns, M_visc_columns, M_pres_columns, Mass_columns, header, totalBound = _readHeader(entrada)

fileLines = _readFile(entrada, header)

uniqueIndex, index = _uniqueLines(fileLines)

time_steps, accumulated_array = _timeArrays(uniqueIndex, index)

R = _rotationMatrix(u, sin_phi, cos_phi)

typfo = _forcesTypes(bound, variabound, porous)

array_drag, array_lift, array_side, array_roll, array_pitch, array_yaw = _calculateFnM(uniqueIndex, fileLines,
	bound, variabound, F_visc_columns, F_pres_columns, F_vari_columns,
	F_poro_columns, M_visc_columns, M_pres_columns, phi, density)

sum_drag, sum_side, sum_lift, sum_roll, sum_pitch, sum_yaw, tu, initial_step, final_step = _timeSummation(initial_time, time,
	time_steps, array_drag, array_lift, array_side, array_roll, array_pitch, array_yaw,M_visc_columns, M_pres_columns)

average_drag, average_side, average_lift, average_roll, average_pitch, average_yaw, rms_drag, rms_side, rms_lift, rms_roll, rms_pitch, rms_yaw = _avNrms(sum_drag, sum_side, sum_lift, sum_roll, sum_pitch, sum_yaw, tu, M_visc_columns, M_pres_columns)

_printCoeff(average_drag, average_side, average_lift, average_roll, average_pitch, average_yaw, rms_drag, rms_side,
	rms_lift, rms_roll, rms_pitch, rms_yaw, M_visc_columns, M_pres_columns)

plot_drag, plot_lift, plot_side, plot_time, plot_roll, plot_pitch, plot_yaw, av_initial_time, av_final_time = _plotArrays(accumulated_array, array_drag, array_side, array_lift, initial_step, final_step, M_visc_columns, M_pres_columns)

_plots(superTitle, plot_time, av_initial_time, av_final_time, plot_drag, average_drag, rms_drag,
	plot_side, average_side, rms_side, plot_lift, average_lift, rms_lift, plot_roll, average_roll, rms_roll,
	plot_pitch, average_pitch, rms_pitch, plot_yaw, average_yaw, rms_yaw, M_visc_columns, M_pres_columns)

if len(bound_mass) != 0:
    array_mass = _calculateMFR(uniqueIndex, fileLines, bound_mass, Mass_columns)

    _plotMass(accumulated_array, array_mass, plot_time)

plt.show()
