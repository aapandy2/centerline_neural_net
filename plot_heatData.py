import scipy.io
import numpy as np
import pylab as pl
from PIL import Image, ImageDraw
import subprocess
from scipy.integrate import simps

CHOOSE_DATASET = 3

#load in .mat file as python dictionary
if(CHOOSE_DATASET == 1):
	heatData = scipy.io.loadmat('/tigress/LEIFER/PanNeuronal/20180430/BrainScanner20180430_141614/heatData.mat')
elif(CHOOSE_DATASET == 2):
	heatData = scipy.io.loadmat('/tigress/LEIFER/PanNeuronal/20180502/BrainScanner20180502_111718/heatData.mat')
elif(CHOOSE_DATASET == 3):
	heatData = scipy.io.loadmat('/tigress/LEIFER/PanNeuronal/20180504/BrainScanner20180504_135752/heatData.mat')
else:
	print 'NEED TO CHOOSE CORRECT DATASET INDEX.  CHOOSE 1, 2, or 3'

velocity        = heatData['behavior']['v'][0][0]
ethogram        = heatData['behavior']['ethogram'][0][0]
x_pos           = heatData['behavior']['x_pos'][0][0]
y_pos           = heatData['behavior']['y_pos'][0][0]
pc1_2           = heatData['behavior']['pc1_2'][0][0]
pc3             = heatData['behavior']['pc_3'][0][0]
acorr           = heatData['acorr']
cgIdx           = heatData['cgIdx']
cgIdxRev        = heatData['cgIdxRev']
G2              = heatData['G2']
gPhotoCorr      = heatData['gPhotoCorr']
gRaw            = heatData['gRaw']
hasPointsTime   = heatData['hasPointsTime']
R2              = heatData['R2']
Ratio2          = heatData['Ratio2']
rPhotoCorr      = heatData['rPhotoCorr']
rRaw            = heatData['rRaw']
XYZcoord        = heatData['XYZcoord']
pc1 = pc1_2[:, 0]
pc2 = pc1_2[:, 1]

try:
	flagged_neurons = heatData['flagged_neurons']
	flagged_volumes = heatData['flagged_volumes']
	print 'flagged neurons = ', flagged_neurons
	print 'flagged volumes = ', flagged_volumes
except:
	pass

# Set plot parameters to make beautiful plots
pl.rcParams['figure.figsize']  = 10, 10
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 15
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'large'
pl.rcParams['axes.labelsize']  = 'large'

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad']  = 8
pl.rcParams['xtick.minor.pad']  = 8
pl.rcParams['xtick.color']      = 'k'
pl.rcParams['xtick.labelsize']  = 'large'
pl.rcParams['xtick.direction']  = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad']  = 8
pl.rcParams['ytick.minor.pad']  = 8
pl.rcParams['ytick.color']      = 'k'
pl.rcParams['ytick.labelsize']  = 'large'
pl.rcParams['ytick.direction']  = 'in'

def plot_2D_arrays():
	#in the 3rd dataset the velocity and positions suddenly jump off
	#of range of possible values; mask these erroneous values
	if(CHOOSE_DATASET == 3):
		global x_pos
		global y_pos
		global velocity
		x_pos    = np.ma.masked_where(x_pos > 0, x_pos)
		y_pos    = np.ma.masked_where(y_pos > 0, y_pos)
		velocity = np.ma.masked_where(np.abs(velocity) > 1e-3, velocity)

	pl.plot(velocity)
	pl.title('Velocity')
	pl.show()
	pl.plot(ethogram)
	pl.title('Ethogram')
	pl.show()
	pl.plot(x_pos)
	pl.title('x position')
	pl.show()
	pl.plot(y_pos)
	pl.title('y position')
	pl.show()
	pl.plot(x_pos, y_pos)
	pl.title('Combined x and y position')
	pl.show()
	
	pl.plot(pc1_2[:, 0])
	pl.title('pc1')
	pl.show()
	pl.plot(pc1_2[:, 1])
	pl.title('pc2')
	pl.show()
	pl.plot(pc3)
	pl.title('pc3')
	pl.show()
	pl.plot(hasPointsTime)
	pl.title('hasPointsTime')
	pl.show()
	return 0

def make_ratio2_movie():
	num_neurons   = np.shape(Ratio2)[0]
	num_timesteps = np.shape(Ratio2)[1]

	command0 = subprocess.Popen('mkdir temp_folder/'.split(), stdout=subprocess.PIPE)
	command0.wait()

	for i in range(0, num_timesteps):
		if(i % 50 == 0):
			print i
		pl.bar(np.arange(num_neurons), Ratio2[:, i])
		pl.savefig('temp_folder/img%04d' % i + '.png')
		pl.close()
		pl.clf()

	#make movie
	command1 = subprocess.Popen('ffmpeg -i temp_folder/img%04d.png neural_activity_movie.mp4'.split(), stdout=subprocess.PIPE)
	command1.wait()
	command2 = subprocess.Popen('rm -r temp_folder/'.split(), stdout=subprocess.PIPE)
	command2.wait()

	return 0


def principal_comp_plots():
	pl.plot(hasPointsTime, np.abs(pc1), label='Eigenworm 1')
	pl.plot(hasPointsTime, np.abs(pc2), label='Eigenworm 2')
	pl.plot(hasPointsTime, np.abs(pc3), label='Eigenworm 3')
	pl.xlabel('Time (s)')
	pl.ylabel('$$|\mathrm{Principal~Component}|$$')
	pl.legend(loc='upper left')
	pl.show()
	
	pc1_sq_sum = sum(np.abs(pc1))
	pc2_sq_sum = sum(np.abs(pc2))
	pc3_sq_sum = sum(np.abs(pc3))
	labels = ['Eigenworm 1', 'Eigenworm 2', 'Eigenworm 3']
	pc_sqs = [pc1_sq_sum, pc2_sq_sum, pc3_sq_sum[0]]
	pl.bar(np.arange(len(pc_sqs)), pc_sqs)
	pl.xticks(np.arange(len(pc_sqs)), labels)
	pl.ylabel('Weighted Sum of Principal Component')
	pl.show()

	return 0

def print_total_dist():
	#masking the erroneous velocity values screws up integration, so just
	#zero them out
	if(CHOOSE_DATASET == 3):
		velocity[np.abs(velocity) > 1e-3] = 0.

	total_dist = simps(np.abs(velocity.flatten()), hasPointsTime.flatten())
#	print total_dist
	return total_dist

def plot_total_dist():
	#these are from running print_total_dist() on the 3 datasets.
	total_dist_3 = 0.0251279429168839 
	total_dist_2 = 0.12310450291336447 
	total_dist_1 = 0.1681653523599813 
	total_dist_array = [total_dist_1, total_dist_2, total_dist_3]
	label_array = ['Day 1', 'Day 3', 'Day 5']
	pl.bar(np.arange(len(total_dist_array)), total_dist_array)
	pl.xticks(np.arange(len(total_dist_array)), label_array)
	pl.ylabel('Total Distance Traveled')
	pl.show()
	return 0

def print_mean_sq_velocity():
	#masking the erroneous velocity values screws up integration, so just
        #zero them out
        if(CHOOSE_DATASET == 3):
                velocity[np.abs(velocity) > 1e-3] = 0.

	mean_sq_vel = np.mean(velocity.flatten()**2.)
#	print mean_sq_vel
	return mean_sq_vel

def plot_mean_sq_velocity():
        #these are from running print_mean_sq_velocity() on the 3 datasets.
	mean_sq_vel_1 = 1.356742646203952e-07
	mean_sq_vel_2 = 6.889257795283694e-08 
	mean_sq_vel_3 = 7.875307026609627e-09 
        mean_sq_v_array = [mean_sq_vel_1, mean_sq_vel_2, mean_sq_vel_3]
        label_array = ['Day 1', 'Day 3', 'Day 5']
        pl.bar(np.arange(len(mean_sq_v_array)), mean_sq_v_array)
        pl.xticks(np.arange(len(mean_sq_v_array)), label_array)
        pl.ylabel('$$\\mathrm{Mean~Square~Velocity~} \\langle v^2 \\rangle$$')
        pl.show()
        return 0

#plot_2D_arrays()
#principal_comp_plots()
#print_total_dist()
#plot_total_dist()
#plot_mean_sq_velocity()
