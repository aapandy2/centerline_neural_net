import scipy.io
import numpy as np
import pylab as pl
from PIL import Image, ImageDraw
import subprocess

CHOOSE_DATASET = 2

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

try:
	flagged_neurons = heatData['flagged_neurons']
	flagged_volumes = heatData['flagged_volumes']
	print 'flagged neurons = ', flagged_neurons
	print 'flagged volumes = ', flagged_volumes
except:
	pass

if(CHOOSE_DATASET == 3):
	#special to 3rd dataset; mask outliers (x and y positions spontaneously jump by 1e7 at a couple timesteps)
	x_pos = np.ma.masked_where(x_pos > 0, x_pos)
	y_pos = np.ma.masked_where(y_pos > 0, y_pos)
	velocity = np.ma.masked_where(np.abs(velocity) > 1e-3, velocity)

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

#make_ratio2_movie() #WARNING: this will take like an hour
