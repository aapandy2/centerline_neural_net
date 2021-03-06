import subprocess
import os
from PIL import Image, ImageDraw
import numpy

new_folder_number = '4'

#LowMag_directory = '/tigress/LEIFER/PanNeuronal/20180510/BrainScanner20180510_105546/LowMagBrain20180510_105558/'
#LowMag_directory = '/tigress/LEIFER/PanNeuronal/20180511/BrainScanner20180511_134913/LowMagBrain20180511_134913/'
#LowMag_directory = '/tigress/LEIFER/PanNeuronal/20180518/BrainScanner20180518_093125/LowMagBrain20180518_093125/'
#LowMag_directory = '/tigress/LEIFER/PanNeuronal/20180517/BrainScanner20180517_152936/LowMagBrain20180517_152936/'
#LowMag_directory = '/tigress/LEIFER/PanNeuronal/20180523/BrainScanner20180523_141946/LowMagBrain20180523_141946/'
LowMag_directory = '/tigress/LEIFER/PanNeuronal/20180530/BrainScanner20180530_145043/LowMagBrain20180530_144827/'

scratch_directory = '/scratch/gpfs/apandya/'

#from https://stackoverflow.com/questions/3844430/how-to-get-the-duration-of-a-video-in-python
def getLength(filename):
  result = subprocess.Popen(["ffprobe", filename],
    stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  return [x for x in result.stdout.readlines() if "Duration" in x]

ffprobe_out_str = str(getLength(LowMag_directory + 'cam1.avi'))
duration_loc = ffprobe_out_str.find('Duration: ')
cam1_length_str = ffprobe_out_str[10 + duration_loc:10 + duration_loc + 8]

bashCommand1 = 'mkdir cam1_frames_unshrunk'
#bashCommand2 = 'ffmpeg -i ../cam1.avi -start_number 0 -vf fps=30 -ss 00:00:00 -to ' + cam1_length_str + ' cam1_frames_%05d.png'
bashCommand2 = 'ffmpeg -i ' + LowMag_directory + 'cam1.avi -start_number 0 -vf fps=30 -ss 00:00:00 -to ' + cam1_length_str + ' cam1_frames_%05d.png'
bashCommand3 = 'mkdir cam1_frames_resized' + new_folder_number
bashCommand4 = 'rm -r cam1_frames_unshrunk'

def count_files(dir):
    return len([1 for x in list(os.listdir(dir))])

#process1 = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE, cwd=LowMag_directory)
process1 = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE, cwd=scratch_directory)
process1.wait()
#process2 = subprocess.Popen(bashCommand2.split(), stdout=subprocess.PIPE, cwd=(LowMag_directory + 'cam1_frames/'))
process2 = subprocess.Popen(bashCommand2.split(), stdout=subprocess.PIPE, cwd=(scratch_directory + 'cam1_frames_unshrunk/'))
process2.wait()
#process3 = subprocess.Popen(bashCommand3.split(), stdout=subprocess.PIPE, cwd=LowMag_directory)
process3 = subprocess.Popen(bashCommand3.split(), stdout=subprocess.PIPE, cwd=scratch_directory)
process3.wait()

#num_images = count_files(LowMag_directory + 'cam1_frames/')
num_images = count_files(scratch_directory + 'cam1_frames_unshrunk/')

print 'number of images:', num_images

for i in range(num_images):
        if(i % 500 == 0):
                print 'shrinking frame number: ', i
        frame_number = i
#        frames_directory = LowMag_directory + 'cam1_frames/'
	frames_directory = scratch_directory + 'cam1_frames_unshrunk/'
        frame = frames_directory + 'cam1_frames_' + '%05d'%frame_number + '.png'

        img = Image.open(frame)
        img_original_dim = 1088 #image starts 1088 x 1088 pixels
        img_scaled_dim  = 300 #scale image to 300 x 300 pixels
        img_scaling_factor = float(img_scaled_dim)/float(img_original_dim)

        img_resized = img.resize((img_scaled_dim, img_scaled_dim))
#        img_resized.save(LowMag_directory + 'cam1_frames_resized/' + 'cam1_frames_' + '%05d'%frame_number + '_scaled.png', "PNG")
	img_resized.save(scratch_directory + 'cam1_frames_resized' + new_folder_number + '/' + 'cam1_frames_' + '%05d'%frame_number + '_scaled.png', "PNG")

#process4 = subprocess.Popen(bashCommand4.split(), stdout=subprocess.PIPE, cwd=LowMag_directory)
process4 = subprocess.Popen(bashCommand4.split(), stdout=subprocess.PIPE, cwd=scratch_directory)
process4.wait()

print 'frames extracted and resized to 300x300'
