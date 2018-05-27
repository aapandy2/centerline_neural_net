import subprocess
import os
from PIL import Image, ImageDraw
import numpy

LowMag_directory = '/tigress/LEIFER/PanNeuronal/20180510/BrainScanner20180510_105546/LowMagBrain20180510_105558/'

#from https://stackoverflow.com/questions/3844430/how-to-get-the-duration-of-a-video-in-python
def getLength(filename):
  result = subprocess.Popen(["ffprobe", filename],
    stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  return [x for x in result.stdout.readlines() if "Duration" in x]

ffprobe_out_str = str(getLength(LowMag_directory + 'cam1.avi'))
duration_loc = ffprobe_out_str.find('Duration: ')
cam1_length_str = ffprobe_out_str[10 + duration_loc:10 + duration_loc + 8]

bashCommand1 = 'mkdir cam1_frames'
bashCommand2 = 'ffmpeg -i ../cam1.avi -start_number 0 -vf fps=30 -ss 00:00:00 -to ' + cam1_length_str + ' cam1_frames_%05d.png'
bashCommand3 = 'mkdir cam1_frames_resized'
bashCommand4 = 'rm -r cam1_frames'

def count_files(dir):
    return len([1 for x in list(os.listdir(dir))])

process1 = subprocess.Popen(bashCommand1.split(), stdout=subprocess.PIPE, cwd=LowMag_directory)
process1.wait()
process2 = subprocess.Popen(bashCommand2.split(), stdout=subprocess.PIPE, cwd=(LowMag_directory + 'cam1_frames/'))
process2.wait()
process3 = subprocess.Popen(bashCommand3.split(), stdout=subprocess.PIPE, cwd=LowMag_directory)
process3.wait()

num_images = count_files(LowMag_directory + 'cam1_frames/')

for i in range(num_images):
        if(i % 500 == 0):
                print 'shrinking frame number: ', i
        frame_number = i
        frames_directory = LowMag_directory + 'cam1_frames/'
        frame = frames_directory + 'cam1_frames_' + '%05d'%frame_number + '.png'

        img = Image.open(frame)
        img_original_dim = 1088 #image starts 1088 x 1088 pixels
        img_scaled_dim  = 300 #scale image to 150 x 150 pixels
        img_scaling_factor = float(img_scaled_dim)/float(img_original_dim)

        img_resized = img.resize((img_scaled_dim, img_scaled_dim))
        img_resized.save(LowMag_directory + 'cam1_frames_resized/' + 'cam1_frames_' + '%05d'%frame_number + '_scaled.png', "PNG")

process4 = subprocess.Popen(bashCommand4.split(), stdout=subprocess.PIPE, cwd=LowMag_directory)
process4.wait()

print 'frames extracted and resized to 300x300'
