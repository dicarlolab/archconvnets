import os
import numpy as np
from os import listdir
from os.path import isfile, join
import PIL
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
import scipy.io as sio
from scipy.io import loadmat
from scipy.stats.mstats import zscore
import scipy.io as sio
import numpy as np
import tabular as tb
import copy
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import random
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import random
from scipy.stats import rankdata
import random
import pickle as pk
import time
import matplotlib.image as mpimg
import Image

cats = ['ApplyEyeMakeup','ApplyLipstick','Archery','BabyCrawling','BalanceBeam','BandMarching','BaseballPitch','Basketball','BasketballDunk','BenchPress','Biking','Billiards','BlowDryHair','BlowingCandles','BodyWeightSquats','Bowling','BoxingPunchingBag','BoxingSpeedBag','BreastStroke','BrushingTeeth','CleanAndJerk','CliffDiving','CricketBowling','CricketShot','CuttingInKitchen','Diving','Drumming','Fencing','FieldHockeyPenalty','FloorGymnastics','FrisbeeCatch','FrontCrawl','GolfSwing','Haircut','Hammering','HammerThrow','HandstandPushups','HandstandWalking','HeadMassage','HighJump','HorseRace','HorseRiding','HulaHoop','IceDancing','JavelinThrow','JugglingBalls','JumpingJack','JumpRope','Kayaking','Knitting','LongJump','Lunges','MilitaryParade','Mixing','MoppingFloor','Nunchucks','ParallelBars','PizzaTossing','PlayingCello','PlayingDaf','PlayingDhol','PlayingFlute','PlayingGuitar','PlayingPiano','PlayingSitar','PlayingTabla','PlayingViolin','PoleVault','PommelHorse','PullUps','Punch','PushUps','Rafting','RockClimbingIndoor','RopeClimbing','Rowing','SalsaSpin','ShavingBeard','Shotput','SkateBoarding','Skiing','Skijet','SkyDiving','SoccerJuggling','SoccerPenalty','StillRings','SumoWrestling','Surfing','Swing','TableTennisShot','TaiChi','TennisSwing','ThrowDiscus','TrampolineJumping','Typing','UnevenBars','VolleyballSpiking','WalkingWithDog','WallPushups','WritingOnBoard','YoYo']

p = '/export/storage/UCF-101/'
n_per_cat = 2

n_files = len(cats)*n_per_cat
n_frames = 130

imgs = np.zeros((3*32*32, n_files*n_frames),dtype='uint8')
labels = np.zeros(n_files*n_frames, dtype='int')

file_ind = 0
global_frame = 0
for cat in cats:
	print '----------------', cat
	
	for m in range(n_per_cat):
		os.system('rm /tmp/image-*.png -f')
		os.system('ffmpeg -i ' + p + cat + '/' + 'v_' + cat + '_g0' + str(m+1) + '_c01.avi' + ' -r 30 -f image2 /tmp/image-%3d.png')
		
		for frame in range(1,n_frames+1):
			try:
				frame_string = '%03i' % frame
				imgs[:,global_frame] = np.asarray(PIL.Image.open('/tmp/image-' + frame_string + '.png').resize((32,32))).transpose((2,0,1)).ravel()
				labels[global_frame] = file_ind
				global_frame += 1
			except:
				break
		
		file_ind += 1

imgs = imgs[:,:global_frame]
labels = labels[:global_frame]
file = open('/home/darren/archconvnets/archconvnets/unsupervised/movies/pre_segmented_nat/movies_batch','w')
pk.dump({'data':imgs, 'mean': imgs.mean(1)[:,np.newaxis], 'labels': labels}, file)
file.close()


