from archconvnets.unsupervised.DFT import DFT_matrix_2d
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.io import loadmat
from archconvnets.unsupervised.grad_transpose import test_grad_transpose
from archconvnets.unsupervised.grad_fourier_l1 import test_grad_fourier_l1
from archconvnets.unsupervised.grad_channel_corr import test_grad_channel_corr
from archconvnets.unsupervised.grad_second_order import test_grad_second_order
from scipy.io import savemat
from scipy.stats.mstats import zscore
import PIL
import PIL.Image
import time
import numpy as np

def pinv(F):
	return np.dot(F.T, np.linalg.inv(np.dot(F,F.T)))

def test_F(F):
	sparsity = 0
	sse = 0
	batch = 9001
	for batch in range(9001,9001+3):#9011):
		batch_data =  np.load('/storage/batch128_img138_full/data_batch_' + str(batch))['data'].reshape((3,138,138,128)).transpose((3,1,2,0))[:,66:66+7,66:66+7]
		for step in range(128):
		        patch = batch_data[img].ravel()
			patch = zscore(patch)
			Ft = pinv(F)
		        sse += np.sum((patch - np.dot(Ft, np.dot(F,patch)))**2)
			sparsity += np.sum(np.abs(np.dot(F,patch)))
	l, g =  test_grad_transpose(F.T, 3, 7, n_out)
	lf, gf = test_grad_fourier_l1(F.T, 3, 7, n_out, X, X2)
	lc, gc = test_grad_channel_corr(F.T, 3, 7, n_out)
	ls, gs = test_grad_second_order(F.T, 3, 7, n_out, c_mat_input)
	loss = sse + lambds*sparsity + lambdt*l + lambdf*lf + lambdc*lc
	print 'recon:',sse, 'sparsity:',lambdf*sparsity, 'transpose:',lambdt*l, 'fourier:',lambdf*lf, 'loss:',loss, 'transpose: ',np.mean(np.abs(1-pdist(F.T,'correlation'))),  img_t
	print 'channel corr:',lambdc*lc, 'second order:',lambds*ls
	sparsities.append(sparsity)
	sses.append(sse)
	transposes.append(l)
	fouriers.append(lf)
	losses.append(loss)
	channel_corrs.append(lc)
	second_orders.append(ls)

####### fourier
X = np.real(DFT_matrix_2d(7))
X2 = np.imag(DFT_matrix_2d(7))

###### second order
c_mat_input = np.squeeze(loadmat('/home/darren/pixels_zscore.mat')['f'])
######
	
img_t = 0
n_out = 32
n_in = 147
F = zscore(np.random.random((n_out,n_in)),axis=None)

sparsities = []
sses = []
transposes = []
fouriers = []
losses = []
channel_corrs = []
second_orders = []

lambds = 1e-1#e1#e1    # sparsity
lambdt = 2.5e1#5        # transpose
lambdf = 1e-1#8         # fourier
lambdc = 1
lambds = 1
# todo: slowness

eps = 5e-2#3#12
n_imgs_batch = 3#25#128

data_mean = np.load('/storage/batch128_img138_full/batches.meta')['data_mean'].reshape((3,138,138))[:,:,:,np.newaxis]

for batch in range(9000):
	t_start = time.time()
	F = F.reshape((n_out,n_in))
	batch_data =  np.load('/storage/batch128_img138_full/data_batch_' + str(batch))['data'].reshape((3,138,138,128)) - data_mean
	batch_data = batch_data.transpose((3,1,2,0))[:,66:66+7,66:66+7]
	Ft = pinv(F)
	one_m_Ft_F = 1 - np.dot(Ft,F)
	one_m_F_Ft = 1 - np.dot(F,Ft)
	Ft_Ftt = np.dot(Ft, Ft.T)
	grad = np.zeros((n_out,n_in))
	grad_recon = np.zeros((n_out,n_in))
	grad_sparse = np.zeros((n_out,n_in))
	
	################# 
	for img in range(n_imgs_batch):
		img_t += 1

		patch = batch_data[img].ravel()
		patch = zscore(patch)

		F_patch = np.dot(F,patch)
		recon = np.dot(Ft,F_patch)
		recon_m_patch = recon - patch
		for i_val in range(n_out):
			for j_val in range(n_in):
				dAdp = np.zeros((n_out,n_in))
				dAdp[i_val,j_val] = 1

				dApdp = -np.dot(np.dot(Ft, dAdp), Ft) 
				dApdp += np.dot(np.dot(Ft_Ftt, dAdp.T), one_m_F_Ft)  
				dApdp += np.dot(np.dot(np.dot(one_m_Ft_F, dAdp.T), Ft.T), Ft)
				
				d_recon_dFij = np.dot(dApdp, F_patch) + Ft[:,i_val]*patch[j_val]
				grad_temp = np.sum(d_recon_dFij*recon_m_patch)
				if np.isnan(grad_temp) == False:
					grad_recon[i_val,j_val] += grad_temp
					grad_sparse[i_val,j_val] += patch[j_val]*np.sign(patch[j_val]*F[i_val,j_val])
	l, g =  test_grad_transpose(F.T, 3, 7, n_out)
	g = g.reshape((n_in, n_out)).T
	grad_f = lambdt * g
	
	lf, gf = test_grad_fourier_l1(F.T, 3, 7, n_out, X, X2)
	gf = gf.reshape((n_in, n_out)).T

	lc, gc = test_grad_channel_corr(F.T, 3, 7, n_out)
	gc = gc.reshape((n_in, n_out)).T
	
	ls, gs = test_grad_second_order(F.T, 3, 7, n_out, c_mat_input)
	gs = gs.reshape((n_in, n_out)).T
	
	grad_f += lambdf * gf
	grad_f += lambdc * gc
	grad_f += lambds * gs
	grad_f += lambds * grad_sparse
	grad_f += grad_recon
	
	F -= eps*grad_f/n_imgs_batch
	Ft = pinv(F)
	test_F(F)

	print 'recon:',eps*np.sum(np.abs(grad_recon)), 'sparsity:',eps*np.sum(np.abs(grad_sparse)), 'transpose:',eps*lambdt*np.sum(np.abs(g)), 'fourier:',eps*lambdf*np.sum(np.abs(gf)), 'loss:',eps*np.sum(grad_recon+lambds*grad_sparse+lambdt*g+lambdf*gf), 'filters: ', np.sum(np.abs(F))
	print 'channel corr:',eps*lambdc*np.sum(np.abs(gc)),'second order:',eps*lambds*np.sum(np.abs(gs))
	print time.time() - t_start

	filters_cat = np.zeros((3*7,7*n_out))
	F = F.reshape((n_out,7,7,3))
	for filter in range(n_out):
		filters_cat[0*7:1*7,filter*7:(filter+1)*7] = F[filter,:,:,0]
		filters_cat[1*7:2*7,filter*7:(filter+1)*7] = F[filter,:,:,1]
		filters_cat[2*7:3*7,filter*7:(filter+1)*7] = F[filter,:,:,2]
	savemat('/home/darren/stats_bag_filters.mat',{'F': F, 'filters_cat': filters_cat, 'sparsities':sparsities,'sses':sses, 'transposes': transposes,
		'fouriers': fouriers, 'losses': losses, 'channel_corrs':channel_corrs,'second_orders':second_orders})




