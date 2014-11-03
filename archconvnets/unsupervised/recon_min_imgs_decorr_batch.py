from archconvnets.unsupervised.DFT import DFT_matrix_2d
from scipy.spatial.distance import pdist
from archconvnets.unsupervised.grad_transpose import test_grad_transpose
from archconvnets.unsupervised.grad_fourier_l1 import test_grad_fourier_l1
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
	loss = sse + lambd*sparsity + lambdt*l + lambf*lf
	print 'recon:',sse, 'sparsity:',sparsity, 'transpose:',lambdt*l, 'fourier:',lambf*lf, 'loss:',loss, 'transpose: ',np.mean(np.abs(1-pdist(F.T,'correlation'))),  img_t
	sparsities.append(sparsity)
	sses.append(sse)
	transposes.append(l)
	fouriers.append(lf)
	losses.append(loss)

####### fourier
X = np.real(DFT_matrix_2d(7))
X2 = np.imag(DFT_matrix_2d(7))

	
img_t = 0
n_out = 32
n_in = 147
F = np.random.random((n_out,n_in))
patch = np.random.random(n_in)

sparsities = []
sses = []
transposes = []
fouriers = []
losses = []

lambd = 0#1#e1#e1
lambdt = 1e4#5
lambf = 1e1#8
eps = 5e-3#12
grad = np.zeros((n_out,n_in))
n_imgs_batch = 5#128
for batch in range(9000):
	t_start = time.time()
	F = F.reshape((n_out,n_in))
	batch_data =  np.load('/storage/batch128_img138_full/data_batch_' + str(batch))['data'].reshape((3,138,138,128)).transpose((3,1,2,0))[:,66:66+7,66:66+7]
	Ft = pinv(F)
	one_m_Ft_F = 1 - np.dot(Ft,F)
	one_m_F_Ft = 1 - np.dot(F,Ft)
	Ft_Ftt = np.dot(Ft, Ft.T)
	
	for img in range(n_imgs_batch):#128):
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
					grad[i_val,j_val] += grad_temp
					grad[i_val,j_val] += lambd*patch[j_val]*np.sign(patch[j_val]*F[i_val,j_val])
	l, g =  test_grad_transpose(F.T, 3, 7, n_out)
	g = g.reshape((n_in, n_out)).T
	grad_f = lambdt * g
	
	lf, gf = test_grad_fourier_l1(F.T, 3, 7, n_out, X, X2)
	gf = gf.reshape((n_in, n_out)).T
	grad_f += lambf * gf
	grad_f += grad
	
	F -= eps*grad_f/n_imgs_batch
	Ft = pinv(F)
	test_F(F)

	print 'recon:',eps*np.sum(np.abs(grad)), 'sparsity:',0, 'transpose:',eps*lambdt*np.sum(np.abs(g)), 'fourier:',eps*lambf*np.sum(np.abs(gf)), 'loss:',eps*np.sum(grad+lambdt*g+lambf*gf), 'filters: ', np.sum(np.abs(F))
	print time.time() - t_start

	filters_cat = np.zeros((3*7,7*n_out))
	F = F.reshape((n_out,7,7,3))
	for filter in range(n_out):
		filters_cat[0*7:1*7,filter*7:(filter+1)*7] = F[filter,:,:,0]
		filters_cat[1*7:2*7,filter*7:(filter+1)*7] = F[filter,:,:,1]
		filters_cat[2*7:3*7,filter*7:(filter+1)*7] = F[filter,:,:,2]
	savemat('/home/darren/autoencode_transpose.mat',{'F': F, 'filters_cat': filters_cat, 'sparsities':sparsities,'sses':sses, 'transposes': transposes,
		'fouriers': fouriers, 'losses': losses})




