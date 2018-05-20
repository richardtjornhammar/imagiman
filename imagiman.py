from pylab import figure, show, rand
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy as np
from scipy import signal as sg
from scipy.stats import rankdata

def gaussian_kernel( size, sigma ):
    size	= int(size*0.5)
    x, y	= np.mgrid[-size:size, -size:size] #:size+1
    g		= np.exp( -0.5/float(sigma**2)*(x**2+y**2) )
    return g / g.sum()

def gaussian_bfact ( size, bfact ):
    x, y	= np.mgrid[-size:size, -size:size] #:size+1
    g		= np.exp( -float(bfact)*(x**2+y**2) )
    return g / g.sum()

def kurtosis(imgmap):
	fimgm		= np.mean(imgmap)
	dev		= imgmap - fimgm
	kurt		= np.mean(dev**4)/(np.mean(dev**2))**2
	return kurt

def img_svd_recon(imgmap,i,j):
	U, s, V 	= np.linalg.svd(imgmap)
	s[i:j]		= 0
	S		= np.zeros( (U.shape[0], V.shape[0]) )
	S[ :V.shape[0], :V.shape[0] ] = np.diag(s)
	recon_imgmap	= np.dot(U, np.dot(S, V))
	return recon_imgmap

def img_B_scale( B, imap ):
	new_map = imap
	b_map = np.exp(-B*imap[:,:])
	if np.isnan( b_map.any() ) :
		print ( "IS NAN" )
	else:
		new_map *= b_map
	return new_map

def img_cdf ( imgmap ):
	n,m		= imgmap.shape
	int_img 	= np.round(imgmap)
	he_n		= n*m*1.0
	ni_max  	= np.max(int_img)
	ni_min  	= np.min(int_img)
	print ("INFO::CDFCALC",ni_max,np.max(imgmap))
	p		= np.array([(sum(sum(int_img==0)))/he_n])
	cdf		= p
	for i in np.nditer( np.arange(0,ni_max,1)) :
		p_tmp	= (sum(sum(int_img==i)))
		p	= np.append(p,p_tmp/he_n)
		cdf	= np.append(cdf,sum(p))
	return cdf

def img_prob ( imgmap ):
	n,m		= imgmap.shape
	int_img 	= np.round(imgmap)
	he_n		= n*m*1.0
	ni_max  	= np.max(int_img)
	ni_min  	= np.min(int_img)
	p		= np.array([(sum(sum(int_img==0)))/he_n])
	cdf		= p
	for i in np.nditer( np.arange(0,ni_max,1)) :
		p_tmp	= (sum(sum(int_img==i)))
		p	= np.append(p,p_tmp/he_n)
	return p

def img_histogram_equalisation(imgmap,cdf):
	I_max 	= np.max(imgmap)
	int_img = np.round(imgmap)
	he_img	= np.zeros( (int_img.shape[0], int_img.shape[1]) )
	for i in range(0,int_img.shape[0]-1,1) :
		for j in range(0,int_img.shape[1]-1,1) :
			he_img[i,j]=cdf[ int_img[i,j] ]
	return he_img*I_max

def img_deequalisation(imgmap,p):
	imax 	= p.shape[0]
	I_max 	= np.max(imgmap)

	print (imax, I_max, p[p.shape[0]-1], p.shape[0]-1)

	int_img = np.round( imgmap * (imax-1.0) / I_max )
	print (np.max(int_img))

	he_img	= np.zeros( (int_img.shape[0], int_img.shape[1]) )
	for i in range(0,int_img.shape[0]-1,1) :
		for j in range(0,int_img.shape[1]-1,1) :
			he_img[i,j]=p[ int_img[i,j] ]
	return he_img#*I_max

def img_cdf_scale(imgmap,cdf,power):
	I_max 	= np.max(imgmap)
	int_img = np.round(imgmap)
	he_img	= np.zeros( (int_img.shape[0], int_img.shape[1]) )
	for i in range(0,int_img.shape[0]-1,1) :
		for j in range(0,int_img.shape[1]-1,1) :
			val=cdf[int_img[i,j]]
			he_img[i,j]=int_img[i,j]*val**power
	return he_img

def transform_cdf ( cdf1, cdf2 ):
	cdf_transform = 0
	for i in range( 1, cdf1.shape[0] ):
		J		= np.argmin( np.abs(cdf1[i]-cdf2) )
		cdf_transform	= np.append( cdf_transform, J )
	return cdf_transform

def img_equalisation(imgmap, cdf_trans):
	I_max 	= np.max(imgmap)
	int_img = np.round(imgmap)
	he_img	= np.zeros( (int_img.shape[0], int_img.shape[1]) )
	for i in range(0,int_img.shape[0]-1,1) :
		for j in range(0,int_img.shape[1]-1,1) :
			he_img[i,j] = cdf_trans[int_img[i,j]]
	return he_img*I_max

def img_svd_denoise_train(imgmap, Nsvd, keep_low):
	n,m		= imgmap.shape
	I_max		= np.max(imgmap)
	recon_img1	= img_svd_recon(imgmap,Nsvd,n) 	# KEEP Nsvd first values
	recon_img2	= img_svd_recon(imgmap,0,Nsvd)	# ZERO Nsvd first values
	recon_img4	= np.zeros(recon_img1.shape)
	if keep_low == 1 :
		for i in range(1,Nsvd):
			recon_img1	 = img_svd_recon(imgmap,Nsvd,n)
			recon_img2	 = img_svd_recon(imgmap,0,Nsvd)
			j		 = i # 2 #Nsvd+1-i
			dsp_img		 = sg.convolve( recon_img2 , np.ones((2*j+1,2*j+1)))
			recon_img2	 = dsp_img[j:n+j,j:n+j]
			recon_img4	+= (recon_img1/np.max(recon_img1)+recon_img2/np.max(recon_img2)) * I_max
	else :
		recon_img4	= recon_img1/np.max(recon_img1)*I_max
	return recon_img4

def img_non_edge_blur_smooth(imap):
	n,m		= imap.shape
	imap_o		= imap*0.0;
	tmp_img		= imap_o

	fimgm		= np.mean(imap)
	dev		= imap - fimgm
	kurt		= np.mean(dev**4)/(np.mean(dev**2))**2
	print ("INFOINFO::", kurt, np.std(imap))

	sigmaI	 	= np.std(imap)**2
	sigmap		= np.sqrt(kurt)**2
	npix 		= int( round(kurt*5) )

	n_dr_map	= imap_o

	for i in range(-npix,npix):
		for j in range(-npix,npix):
			n_dr_map	= imap
			D2 		= 0
			n_dr_map = np.roll(n_dr_map,j,axis=0)
			n_dr_map = np.roll(n_dr_map,i,axis=1)
			D2		 = i**2+j**2
			tmp_img		+= np.exp(-0.5*(n_dr_map-imap)**2/sigmaI)*np.exp(-0.5*D2/sigmap)*n_dr_map
	print (np.max(tmp_img), np.min(tmp_img))
	if 0:
		mat_rand	= np.random.random((3,3))
		mat_rand[1,1]	= 1.0
		s_rand		= sum(sum(mat_rand))*0.75
		t_img   	= abs(sg.convolve( imap, mat_rand/s_rand ))
		tmp_img		+= t_img[1:n+1,1:m+1]

	return tmp_img

def img_non_edge_blur_smooth_norm(imap):

	n,m		= imap.shape
	imap_o		= imap*0.0;
	tmp_img		= imap_o

	fimgm		= np.mean(imap)
	dev		= imap - fimgm
	kurt		= np.mean(dev**4)/(np.mean(dev**2))**2
	print ("INFOINFO::", kurt, np.std(imap))

	kurt		= 1.0*kurt
	sigmaI1	 	= 1.0*np.std(imap)**2
	sigmaI2	 	= 1.0*np.std(imap)**2
	sigmap		= np.sqrt(kurt)**2
	npix 		= int( round(kurt*5) )

	n_dr_map	= imap_o
	sq2g = np.sqrt(2.0*np.pi)
	iG1 	= 1.0/(sq2g*np.sqrt(sigmaI1))
	iG2 	= 1.0/(sq2g*np.sqrt(sigmaI2))
	iG3 	= 1.0/(sq2g*np.sqrt(sigmap ))
	W	= 0.0
	for i in range(-npix,npix):
		for j in range(-npix,npix):
			n_dr_map	= imap
			D2 		= 0
			n_dr_map = np.roll(n_dr_map,j,axis=0)
			n_dr_map = np.roll(n_dr_map,i,axis=1)
			I2		= n_dr_map*imap
			D2		= i**2+j**2
			w 		= np.exp(-0.5*(n_dr_map-imap)**2/sigmaI1)*iG1*np.exp(-0.5*D2/sigmap)*iG2*np.exp(-0.5*I2/sigmaI2)*iG1*iG3
			W		+= w
			tmp_img		+= w*n_dr_map
	tmp_img /= W
	print (np.max(tmp_img), np.min(tmp_img), np.mean(tmp_img))
	return tmp_img

def img_svd_denoise(imgmap, Nsvd, keep_low):
	n,m		= imgmap.shape
	I_max		= np.max(imgmap)
	recon_img1	= img_svd_recon(imgmap,Nsvd,n) 	# KEEP Nsvd first values
	recon_img2	= img_svd_recon(imgmap,0,Nsvd)	# ZERO Nsvd first values
	if keep_low == 1 :
		d_img=recon_img2
#		for i in range(1,5):
		j		= 2 #np.ceil(rand(1)*3)
		dsp_img		= sg.convolve( d_img ,  np.ones((2*j+1,2*j+1)) ) #3-1,5-2,7-3,9-4,11-5
		d_img		= dsp_img[j:n+j,j:n+j]
		recon_img4	= (recon_img1/np.max(recon_img1)+d_img/np.max(d_img))*I_max
	else :
		recon_img4	= recon_img1/np.max(recon_img1)*I_max
	return recon_img4

kernel_edge_detect1 = np.array([[ 1., 0.,-1.],
                                [ 0., 0., 0.],
                                [-1., 0., 1.]])

kernel_edge_detect2 = np.array([[ 0., 1., 0.],
                                [ 1.,-4., 1.],
                                [ 0., 1., 0.]])

kernel_edge_detect3 = np.array([[-1.,-1.,-1.],
                                [-1., 8.,-1.],
                                [-1.,-1.,-1.]])

kernel_deriv 	    = np.array([[-1.,-2.,-1.],
                                [-2.,12.,-2.],
                                [-1.,-2.,-1.]])

kernel_sharpen = np.array([[ 0.,-1., 0.],
                           [-1., 5.,-1.],
                           [ 0.,-1., 0.]])

kernel_sharpen2 = np.array([[-1.,-1.,-1.],
                            [-1., 9.,-1.],
                            [-1.,-1.,-1.]])


kernel_blur = np.array([[ 1.0 , 1.0 , 1.0 ],
                        [ 1.0 , 1.0 , 1.0 ],
                        [ 1.0 , 1.0 , 1.0 ]])

kernel_umask = np.array([[ 1., 1., 1.],
                         [ 1.,-8., 1.],
                         [ 1., 1., 1.]])

kernel_Gy =   np.array([[-1., -2., -1.],
                        [ 0.,  0.,  0.],
                        [ 1.,  2.,  1.]])

kernel_Gx =   np.array([[-1.,  0.,  1.],
                        [-2.,  0.,  2.],
                        [-1.,  0.,  1.]])

kernel_Gxy =   np.array([[-2., -1.,  0.],
                         [-1.,  0.,  1.],
                         [ 0.,  1.,  2.]])

kernel_Gyx =   np.array([[ 0.,  1.,  2.],
                         [-1.,  0.,  1.],
                         [-2., -1.,  0.]])


kernel_self =   np.array([[ 0.,  0.,  0.],
                          [ 0.,  1.,  0.],
                          [ 0.,  0.,  0.]])

def img_deriv(A_img):
	mag	= 1E0
	iamax   = np.max(A_img)
	D_img   = abs(sg.convolve( A_img,	kernel_Gx *mag ))
	D_img  += abs(sg.convolve( A_img,	kernel_Gy *mag ))
	D_img  += abs(sg.convolve( A_img,  -1.0*kernel_Gx *mag ))
	D_img  += abs(sg.convolve( A_img,  -1.0*kernel_Gy *mag ))
	D_img  += abs(sg.convolve( A_img,       kernel_Gxy*mag ))
	D_img  += abs(sg.convolve( A_img,       kernel_Gyx*mag ))
	D_img  += abs(sg.convolve( A_img,  -1.0*kernel_Gxy*mag ))
	D_img  += abs(sg.convolve( A_img,  -1.0*kernel_Gyx*mag ))
	idmax   = np.max(D_img)

	p,q = D_img.shape
	n,m = A_img.shape
	fx=round((p-n)*0.5)
	fy=round((q-m)*0.5)
	return D_img[fx:n+fx,fy:m+fy]*iamax/idmax
#
img 		= mpimg.imread('RichTjorn.bmp');		magnitude 	= 1E+3
#img 		= mpimg.imread('crew89_noisy_25.png');		magnitude 	= 1E-1
#img 		= mpimg.imread('lizard_noisy.bmp');		magnitude 	= 1E-1
#img 		= mpimg.imread('noisy_voc_best_002.bmp');	magnitude 	= 1E-1
#img 		= mpimg.imread('chest_xray.bmp');		magnitude 	= 1E-3
#
narrimgsh	= np.array(img.shape)
DIM 		= narrimgsh.shape[0]

if DIM==3:
	n, m, c 	= img.shape
else:
	n, m		= img.shape

W		= ( rand(n,m) + rand(n,m) + rand(n,m)
		+   rand(n,m) + rand(n,m) + rand(n,m) )/6.0

print ("SNR")
print (np.mean(img)/np.std(W*magnitude)**2)

if DIM==3 :
	flat_img	= np.mean(img,2)
else:
	flat_img	= img

noisy_img  = flat_img+W*magnitude
noisy_img /= np.max(noisy_img)/256.0

nm = np.shape(noisy_img)
linimg = noisy_img.reshape(-1)
rimg = rankdata(linimg,'average').reshape(nm)/float(len(linimg))

print('RANKED INTENSITY RANGE:', np.min(rimg),np.max(rimg) )

A_img = noisy_img
#A_img = rimg

figure(1)
imgplot1 = plt.imshow( noisy_img , cmap = cm.Greys_r )

figure(2)
neb_img	= img_non_edge_blur_smooth_norm( A_img )
imgplot1 = plt.imshow( rimg , cmap = cm.Greys_r )

figure(3)
deb_img = img_non_edge_blur_smooth_norm( rimg )
imgplot1 = plt.imshow( deb_img , cmap = cm.Greys_r )

figure(4)
imgplot1 = plt.imshow( neb_img , cmap = cm.Greys_r )

show()
