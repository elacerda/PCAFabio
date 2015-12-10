#!//anaconda/bin/python
__author__ = 'cid'

import numpy as np
import matplotlib.pyplot as plt
import pyfits

# Reading Fabio's table
hdu = pyfits.open('parameters-SDSS-Starlight-wise.fits')
fab = hdu[1].data

# obs: [:] ==> np.copy(..)
fab_Hb = fab['hbeta_flux'][:]
fab_O3 = fab['oiii_flux'][:]
fab_N2 = fab['nii_flux'][:]
fab_Ha = fab['halpha'][:]
fab_WHa = fab['WHa'][:]
fab_W2 = fab['w2_flux'][:]
fab_W3 = fab['w3_flux'][:]
fab_vdisp = fab['vdisp'][:]
fab_M = fab['mcor_gal'][:]

hdu.close()

# rename shit
nGal = len(fab_Hb)
m = np.ones(nGal,dtype=np.bool)
m = np.bitwise_and( m , np.greater(fab_Hb,0) )
m = np.bitwise_and( m , np.greater(fab_O3,0) )
m = np.bitwise_and( m , np.greater(fab_Ha,0) )
m = np.bitwise_and( m , np.greater(fab_N2,0) )
m = np.bitwise_and( m , np.greater(fab_WHa,0) )
m = np.bitwise_and( m , np.greater(fab_W2,0) )
m = np.bitwise_and( m , np.greater(fab_W3,0) )
m = np.bitwise_and( m , ~np.isnan(fab_Hb) )
m = np.bitwise_and( m , ~np.isnan(fab_O3) )
m = np.bitwise_and( m , ~np.isnan(fab_Ha) )
m = np.bitwise_and( m , ~np.isnan(fab_N2) )
m = np.bitwise_and( m , ~np.isnan(fab_WHa) )
m = np.bitwise_and( m , ~np.isnan(fab_W2) )
m = np.bitwise_and( m , ~np.isnan(fab_W3) )

O3Hb = np.log10(fab_O3[m] / fab_Hb[m])
N2Ha = np.log10(fab_N2[m] / fab_Ha[m])
HaHb = np.log10(fab_Ha[m] / fab_Hb[m])
WHa  = np.log10(fab_WHa[m])
W2W3 = np.log10(fab_W3[m] / fab_W2[m])

rO3Hb = (O3Hb - O3Hb.mean()) / O3Hb.std()
rN2Ha = (N2Ha - N2Ha.mean()) / N2Ha.std()
rHaHb = (HaHb - HaHb.mean()) / HaHb.std()
rWHa  = (WHa - WHa.mean())   / WHa.std()
rW2W3 = (W2W3 - W2W3.mean()) / W2W3.std()

################ Ed
from scipy import linalg
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 10
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
nVars = 5
nMeasurements = len(rO3Hb)
########
'''
 v = variable
 m = measurements / galaxy
 e = eigenvalue index
'''
################# plots #################
mat__vm = np.empty((nVars, nMeasurements), dtype = np.float_)
mat__vm[0] = rO3Hb
mat__vm[1] = rN2Ha
mat__vm[2] = rHaHb
mat__vm[3] = rWHa
mat__vm[4] = rW2W3
ddof = nVars - 1
covMat__vv = mat__vm.dot(mat__vm.T) / ddof
eValues__e, eVectors__ve = linalg.eigh(covMat__vv)
S = np.argsort(eValues__e)[::-1]
eValuesS__e = eValues__e[S]
eVectorsS__ve = eVectors__ve[S]
projected_mat__me = np.dot(mat__vm.T, eVectorsS__ve)
nEig = nVars
PC__em = np.empty((nEig, nMeasurements), dtype = np.float_)
for i in xrange(nEig):
    PC__em[i] = projected_mat__me[:,i]
################# plots #################
kwargs_scatter = dict(alpha=0.9,edgecolor='none',cmap='spectral')
f = plt.figure()
shape = (nEig, nEig)

for i in xrange(nEig):
    ini_i = i
    for j in range(i, nEig):
        ax = plt.subplot2grid(shape, loc = (i, j))
        ax.scatter(PC__em[j], PC__em[i], c = W2W3, **kwargs_scatter)
        if j > ini_i:
            plt.setp(ax.get_xticklabels(), visible = False)
            plt.setp(ax.get_yticklabels(), visible = False)

f.subplots_adjust(hspace = 0.003, wspace = 0.003)
f.savefig('teste.png')
plt.close(f)
