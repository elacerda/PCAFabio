#!/usr/bin/python
__author__ = ['cid@astro.ufsc.br', 'lacerda@astro.ufsc.br']

import sys
import pyfits
import numpy as np
import matplotlib as mpl
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib.pyplot import MaxNLocator

#debug = True
debug = False


def PCA(arr, reduced = False, arrMean = False, arrStd = False, sort = True):
    '''
    ARR array must have shape (measurements, variables)
    reduced = True:
        each var = (var - var.mean()) / var.std()
    '''
    arr__mv = arr
    nMeasurements, nVars = arr__mv.shape    
    if not arrMean or not arrMean.any():
        arrMean__m = arr.mean(axis = -1)
    else:
        arrMean__m = arrMean
    if not reduced:
        diff__mv = arr__mv - arrMean__m
    else:
        if not arrStd or not arrStd.any():
            arrStd__m = arr.std(axis = -1)
        else:
            arrStd__m = arrStd  
        diff__mv = np.asarray([ v / arrStd__m for v in (arr__mv.T - arrMean__m) ]).T
    covMat__vv = (diff__mv.T).dot(diff__mv) / (nVars - 1)
    eigVal__e, eigVec__ve = eigh(covMat__vv)
    eigValS__e = eigVal__e
    eigVecS__ve = eigVec__ve
    if sort:
        S = np.argsort(eigVal__e)[::-1]
        eigValS__e = eigVal__e[S]
        eigVecS__ve = eigVec__ve[:, S]
    return diff__mv, arrMean__m, arrStd__m, covMat__vv, eigValS__e, eigVecS__ve


if __name__ == '__main__':
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
    m = np.ones(nGal, dtype = np.bool)
    m = np.bitwise_and(m, np.greater(fab_Hb, 0))
    m = np.bitwise_and(m, np.greater(fab_O3, 0))
    m = np.bitwise_and(m, np.greater(fab_Ha, 0))
    m = np.bitwise_and(m, np.greater(fab_N2, 0))
    m = np.bitwise_and(m, np.greater(fab_WHa, 0))
    m = np.bitwise_and(m, np.greater(fab_W2, 0))
    m = np.bitwise_and(m, np.greater(fab_W3, 0))
    m = np.bitwise_and(m, ~np.isnan(fab_Hb))
    m = np.bitwise_and(m, ~np.isnan(fab_O3))
    m = np.bitwise_and(m, ~np.isnan(fab_Ha))
    m = np.bitwise_and(m, ~np.isnan(fab_N2))
    m = np.bitwise_and(m, ~np.isnan(fab_WHa))
    m = np.bitwise_and(m, ~np.isnan(fab_W2))
    m = np.bitwise_and(m, ~np.isnan(fab_W3))
    
    O3Hb = np.log10(fab_O3[m] / fab_Hb[m])
    N2Ha = np.log10(fab_N2[m] / fab_Ha[m])
    HaHb = np.log10(fab_Ha[m] / fab_Hb[m])
    WHa = np.log10(fab_WHa[m])
    W2W3 = np.log10(fab_W3[m] / fab_W2[m])
    
    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    # rO3Hb = (O3Hb - O3Hb.mean()) / O3Hb.std()
    # rN2Ha = (N2Ha - N2Ha.mean()) / N2Ha.std()
    # rHaHb = (HaHb - HaHb.mean()) / HaHb.std()
    # rWHa = (WHa - WHa.mean()) / WHa.std()
    # rW2W3 = (W2W3 - W2W3.mean()) / W2W3.std()
    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    ################ Ed
    '''
     v = variable
     m = measurements / galaxy
     e = eigenvalue index
    '''
    ################# PCA #################
    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    # mat__mv = np.column_stack([rO3Hb, rN2Ha, rHaHb, rWHa, rW2W3])
    # nMeasurements, nVars = mat__mv.shape
    # covMat__vv = (mat__mv.T).dot(mat__mv) / (nVars - 1)
    # eValues__e, eVectors__ve = eigh(covMat__vv)
    # S = np.argsort(eValues__e)[::-1]
    # eValuesS__e = eValues__e[S]
    # eVectorsS__ve = eVectors__ve[:, S]
    #EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    mat__mv = np.column_stack([O3Hb, N2Ha, HaHb, WHa, W2W3])
    nMeasurements, nVars = mat__mv.shape
    aux = PCA(mat__mv, reduced = True)
    diff__mv, arrMean__m, arrStd__m, covMat__vv, eValuesS__e, eVectorsS__ve = aux
    projected_mat__me = np.dot(mat__mv, eVectorsS__ve)
    nEig = nVars
    PC__em = np.empty((nEig, nMeasurements), dtype = np.float_)
    for i in xrange(nEig):
        PC__em[i, :] = projected_mat__me[:, i]
    ################# plots #################
    if debug:
        sys.exit(1)
    fs = 15
    cmap = 'spectral'
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Times New Roman'
    mpl.rcParams['font.size'] = fs
    mpl.rcParams['xtick.labelsize'] = fs
    mpl.rcParams['ytick.labelsize'] = fs
    kwargs_scatter = dict(alpha = 0.9, edgecolor = 'none', cmap = cmap)
    kwargs_text = dict(fontsize = fs - 2, color = 'k', verticalalignment = 'top', horizontalalignment = 'right', bbox = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.))
    f = plt.figure()
    f.set_size_inches(10,10)
    shape = (nEig, nEig)
    prune = None
    z = W2W3
    orientation = 'horizontal'
    for i in xrange(nEig):
        row = i
        PCrow = PC__em[row]
        ax = plt.subplot2grid(shape, loc = (row, row))
        ax.hist(PCrow, bins=30, orientation = orientation);
        plt.setp(ax.get_xticklabels(), visible = False)
        ax.yaxis.set_major_locator(MaxNLocator(4, prune = prune))
        prune = 'upper'
        ax.set_ylabel('PC%d' % (i + 1))
        evprc = 100. * eValuesS__e[i] / eValuesS__e.sum()
        ax.text(0.99, 0.95, '%.0f %%' % evprc, transform = ax.transAxes, **kwargs_text)
        for j in xrange(i + 1, nEig):
            col = j
            PCcol = PC__em[col]
            ax = plt.subplot2grid(shape, loc = (row, col))
            ax.scatter(PCcol, PCrow, c = z, **kwargs_scatter)
            ax.set_xlabel('PC%d' % (j + 1))
            if j > i:
                plt.setp(ax.get_xticklabels(), visible = False)
                plt.setp(ax.get_yticklabels(), visible = False)
            ax.grid()
    f.subplots_adjust(hspace = 0, wspace = 0)
    f.savefig('PCsvsPCs_W2W3.png')
    plt.close(f)
