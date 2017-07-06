import numpy as np
import funct as polfun
from astropy.io import fits
from itertools import product as carthprod
import shutil
import os

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate
from scipy.stats import poisson
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset




# Specify data and filename
datadir = "/home/bjung/Documents/Leiden_University/brp/data_red/calib_data"
scidatadir = datadir + "/sorted/NGC4696,IPOL"
sci_dirs = [scidatadir + "/CHIP1"]
testdata = sci_dirs[0] + "/tpl8/corrected2/FORS2.2011-05-04T01:31:46.334_COR.fits" # j=7, k=1
# Load testdata
header, data = polfun.extract_data(testdata)
# Directory for saving plots
plotdir = "/home/bjung/Documents/Leiden_University/brp/data_red/plots/"
imdir = "/home/bjung/Documents/Leiden_University/brp/data_red/images/"
tabledir = "/home/bjung/Documents/Leiden_University/brp/data_red/tables/"




# Define grid
tabularasa = np.zeros([90,90])
xgrid, ygrid = np.meshgrid(np.arange(90), np.arange(90))
# Set test Gaussian
testOgauss = polfun.gaussian2d([xgrid,ygrid], 45, 45, 0, 2, 2, 3).reshape(tabularasa.shape)
testEgauss = polfun.gaussian2d([xgrid,ygrid], 45, 45.3, 0, 2, 2, 3).reshape(tabularasa.shape)
polfun.savefits(testOgauss, imdir+"/gradmethtest", "testO")
polfun.savefits(testEgauss, imdir+"/gradmethtest", "testE")
polfun.savefits(testOgauss-testEgauss, imdir+"/gradmethtest", "testslitdiff")
polfun.saveim_png(testOgauss, plotdir+"/gradmethtest", "testO", colmap="afmhot")
polfun.saveim_png(testEgauss, plotdir+"/gradmethtest", "testE", colmap="afmhot")
polfun.saveim_png(testOgauss-testEgauss, plotdir+"/gradmethtest", "testslitdiff", colmap="afmhot")



cval_prev, dval_prev = 0, 0
for itno in range(5):

    # Define c-ranges
    cstart, cend = np.round([cval_prev - 2/(itno+1), cval_prev + 2/(itno+1)], 2)
    crange = np.linspace(cstart, cend, 49)
    cstep = np.round(crange[1]-crange[0], 3)
    print("\tcrange:\t", cstart, cend, cstep)
    
    # Define d-ranges
    dstart, dend = np.round([dval_prev - 2/(itno+1), dval_prev + 2/(itno+1)], 2)
    drange = np.linspace(dstart, dend, 49)
    dstep = np.round(drange[1]-drange[0], 3)
    print("\tdrange:\t", dstart, dend, dstep)
    
    # Compute the c and d parameters which optimize overlap using gradient method
    gradopt_old, Qopt, opt_cd = polfun.offsetopt_cd(testOgauss, testEgauss, crange, drange,
                                [45,45], 10, iteration=itno, savetofits=True, 
                                pltsavedir=plotdir+"/gradmethtest", 
                                imsavedir=imdir+"/gradmethtest")
    cval_prev, dval_prev = opt_cd
    print("Qopt, opt_cd:\t\t", Qopt, opt_cd)






