import numpy as np
import funct as polfun
from astropy.io import fits
from itertools import product as cartprod
import shutil
import os
import re

from astropy.stats import sigma_clipped_stats
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
#from photutils import Background2D, SigmaClip, MedianBackground

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib import cm
#from scipy.ndimage import filters as scifilt
from scipy import interpolate
from scipy.stats import poisson
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset








# Specify data and filename
datadir = "/home/bjung/Documents/Leiden_University/brp/data_red/calib_data"
# Specify standard star directories
stddatadir = datadir + "/sorted/STD,IPOL"
std_dirs = [stddatadir + "/Vela1_95/CHIP1", stddatadir + "/WD1615_154/CHIP1"]
teststddata = [std_dirs[0] + "/tpl3/FORS2.2011-05-04T00:05:36.569.fits", # RETA POSANG 45 deg
               std_dirs[1] + "/tpl2/FORS2.2011-05-04T05:33:58.533.fits"] # RETA POSANG 45 deg
# Specify science data directories
scidatadir = datadir + "/sorted/NGC4696,IPOL"
sci_dirs = [scidatadir + "/CHIP1"]
testscidata = sci_dirs[0] + "/tpl8/FORS2.2011-05-04T01:31:46.334.fits" # RETA POSANG 45 deg # j=7, k=1
# Combine data dirs in list
testdata_fnames = [teststddata[0], teststddata[1], testscidata]
# Load testdata
headerVELA, dataVELA = polfun.extract_data(teststddata[0])
headerWD, dataWD = polfun.extract_data(teststddata[1])
headerNGC, dataNGC = polfun.extract_data(testscidata)
# Directory for saving plots
plotdir = "/home/bjung/Documents/Leiden_University/brp/data_red/plots"
imdir = "/home/bjung/Documents/Leiden_University/brp/data_red/images"
npsavedir = "/home/bjung/Documents/Leiden_University/brp/data_red/npsaves"
tabledir = "/home/bjung/Documents/Leiden_University/brp/data_red/tables"

# Specify bias and masterflat
header, Mbias = polfun.extract_data(datadir + "/masterbias.fits")
header, Mflat_norm = polfun.extract_data(datadir + "/masterflats/masterflat_norm_FLAT,LAM_IPOL_CHIP1.fits")



# Aproximate coordinates of selection of stars within CHIP1 of 'Vela1_95' and 'WD1615_154'. Axis 1 specifies the different stars within the std_dir; axis 2 specifies the x, y1, y2 coordinate of the specific star (with y1 specifying the y coordinate on the upper slit and y2 indicating the y coordinate on the lower slit) as well as the aproximate stellar radius and the slit pair number (numbered 1 to 5 from lower to upper slit pair) which the star lies on. NOTE: THE LAST LIST WITHIN AXIS1 IS A SKY APERTURE!!!
star_lst_sci = [[335, 904, 807, 5, 5], [514, 869, 773, 7, 5], [1169, 907, 811, 5, 5], 
                [1383, 878, 782, 7, 5], [341, 694, 599, 10, 4], [370, 702, 607, 11, 4], 
                [362, 724, 630, 5, 4], [898, 709, 609, 8, 4], [1836, 707, 611, 6, 4], 
                [227, 523, 429, 6, 3], [354, 498, 404, 10, 3], [376, 512, 418, 8, 3], 
                [419, 525, 431, 7, 3], [537, 491, 392, 7, 3], [571, 541, 446, 8, 3], 
                [1096, 510, 416, 5, 3], [1179, 530, 436, 8, 3], [487, 320, 226, 7, 2], 
                [637, 331, 238, 6, 2], [1214, 345, 252, 6, 2], [1248, 326, 233, 6, 2], 
                [1663, 308, 217, 9, 2], [326, 132, 40, 5, 1], [613, 186, 94, 10, 1], 
                [634, 184, 91, 9, 1], [642, 134, 41, 7, 1], [838, 175, 82, 8, 1], 
                [990, 140, 48, 11, 1], [1033, 157, 65, 9, 1], [1172, 147, 55, 7, 1], 
                [1315, 164, 71, 8, 1], [1549, 164, 72, 13, 1]] #[pixel] # 32 stars
star_lst_stdVELA = [[1034, 347, 251, 15, 2], [1177, 368, 273, 8, 2], [319, 345, 250, 5, 2], [281, 499, 403, 6, 3], [414, 139, 45, 12, 1], [531, 706, 609, 5, 4], [1583, 322, 229, 3, 2], [1779, 321, 224, 4, 2], [1294, 725, 627, 4, 4], [1501, 719, 622, 7, 4]] #[pixel] # 10 stars
star_lst_stdWD = [[1039, 347, 253, 12, 2], [599, 541, 446, 5, 3], [365, 700, 604, 5, 4], [702, 903, 806, 6, 5], [801, 136, 43, 4, 1], [1055, 133, 43, 4, 1], [1186, 130, 37, 4, 1], [1132, 685, 592, 3, 4], [1222, 685, 592, 4, 4], [1395, 679, 587, 4, 4], [1413, 912, 816, 5, 5], [1655, 542, 449, 5, 3], [1643, 512, 417, 5, 3], [1632, 190, 97, 6, 1], [1608, 178, 85, 4, 1]] #[pixel] # 15 stars           
# Combine star lists
star_lsts = [star_lst_stdVELA, star_lst_stdWD, star_lst_sci]


# Set background offset ranges
bkgoffsets = np.round(np.arange(-5,7.5,2.5),1)
# Define the x- and y-ranges corresponding to the chip
chip_xyranges = [[183,1868],[25,934]]
# Range of aperture radii
r_range = np.arange(1, 16) #[pixels]
# Pixel scale
pixscale = 0.126 #[arcsec/pixel]
# Boolean variables for turning on/off the computations
detsplines1d, detsplines2d = False, False
detoverlaps1d, detoverlaps2d = True, True
recompute_fluxlsts = True


# Cut and append slits
slitsNGC, upedges, lowedges, gapw, slitshapes = polfun.cut_to_slits(dataNGC)
slitsNGC = [slitsNGC[i][0:np.min(slitshapes[:,0]),
                        0:np.min(slitshapes[:,1])] for i in range(0,10,1)] # Enforce same x-shape
slitsVELA = [dataVELA[lowedges[i]:upedges[i],
                      chip_xyranges[0][0]:chip_xyranges[0][0]+np.min(slitshapes[:,1])] 
             for i in range(len(lowedges))]
slitsWD = [dataWD[lowedges[i]:upedges[i],
                  chip_xyranges[0][0]:chip_xyranges[0][0]+np.min(slitshapes[:,1])]
           for i in range(len(lowedges))]
# Append to list
slits_lst = [slitsVELA, slitsWD, slitsNGC]



# Determine slitwidths
upedges, lowedges, gapw = [np.array(temp) for temp in [upedges, lowedges, gapw]]
slitwidths = upedges-lowedges





'''
####################### EVALUATION OF DIFFERENT INTERPOLATION METHODS #######################
# 1d spline #TODO
tck_1d = interpolate.splrep(xpoints, testslitO[27], s=0)
interpO3_1d = interpolate.splev(interpx, tck_1d, der=0)
# Compute the relative interpolation errors
diff1_norm = (interpO[270,xpoints*10] - testslitO[27]) / testslitO[27]
diff3_norm = (interpO3_1d[xpoints*10] - testslitO[27]) / testslitO[27] 
# PLOT
plt.figure()
plt.plot(xpoints, diff1_norm, 'r', label='2D cubic spline')
plt.plot(xpoints, diff3_norm, 'g', label='1D cubic spline')
plt.legend(loc='best')
plt.xlabel(r"X [pixels]", fontsize=20), plt.ylabel(r"Relative Error [--]", fontsize=20)
plt.title("Interpolation Errors")
plt.savefig(plotdir + "/interp/interpErrs.png")
plt.show()
plt.close()
####################### END evaluation of different interpolation methods #######################
''' # TODO EVALUATE MORE FORMS OF INTERPOLATION





# Iterate over all objects (Vela1_95, WD1615_154, NGC4696)
for i, objdir in enumerate([std_dirs[0], std_dirs[1], sci_dirs[0]]):
    object_ = objdir.split("/")[-2]
    print("OBJECT:\t{}".format(object_))
    
    if i != 2:
        print("\nOnly NGC4696,IPOL used!!!\n")
        continue
    
    # Load the c- and d-scapes
    Xscape_Jk = np.load(npsavedir+"/{}/totoffsX_i{}Jk.npy".format(object_,i+1)) #For CHECK
    Yscape_Jk = np.load(npsavedir+"/{}/totoffsY_i{}Jk.npy".format(object_,i+1)) #For CHECK
    dxscape_Jk = np.load(npsavedir+"/{}/dxscape_i{}Jk.npy".format(object_,i+1))
    dyscape_Jk = np.load(npsavedir+"/{}/dyscape_i{}Jk.npy".format(object_,i+1))    
    cscape_Jk = np.load(npsavedir+"/{}/cscape_i{}Jk.npy".format(object_,i+1))
    dscape_Jk = np.load(npsavedir+"/{}/dscape_i{}Jk.npy".format(object_,i+1))
    print("c- and d-scapes loaded!")
    
    # Select slits and starlist corresponding to current object
    star_lst = np.array(star_lsts[i])
    
    # Create list with templates
    tpl_dirlst, tpl_flst = polfun.mk_lsts(objdir)
    tplNRlst = [int(re.findall('\d+', temp)[0]) for temp in tpl_dirlst]
    tpl_dirlst = [m[0] for m in sorted(zip(tpl_dirlst, tplNRlst), key=lambda l: l[1])] #Sort list
    
    
    # Define plot save directories
    temp = objdir.split(datadir+"/sorted/")[1]
    pltsavedir = plotdir +"/"+ temp.split("/")[0]
    imsavedir = imdir +"/"+ temp.split("/")[0]
    if pltsavedir.split("/")[-1] == "STD,IPOL":
        pltsavedir = plotdir +"/"+ temp.split("/")[1]
        imsavedir = imdir +"/"+ temp.split("/")[1]
    datasavedir = npsavedir+"/"+imsavedir.split("/")[-1]
    print("\n\n{}\n\n".format(imsavedir.split("/")[-1]))        
    
    
    # Iterate over all templates
    slitoffs_JKs, exptimes_JK = np.zeros([11,4,5,2]), np.zeros([11,4])
    O__JKb, E__JKb, OplusE__JKb, O_E__JKb, O_E_grad__JKb = [], [], [], [], []
    UQPphi__Jb, sigma_UQPphi__Jb = [], []
    pixoffxy__JK, cdintpxy__JK, filters_Jk, retangles_Jk, J = [], [], [], [], 0
    for j, tpl_name in enumerate(tpl_dirlst):
        print("\tTPL:\t{}".format(tpl_name))
        tpl_dir = objdir + '/' + tpl_name
        print("DEBUG:\t{}".format(J))
        
              
        # Load or recompute intermediate results
        if not recompute_fluxlsts:
            O__JKb = np.load(datasavedir+"/O__i{}Jb.npy".format(i+1))
            E__JKb = np.load(datasavedir+"/E__i{}Jb.npy".format(i+1))
            OplusE__JKb = np.load(datasavedir+"/OplusE__i{}JKb.npy".format(i+1))
            O_E__JKb = np.load(datasavedir+"/O_E__i{}JKb.npy".format(i+1))
            O_E_grad__JKb = np.load(datasavedir+"/O_E_grad__i{}JKb.npy".format(i+1))        
            UQPphi__Jb = np.load(datasavedir+"/UQPphi__i{}Jb.npy".format(i+1))
            sigma_UQPphi__Jb = np.load(datasavedir+"/sigmaUQPphi__i{}Jb.npy".format(i+1))
            print("LOADED flux and stokes lists!")
            pixoffxy__JK = np.load(datasavedir+"/pixoffxy_i{}JKs.npy".format(i+1))
            pixoffxy__JK = np.load(datasavedir+"/cdintpxy_i{}JKs.npy".format(i+1))
            slitoffs_JKs = np.load(datasavedir+"/slitoffs_i{}JKs.npy".format(i+1)) 
            filters_Jk = np.load(datasavedir+"/filters_i{}Jk.npy".format(i+1))
            exptimes_JK = np.load(datasavedir+"/exptimes_i{}JK.npy".format(i+1))
            print("LOADED offset, filter and integration times lists!")
            scapexarcs = np.load(npsavedir+"/scapexarcs.npy")
            scapeyarcs = np.load(npsavedir+"/scapeyarcs.npy")
            scapexslitarcs = np.load(datasavedir+
                    "/tpl{}/exp{}/slitp2/scapexslitarcs_i{}j{}k{}slitp{}.npy".format(1,1,i+1,1,1,2))
            scapeyslitarcs = np.load(datasavedir+
                    "/tpl{}/exp{}/slitp2/scapeyslitarcs_i{}j{}k{}slitp{}.npy".format(1,1,i+1,1,1,2))
            print("Loaded image axes ranges!")
            break     
        #TODO TODO TODO DO THINGS
        
        
        # Create a list with filenames of files stored within tpldir
        expdir_lst, expfile_lst = polfun.mk_lsts(tpl_dir)
        expfile_lst = np.sort(expfile_lst)
        
        # Skip non-usable templates (non-usable templates should be put in a folder "skipped" or an equivalent directory which doesn't start with the string "tpl") or incomplete templates.
        if ((len(expfile_lst) != 4) or
            (objdir.split("/")[-2] == "Vela1_95" and tpl_name in ["tpl1", "tpl2", "tpl3"]) or
            (objdir.split("/")[-2] == "NGC4696,IPOL" and tpl_name == "tpl5")): #TODO INCLUDE TPL5
            print("\t skipped")
            continue
        
        
        # Iterate over all exposures
        slitoffs_ks = np.zeros([4,5,2], dtype=int)
        pixoffxy__k, cdintpxy__k, filters_k, retangles_k, exptimes_k = [], [], [], [], []
        O__kb, E__kb, O_E__kb, OplusE__kb, O_E_grad__kb = [], [], [], [], []
        for k, fname in enumerate(expfile_lst):
            print("\n\t\t {}".format(fname))
            
            # Select the c- and d-scapes corresponding to the current exposure
            Xscape, Yscape = Xscape_Jk[J,k], Yscape_Jk[J,k] #CHECK
            dxscape, dyscape = dxscape_Jk[J,k], dyscape_Jk[J,k]
            cscape, dscape = cscape_Jk[J,k], dscape_Jk[J,k]
            
            # Extract data
            header, data = polfun.extract_data(tpl_dir +'/'+ fname)
            filtername = header["HIERARCH ESO INS FILT1 NAME"]
            filters_k.append(filtername)
            print("DEBUG filtername:\t{}".format(filtername))
            retangles_k.append(header["HIERARCH ESO INS RETA2 POSANG"]) #dtype float
            exptime = header["EXPTIME"]
            exptimes_k.append(exptime)
            # De-biasing and flat-fielding corrections
            data = (data - Mbias) / Mflat_norm
            # Division by exposure time
            data = data / exptime
            # Extract the slits
            minNy = np.min(np.diff(np.dstack([lowedges,upedges])[0]))
            slits = [data[lowedges[m]:lowedges[m]+minNy,
                     chip_xyranges[0][0]:chip_xyranges[0][0]+np.min(slitshapes[:,1])]
                     for m in range(len(lowedges))]  
            # Select the O and E slits containing NGC4696      
            Oylow, Oyup = lowedges[3], upedges[3]            
            slitsNGC = [slits[2],slits[3]]
            slitE, slitO = slitsNGC
            
            
            # Diagnostic plot
            '''
            f, axarr = plt.subplots(2, sharex=True)
            axarr[0].imshow(slitO, origin='lower')
            axarr[0].set_title('O')
            axarr[1].imshow(Ocorr, origin='lower')
            axarr[1].set_title('O-bgO')
            plt.show()
            plt.close()
            '''
            
            
            # Determine 1d polynomial interpolations over median c- and d-scapes
            pixoffxy, cdintp_xy = [], []
            for totscape, pixscape, subscape, pltsavetitle, plttitle in zip([Xscape,Yscape], 
                                                                  [dxscape, dyscape], 
                                                                  [cscape, dscape], 
                                                                  ["cinterp", "dinterp"],
                                                                  [r"c", r"d"]):
                
                # Extract the x- and y-coordinates corresponding to points 
                # in c- and d-scapes
                yxcoord = np.argwhere(~np.isnan(subscape))
                points = np.dstack(yxcoord)[0]
                x, y = points[1,:], points[0,:]
                # Compute gridpoints for evaluation
                scapex = np.arange(0,subscape.shape[1],1) 
                scapey = np.arange(0,subscape.shape[0],1)
                scapexarcs, scapeyarcs = (scapex - np.median(scapex))*.126, scapey*.126
                scapexarcs_grid, scapeyarcs_grid = np.meshgrid(scapexarcs, scapeyarcs)  
                # Determine the values of the points within c- and d-scape
                pixval = pixscape[y,x]
                val = subscape[y,x]
                
                
                # Select the x- and y-coordinates of c- and d-values on the current slit
                slitmask1 = (y>Oylow)*(y<Oyup)
                slitmask2 = (scapey>Oylow)*(scapey<Oyup)
                xslit, yslit, valslit = x[slitmask1], y[slitmask1], val[slitmask1]
                pixvalslit = pixval[slitmask1]
                pixoffxy.append(int(np.median(pixvalslit)))
                # Rescale to arcseconds
                xslitarcs = 0.126*(xslit - np.median(scapex))
                yslitarcs = 0.126*(yslit)
                scapexslitarcs, scapeyslitarcs = scapexarcs, scapeyarcs[slitmask2]
                scapexslitarcs_grid, scapeyslitarcs_grid = np.meshgrid(scapexslitarcs,
                                                                       scapeyslitarcs) 
                
                # Determine rms errors for the offset values
                '''
                valrmse = np.sqrt((valslit - np.median(valslit))**2)
                rmse = np.sqrt(np.mean((valslit - np.median(valslit))**2))
                # Mask values with a valrmse > 3*rmse
                valmask = (valrmse < 3*rmse)
                '''
                
                
                
                ########### SINGLE-SLIT UNIVARIATE cubic spline ###########
                # Univariate cubic splines using only slit datapoints
                duplmask = np.unique(xslit, return_index=True)[1]
                cubspl1d_f = interpolate.interp1d(xslitarcs[duplmask], valslit[duplmask],
                                                  kind='cubic')
                new_scapexslitarcs = np.arange(np.min(xslitarcs),np.max(xslitarcs),0.05)
                cubspl1d_v = cubspl1d_f(new_scapexslitarcs)
                cubspl1d_val = np.tile(cubspl1d_v, [len(scapeyslitarcs),1])
                
                # Contour the gridded data, plotting dots at the randomly spaced data points.
                levels = np.arange(np.min(cubspl1d_val),np.max(cubspl1d_val)+0.1,0.0025)
                CS = plt.contour(new_scapexslitarcs,scapeyslitarcs,cubspl1d_val,
                                 levels,linewidths=0)
                CS = plt.contourf(new_scapexslitarcs,scapeyslitarcs,cubspl1d_val, 
                                  levels, vmin=-1, vmax=1)
                # Plot data points
                plt.scatter(xslitarcs, yslitarcs, marker='o', s=50, c=valslit, 
                            cmap=CS.cmap, norm=CS.norm)
                plt.colorbar() # draw colorbar
                plt.xlabel(r"X [arcsec]", fontsize=20), plt.ylabel(r"Y [arcsec]", fontsize=20)
                plt.title(r"{}".format(plttitle), fontsize=26)
                plt.savefig(pltsavedir+"/{}_j{}k{}".format(pltsavetitle,j,k))
                #plt.show()  
                plt.close()   
                
                # Save to numpy savefiles
                polfun.savenp(cubspl1d_val, datasavedir+"/tpl{}/exp{}".format(j+1,k+1), 
                              "{}cubspl1d_i{}j{}k{}slitp{}".format(pltsavetitle,i+1,j+1,k+1,1))
                # Append cubic spline interpolation to list
                cdintp_xy.append(cubspl1d_val)
            
            # Append offset parameters
            pixoffxy__k.append(pixoffxy), cdintpxy__k.append(cdintp_xy)
            
            
            # Save axes ranges
            polfun.savenp(scapexarcs, npsavedir, "scapexarcs")
            polfun.savenp(scapeyarcs, npsavedir, "scapeyarcs")
            polfun.savenp(scapexslitarcs, datasavedir+"/tpl{}/exp{}/slitp2".format(j+1,k+1),
                          "scapexslitarcs_i{}j{}k{}slitp{}".format(i+1,j+1,k+1,2))
            polfun.savenp(scapeyslitarcs, datasavedir+"/tpl{}/exp{}/slitp2".format(j+1,k+1),
                          "scapeyslitarcs_i{}j{}k{}slitp{}".format(i+1,j+1,k+1,2))
            
            print("DEBUG cdoptoffs_star25(newnr18):\t{}, {}".format(cdintp_xy[0][21,302], 
                                                                    cdintp_xy[1][21,302]))
            
            
            # Determine backgrounds
            bkg_i, bgsavenames = [], ["E","O"]
            for s, im in enumerate([slitE,slitO]):
                bkg_mean, bkg_median, bkg_std = sigma_clipped_stats(im, sigma=3.0, iters=10)
                #print("DEBUG background:\t{}, {}, {}".format(bkg_mean, bkg_median, bkg_std))
                polfun.savefits(im, imsavedir+"/tpl{}/exp{}".format(j+1,k+1), 
                                "{}_j{}k{}".format(bgsavenames[s],j+1,k+1))
                polfun.savefits(im-bkg_median, 
                                imsavedir+"/tpl{}/exp{}".format(j+1, k+1), 
                                "{}-bg{}_j{}k{}b{}".format(bgsavenames[s],bgsavenames[s],
                                                           j+1,k+1,round(bkg_median,1)))
                
                bkg_i.append(bkg_median)
            bkgE, bkgO = bkg_i
            
            
            # Evaluate different background levels
            O__b, E__b, O_E__b, OplusE__b, O_E_grad__b = [], [], [], [], []
            for b, bkgoffs in enumerate(bkgoffsets): # Evaluate different background levels
                
                # Offset background level
                Ecorr, Ocorr = slitE-(bkgE-bkgoffs), slitO-(bkgO+bkgoffs)
                O__b.append(Ocorr), E__b.append(Ecorr)
                # Determine pixel-by-pixel Stokes parameters
                O_E, OplusE, O_E_grad = polfun.detslitdiffnorm([Ecorr,Ocorr], 
                                                          pixoffs=pixoffxy, suboffs_i=cdintp_xy, 
                                                          savefigs=True, 
                                                          plotdirec = pltsavedir+
                                                                "/tpl{}/exp{}".format(j+1,k+1),
                                                          imdirec = imsavedir+
                                                                "/tpl{}/exp{}".format(j+1,k+1),
                                                          suffix = "_j{}k{}slitp{}b{}".format(
                                                                    j+1,k+1,1,bkgoffs))
                
                # Diagnostic plot
                '''
                f, axarr = plt.subplots(2, sharex=True)
                O_Eplt= axarr[0].imshow(O_E, origin='lower', cmap='afmhot', vmin=-1, vmax=1)
                axarr[0].set_title('O_E')
                O_E_gradplt = axarr[1].imshow(O_E_grad, origin='lower', cmap='afmhot', 
                                              vmin=-1, vmax=1)
                axarr[1].set_title('O-E-grad')
                #plt.show()
                plt.close()
                '''
                
                # Append to lists
                O_E__b.append(O_E), OplusE__b.append(OplusE), O_E_grad__b.append(O_E_grad)
            
            
                # TODO Use the c- and d-point coordinates of the first exposures instead???
                # TODO -> NO! Since some of the edges of the slits are cut off during detslitdiffnorm
                # Determine slitwise stellar coordinates for stacking
                if b == (len(bkgoffsets)-1)//2: # Only run once
                    xcents = np.sort(xslit-chip_xyranges[0][0])
                    ycents = [m[0] for m in sorted(zip(yslit-Oylow, xslit), 
                              key=lambda l: l[1])]             
                    for starno, [xcent,ycent] in enumerate(zip(xcents,ycents)):
                        
                        # Determine approximate location
                        starcent = polfun.find_center([xcent,ycent], 
                                                      OplusE__b[b], 15)
                        
                        # APPROXIMATE ALLOCATION CHECK
                        '''
                        plt.figure()
                        plt.imshow(OplusE, origin='lower', cmap='rainbow')
                        plt.colorbar()
                        plt.scatter(starcent[0], starcent[1], s=30, c='b')
                        plt.show()
                        plt.close() 
                        '''        
                        
                        # Determine exposure offsets w.r.t. exposure (J,k)=(0,0)
                        if J == 0 and k == 0:
                            slitO_J1k1 = slitO
                            cutxmin, cutxmax = starcent[0]-35, starcent[0]+35
                            cutymin = max(0, starcent[1]-35)
                            cutymax = min(OplusE__b[b].shape[0]-1,starcent[1]+35)
                            cutoutcent = (starcent - np.rint([cutxmin,cutymin])).astype(int)  
                            
                        else:
                            cutout1 = slitO[cutymin:cutymax, cutxmin:cutxmax]
                            cutout2 = slitO_J1k1[cutymin:cutymax, cutxmin:cutxmax] 
                            
                            # GENERAL ALLOCATION CHECK
                            '''
                            plt.figure()
                            plt.imshow(cutout1, origin='lower', cmap='rainbow')
                            plt.colorbar()
                            plt.scatter(cutoutcent[0], cutoutcent[1], s=30, c='b')
                            plt.show()
                            plt.close()  
                            '''
                                   
                            [offsetopt, 
                             well, 
                             alignedim_well] = polfun.offsetopt_well([cutout1,cutout2],
                                                                     np.arange(-10,11),
                                                                     np.arange(-10,11), 
                                                                     cutoutcent, 10, 
                                                                     saveims=False) 
                            
                            # Diagnostic plot
                            '''
                            plt.imshow(alignedim_well, origin='lower', cmap='afmhot')
                            plt.colorbar()
                            plt.show()
                            plt.close()
                            '''
                        
                            # Append results to slitoffs
                            slitoffs_ks[k,starno] =  offsetopt
            ################ End iteration through all backgrounds ################
            
            # Append results to lists
            O_E__kb.append(O_E__b), OplusE__kb.append(OplusE__b), O_E_grad__kb.append(O_E_grad__b)
            O__kb.append(O__b), E__kb.append(E__b)
        ################ End iteration through all exposures ################
        
        
        
        # Sort lists
        imlists__Kb = []
        imlists__kb = [O__kb, E__kb, O_E__kb, OplusE__kb, O_E_grad__kb, 
                       exptimes_k, pixoffxy__k, cdintpxy__k, slitoffs_ks]
        for imlst in imlists:
            imlists__Kb.append(np.array([m[0] for m in 
                               sorted(zip(imlst, retangles_k), key=lambda l: l[1])]))
        [O__Kb, E__Kb, O_E__Kb, OplusE__Kb, O_E_grad__Kb, 
         exptimes_K, pixoffxy__K, cdintpxy__K, slitoffs_JKs[J]] = imlists__Kb
        
        '''
        # Sort exposure lists according to ascending retarder waveplate angle
        print("DEBUG retangles:\t {} \t {}".format(retangles_k,type(retangles_k[0])))
        slitoffs_JKs[J] = [m[0] for m in sorted(zip(slitoffs_ks, retangles_k), key=lambda l: l[1])]
        # Save intermediate results
        polfun.savenp(slitoffs_JKs[J], datasavedir+"/tpl{}".format(J+1), 
                      "slitoffs_i{}J{}Ks".format(i+1,J+1))
        '''#TODO REMOVE
        
        # Determine exposure-wise offsets
        print("DEBUG Qoffsxlst:\t {}".format(np.diff(slitoffs_JKs[J,[2,0],:,0],axis=0)))
        print("DEBUG Qoffsylst:\t {}".format(np.diff(slitoffs_JKs[J,[2,0],:,1],axis=0)))
        print("DEBUG Uoffsxlst:\t {}".format(np.diff(slitoffs_JKs[J,[3,1],:,0],axis=0)))
        print("DEBUG Uoffsylst:\t {}".format(np.diff(slitoffs_JKs[J,[3,1],:,1],axis=0)))
        Qoffsx = int(np.median(np.diff(slitoffs_JKs[J,[2,0],:,0],axis=0)))
        Qoffsy = int(np.median(np.diff(slitoffs_JKs[J,[2,0],:,1],axis=0)))
        Uoffsx = int(np.median(np.diff(slitoffs_JKs[J,[3,1],:,0],axis=0)))
        Uoffsy = int(np.median(np.diff(slitoffs_JKs[J,[3,1],:,1],axis=0)))
        QUoffsx = int(np.median(np.diff(slitoffs_JKs[J,[1,0],:,0],axis=0)))
        QUoffsy = int(np.median(np.diff(slitoffs_JKs[J,[1,0],:,1],axis=0)))
        
        # Append to lists
        O__JKb.append(O__Kb), E__JKb.append(E__Kb)
        O_E__JKb.append(O_E__Kb), OplusE__JKb.append(OplusE__Kb), O_E_grad__JKb.append(O_E_grad__Kb)
        # Determine the normalized slit difference
        slitdiffnorm__Kb = O_E_grad__Kb / OplusE__Kb
        
        
        # Determine the Stokes parameter and polarization degrees for all background levels
        if filtername == "b_HIGH":
            offsetangle = 2.*1.54
        elif filtername == "v_HIGH":
            offsetangle = 2.*1.8
        UQPphi__b, sigma_UQPphi__b = [], []
        for b in np.arange(len(bkgoffsets)):
            # Determine U/I, Q/I, P_L and phi_L
            UQPphi, sigma_UQPphi = polfun.detpol(slitdiffnorm__Kb[:,b], 
                                                 np.nanmedian(np.sqrt(OplusE__Kb[:,b]),axis=0), 
                                                 offsxy0__45=[Qoffsx,Qoffsy], 
                                                 offsxy22_5__67_5=[Uoffsx,Uoffsy], 
                                                 offsxy0__22_5=[QUoffsx, QUoffsy],
                                                 corran=offsetangle)
            # Append results to list
            UQPphi__b.append(UQPphi), sigma_UQPphi__b.append(sigma_UQPphi)
            # Save results to fits files
            fsavenames = ["U", "Q", "P", "sigmaU", "sigmaQ", "sigmaP"]
            for m in range(3):
                polfun.savefits(UQPphi[m], 
                                imsavedir+"/tpl{}/bkgoffs{}".format(j+1,bkgoffsets[b]), 
                                "{}_i{}j{}b{}".format(fsavenames[m],i+1,j+1,bkgoffsets[b]))
                polfun.savefits(sigma_UQPphi[m], 
                                imsavedir+"/tpl{}/bkgoffs{}".format(j+1,bkgoffsets[b]),
                                "sigma{}_i{}j{}b{}".format(fsavenames[3+m],i+1,j+1,bkgoffsets[b]))
        
        
        # Append results to template list
        UQPphi__Jb.append(UQPphi__b), sigma_UQPphi__Jb.append(sigma_UQPphi__b)
        # Update filter and retangle lists
        filters_Jk.append(filters_k), retangles_Jk.append(retangles_k)
        # Update offset lists
        pixoffxy__JK.append(pixoffxy__K), cdintpxy__JK.append(cdintpxy__K)
        # Add one to index
        J += 1
    
    # Save intermediate result
    if recompute_fluxlsts:
        polfun.savenp(UQPphi__Jb, datasavedir, "UQPphi__i{}Jb".format(i+1))
        polfun.savenp(sigma_UQPphi__Jb, datasavedir, "sigma_UQPphi__i{}Jb".format(i+1))
        polfun.savenp(O__JKb, datasavedir, "O__i{}Jb".format(i+1))
        polfun.savenp(E__JKb, datasavedir, "E__i{}Jb".format(i+1))
        polfun.savenp(OplusE__JKb, datasavedir, "OplusE__i{}JKb".format(i+1))
        polfun.savenp(O_E__JKb, datasavedir, "O_E__i{}JKb".format(i+1))
        polfun.savenp(O_E_grad__JKb, datasavedir, "O_E_grad__i{}JKb".format(i+1))   
        polfun.savenp(slitoffs_JKs, datasavedir, "slitoffs_i{}JKs".format(i+1)) 
        polfun.savenp(filters_Jk, datasavedir, "filters_i{}Jk".format(i+1)) 
        polfun.savenp(exptimes_JK, datasavedir, "exptimes_i{}JK".format(i+1)) 
        polfun.savenp(pixoffxy__JK, datasavedir, "pixoffxy_i{}JK".format(i+1)) 
        polfun.savenp(cdintpxy__JK, datasavedir, "cdinptxy__i{}JK".format(i+1)) 
        print("Saved flux lists and Stokes lists!")
    ###################### FINISHED CALCULATIONS
    
    
    
    
    
    # Extract box region parameters
    regfiles = 1*["/fils6"]
    regfiles = [imsavedir + regfile + "_{}.reg".format(tempitno+1) 
                for tempitno,regfile in enumerate(regfiles)]
    filorxy_lst, filorangles_lst, boxparms = [], [], []
    for regfileno, regfile in enumerate(regfiles):
        boxfile = open(regfile, 'r')
        
        parms = []
        for linenr, line in enumerate(boxfile):
            # Skip first three lines
            if linenr in range(3):
                continue
            
            # Extract centres
            templst = line.split(',')
            xcent, ycent = float(templst[0].split('(')[1]), float(templst[1])
            parms.append([xcent,ycent])
            sizex, sizey = float(templst[2]), float(templst[3])
            angle =  float(templst[4].split(")")[0])
            parms[linenr-3].extend([sizex, sizey ,angle])
        
        # Sort parameter list according to ascending X-coordinate
        parms = np.array(parms)
        parms = np.array([m[0] for m in sorted(zip(parms, parms[:,0]), key=lambda l: l[1])])
        if regfileno == 0:
            minrowind = (parms[:,1]).argmin()
            parms[minrowind::] = np.array([m[0] for m in sorted(zip(parms[minrowind::], 
                                                                    parms[minrowind::][:,1]), 
                                                                key=lambda l: l[1])])
        
        # Fit cubic splines through box centers
        tck, u = interpolate.splprep(parms[:,0:2].T, u=None, s=0.0)
        u_new = np.linspace(u.min(), u.max(), 1000)
        filorxy = np.array(interpolate.splev(u_new, tck, der=0)).T 
        filordxdy = np.diff(filorxy,axis=0) 
        filorangle = (180/np.pi) * np.array([polfun.detangle(vec,np.array([0,1])) 
                                             for vec in filordxdy]) # Deg
        filorangle = np.where(~((filordxdy[:,0]>0)*(filordxdy[:,1]>0)), filorangle, 180-filorangle)
        
        # Append results to lists
        boxparms.append(parms)
        filorxy_lst.extend(filorxy[0:-1]), filorangles_lst.extend(filorangle)
        
        # Diagnostic plot
        '''
        filorxy_arr = np.array(filorxy_lst)
        plt.scatter(filorxy_arr[:,0], filorxy_arr[:,1])
        plt.show()
        '''
    
    
    # Transform into arrays
    UQPphi__Jb, sigma_UQPphi__Jb = np.array([UQPphi__Jb, sigma_UQPphi__Jb])
    filters_Jk = np.array(filters_Jk)
    
    
    # Stack exposures
    fUQfil_mv = np.tile(np.nan, [2,2,21,2])
    Vmask_J, Bmask_J = (filters_Jk[:,0]=="v_HIGH"), (filters_Jk[:,0]=="b_HIGH")
    filMmasks__bf, boxmasks__bf, reghists__bfUQ = [], [], []
    boxavpLphiL__bf, boxvarpLphiL__bf, stackedUQPphi__bf, overlmasks__bf = [], [], [], []
    for b in np.arange(len(bkgoffsets)):
        filMmasks_f, overlmasks__f, boxmasks_f, reghists_fUQ = [], [], [], []
        boxavpLphiL__f, boxvarpLphiL__f, stackedUQPphi__f, smoothedUQPphi__f = [], [], [], [] 
        for filternr, [filtermask_J, filtername] in enumerate(zip([Vmask_J, Bmask_J],
                                                                  ["v_HIGH", "b_HIGH"])):
            '''
            # Stack raw O and E exposures
            O__K, E__K = [], []
            O__JK, E__JK = O__JKb[:,:,b], E__JKb[:,:,b]
            for K in range(4):
                tempoffs = pixoffxy__JK[:,K] - pixoffxy__JK[0,K] #TODO NO! THIS IS WRONG. You need to use the offsets between the individual templates
                O__K.append(polfun.stackim(O__JK[:,K], tempoffs, returnmask=False))
                E__K.append(polfun.stackim(E__JK[:,K], tempoffs, returnmask=False))
            '''
            
            
            # Select all templates corresponding to current filter
            U_J, Q_J = UQPphi__Jb[:,b,0], UQPphi__Jb[:,b,1]
            pL_J, phiL_J = UQPphi__Jb[:,b,2], UQPphi__Jb[:,b,3]
            slitoffs_JK = np.mean(slitoffs_JKs, axis=2)
            # Stack image templates for Stokes parameters
            stacked_lst, smoothed_lst, stackedsavenames = [], [], ["U","Q","pL","phiL"]
            for tempno, tempim in enumerate([U_J,Q_J,pL_J,phiL_J]):
                [tempimstacked, overlmask, 
                 embuplcorn, newembuplcorn] = polfun.stackim(tempim[filtermask_J], 
                                                             slitoffs_JK[filtermask_J,0],
                                                             returnmask=True)
                overlmask = overlmask[embuplcorn[0]:embuplcorn[0]+tempim.shape[1],
                                      embuplcorn[1]:embuplcorn[1]+tempim.shape[2]]
                #print("DEEEEBUG:\t{}".format(overlmask.shape))
                # Save results as fitsfiles
                polfun.savefits(tempimstacked, imsavedir+"/bkgoffs{}".format(bkgoffsets[b]), 
                                "{}_stacked__i{}b{}f{}".format(stackedsavenames[tempno], i+1, 
                                                               bkgoffsets[b],filternr+1))
                # Append results to lists
                stacked_lst.append(tempimstacked)
            
            overlmasks__f.append(overlmask)   
            Q_stacked, Ustacked, pL_stacked, phiL_stacked = stacked_lst 
            stackedUQPphi__f.append(stacked_lst), smoothedUQPphi__f.append(smoothed_lst)
            # TODO Determine stacked array errors #TODO MAYBE REMOVE SMOOTHED versions?????
            
            
            # Save results
            ''' 
            polfun.savenp(U_stacked, datasavedir, "U_stacked__i{}f{}".format(i+1,filternr+1))      
            polfun.savenp(Q_stacked, datasavedir, "Q_stacked__i{}f{}".format(i+1,filternr+1))
            polfun.savenp(pL_stacked, datasavedir, "pL_stacked__i{}f{}".format(i+1,filternr+1))      
            polfun.savenp(phiL_stacked, datasavedir, "phiL_stacked__i{}f{}".format(i+1,filternr+1))
            polfun.savefits(Q_stacked, imsavedir+"/bkgoffs{}".format(bkgoffsets[b]), 
                            "Q_stacked__i{}b{}f{}".format(i+1,bkgoffsets[b],filternr+1))
            polfun.savefits(pL_stacked, imsavedir+"/bkgoffs{}".format(bkgoffsets[b]),  
                            "pL_stacked__i{}b{}f{}".format(i+1,bkgoffsets[b],filternr+1))
            polfun.savefits(phiL_stacked, imsavedir+"/bkgoffs{}".format(bkgoffsets[b]),  
                            "phiL_stacked__i{}b{}f{}".format(i+1,bkgoffsets[b],filternr+1))
            '''
            print("Saved stacked results!")
            
            
            # Create box region histogram statistics and mastermask
            boxparms_all = np.concatenate(boxparms)
            boxcents = boxparms_all[:,0:2].astype(int) #pix
            boxsizes = boxparms_all[:,2:4].astype(int) #pix
            boxrots = boxparms_all[:,4] #deg
            I = np.abs(polfun.mask2d(np.array(OplusE__JKb)[0,0,b], overlmask))
            polfun.savefits(I, imsavedir+"/bkgoffs{}".format(bkgoffsets[b]),
                            "I__i{}b{}f{}".format(i+1,bkgoffsets[b],filternr+1))
            
            # Recalibrate the box centers and the stacking mask
            '''
            boxcents_cal = np.array(boxcents) - (pLnewembuplcorn-pLoldembuplcorn)[[1,0]]
            stackmask_cal = pLmask[pLoldembuplcorn[0]:pLoldembuplcorn[0]+I.shape[0],
                                   pLoldembuplcorn[1]:pLoldembuplcorn[1]+I.shape[1]]
            '''
            
            # Determine regional counts
            boxavpLphiL_lst, boxvarpLphiL_lst, boxmask_lst, valmask_lst = [], [], [], []
            for boxnr, [boxcent, boxsize, boxrot] in enumerate(zip(boxcents,boxsizes,boxrots)):
                # Determine the regional masks
                pLmasked, boxmask = polfun.createrectmask(pL_stacked, 
                                                          boxcent, boxsize, (np.pi/180)*boxrot)
                boxmask_lst.append(~boxmask)
                # Extract counts
                boxpL, boxphiL = pL_stacked[boxmask], phiL_stacked[boxmask]
                # Append box averages and variances
                
                boxavpLphiL_lst.append(np.mean(np.array([boxpL, boxphiL]), axis=1))
                boxvarpLphiL_lst.append(np.var(np.array([boxpL, boxphiL]), axis=1))
                
                #print("Mean, var:\t{} , {}".format(mean, var))
                valmask = pL_stacked>0.008 #(mean-np.sqrt(var)) #TODO TODO TODO CHANGED 18-08-17
                valmask_lst.append(~valmask)
                
            
            # Write box mean results to table
            polfun.writetable(np.array(boxavpLphiL_lst), np.sqrt(boxvarpLphiL_lst), 
                         ["Region {}".format(tempind) for tempind in range(len(boxavpLphiL_lst))],
                         ["", r"$P_L$", r"\phi_L"], tabledir+"/bkgoffs{}".format(bkgoffsets[b]),
                         "boxPlPhil__f{}".format(filternr+1), rounddig=3, overwrite=True)
            
            # Store all box masks
            boxavpLphiL__f.append(boxavpLphiL_lst), boxvarpLphiL__f.append(boxavpLphiL_lst)
            boxmasks_f.append(boxmask_lst)
            
            # Form 2Dmastermask (contains np.nan where non-filament and 1 where filament)
            valMmask = ~(np.prod(valmask_lst, axis=0).astype(int))
            boxMmask = ~(np.prod(boxmask_lst, axis=0).astype(int))
            filMmask = np.where(boxMmask*valMmask==1, boxMmask*valMmask, np.nan)
            filMmasks_f.append(filMmask)
            
        # Append results for currently evaluated background to lists
        '''
        boxavpLphiL__bf.append(boxavpLphiL__f), boxvarpLphiL__bf.append(boxavpLphiL__f) 
        stackedUQPphi__bf.append([stackedUQPphi__f]), overlmasks__bf.append(overlmasks__f)
        '''
        
        
        
        
        
        
        # Plot I, Qbar, Ubar next to each other for both b and v
        exptimev, exptimeb = exptimes_JK[[0,4],0]
        # Define plot arrays
        axarrs, imswinds = np.tile(None,[2,2,3]), np.tile(None,[2,2,3])
        gs1, gs2 = gridspec.GridSpec(8,3), gridspec.GridSpec(8,3)
        gs1.update(right=0.95, top=0.95, left=0.05, bottom=0.55, wspace=0.3, hspace=2)
        gs2.update(right=0.95, top=0.45, left=0.05, bottom=0.05, wspace=0.3, hspace=2)
        # Define plotdata array
        stackshape = stackedUQPphi__f[0][1].shape
        phiplot1 = stackedUQPphi__f[0][3] * filMmasks_f[0][0:stackedUQPphi__f[0][3].shape[0],
                                                          0:stackedUQPphi__f[0][3].shape[1]]
        phiplot2 = stackedUQPphi__f[1][3] * filMmasks_f[1][0:stackedUQPphi__f[1][3].shape[0],
                                                          0:stackedUQPphi__f[1][3].shape[1]]
        pltlst = [[ [np.arcsinh(OplusE__JKb[0][0][b][0:stackshape[0],0:stackshape[1]]/exptimev), 
                     stackedUQPphi__f[0][1], stackedUQPphi__f[0][0]], 
                    [np.nan, stackedUQPphi__f[0][2], phiplot1] ],

                  [ [np.arcsinh(OplusE__JKb[4][0][b][0:stackshape[0],0:stackshape[1]]/exptimeb), 
                     stackedUQPphi__f[1][1], stackedUQPphi__f[1][0]], 
                    [np.nan, stackedUQPphi__f[1][2], phiplot2] ]]
        plttitles = [ [r"I",r"Q/I",r"U/I"], [None, r"$P_L$", r"$\phi_L$"] ]
        plttitles = [plttitles,plttitles]
        
        # Set color maps
        colormaps = np.tile('afmhot', [2,2,3])
        colormaps[:,:,0], colormaps[:,1,2] = 'Greys', 'jet'
        # Set color range limits
        vminarr, vmaxarr = np.tile(-.015, [2,2,3]), np.tile(.015, [2,2,3])
        vminarr[:,0,0] = -10/np.array([exptimev,exptimev])
        vmaxarr[:,0,0] = 150/np.array([exptimev,exptimev])
        vminarr[:,1,2], vmaxarr[:,1,2] = 0, 180
        
        # Plot data
        fig = plt.figure(figsize=(24,18))
        for gsnr, grspec in enumerate([gs1,gs2]):
            for [pltrownr,pltcolnr], [gsrow, gscol] in zip(cartprod(range(2),range(3)), 
                                                           cartprod([0,4],range(3))):
                if not (pltrownr == 1 and pltcolnr == 0): 
                    axarrs[gsnr][pltrownr][pltcolnr] = fig.add_subplot(grspec[gsrow:gsrow+4,gscol])
                    imswinds[gsnr][pltrownr][pltcolnr] = axarrs[gsnr][pltrownr][pltcolnr].imshow(
                                                    pltlst[gsnr][pltrownr][pltcolnr][:,734:936],
                                                    origin='lower', 
                                                    cmap=colormaps[gsnr,pltrownr,pltcolnr],
                                                    vmin=vminarr[gsnr,pltrownr,pltcolnr],
                                                    vmax=vmaxarr[gsnr,pltrownr,pltcolnr], 
                                                    extent=[np.min(scapexslitarcs[734:936]),
                                                            np.max(scapexslitarcs[734:936]),
                                                    np.min(scapeyslitarcs),np.max(scapeyslitarcs)])
                    
                    if pltrownr != 0:
                        axarrs[gsnr][pltrownr][pltcolnr].set_xlabel(r"X [arcsec]", fontsize=20)
                    axarrs[gsnr][pltrownr][pltcolnr].set_ylabel(r"Y [arcsec]", fontsize=20)
                    axarrs[gsnr][pltrownr][pltcolnr].set_title(r"{}".format(
                                                               plttitles[gsnr][pltrownr][pltcolnr]),
                                                               fontsize=26)
                    if pltrownr == 0 and pltcolnr == 0:
                        axarrs[gsnr][pltrownr][pltcolnr].set_xlabel(r"X [arcsec]", fontsize=20)
                    '''
                    if pltrownr != 1:
                        plt.setp(axarrs[gsnr,pltrownr,pltcolnr].get_xticklabels(), visible=False) 
                    if pltcolnr != 0:
                        plt.setp(axarrs[gsnr,pltcolnr].get_yticklabels(), visible=False) 
                    else:
                        axarrs[]
                    ''' #TODO REMOVE tick labels for non-boundary windows?
                
            # DoLP color bar
            cbaxes1, cbaxes2 = plt.subplot(grspec[5,0]), plt.subplot(grspec[6,0])
            cb1 = plt.colorbar(imswinds[0][0][1], orientation='horizontal', 
                               cax = cbaxes1)#, ticks=np.round(np.arange(-.2,.2,.05),2)) 
            #fig.colorbar(imswinds[0][0][1], cax=cbaxes1)  
            cb2 = plt.colorbar(imswinds[0][1][2], orientation='horizontal', 
                               cax = cbaxes2)#, ticks=np.round(np.arange(-180,180,60),0))  
            #fig.colorbar(imswinds[0][1][2], cax=cbaxes2)   
            
        fig.text(0.015, 0.75, r"v_HIGH", fontsize=26, ha="center", va="center", rotation=90) 
        fig.text(0.015, 0.25, r"b_HIGH", fontsize=26, ha="center", va="center", rotation=90) 
        #plt.savefig(pltsavedir+"/UQ_i{}fALLv2".format(i+1,filternr+1))
        plt.savefig(pltsavedir+"/IQbarUbarPlPhil_b{}f.png".format(bkgoffsets[b],bkgoffsets[b]))
        #plt.show()
        plt.close()
        
        
        
        
        
        # Define intensities in both filters   
        I__f = np.array([OplusE__JKb[0][0][b]/exptimev, OplusE__JKb[4][0][b]/exptimeb])
        # Select filaments for plots
        for filsel, filselname in zip([[0,863],[863,-1],[0,-1]], ['LEFT','RIGHT','']): 
            # Scatter plot Theta_orient versus phi_L
            fig, ax = plt.subplots(1)
            filternames, filtersymbs = ["v_HIGH", "b_HIGH"], ['v', 'o']
            for filternr, filMmask in enumerate(filMmasks_f):        
                
                # Select filament A_V values
                kernel = Gaussian2DKernel(stddev=9)
                smoothedI = convolve(I__f[filternr], kernel)
                Av = I__f[filternr] - smoothedI #2.5*np.log10(smoothedI/I__f[filternr])
                # Reshape to Pl-shape
                Av = polfun.mask2d(Av, overlmasks__f[filternr])
                
                # Save results
                polfun.savefits(smoothedI, imsavedir+"/bkgoffs{}".format(bkgoffsets[b]),
                                "smoothedI__b{}f{}".format(bkgoffsets[b],filternr+1))            
                polfun.savefits(Av, imsavedir+"/bkgoffs{}".format(bkgoffsets[b]),
                                "Av__b{}f{}".format(bkgoffsets[b],filternr+1))
                
                # Select filament polarization degree values and extinction values
                pL = stackedUQPphi__f[filternr][2]
                pLfil = pL*filMmask[0:pL.shape[0],0:pL.shape[1]]
                pLfil = np.where(pLfil>=0.009, pLfil, np.nan)[:,filsel[0]:filsel[1]] # select only significant pL's
                Avfil = (Av*filMmask[0:Av.shape[0],0:Av.shape[1]])[:,filsel[0]:filsel[1]]
                
                avim = ax.scatter(Avfil.flatten(), 100*pLfil.flatten(), 
                                  label=filternames[filternr], marker=filtersymbs[filternr], 
                                  c=100*pLfil.flatten(), vmin=-1.5, vmax=1.5, cmap='afmhot')
            
            plt.colorbar(avim)
            ax.set_xlabel(r"$A_V$ [mag]", fontsize=20)        
            ax.set_ylabel(r"$P_L$ [$\%$]", fontsize=20)
            ax.legend(loc='best')
            plt.savefig(pltsavedir+"/{}filsAvPl_b{}f.png".format(filselname, bkgoffsets[b]))
            #plt.show()
            plt.close()     
            


        # Scatter plot Theta_orient versus phi_L
        fig, ax = plt.subplots(1)
        filternames, filtersymbs = ["v_HIGH", "b_HIGH"], ['v', 'o']
        #fitnames, fitsymbs = ["v_HIGH", "b_HIGH"], ['k-', 'k--']
        for filternr, filMmask in enumerate(filMmasks_f):
            
            # Select filament polarization angle values
            phiL = stackedUQPphi__f[filternr][3]
            phiLfil = phiL*filMmask[0:phiL.shape[0],0:phiL.shape[1]]
            
            # Bin the polarization angles
            phiLfil_binned = phiLfil
            '''
            phiLfil_binned = np.tile(np.nan,phiLfil.shape)
            for row, col in cartprod(np.arange(phiLfil.shape[0]),np.arange(phiLfil.shape[1])):
                phiLfil_binned[row,col] = np.nanmedian(phiLfil[row-1:row+1,col-1:col+1])
            '''
            
            # Determine orientation angles corresponding to each filament pixel
            filxy = np.fliplr(np.argwhere(phiLfil_binned != np.nan)) #row, col
            filangles = np.tile(np.nan, phiLfil_binned.shape)
            for xy in filxy:
                
                if np.isnan(phiLfil[xy[1],xy[0]]):
                    continue
                
                splevdists = np.sqrt(np.sum((filorxy_lst - xy)**2, axis=1))
                minslevdistarg = splevdists.argmin()
                filangles[xy[1],xy[0]] = filorangles_lst[minslevdistarg]
                    
            
            #filangles = np.where(filangles >= 0, filangles, filangles+180) 
            #avim = ax1.scatter() #TODO INCLUDE RADIAL VERSION IN THIS FOR LOOP
            
            
            # Include linear regression line
            '''
            nanmask = np.isnan(phiLfil_binned.flatten())
            phiLfil_sorted = np.sort(phiLfil_binned.flatten()[~nanmask])
            filangles_sorted = [m[0] for m in sorted(zip(filangles.flatten()[~nanmask], 
                                phiLfil_binned.flatten()[~nanmask]), key=lambda l: l[1])]
            linfit = np.polyfit(phiLfil_sorted, np.array(filangles_sorted), 1)
            linfit_fn = np.polyval(linfit, np.linspace(0,180,1000)) 
            ''' #TODO REMOVE
            # Plot results
            orim = ax.scatter(phiLfil_binned.flatten(), filangles.flatten(), 
                              label=filternames[filternr], marker=filtersymbs[filternr], 
                              c=phiLfil.flatten(), vmin=0, vmax=180, cmap='jet')
            linfitplot = ax.plot(np.linspace(0,180,1000), np.linspace(0,180,1000), 'k-') 
                                 #linfit_fn, fitsymbs[filternr], label=fitnames[filternr])
            # Save results
            polfun.savefits(filangles, imsavedir, "filangles_f{}".format(filternr+1))
            polfun.savefits(phiLfil, imsavedir, "phiLfil_f{}".format(filternr+1))
            
            
            # Create boxed polarization angle layouts
            '''
            boxedphiL = np.tile(np.nan, phiLfil_binned.shape)
            Utemp, Qtemp = stackedUQPphi__f[filternr][0:2]        
            for boxmask in boxmasks_f[filternr]:
                
                # Determine median U and Q in boxes
                Uboxtemp, Qboxtemp = np.median(Utemp[~boxmask]), np.median(Qtemp[~boxmask])
                boxedphiL[~boxmask] = (180/np.pi) * (0.5*np.arctan(Uboxtemp/Qboxtemp))
            
            polfun.savefits(boxedphiL, imsavedir, "boxedphiL_f{}".format(filternr))
            '''

        plt.colorbar(orim)
        ax.set_xlim(xmin=0), ax.set_ylim(ymin=0)
        ax.set_xlabel(r"$\phi_L [^{\circ}]$", fontsize=20)
        ax.set_ylabel(r"$\Theta [^{\circ}]$", fontsize=20)
        ax.legend(loc='best')
        plt.savefig(pltsavedir+"/ThetaPhil_b{}f.png".format(bkgoffsets[b]))
        #plt.show()
        plt.close()
        
        
        
        
        '''
        # Scatter plot Theta_radial vs phi_L
        fig, ax = plt.subplots(1)
        filternames, filtersymbs = ["v_HIGH", "b_HIGH"], ['v', 'o']
        for filternr, filMmask in enumerate(filMmasks_f):
            
            # Select filament polarization angle values
            phiL = stackedUQPphi__f[filternr][3]
            phiLfil = phiL*filMmask[0:phiL.shape[0],0:phiL.shape[1]]
            
            # Bin the polarization angles
            phiLfil_binned = np.tile(np.nan,phiLfil.shape)
            for row, col in cartprod(np.arange(phiLfil.shape[0]),np.arange(phiLfil.shape[1])):
                phiLfil_binned[row,col] = np.nanmedian(phiLfil[row-1:row+1,col-1:col+1])
            
            # Determine orientation angles corresponding to each filament pixel
            filxy = np.fliplr(np.argwhere(phiLfil_binned != np.nan)) #row, col
            filangles = np.tile(np.nan, phiLfil_binned.shape)
            for xy in filxy:
                
                if np.isnan(phiLfil[xy[1],xy[0]]):
                    continue
                
                splevdists = np.sqrt(np.sum((filorxy - xy)**2, axis=1))
                minslevdistarg = splevdists.argmin()
                if minslevdistarg == filorangle.shape[0]:
                    filangles[xy[1],xy[0]] = filorangle[-1]
                else:
                    filangles[xy[1],xy[0]] = filorangle[minslevdistarg]
                    
            
            filangles = np.where(filangles >= 0, filangles, filangles+180) 
            orim = ax.scatter(filangles.flatten(), phiLfil_binned.flatten(),
                              label=filternames[filternr], marker=filtersymbs[filternr], 
                              c=filangles.flatten(), vmin=0, vmax=180, cmap='jet')
                       
            polfun.savefits(filangles, imsavedir, "filangles_f{}".format(filternr))
            polfun.savefits(phiLfil, imsavedir, "phiLfil_f{}".format(filternr))
            
            
            
            # Create boxed polarization angle layouts
            boxedphiL = np.tile(np.nan, phiLfil_binned.shape)
            Utemp, Qtemp = stackedUQPphi__f[filternr][0:2]        
            for boxmask in boxmasks_f[filternr]:
                
                # Determine median U and Q in boxes
                Uboxtemp, Qboxtemp = np.median(Utemp[~boxmask]), np.median(Qtemp[~boxmask])
                boxedphiL[~boxmask] = (180/np.pi) * (0.5*np.arctan(Uboxtemp/Qboxtemp))
            
            polfun.savefits(boxedphiL, imsavedir, "boxedphiL_f{}".format(filternr))
        
            
        plt.colorbar(orim)
        ax.set_xlabel(r"$\Theta [^{\circ}]$", fontsize=20)
        ax.set_ylabel(r"$\phi_L [^{\circ}]$", fontsize=20)
        plt.legend(loc='best')
        plt.savefig(pltsavedir+"/test2.png")
        #plt.show()
        plt.close()    
        '''
            
    
    
    
    
    
    
    
    
    
    
    
    
    
