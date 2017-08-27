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
#from photutils import Background2D, SigmaClip, MedianBackground

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib import cm
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


# Define the x- and y-ranges corresponding to the chip
chip_xyranges = [[183,1868],[25,934]]
# Range of aperture radii
r_range = np.arange(1, 16) #[pixels]
# Pixel scale
pixscale = 0.126 #[arcsec/pixel]
# Boolean variables for turning on/off the computations
detsplines1d, detsplines2d = False, False
detoverlaps1d, detoverlaps2d = True, True
recompute_fluxlsts = False


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





# Compute the offset wells for all stars within all exposures of all templates
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
    filters_Jk, retangles_Jk, J = [], [], 0
    slitoffs_JKs, exptimes_JK = np.zeros([11,4,5,2]), np.zeros([11,4])
    OplusE__JK, O_E__JK, O_E_grad__JK, UQPphi_J, sigma_UQPphi_J = [], [], [], [], []
    for j, tpl_name in enumerate(tpl_dirlst):
        print("\tTPL:\t{}".format(tpl_name))
        tpl_dir = objdir + '/' + tpl_name
        print("DEBUG:\t{}".format(J))
        
        
        # Load or recompute intermediate results
        if not recompute_fluxlsts:
            UQPphi_J = np.load(datasavedir+"/UQPphi__i{}JK.npy".format(i+1))
            sigma_UQPphi_J = np.load(datasavedir+"/sigma_UQPphi__i{}JK.npy".format(i+1))
            OplusE__JK = np.load(datasavedir+"/OplusE__i{}JK.npy".format(i+1))
            O_E__JK = np.load(datasavedir+"/O_E__i{}JK.npy".format(i+1))
            O_E_grad__JK = np.load(datasavedir+"/O_E_grad__i{}JK.npy".format(i+1))
            print("LOADED flux and stokes lists!")
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
        filters_k, retangles_k, exptimes_k = [], [], []
        O_Elst, OplusElst, O_E_gradlst, xcoords_ks, ycoords_ks = [], [], [], [], []
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


            # Subtract backgrounds
            imcorr_lst, bgsavenames = [], ["E","O"]
            for s, im in enumerate([slitE,slitO]):
                bkg_mean, bkg_median, bkg_std = sigma_clipped_stats(im, sigma=3.0, iters=10)
                print("DEBUG background:\t{}, {}, {}".format(bkg_mean, bkg_median, bkg_std))
                
                polfun.savefits(im, imsavedir+"/tpl{}/exp{}".format(j+1,k+1), 
                                "{}_j{}k{}".format(bgsavenames[s],j+1,k+1))
                #polfun.savefits(bkg.background, imsavedir+"/tpl{}/exp{}".format(j+1,k+1), 
                #                "bg{}_j{}k{}".format(bgsavenames[s],j+1,k+1))
                polfun.savefits(im - bkg_median, imsavedir+"/tpl{}/exp{}".format(j+1,k+1), 
                                "{}-bg{}_j{}k{}".format(bgsavenames[s],bgsavenames[s],j+1,k+1))
                
                imcorr_lst.append(im - bkg_median)#bkg.background)
            Ecorr, Ocorr = imcorr_lst
            
            
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
            pixoffxy, cdintp_xy, xcoords_ts, ycoords_ts = [], [], [], []
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
                # Append sorted x- and y-coordinates to slits
                xcoords_ts.append(np.sort(xslit)), ycoords_ts.append(np.sort(yslit))
                # Rescale to arcseconds
                xslitarcs = 0.126*(xslit - np.median(scapex))
                yslitarcs = 0.126*(yslit)
                scapexslitarcs, scapeyslitarcs = scapexarcs, scapeyarcs[slitmask2]
                scapexslitarcs_grid, scapeyslitarcs_grid = np.meshgrid(scapexslitarcs,
                                                                       scapeyslitarcs) 
                
                # Determine rms errors for the offset values
                valrmse = np.sqrt((valslit - np.median(valslit))**2)
                rmse = np.sqrt(np.mean((valslit - np.median(valslit))**2))
                # Mask values with a valrmse > 3*rmse
                valmask = (valrmse < 3*rmse)
                
                
                
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
            
            
            
            # Save axes ranges
            polfun.savenp(scapexarcs, npsavedir, "scapexarcs")
            polfun.savenp(scapeyarcs, npsavedir, "scapeyarcs")
            polfun.savenp(scapexslitarcs, datasavedir+"/tpl{}/exp{}/slitp2".format(j+1,k+1),
                          "scapexslitarcs_i{}j{}k{}slitp{}".format(i+1,j+1,k+1,2))
            polfun.savenp(scapeyslitarcs, datasavedir+"/tpl{}/exp{}/slitp2".format(j+1,k+1),
                          "scapeyslitarcs_i{}j{}k{}slitp{}".format(i+1,j+1,k+1,2))
            
            print("DEBUG cdoptoffs_star25(newnr18):\t{}, {}".format(cdintp_xy[0][21,302], 
                                                                    cdintp_xy[1][21,302]))
                
            
            
            # Determine pixel-by-pixel Stokes parameters
            O_E, OplusE, O_E_grad = polfun.detslitdiffnorm([Ecorr,Ocorr], 
                                                      pixoffs=pixoffxy, suboffs_i=cdintp_xy, 
                                                      savefigs=True, 
                                                      plotdirec = pltsavedir+
                                                            "/tpl{}/exp{}".format(j+1,k+1),
                                                      imdirec = imsavedir+
                                                            "/tpl{}/exp{}".format(j+1,k+1),
                                                      suffix = "_j{}k{}slitp{}".format(j+1,k+1,1))
        
        
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
            O_Elst.append(O_E), OplusElst.append(OplusE), O_E_gradlst.append(O_E_grad)
            
            
            # Determine slitwise stellar coordinates for stacking
            xcents = np.sort(xslit-chip_xyranges[0][0])
            ycents = [m[0] for m in sorted(zip(yslit-Oylow, xslit), key=lambda l: l[1])]             
            for starno, [xcent,ycent] in enumerate(zip(xcents,ycents)):
                
                starcent = polfun.find_center([xcent,ycent], OplusE, 15)
                #slitx_ks[k,starno], slity_ks[k,starno] = starcent #TODO REMOVE
                
                # GENERAL ALLOCATION CHECK
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
                    cutymin, cutymax = max(0, starcent[1]-35), min(OplusE.shape[0]-1, starcent[1]+35)
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
                           
                    offsetopt, well, alignedim_well = polfun.offsetopt_well([cutout1,cutout2], 
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
            
        
        # Sort exposure lists according to ascending retarder waveplate angle
        print("DEBUG retangles:\t {} \t {}".format(retangles_k,type(retangles_k[0])))
        slitoffs_JKs[J] = [m[0] for m in sorted(zip(slitoffs_ks, retangles_k), key=lambda l: l[1])]
        #slity_JKs[J] = [m[0] for m in sorted(zip(slity_ks, retangles_k), key=lambda l: l[1])] #TODO REMOVE
        # Save intermediate results
        polfun.savenp(slitoffs_JKs[J], datasavedir+"/tpl{}".format(J+1), 
                      "slitoffs_i{}J{}Ks".format(i+1,J+1))
        
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
        
        # Sort slit difference lists
        O_E__K = [m[0] for m in sorted(zip(O_Elst, retangles_k), key=lambda l: l[1])] 
        OplusE__K = [m[0] for m in sorted(zip(OplusElst, retangles_k), key=lambda l: l[1])] 
        O_E_grad__K = [m[0] for m in sorted(zip(O_E_gradlst, retangles_k), key=lambda l: l[1])] 
        exptimes_JK[J] = [m[0] for m in sorted(zip(exptimes_k, retangles_k), key=lambda l: l[1])] 
        # Append to lists
        O_E__JK.append(O_E__K), OplusE__JK.append(OplusE__K), O_E_grad__JK.append(O_E_grad__K)
        
        print("DEBUG O_Eshapes (-,+,-_-grad):\t{}, {}, {}".format(np.array(O_E__K).shape,
                                                              np.array(OplusE__K).shape,
                                                              np.array(O_E_grad__K).shape))
        slitdiffnorm__K = np.array(O_E_grad__K) / np.array(OplusE__K)
        
        # Determine the Stokes parameter and polarization degrees
        if filtername == "b_HIGH":
            offsetangle = 2.*1.54
        elif filtername == "v_HIGH":
            offsetangle = 2.*1.8
        UQPphi, sigma_UQPphi = polfun.detpol(slitdiffnorm__K, 
                                             np.nanmedian(np.sqrt(OplusE__K),axis=0), 
                                             offsxy0__45=[Qoffsx,Qoffsy], 
                                             offsxy22_5__67_5=[Uoffsx,Uoffsy], 
                                             offsxy0__22_5=[QUoffsx, QUoffsy],
                                             corran=offsetangle)
        # Append results to list
        UQPphi_J.append(UQPphi), sigma_UQPphi_J.append(sigma_UQPphi)
        # Save results to fits files
        polfun.savefits(UQPphi[1], imsavedir+"/tpl{}".format(j+1), "Q_i{}j{}".format(i+1,j+1))
        polfun.savefits(UQPphi[0], imsavedir+"/tpl{}".format(j+1), "U_i{}j{}".format(i+1,j+1))
        polfun.savefits(UQPphi[2], imsavedir+"/tpl{}".format(j+1), "P_i{}j{}".format(i+1,j+1))
        polfun.savefits(sigma_UQPphi[1], imsavedir+"/tpl{}".format(j+1), 
                        "sigmaQ_i{}j{}".format(i+1,j+1))
        polfun.savefits(sigma_UQPphi[0], imsavedir+"/tpl{}".format(j+1), 
                        "sigma_U_i{}j{}".format(i+1,j+1))
        polfun.savefits(sigma_UQPphi[2], imsavedir+"/tpl{}".format(j+1), 
                        "sigma_P_i{}j{}".format(i+1,j+1))
        
        
        
        # Update filter and retangle lists
        filters_Jk.append(filters_k), retangles_Jk.append(retangles_k)
        # Add one to index
        J += 1
    
    # Save intermediate results
    if recompute_fluxlsts:
        polfun.savenp(UQPphi_J, datasavedir, "UQPphi__i{}JK".format(i+1))
        polfun.savenp(sigma_UQPphi_J, datasavedir, "sigma_UQPphi__i{}JK".format(i+1))
        polfun.savenp(OplusE__JK, datasavedir, "OplusE__i{}JK".format(i+1))
        polfun.savenp(O_E__JK, datasavedir, "O_E__i{}JK".format(i+1))
        polfun.savenp(O_E_grad__JK, datasavedir, "O_E_grad__i{}JK".format(i+1))   
        polfun.savenp(slitoffs_JKs, datasavedir, "slitoffs_i{}JKs".format(i+1)) 
        polfun.savenp(filters_Jk, datasavedir, "filters_i{}Jk".format(i+1)) 
        polfun.savenp(exptimes_JK, datasavedir, "exptimes_i{}JK".format(i+1)) 
        print("Saved flux lists and Stokes lists!")
    ###################### FINISHED CALCULATIONS
    
    
    
    
    
    # Create filament orientation polynomial
    filcontsfile = open(imsavedir+"/filcontours.reg", 'r')
    boxfile = open(imsavedir+"/fils6.reg", 'r')
    parms_lst = []
    for filenr, f in enumerate([filcontsfile,boxfile]):
        parms = []
        for linenr, line in enumerate(f):
            # Skip first three lines
            if linenr in range(3):
                continue
            
            # Extract centres
            templst = line.split(',')
            xcent, ycent = float(templst[0].split('(')[1]), float(templst[1])
            parms.append([xcent,ycent])
            if filenr == 1:
                sizex, sizey = float(templst[2]), float(templst[3])
                angle =  float(templst[4].split(")")[0])
                parms[linenr-3].extend([sizex, sizey ,angle])
        
        # Sort parameter list according to ascending X-coordinate
        parms = np.array(parms)
        parms = np.array([m[0] for m in sorted(zip(parms, parms[:,0]), key=lambda l: l[1])])
        minrowind = (parms[:,1]).argmin()
        parms[minrowind::] = np.array([m[0] for m in sorted(zip(parms[minrowind::], 
                                                                parms[minrowind::][:,1]), 
                                                            key=lambda l: l[1])])
        parms_lst.append(np.array(parms))

    # Extract lists
    contparms, boxparms = parms_lst   
    
    
    
    
    
    # Transform into arrays
    UQPphi_J, sigma_UQPphi_J = np.array([UQPphi_J, sigma_UQPphi_J])
    filters_Jk = np.array(filters_Jk)
    # SHIFTS BETWEEN TEMPLATES ARE ZERO?!
    '''
    #xcoords_JKs, ycoords_JKs = np.array(xcoords_JKs), np.array(ycoords_JKs) #TODO REMOVE
    for coords_JKs in [xcoords_JKs, ycoords_JKs]:
        shiftsQ_J = np.cumsum(np.median(np.diff(coords_JKs[:,[0,2],:], axis=1), axis=[1,2]))
        shiftsU_J = np.cumsum(np.median(np.diff(coords_JKs[:,[1,3],:], axis=1), axis=[1,2]))
        print("DEBUG shiftsQ:\t{}".format(shiftsQ_J))
        print("DEBUG shiftsU:\t{}".format(shiftsU_J))
    ''' #TODO REMOVE
    # Stack exposures
    fUQfil_mv = np.tile(np.nan, [2,2,21,2])
    Vmask_J, Bmask_J = (filters_Jk[:,0]=="v_HIGH"), (filters_Jk[:,0]=="b_HIGH")
    stacked_fUQPphi, filMmasks_f, boxmasks_f, reghists_fUQ = [], [], [], []
    for filternr, [filtermask_J, filtername] in enumerate(zip([Vmask_J, Bmask_J],
                                                            ["v_HIGH", "b_HIGH"])):
        
        # Select all templates corresponding to current filter
        U_J, Q_J, pL_J, phiL_J = UQPphi_J[:,0], UQPphi_J[:,1], UQPphi_J[:,2], UQPphi_J[:,3]
        slitoffs_JK = np.mean(slitoffs_JKs, axis=2)
        '''
        Istacked_K, Imask_K, Ioldembupcorn_K, Inewembuplcorn = [], [], [], []
        for K in range(4):
            Istacked, Imask, Ioldembuplcorn, Inewembuplcorn = polfun.stackim(
                                                                          OplusE__JK[filtermask_J][K],
                                                                          slitoffs_JK[filtermask_J,K],
                                                                          returnmask=True)
            Istacked_K.append(Istacked), Imask_K.append(Imask)
            Ioldembupcorn_K.append(Ioldembuplcorn), Inewembuplcorn_K.append(Inewembuplcorn)
        '''# TODO Maybe not necessary
        U_stacked, Umask, Uoldembuplcorn, Unewembuplcorn = polfun.stackim(U_J[filtermask_J],
                                                                          slitoffs_JK[filtermask_J,0],
                                                                          returnmask=True)
        Q_stacked, Qmask, Qoldembuplcorn, Qnewembuplcorn = polfun.stackim(Q_J[filtermask_J],
                                                                          slitoffs_JK[filtermask_J,1],
                                                                          returnmask=True)
        pL_stacked, pLmask, pLoldembuplcorn, pLnewembuplcorn = polfun.stackim(pL_J[filtermask_J],
                                                                          slitoffs_JK[filtermask_J,0],
                                                                          returnmask=True)
        phiL_stacked, phiLmask, phiLoldembuplcorn, phiLnewembuplcorn = polfun.stackim(
                                                                          phiL_J[filtermask_J],
                                                                          slitoffs_JK[filtermask_J,0],
                                                                          returnmask=True)
        stacked_fUQPphi.append([U_stacked, Q_stacked, pL_stacked, phiL_stacked])
        print("Finished stacking!")
        # TODO Determine stacked array errors
        
        
        # Save results as npsave files
        polfun.savenp(U_stacked, datasavedir, "U_stacked__i{}f{}".format(i+1,filternr+1))      
        polfun.savenp(Q_stacked, datasavedir, "Q_stacked__i{}f{}".format(i+1,filternr+1))
        polfun.savenp(pL_stacked, datasavedir, "pL_stacked__i{}f{}".format(i+1,filternr+1))      
        polfun.savenp(phiL_stacked, datasavedir, "phiL_stacked__i{}f{}".format(i+1,filternr+1))
        # Save results as fitsfiles
        polfun.savefits(U_stacked, imsavedir, "U_stacked__i{}f{}".format(i+1,filternr+1))
        polfun.savefits(Q_stacked, imsavedir, "Q_stacked__i{}f{}".format(i+1,filternr+1))
        polfun.savefits(pL_stacked, imsavedir, "pL_stacked__i{}f{}".format(i+1,filternr+1))
        polfun.savefits(phiL_stacked, imsavedir, "phiL_stacked__i{}f{}".format(i+1,filternr+1))
        print("Saved stacked results!")
        
        
        
        # Create box region histogram statistics and mastermask
        boxcents, boxsizes = boxparms[:,0:2].astype(int), boxparms[:,2:4].astype(int) #pix
        boxrots = boxparms[:,4] #deg
        I = np.array(OplusE__JK)[0,0]

        # Recalibrate the box centers and the stacking mask
        '''
        boxcents_cal = np.array(boxcents) - (pLnewembuplcorn-pLoldembuplcorn)[[1,0]]
        stackmask_cal = pLmask[pLoldembuplcorn[0]:pLoldembuplcorn[0]+I.shape[0],
                               pLoldembuplcorn[1]:pLoldembuplcorn[1]+I.shape[1]]
        '''

        # Determine regional counts
        boxhist_lst, boxmask_lst, valmask_lst = [], [], []
        for boxnr, [boxcent, boxsize, boxrot] in enumerate(zip(boxcents,boxsizes,boxrots)):
            # Determine the regional masks
            Imasked, boxmask = polfun.createrectmask(pL_stacked, 
                                                     boxcent, boxsize, (np.pi/180)*boxrot)
            boxmask_lst.append(~boxmask)
            # Extract counts
            temp = pL_stacked[boxmask]
            # Determine count histograms
            hist, bins = np.histogram(temp, range=(-.02,.02), bins=41)
            boxhist_lst.append([hist,bins]) 
            
            # Determine Gaussian fit to histogram (based on maximum likelihood)
            mean, var = np.mean(temp), np.var(temp) 
            print("Mean, var:\t{} , {}".format(mean, var))
            valmask = pL_stacked>(mean-np.sqrt(var))
            valmask_lst.append(~valmask)
            
            
        # Store all box masks
        boxmasks_f.append(boxmask_lst)
        
        # Form 2Dmastermask (contains np.nan where non-filament and 1 where filament)
        valMmask = ~(np.prod(valmask_lst, axis=0).astype(int))
        boxMmask = ~(np.prod(boxmask_lst, axis=0).astype(int))
        filMmask = np.where(boxMmask*valMmask==1, boxMmask*valMmask, np.nan)
        filMmasks_f.append(filMmask)
        
        
        
    
    
    # Plot I, Qbar, Ubar next to each other for both b and v
    exptimev, exptimeb = exptimes_JK[[0,4],0]
    # Define plot arrays
    axarrs, imswinds = 2*[2*[3*[[]]]], 2*[2*[3*[[]]]]
    gs1, gs2 = gridspec.GridSpec(8,3), gridspec.GridSpec(8,3)
    gs1.update(right=0.95, top=0.95, left=0.05, bottom=0.55, wspace=0.3, hspace=2)
    gs2.update(right=0.95, top=0.45, left=0.05, bottom=0.05, wspace=0.3, hspace=2)
    # Define plotdata array
    stackshape = stacked_fUQPphi[0][1].shape
    phiplot1 = stacked_fUQPphi[0][3] * filMmasks_f[0][0:stacked_fUQPphi[0][3].shape[0],
                                                      0:stacked_fUQPphi[0][3].shape[1]]
    phiplot2 = stacked_fUQPphi[1][3] * filMmasks_f[1][0:stacked_fUQPphi[1][3].shape[0],
                                                      0:stacked_fUQPphi[1][3].shape[1]]
    pltlst = [[ [np.arcsinh(OplusE__JK[0][0][0:stackshape[0],0:stackshape[1]]/exptimev), 
                 stacked_fUQPphi[0][1], stacked_fUQPphi[0][0]], 
                [np.nan, stacked_fUQPphi[0][2], phiplot1] ],

              [ [np.arcsinh(OplusE__JK[4][0][0:stackshape[0],0:stackshape[1]]/exptimeb), 
                 stacked_fUQPphi[1][1], stacked_fUQPphi[1][0]], 
                [np.nan, stacked_fUQPphi[1][2], phiplot2] ]]
    plttitles = [2*[[ [r"I",r"Q/I",r"U/I"], [None, r"$P_L$", r"$\phi_L$"] ]]][0]

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
    plt.savefig(pltsavedir+"/test.png")
    plt.show()
    plt.close()

    
    
    
    
    # Fit cubic spline through box centers
    tck, u = interpolate.splprep(boxparms[:,0:2].T, u=None, s=0.0)
    u_new = np.linspace(u.min(), u.max(), 1000)
    filorxy = np.array(interpolate.splev(u_new, tck, der=0)).T
    filordxdy = np.diff(filorxy,axis=0)
    filorslope = filordxdy[:,1]/filordxdy[:,0]
    filorangle = (180/np.pi) * np.arctan(filorslope) # Deg

    # Scatter plot Theta_orient vs phi_L
    fig, ax = plt.subplots(1)
    filternames, filtercols = ["v_HIGH", "b_HIGH"], ['r', 'b']
    for filternr, filMmask in enumerate(filMmasks_f):
        
        # Select filament polarization angle values
        phiL = stacked_fUQPphi[filternr][3]
        #phiL = np.where(phiL >= 0, phiL, phiL+180) #TODO WHAT TO DO with negative pol angles
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
            
            splevdists = np.sqrt(np.sum((filorxy - xy)**2, axis=1))
            minslevdistarg = splevdists.argmin()
            if minslevdistarg == filorangle.shape[0]:
                filangles[xy[1],xy[0]] = filorangle[-1]
            else:
                filangles[xy[1],xy[0]] = filorangle[minslevdistarg]
                
        
        filangles = np.where(filangles >= 0, filangles, filangles+180) 
        ax.scatter(filangles.flatten(), phiLfil_binned.flatten(), 
                   label=filternames[filternr], color=filtercols[filternr])
                   
        polfun.savefits(filangles, imsavedir, "filangles_f{}".format(filternr))
        polfun.savefits(phiLfil, imsavedir, "phiLfil_f{}".format(filternr))
        
        
        
        # Create boxed polarization angle layouts
        '''
        boxedphiL = np.tile(np.nan, phiLfil_binned.shape)
        Utemp, Qtemp = stacked_fUQPphi[filternr][0:2]        
        for boxmask in boxmasks_f[filternr]:
            
            # Determine median U and Q in boxes
            Uboxtemp, Qboxtemp = np.median(Utemp[~boxmask]), np.median(Qtemp[~boxmask])
            boxedphiL[~boxmask] = (180/np.pi) * (0.5*np.arctan(Uboxtemp/Qboxtemp))
        
        polfun.savefits(boxedphiL, imsavedir, "boxedphiL_f{}".format(filternr))
        '''
        
    
    ax.set_xlabel(r"$\Theta [^{\circ}]$", fontsize=20)
    ax.set_ylabel(r"$\phi_L [^{\circ}]$", fontsize=20)
    plt.legend(loc='best')
    plt.savefig(pltsavedir+"/test2.png")
    plt.show()
    plt.close()
    
    
    
    
    
    # Scatter plot Theta_radial vs phi_L
    fig, ax = plt.subplots(1)
    filternames, filtercols = ["v_HIGH", "b_HIGH"], ['r', 'b']
    for filternr, filMmask in enumerate(filMmasks_f):
        
        # Select filament polarization angle values
        phiL = stacked_fUQPphi[filternr][3]
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
        ax.scatter(filangles.flatten(), phiLfil_binned.flatten(), 
                   label=filternames[filternr], color=filtercols[filternr])
                   
        polfun.savefits(filangles, imsavedir, "filangles_f{}".format(filternr))
        polfun.savefits(phiLfil, imsavedir, "phiLfil_f{}".format(filternr))
        
        
        
        # Create boxed polarization angle layouts
        '''
        boxedphiL = np.tile(np.nan, phiLfil_binned.shape)
        Utemp, Qtemp = stacked_fUQPphi[filternr][0:2]        
        for boxmask in boxmasks_f[filternr]:
            
            # Determine median U and Q in boxes
            Uboxtemp, Qboxtemp = np.median(Utemp[~boxmask]), np.median(Qtemp[~boxmask])
            boxedphiL[~boxmask] = (180/np.pi) * (0.5*np.arctan(Uboxtemp/Qboxtemp))
        
        polfun.savefits(boxedphiL, imsavedir, "boxedphiL_f{}".format(filternr))
        '''
        
    
    ax.set_xlabel(r"$\Theta [^{\circ}]$", fontsize=20)
    ax.set_ylabel(r"$\phi_L [^{\circ}]$", fontsize=20)
    plt.legend(loc='best')
    plt.savefig(pltsavedir+"/test2.png")
    plt.show()
    plt.close()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
