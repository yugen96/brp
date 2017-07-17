import numpy as np
import funct as polfun
from astropy.io import fits
from itertools import product as carthprod
import shutil
import os
import re

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import Background2D, SigmaClip, MedianBackground

import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
            OplusE__JK = np.load(datasavedir+"/OplusE__i{}JK.npy".format(i+1))
            O_E__JK = np.load(datasavedir+"/O_E__i{}JK.npy".format(i+1))
            O_E_grad__JK = np.load(datasavedir+"/O_E_grad__i{}JK.npy".format(i+1))
            print("LOADED flux and stokes lists!")
            filters_Jk = np.load(datasavedir+"/filters_i{}Jk.npy".format(i+1))
            exptimes_JK = np.load(datasavedir+"/exptimes_i{}JK.npy".format(i+1))
            print("LOADED filter and integration times lists!")
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
            minNy = np.min(np.diff(zip(lowedges,upedges)))
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
                sigma_clip = SigmaClip(sigma=3., iters=10)
                bkg_estimator = MedianBackground()
                bkg = Background2D(im, (20, 20), filter_size=(8,8),
                                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
                '''
                for q in [8]:
                    bkg = Background2D(im, (20, 20), filter_size=(q, q),
                                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
                    if k+1 == 1:
                        polfun.savefits(bkg.background, imsavedir+"/temp", 
                                        "bg{}{}_j{}k{}".format(bgsavenames[s],q,j+1,k+1))
                        polfun.savefits(im - bkg.background, imsavedir+"/temp", 
                                        "{}-bg{}{}_j{}k{}".format(bgsavenames[s],bgsavenames[s],q,j+1,k+1))
                        polfun.savefits(im, imsavedir+"/temp", 
                                "{}_j{}k{}".format(bgsavenames[s],j+1,k+1))
                '''
                
                polfun.savefits(im, imsavedir+"/tpl{}/exp{}".format(j+1,k+1), 
                                "{}_j{}k{}".format(bgsavenames[s],j+1,k+1))
                polfun.savefits(bkg.background, imsavedir+"/tpl{}/exp{}".format(j+1,k+1), 
                                "bg{}_j{}k{}".format(bgsavenames[s],j+1,k+1))
                polfun.savefits(im - bkg.background, imsavedir+"/tpl{}/exp{}".format(j+1,k+1), 
                                "{}-bg{}_j{}k{}".format(bgsavenames[s],bgsavenames[s],j+1,k+1))
                
                imcorr_lst.append(im - bkg.background)
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
                print("DEBUG pixvalslit:\t{}".format(pixvalslit))
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
            noncorr_lst = polfun.detslitdiffnorm([slitE,slitO], 
                                                 pixoffs=pixoffxy, suboffs_i=cdintp_xy, 
                                                 savefigs=True, 
                                                 plotdirec = pltsavedir+
                                                             "/tpl{}/exp{}".format(j+1,k+1),
                                                 imdirec = imsavedir+
                                                           "/tpl{}/exp{}".format(j+1,k+1),
                                                 suffix = "_NONbgCORj{}k{}slitp{}".format(j+1,k+1,1))
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
            O_E_gradplt = axarr[1].imshow(O_E_grad, origin='lower', cmap='afmhot', vmin=-1, vmax=1)
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
        
        '''
        xcoords_Ks = [m[0] for m in sorted(zip(xcoords_ks, retangles_k), key=lambda l: l[1])] 
        ycoords_Ks = [m[0] for m in sorted(zip(ycoords_ks, retangles_k), key=lambda l: l[1])]
        # Transform into arrays and append to list
        xcoords_Ks, ycoords_Ks = np.array(xcoords_Ks), np.array(ycoords_Ks) 
        xcoords_JKs.append(xcoords_Ks), ycoords_JKs.append(ycoords_Ks)
        # Determine exposure-wise offsets
        print("DEBUG Qoffsxlst:\t {}".format(np.diff(xcoords_Ks[[0,2]],axis=0)))
        print("DEBUG Qoffsylst:\t {}".format(np.diff(ycoords_Ks[[0,2]],axis=0)))
        print("DEBUG Uoffsxlst:\t {}".format(np.diff(xcoords_Ks[[1,3]],axis=0)))
        print("DEBUG Uoffsylst:\t {}".format(np.diff(ycoords_Ks[[1,3]],axis=0)))
        Qoffsx = int(np.median(np.diff(xcoords_Ks[[0,2]],axis=0)))
        Qoffsy = int(np.median(np.diff(ycoords_Ks[[0,2]],axis=0)))
        Uoffsx = int(np.median(np.diff(xcoords_Ks[[1,3]],axis=0)))
        Uoffsy = int(np.median(np.diff(ycoords_Ks[[1,3]],axis=0)))
        '''
        
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
    polfun.savenp(UQPphi_J, datasavedir, "UQPphi__i{}JK".format(i+1))
    polfun.savenp(OplusE__JK, datasavedir, "OplusE__i{}JK".format(i+1))
    polfun.savenp(O_E__JK, datasavedir, "O_E__i{}JK".format(i+1))
    polfun.savenp(O_E_grad__JK, datasavedir, "O_E_grad__i{}JK".format(i+1))    
    polfun.savenp(filters_Jk, datasavedir, "filters_i{}Jk".format(i+1)) 
    polfun.savenp(exptimes_JK, datasavedir, "exptimes_i{}JK".format(i+1)) 
    ###################### FINISHED CALCULATIONS
    
    
    
    
    
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
    fUQstar_mv, fUQfil_mv = np.tile(np.nan, [2,2,6,2]), np.tile(np.nan, [2,2,5,2])
    stacked_fUQ, fUQfil_lst, fUQstar_lst = [], [], []
    Vmask_J, Bmask_J = (filters_Jk[:,0]=="v_HIGH"), (filters_Jk[:,0]=="b_HIGH")
    for filternr, [filtermask_J, filtername] in enumerate(zip([Vmask_J, Bmask_J],
                                                            ["v_HIGH", "b_HIGH"])):
        
        # Select all templates corresponding to current filter
        U_J, Q_J = UQPphi_J[:,0], UQPphi_J[:,1]
        slitoffs_JK = np.mean(slitoffs_JKs, axis=2)
        U_stacked, Umask, Uoldembuplcorn, Unewembuplcorn = polfun.stackim(U_J[filtermask_J],
                                                                          slitoffs_JK[filtermask_J,0],
                                                                          returnmask=True)
        Q_stacked, Qmask, Qoldembuplcorn, Qnewembuplcorn = polfun.stackim(Q_J[filtermask_J],
                                                                          slitoffs_JK[filtermask_J,1],
                                                                          returnmask=True)
        stacked_fUQ.append([U_stacked, Q_stacked])
        # TODO Determine stacked array errors
        
        
        # Diagnostic plot
        '''
        fig, axarr = plt.subplots(2, sharex=True)
        im1 = axarr[0].imshow(U_stacked, origin='lower', cmap='afmhot', vmin=-.15, vmax=.15)
        im2 = axarr[1].imshow(Q_stacked, origin='lower', cmap='afmhot', vmin=-.15, vmax=.15)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im1, cax=cbar_ax)
        plt.show()
        plt.close()
        '''
        
        
        # Save results as npsave files
        polfun.savenp(U_stacked, datasavedir, "U_stacked__i{}f{}".format(i+1,filternr+1))      
        polfun.savenp(Q_stacked, datasavedir, "Q_stacked__i{}f{}".format(i+1,filternr+1))
        # Save results as fitsfiles
        polfun.savefits(U_stacked, imsavedir, "U_stacked__i{}f{}".format(i+1,filternr+1))
        polfun.savefits(Q_stacked, imsavedir, "Q_stacked__i{}f{}".format(i+1,filternr+1))
        
        
        # Diagnostic plot
        '''
        fig, axarr = plt.subplots(2, sharex=True)
        im1 = axarr[0].imshow(U_stacked, origin='lower', cmap='afmhot', vmin=-.1, vmax=.1,
                              extent=[np.min(scapexslitarcs),np.max(scapexslitarcs),
                                      np.min(scapeyslitarcs),np.max(scapeyslitarcs)])
        axarr[0].set_ylim(bottom=np.min(scapeyslitarcs), top=np.max(scapeyslitarcs))
        axarr[0].set_title('U',fontsize=26)
        axarr[0].set_ylabel('Y [arcsec]',fontsize=20)  
        im2 = axarr[1].imshow(Q_stacked, origin='lower', cmap='afmhot', vmin=-.1, vmax=.1,
                              extent=[np.min(scapexslitarcs),np.max(scapexslitarcs),
                                      np.min(scapeyslitarcs),np.max(scapeyslitarcs)])
        axarr[1].set_ylim(bottom=np.min(scapeyslitarcs), top=np.max(scapeyslitarcs))
        axarr[1].set_title('Q',fontsize=26)
        axarr[1].set_ylabel('Y [arcsec]',fontsize=20)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im1, cax=cbar_ax)
        plt.savefig(pltsavedir+"/UQ_i{}f{}".format(i+1,filternr+1))
        plt.show()
        plt.close()
        '''
     
        
        
        
        # Create flattened array with the pixel values for the selected filaments
        boxcents = np.array([[850,64],[846,35],[871,45],[818,39],[796,47]]) #pix
        boxsizes = np.array([[17,9],[26,8],[13,6],[20,8],[15,4]]) #pix
        boxrots = np.array([0, 0, 45, 165, 160]) #deg
        uplims = [300,15,16,6,6]
        I = np.array(OplusE__JK)[0,0]
        UQfil_lst = []
        for ind, [polpar, stackmask, 
                  oldembulcorn, newembulcorn] in enumerate(zip([U_stacked, Q_stacked],[Umask,Qmask],
                                                               [Uoldembuplcorn, Qoldembuplcorn],
                                                               [Unewembuplcorn, Qnewembuplcorn])):
            
            # Recalibrate the box centers and the stacking mask
            boxcents_cal = np.array(boxcents) - (newembulcorn-oldembulcorn)[[1,0]]
            stackmask_cal = stackmask[oldembulcorn[0]:oldembulcorn[0]+I.shape[0],
                                      oldembulcorn[1]:oldembulcorn[1]+I.shape[1]]
            
            # Determine regional counts
            polreg_lst = []
            for boxnr, [boxcent, boxsize, boxrot] in enumerate(zip(boxcents_cal,boxsizes,boxrots)):
                # Determine the regional masks
                Imasked, filmask = polfun.createrectmask(polfun.mask2d(I,stackmask_cal), 
                                                         boxcent, boxsize, (np.pi/180)*boxrot)
                valmask = (Imasked < uplims[boxnr]) # Apply sharp masking to select filament pixels
                # Extract counts
                temp = polpar[filmask]
                polreg_lst.append(temp[valmask])
            
            # Append to list specifying U and Q for each filament
            UQfil_lst.append(np.array(polreg_lst))
        # Append to list which distinguishes between filters
        fUQfil_lst.append(UQfil_lst)
        
        
        
        
        
        # Create flattened array with the pixel values for the selected stellar regions
        circaps = [[556,47,6],[453,33,4],[303,21,6],[1547,55,5],[1479,9,7],[1064,28,5]]
        I = np.array(OplusE__JK)[0,0]
        UQstar_lst = [] 
        for ind, [polpar, stackmask, 
                  oldembulcorn, newembulcorn] in enumerate(zip([U_stacked, Q_stacked],[Umask,Qmask],
                                                               [Uoldembuplcorn, Qoldembuplcorn],
                                                               [Unewembuplcorn, Qnewembuplcorn])):
            
            # Recalibrate the aperture centers and the stacking mask
            apcents_cal = np.array(circaps)[:,0:2] - (newembulcorn-oldembulcorn)[[1,0]]
            stackmask_cal = stackmask[oldembulcorn[0]:oldembulcorn[0]+I.shape[0],
                                      oldembulcorn[1]:oldembulcorn[1]+I.shape[1]]
            
            # Determine regional counts
            polstar_lst = []
            for stellarnr in range(len(circaps)):
                
                # Select aperture center and radius
                apcent, apR = apcents_cal[stellarnr], np.array(circaps)[stellarnr,2]
                # Determine the regional masks
                _, starmask = polfun.cmask(I, apcent, apR)
                starmask_cal = polfun.mask2d(starmask,stackmask_cal)
                # Extract counts
                temp = polpar[starmask_cal]
                polstar_lst.append(temp)
                
                # Diagnostic plot
                '''
                plt.imshow(polfun.mask2d(I, stackmask_cal), origin='lower', cmap='afmhot', alpha=0.5)
                plt.imshow(starmask_cal, origin='lower', cmap='Greys', alpha=0.5)#, alpha=0.5)
                #plt.scatter(center[0], center[1], color='k', s=50)
                plt.show()
                plt.close()
                '''
                
            # Append to list specifying U and Q for each star
            UQstar_lst.append(np.array(polstar_lst))
        # Append to list which distinguishes between filters
        fUQstar_lst.append(np.array(UQstar_lst))
        
        
        
        
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
        # Create filament histograms for U/I and Q/I
        # Set up  new figure
        fig, axarr = plt.subplots(2, 5, sharex='col', sharey='row', figsize=(20,8))
        # Set bin ranges and initialize maximum frequency arrays
        binrange, nrbins, truemaxfreq = (-0.1,0.1), 21, 0
        for ind in range(2): 
            
            # Plot Stokes parameter for each selected region
            UQPhists = []
            for filNR, poldata in enumerate(fUQfil_lst[filternr][ind]): 
                
                #weights = np.ones_like(poldata)/float(len(poldata))
                hist, bins = np.histogram(poldata, range=binrange, bins=nrbins)#, weights=weights)
                bincents = (bins[:-1] + bins[1:]) / 2
                axarr[ind][filNR].bar(bincents, hist, #color=filUQcolors[filNR,ind],# alpha=0.5, 
                                      align='center', width=0.7*(bins[1]-bins[0])) 
                                      #label=r"X={}".format(polparlabs[ind]))
                                 
                # Plot Gaussian fit
                mean, var = np.mean(poldata), np.var(poldata)
                print("Mean, var:\t{} , {}".format(mean, var))
                fUQfil_mv[filternr,ind,filNR,:] = [mean,var] # Append to array
                pdf_x = np.linspace(np.min(poldata), np.max(poldata), 100)
                pdf_y = (1./np.sqrt(2*np.pi*var)) * np.exp((-0.5*(pdf_x-mean)**2) / var)
                axarr[ind][filNR].plot(pdf_x, pdf_y, 'k--')
                axarr[ind][filNR].set_xlim(xmin=-0.15, xmax=0.15)
                combfreqs = np.concatenate([pdf_y,hist])
                maxfreq = np.max(combfreqs).astype(int)
                if maxfreq > truemaxfreq:
                    truemaxfreq = maxfreq
                    axarr[ind][filNR].set_ylim(ymin=0, ymax=1.2*truemaxfreq) 
                #axarr[ind][filNR].legend(loc='upper left')
                plt.grid()
                #axarr[filNR].set_title(r"Filament {}".format(filNR+1))
                #axarr[filNR].legend(loc='upper right')
                #axarr[ind][4].set_xlabel(r"X/I", fontsize=20)
        
        # Plot y-labels and y-titles
        panelrowtitles, rowtitlelocs = [r"Q/I", r"U/I"], [0.75, 0.25]
        for ind in range(2):
            axarr[ind][0].set_ylabel(r"Frequency [--]", fontsize=20)
            fig.text(0.03, rowtitlelocs[ind], panelrowtitles[ind], fontsize=26, 
                     ha="center", va="center", rotation=90) 
        # Plot x-labels and x-titles
        for filNR in range(len(fUQfil_lst[filternr][1])):
            axarr[0][filNR].set_title(r"Filament {}".format(filNR), fontsize=26)    
            axarr[1][filNR].set_xlabel(r"X/I", fontsize=20)
        
        #plt.tight_layout()
        plt.savefig(pltsavedir + "/filhists_i{}f{}NEW.png".format(i+1, filternr+1))
        plt.show()
        plt.close()   
        
                    
        '''
        # Create filament histograms for U/I and Q/I
        filUQcolors = np.array([['#660000','#000066'], ['#ff0000','#0033cc'], ['#cc3300','#0099cc'], 
                                ['#ff6600','#00cc66'], ['#ff9933','#009933'], ['#ffcc00','#003300']])
        polparlabs, suffixes = [r"U", r"Q", r"$P_L$ [%]"], ["U", "Q", "P"]
        # Set up  new figure
        fig, axarr = plt.subplots(4, sharex='col', sharey='row', figsize=(20,8))
        # Set bin ranges and initialize maximum frequency arrays
        binrange, nrbins, truemaxfreq = (-0.1,0.1), 21, 0
        for ind in range(2): 
            
            # Plot Stokes parameter for each selected region
            UQPhists = []
            for filNR, poldata in enumerate(fUQfil_lst[filternr][ind]): 
                
                #weights = np.ones_like(poldata)/float(len(poldata))
                hist, bins = np.histogram(poldata, range=binrange, bins=nrbins)#, weights=weights)
                bincents = (bins[:-1] + bins[1:]) / 2
                axarr[filNR].bar(bincents, hist, color=filUQcolors[filNR,ind],# alpha=0.5, 
                                      align='center', width=0.7*(bins[1]-bins[0]), 
                                      label=r"X={}".format(polparlabs[ind]))
                                 
                # Plot Gaussian fit
                mean, var = np.mean(poldata), np.var(poldata)
                print("Mean, var:\t{} , {}".format(mean, var))
                fUQfil_mv[filternr,ind,filNR,:] = [mean,var] # Append to array
                pdf_x = np.linspace(np.min(poldata), np.max(poldata), 100)
                pdf_y = (1./np.sqrt(2*np.pi*var)) * np.exp((-0.5*(pdf_x-mean)**2) / var)
                axarr[filNR].plot(pdf_x, pdf_y, 'k--')
                axarr[filNR].set_xlim(xmin=-0.15, xmax=0.15)
                combfreqs = np.concatenate([pdf_y,hist])
                maxfreq = np.max(combfreqs).astype(int)
                if maxfreq > truemaxfreq:
                    truemaxfreq = maxfreq
                    axarr[filNR].set_ylim(ymin=0, ymax=1.2*truemaxfreq) 
                axarr[filNR].legend(loc='upper left')
                plt.grid()
                #axarr[filNR].set_title(r"Filament {}".format(filNR+1))
                #axarr[filNR].legend(loc='upper right')
        
        axarr[2].set_ylabel(r"Frequency", fontsize=20)
        axarr[4].set_xlabel(r"X/I", fontsize=20)
        axarr[0].set_title(r"Filament stokes distribution", fontsize=26)    
        #plt.tight_layout()
        plt.savefig(pltsavedir + "/filhists{}_i{}f{}.png".format(suffixes[ind], i+1, filternr+1))
        plt.show()
        plt.close() 
        '''
        #TODO Make panels with b and v plotted together..
        
        
        
        # Create stellar histograms for U/I and Q/I
        starUQcolors = np.array([['#660000','#000066'], ['#ff0000','#0033cc'], ['#cc3300','#0099cc'], 
                                ['#ff6600','#00cc66'], ['#ff9933','#009933'], ['#ffcc00','#003300']])
        polparlabs, suffixes = [r"U", r"Q", r"$P_L$ [%]"], ["U", "Q", "P"]
        # Set up  new figure
        fig, axarr = plt.subplots(6, sharex=True)
        # Set bin ranges and initialize maximum frequency arrays
        binrange, nrbins, truemaxfreq = (-0.1,0.1), 21, 0
        for ind in range(2): 
            
            # Plot Stokes parameter for each selected region
            for starNR, poldata in enumerate(fUQstar_lst[filternr][ind]): 
                
                #weights = np.ones_like(poldata)/float(len(poldata))
                hist, bins = np.histogram(poldata, range=binrange, bins=nrbins)#, weights=weights)
                bincents = (bins[:-1] + bins[1:]) / 2
                axarr[starNR].bar(bincents, hist, color=starUQcolors[starNR,ind], alpha=0.5, 
                                 align='center', width=0.7*(bins[1]-bins[0]), 
                                 label=r"X={}".format(polparlabs[ind]))
                # Plot Gaussian fit
                mean, var = np.mean(poldata), np.var(poldata)
                fUQstar_mv[filternr,ind,starNR,:] = [mean,var] # Append to array
                pdf_x = np.linspace(np.min(poldata), np.max(poldata), 100)
                pdf_y = (1./np.sqrt(2*np.pi*var)) * np.exp((-0.5*(pdf_x-mean)**2) / var)
                axarr[starNR].plot(pdf_x, pdf_y, 'k--')
                axarr[starNR].set_xlim(xmin=-0.15, xmax=0.15)
                combfreqs = np.concatenate([pdf_y,hist])
                maxfreq = np.max(combfreqs).astype(int)
                if maxfreq > truemaxfreq:
                    truemaxfreq = maxfreq
                    axarr[starNR].set_ylim(ymin=0, ymax=1.2*truemaxfreq) 
                axarr[starNR].legend(loc='upper left')
                plt.grid()
        
        axarr[2].set_ylabel(r"Frequency", fontsize=20)
        axarr[5].set_xlabel(r"X/I", fontsize=20)
        axarr[0].set_title(r"Stellar stokes distribution", fontsize=26)    
        #plt.tight_layout()
        plt.savefig(pltsavedir + "/starhistsUQ_i{}f{}.png".format(i+1, filternr+1))
        plt.show()
        plt.close() 
        
    
    
    # Save the intermediate results
    polfun.savenp(stacked_fUQ, datasavedir, "stacked_fUQ")
    polfun.savenp(fUQfil_lst, datasavedir, "fUQfil_lst")
    polfun.savenp(fUQstar_lst, datasavedir, "fUQstar_lst")
    
    
    
    
                
                            
    # Plot I, Qbar, Ubar next to each other for both b and v
    fig, axarr = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(15,8))
    # Plot total intensities
    exptimev, exptimeb = exptimes_JK[[0,4],0]
    imIb = axarr[0][0].imshow(OplusE__JK[0][0][:,734:986] / exptimev, origin='lower', 
                              cmap='afmhot', vmin=-10/exptimev, vmax=100/exptimev, 
                              extent=[np.min(scapexslitarcs[734:986]),
                                      np.max(scapexslitarcs[734:986]),
                                      np.min(scapeyslitarcs),np.max(scapeyslitarcs)])
    imIv = axarr[1][0].imshow(OplusE__JK[4][0][:,734:986] / exptimeb, origin='lower', 
                              cmap='afmhot', vmin=-10/exptimeb, vmax=100/exptimeb, 
                              extent=[np.min(scapexslitarcs[734:986]),
                                      np.max(scapexslitarcs[734:986]),
                                      np.min(scapeyslitarcs),np.max(scapeyslitarcs)])
    # Plot filament boundaries
    for tempind in range(2):
        for boxnr, [boxcent, boxsize, boxrot] in enumerate(zip(boxcents_cal,boxsizes,boxrots)):
            # Determine box center and size in arcseconds
            boxcentarcs = np.array([scapexslitarcs[boxcent[0]], scapeyslitarcs[boxcent[1]]])
            boxsizearcs = .126*boxsize
            # Rotate lower left corner to correct position
            rotmatrix = np.array([[np.cos(boxrot),-np.sin(boxrot)],
                                  [np.sin(boxrot),np.cos(boxrot)]])  
            boxlowleftarcs_rot = np.matmul(rotmatrix, -.5*boxsizearcs) + boxcentarcs
            print(boxlowleftarcs_rot)
            # Plot box
            axarr[tempind][0].add_patch( patches.Rectangle(boxlowleftarcs_rot, 
                                                           boxsizearcs[0], boxsizearcs[1],
                                                           angle=boxrot, fill=False, 
                                                           color='b', linewidth=3) )

    # Plot normalized Stokes Q
    imQb = axarr[0][1].imshow(stacked_fUQ[0][1][:,734:986], origin='lower', 
                              cmap='afmhot', vmin=-.1, vmax=.1,
                              extent=[np.min(scapexslitarcs[734:986]),
                                      np.max(scapexslitarcs[734:986]),
                                      np.min(scapeyslitarcs),np.max(scapeyslitarcs)])                
    imQv = axarr[1][1].imshow(stacked_fUQ[1][1][:,734:986], origin='lower', 
                              cmap='afmhot', vmin=-.1, vmax=.1,
                              extent=[np.min(scapexslitarcs[734:986]),
                                      np.max(scapexslitarcs[734:986]),
                                      np.min(scapeyslitarcs),np.max(scapeyslitarcs)])
    # Plot normalized Stokes Q
    imUb = axarr[0][2].imshow(stacked_fUQ[0][0][:,734:986], origin='lower', 
                              cmap='afmhot', vmin=-.1, vmax=.1,
                              extent=[np.min(scapexslitarcs[734:986]),
                                      np.max(scapexslitarcs[734:986]),
                                      np.min(scapeyslitarcs),np.max(scapeyslitarcs)])
    imUv = axarr[1][2].imshow(stacked_fUQ[1][0][:,734:986], origin='lower', 
                              cmap='afmhot', vmin=-.1, vmax=.1,
                              extent=[np.min(scapexslitarcs[734:986]),
                                      np.max(scapexslitarcs[734:986]),
                                      np.min(scapeyslitarcs),np.max(scapeyslitarcs)])

    # Set axes labels
    rowtitlelocs = [0.75, 0.25]
    panelrowtitles, panelcoltitles = [r"v_HIGH", r"b_HIGH"], [r"I", r"Q/I", r"U/I"]
    for tempind in range(3):
        axarr[0][tempind].set_title(panelcoltitles[tempind], fontsize=26)
        axarr[1][tempind].set_xlabel(r"X [arcsec]", fontsize=20)

    for tempind in range(2):
        fig.text(0.03,rowtitlelocs[tempind], panelrowtitles[tempind], fontsize=26, 
                 ha="center", va="center", rotation=90) 
        axarr[tempind][0].set_ylabel(r"Y [arcsec]", fontsize=20)

    # Add colorbars
    '''
    cbaxes1 = fig.add_axes([0.13, 0.02, 0.23, 0.03]) 
    cb1 = plt.colorbar(imIb, orientation='horizontal', 
                      cax = cbaxes1, ticks=np.round(np.arange(-10,110,20),0))  
    fig.colorbar(imIb, cax=cbar_ax)      
    '''                
    cbaxes2 = fig.add_axes([0.4, 0.5, 0.5, 0.03]) 
    cb2 = plt.colorbar(imQb, orientation='horizontal', 
                      cax = cbaxes2, ticks=np.round(np.arange(-.1,.15,.05),2))  
    fig.colorbar(imQb, cax=cbar_ax)
    #plt.tight_layout()
    plt.savefig(pltsavedir+"/UQ_i{}fALL".format(i+1,filternr+1))
    plt.show()
    plt.close()
    
    
    
    
    
    # Plot filamentary QvsU points
    filtercolours = ['g', 'b']
    offsangle_lst = 2.*np.array([1.80, 1.54]) #deg
    for filternr in range(2):
        if filternr == 0:
            galcenttag = "Galactic center"
        else:
            galcenttag = None
        fig, ax = plt.subplots(1)     
        fig, ax = polfun.QvsU(fig, ax, 
                              [fUQfil_mv[filternr,1,0,0],np.sqrt(fUQfil_mv[filternr,1,0,1]),
                               fUQfil_mv[filternr,0,0,0],np.sqrt(fUQfil_mv[filternr,0,0,1])],
                              offsetangle=offsangle_lst[filternr],
                              colour='m', tag=galcenttag)     
        fig, ax = polfun.QvsU(fig, ax, 
                              [fUQfil_mv[filternr,1,1::,0],np.sqrt(fUQfil_mv[filternr,1,1::,1]),
                               fUQfil_mv[filternr,0,1::,0],np.sqrt(fUQfil_mv[filternr,0,1::,1])],
                              offsetangle=offsangle_lst[filternr],
                              colour=filtercolours[filternr], 
                              PLPHI=[0.02,0.01,0,0], checkPphi=[True,False], 
                              tag="Dust filaments")
        for filno in range(len(fUQfil_mv[0,0,:,0])):
            filU, filQ = fUQfil_mv[filternr,:,filno,0]
            ax.annotate(r'{}'.format(filno), xy=(filQ, filU), 
                        xytext=(filQ+2e-3, filU+2e-3), fontsize=20)
        
        ax.grid()
        ax.set_xlabel(r"Q/I", fontsize=20), ax.set_ylabel(r"U/I", fontsize=20)                      
        ax.set_title(r"Filament linear stokes", fontsize=26)
        ax.legend(loc='upper right', fancybox=True, framealpha=0.5)
        plt.savefig(pltsavedir + "/filUQ_i{}f{}.png".format(i+1, filternr+1))
        plt.show()
        plt.close()
    
    
    # Plot stellar QvsU points
    for filternr in range(2):
        fig, ax = plt.subplots(1)
        fig, ax = polfun.QvsU(fig, ax, 
                              [fUQstar_mv[filternr,1,:,0],np.sqrt(fUQstar_mv[filternr,1,:,1]),
                               fUQstar_mv[filternr,0,:,0],np.sqrt(fUQstar_mv[filternr,0,:,1])],
                              colour=filtercolours[filternr])
        
        ax.grid()
        ax.set_xlabel(r"Q/I", fontsize=20), ax.set_ylabel(r"U/I", fontsize=20)                      
        ax.set_title(r"Stellar linear stokes", fontsize=26)
        plt.savefig(pltsavedir + "/stellarUQ_i{}f{}.png".format(i+1, filternr+1))
        plt.show()
        plt.close()    
    
    
    
    
    
    
    '''
    QvsU(fig, ax, QU, offsetangle=0., PLPHI=np.zeros(4), checkPphi=[False,False], plot_inset=False, inset_ax=None, colour='b', tag=None):    
    
    ax.errorbar(UQfil_mv[0,:,0], UQfil_mv[1,:,0], xerr=UQfil_mv[0,:,1], yerr=UQfil_mv[1,:,1], 
                fmt='.', markersize = 16.) # STD star    
    
    ax.set_title(r"Filament linear stokes")
    plt.tight_layout()
    plt.show()
    plt.close()
    '''
    
    
    
    '''
    # Create stellar histograms
    starcolors = ['#000066', '#0033cc', '#0099cc', '#00cc66', '#009933', '#003300']
    polparlabs, suffixes = [r"U/I", r"Q/I", r"$P_L$ [%]"], ["U", "Q", "P"]
    for ind in range(2):
        
        # Set up  new figure
        fig, axarr = plt.subplots(6, sharex=True)
        # Set bin ranges
        if ind == 2: 
            binrange, nrbins = (0,20), 20
        else: 
            binrange, nrbins = (-0.2,0.2), 40     
        
        # Plot Stokes paramter for each selected region
        for starNR, starUQ in enumerate(starUQ_lst):
            poldata = starUQ[ind]
            print("DEBUG starpoldata:\t{}".format(poldata))
            axarr[starNR].hist(poldata, color=starcolors[starNR], normed=True, 
                              range=binrange,  bins=nrbins, label=r"Star {}".format(starNR))
            axarr[starNR].set_yticks(np.arange(0,4100,1000))
            # Plot Gaussian fit
            mean, var = np.mean(poldata), np.var(poldata)
            pdf_x = np.linspace(np.min(poldata), np.max(poldata), 100)
            pdf_y = (1./np.sqrt(2*np.pi*var)) * np.exp((-0.5*(pdf_x-mean)**2) / var)
            axarr[starNR].plot(pdf_x, pdf_y, 'k--')
            #axarr[starNR].set_title(r"Star {}".format(starNR+1))
            #axarr[starNR].legend(loc='upper right')
            
        axarr[3].set_ylabel(r"Normalized frequency", fontsize=20)
        axarr[5].set_xlabel(r"{}".format(polparlabs[ind]), fontsize=20)
        #plt.tight_layout()
        plt.savefig(pltsavedir + "/starhists{}_i{}.png".format(suffixes[ind], i+1))
        plt.show()
        plt.close() 
    '''

    
            
            
            

