import numpy as np
import funct as polfun
from astropy.io import fits
from itertools import product as carthprod
import shutil
import os

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import Background2D, SigmaClip, MedianBackground

import matplotlib.pyplot as plt
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
# Boolean variable for turning on/off the computation of the cubic splines
detsplines1d, detsplines2d = False, False
detoverlaps1d, detoverlaps2d = True, True


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





# Determine cubic splines for the median c- and d-scapes
if detsplines1d or detsplines2d:
    '''
    # (Re)compute the median c- and d-scapes if not present as np savefile
    if not os.path.exists(npsavedir+"/cdscapes/cscape_med.npy"):
        
        # Load the original total x and y offset data
        cscape_ijk = np.load(npsavedir+"/cdscapes/cscape_ijk.npy")
        dscape_ijk = np.load(npsavedir+"/cdscapes/dscape_ijk.npy")

        # Determine median pixel+subpixel offsets over all different templates and exposures
        medcscape_i, meddscape_i = [], []
        for i, [cscape_jk, dscape_jk] in enumerate(zip(cscape_ijk, dscape_ijk)):
            medcscape_j, meddscape_j = [], []
            for j, [cscape_k, dscape_k] in enumerate(zip(cscape_jk, dscape_jk)):
                
                # Determine median offsets over all exposures
                medcscape_j.append(np.nanmedian(cscape_k, axis=0))
                meddscape_j.append(np.nanmedian(dscape_k, axis=0))
            
            # Determine median offsets over all templates
            medcscape_i.append(np.nanmedian(medcscape_j, axis=0))
            meddscape_i.append(np.nanmedian(meddscape_j, axis=0))    

        # Determine median offsets over all objects
        cscape_med = np.nanmedian(medcscape_i, axis=0)
        dscape_med = np.nanmedian(meddscape_i, axis=0)
        cscape_medNGCWD = np.nanmedian(medcscape_i[1::], axis=0)
        dscape_medNGCWD = np.nanmedian(meddscape_i[1::], axis=0)
        # Save results
        polfun.savenp(cscape_med, npsavedir+"/cdscapes", "cscape_med")
        polfun.savenp(dscape_med, npsavedir+"/cdscapes", "dscape_med")
        polfun.savenp(medcscape_i[2], npsavedir+"/cdscapes", "cscape_NGCmed")
        polfun.savenp(meddscape_i[2], npsavedir+"/cdscapes", "dscape_NGCmed")  
        polfun.savenp(cscape_medNGCWD, npsavedir+"/cdscapes", "cscape_NGC+WDmed")
        polfun.savenp(dscape_medNGCWD, npsavedir+"/cdscapes", "dscape_NGC+WDmed")
    '''
    '''
for i, [cscape_jk, dscape_jk] in enumerate(zip(cscape_ijk,dscape_ijk)):
    print(i)
    for j, [cscape_k, dscape_k] in enumerate(zip(cscape_jk,dscape_jk)):
        print("\t{}".format(j))
        for k, [cscape, dscape] in enumerate(zip(cscape_k,dscape_k)):
            print("\t\t{}".format(k))
            cscape, dscape = np.array(cscape), np.array(dscape)
            print("\t\t\tc\t", np.nanmin(cscape), np.nanmax(cscape)) 
            print("\t\t\td\t", np.nanmin(dscape), np.nanmax(dscape)) 
    '''       
    
    
    # Load the c- and d-scapes
    cscape_jk = np.load(npsavedir+"/NGC4696,IPOL/cscape_i2.npy")
    dscape_jk = np.load(npsavedir+"/NGC4696,IPOL/dscape_i2.npy")
    print("c- and d-scapes loaded!")
    
    # Compute median offsets
    
    
    
    '''
    # High influence points?
    maskind_c = [38, 27]
    maskind_d = [22, 29, 45, 26, 50, 49, 19, 17, 23, 39, 40, 32, 5]
    ''' 
    
    
    # Determine cubic splines and 3rd order polynomial fits over the whole chip range
    if detsplines2d:
    
        # Plot cubic splines for both c- and d-scape
        polynoms, splines1d, splines, splinesnear = [], [], [], []
        for (scape, pltsavetitle, plttitle) in zip([cscape_med, dscape_med], 
                                                    #[maskind_c, maskind_d],
                                                    ["c-scapeNGC+WD", "d-scapeNGC+WD"],
                                                    [r"$\delta_x + c$", r"$\delta_y + d$"])):
                    
            # Extract the x- and y-coordinates corresponding to points in c- and d-scapes
            xycoord = np.argwhere(~np.isnan(scape))
            points = np.dstack(xycoord)[0]
            x, y = points[1,:], points[0,:]
            # Determine the values of the points within c- and d-scape
            val = scape[y,x]
            # Compute gridpoints for evaluation
            scapex, scapey = np.arange(0,scape.shape[1],1), np.arange(0,scape.shape[0],1)
            scape_xgrid, scape_ygrid = np.meshgrid(scapex, scapey)
            
            # Convert coordinates to arcseconds
            xarcs, yarcs = np.array([x - np.median(scapex), y]) * .126
            scapexarcs, scapeyarcs = (scapex - np.median(scapex))*.126, scapey*.126
            scapexarcs_grid, scapeyarcs_grid = np.meshgrid(scapexarcs, scapeyarcs)    
            
            
            
            ########### WHOLE-CHIP BIVARIATE CUBIC SPLINE ###########
            # Determine cubic spline to the c- and d-values
            scape_i = interpolate.griddata((xarcs, yarcs), val, 
                                           (scapexarcs[None,:], scapeyarcs[:,None]), method='cubic')
            
            # Contour the gridded data, plotting dots at the randomly spaced data points.
            levels = np.arange(-2,2.25,0.25)
            CS = plt.contour(scapexarcs,scapeyarcs,scape_i,levels,linewidths=0.5, colors='k')
            CS = plt.contourf(scapexarcs,scapeyarcs,scape_i,levels, vmin=-2, vmax=2)
            # Plot data points
            plt.scatter(xarcs, yarcs, marker='o', s=50, c=val, cmap=CS.cmap, norm=CS.norm)
            plt.colorbar() # draw colorbar
            # Plot the slit boundaries
            for i, [borderdown, borderup] in enumerate(zip(lowedges, upedges)):
                if i == 0:   
                    plt.plot(scapexarcs, np.tile(.126*borderdown, len(scapexarcs)), color='k')
                elif i != 0 and i != len(lowedges)-1:
                    plt.plot(scapexarcs, np.tile(.126*borderdown, len(scapexarcs)), 
                             linestyle='--', color='k')
                elif i == len(lowedges)-1:
                    plt.plot(scapexarcs, np.tile(.126*borderdown, len(scapexarcs)), 
                             linestyle='--', color='k')
                    plt.plot(scapexarcs, np.tile(.126*borderup, len(scapexarcs)), color='k')
                
            plt.plot(np.tile(-120, len(scapeyarcs)), scapeyarcs, color='k')
            plt.plot(np.tile(120, len(scapeyarcs)), scapeyarcs, color='k')
            plt.xlabel(r"X [arcsec]", fontsize=20), plt.ylabel(r"Y [arcsec]", fontsize=20)
            plt.title(r"{}".format(plttitle), fontsize=26)
            plt.savefig(plotdir+"/"+pltsavetitle+"MED.png")
            #plt.show()
            plt.close()
            
            
            
            '''
            # Linear interpolation over the cubic interpolation (to fill in nan values)
            xycoord_i = np.argwhere(~np.isnan(scape_i))
            points_i = np.dstack(xycoord_i)[0]
            x_i, y_i = [points_i[1,:], points_i[0,:]]
            [xarcs_i, yarcs_i] = .126 * np.array([x_i-np.median(x_i), y_i])
            scape_inear = interpolate.griddata((xarcs_i, yarcs_i), scape_i[y_i, x_i], 
                                           (scapexarcs[None,:], scapeyarcs[:,None]), method='nearest')

            # Contour the gridded data, plotting dots at the randomly spaced data points.
            CS = plt.contour(scapexarcs,scapeyarcs,scape_inear,15,linewidths=0.5, colors='k')
            CS = plt.contourf(scapexarcs,scapeyarcs,scape_inear,15)
            # Plot data points
            plt.scatter(xarcs, yarcs, marker='o', s=50, c=val, cmap=CS.cmap, norm=CS.norm)
            plt.colorbar() # draw colorbar
            # Plot the slit boundaries
            for i, [borderdown, borderup] in enumerate(zip(lowedges, upedges)):
                if i == 0:   
                    plt.plot(scapexarcs, np.tile(.126*borderdown, len(scapexarcs)), color='k')
                elif i != 0 and i != len(lowedges)-1:
                    plt.plot(scapexarcs, np.tile(.126*borderdown, len(scapexarcs)), 
                             linestyle='--', color='k')
                elif i == len(lowedges)-1:
                    plt.plot(scapexarcs, np.tile(.126*borderdown, len(scapexarcs)), 
                             linestyle='--', color='k')
                    plt.plot(scapexarcs, np.tile(.126*borderup, len(scapexarcs)), color='k')
                
            plt.plot(np.tile(-120, len(scapeyarcs)), scapeyarcs, color='k')
            plt.plot(np.tile(120, len(scapeyarcs)), scapeyarcs, color='k')
            plt.xlabel(r"X [arcsec]", fontsize=20), plt.ylabel(r"Y [arcsec]", fontsize=20)
            plt.title(r"{}".format(plttitle), fontsize=26)
            plt.savefig(plotdir+"/"+pltsavetitle+"MEDnear.png")
            plt.show()
            plt.close()
            '''
                    
            
            # Filter out high interpolation values along each row
            '''
            newscape_i = np.tile(np.nan, scape_i.shape)
            for rownr, row in enumerate(scape_i):
                
                # Round row values which are above twice the median row value to the median value
                rowmed = np.nanmedian(row)
                if np.isnan(rowmed):
                    continue
                # Determine the root mean square error of the row 
                rowrms = np.sqrt(np.nanmean(row**2 - rowmed**2))
                # Remove pixels with a rms error greater than the mean rms error
                rowmask = (np.sqrt(row**2 - rowmed**2) > 3*rowrms)
                row[rowmask] = np.tile(rowmed, len(row[rowmask]))
                
                # Overwrite row
                newscape_i[rownr] = row
            
            
            
            # Contour the gridded data, plotting dots at the randomly spaced data points.
            CS = plt.contour(scapexarcs,scapeyarcs,newscape_i,15,linewidths=0.5)
            CS = plt.contourf(scapexarcs,scapeyarcs,newscape_i,15)
            # Plot data points
            plt.scatter(xarcs, yarcs, marker='o', s=50, c=val, cmap=CS.cmap, norm=CS.norm)
            plt.colorbar() # draw colorbar
            # Plot the slit boundaries
            for i, [borderdown, borderup] in enumerate(zip(lowedges, upedges)):
                if i == 0:   
                    plt.plot(scapexarcs, np.tile(.126*borderdown, len(scapexarcs)), color='k')
                elif i != 0 and i != len(lowedges)-1:
                    plt.plot(scapexarcs, np.tile(.126*borderdown, len(scapexarcs)), 
                             linestyle='--', color='k')
                elif i == len(lowedges)-1:
                    plt.plot(scapexarcs, np.tile(.126*borderdown, len(scapexarcs)), 
                             linestyle='--', color='k')
                    plt.plot(scapexarcs, np.tile(.126*borderup, len(scapexarcs)), color='k')
                
            plt.plot(np.tile(-120, len(scapeyarcs)), scapeyarcs, color='k')
            plt.plot(np.tile(120, len(scapeyarcs)), scapeyarcs, color='k')
            plt.xlabel(r"X [arcsec]", fontsize=20), plt.ylabel(r"Y [arcsec]", fontsize=20)
            plt.title(r"{}".format(plttitle), fontsize=26)
            plt.savefig(plotdir+"/"+pltsavetitle+"MEDmaskattempt.png")
            plt.show()      
            plt.close()
            '''  
            
            
            
            ########### WHOLE-CHIP BIVARIATE 3rd ORDER POLYNOMIAL ########### 
            # Third order bivariate polynomial fit
            polynom = polfun.polyfit2d(xarcs, yarcs, val, order=3)
            # Evalutate fitted polynomial at gridpoints
            polyfitdata = polfun.polyval2d(scapexarcs_grid, scapeyarcs_grid, polynom)
            
            # Contour the gridded data, plotting dots at the randomly spaced data points.
            levels = np.arange(-2,2.25,0.25)
            CS = plt.contour(scapexarcs,scapeyarcs,polyfitdata,levels, linewidths=0.5)
            CS = plt.contourf(scapexarcs,scapeyarcs,polyfitdata,levels, vmin=-2, vmax=2)
            # Plot data points
            plt.scatter(xarcs, yarcs, marker='o', s=50, c=val, cmap=CS.cmap, norm=CS.norm)
            plt.colorbar() # draw colorbar
            # Plot the slit boundaries
            for i, [borderdown, borderup] in enumerate(zip(lowedges, upedges)):
                if i == 0:   
                    plt.plot(scapexarcs, np.tile(.126*borderdown, len(scapexarcs)), 
                             linewidth=2, color='k')
                elif i != 0 and i != len(lowedges)-1:
                    plt.plot(scapexarcs, np.tile(.126*borderdown, len(scapexarcs)), 
                             linestyle='--', color='k')
                elif i == len(lowedges)-1:
                    plt.plot(scapexarcs, np.tile(.126*borderdown, len(scapexarcs)), 
                             linestyle='--', color='k')
                    plt.plot(scapexarcs, np.tile(.126*borderup, len(scapexarcs)), 
                             linewidth=2, color='k')
                
            plt.plot(np.tile(-120, len(scapeyarcs)), scapeyarcs, color='k', linewidth=2)
            plt.plot(np.tile(120, len(scapeyarcs)), scapeyarcs, color='k', linewidth=2)
            plt.xlabel(r"X [arcsec]", fontsize=20), plt.ylabel(r"Y [arcsec]", fontsize=20)
            plt.title(r"{}".format(plttitle), fontsize=26)
            plt.savefig(plotdir+"/"+pltsavetitle+"polyfit.png")
            #plt.show()
            plt.close()     
            
            
            # Append spline to list for storage
            polynoms.append(polyfitdata)
            splines.append(scape_i)
            #splinesnear.append(scape_inear)           
            
             
        # Save splines and polynomials to numpy savefile
        polfun.savenp(polynoms, npsavedir+"/interps2d", "polyfits")
        polfun.savenp(splines, npsavedir+"/interps2d", "cubicsplines")
    
    
        
    # Determine splines and polynomials of each individual slit
    if detsplines1d:
            
        # Compute slitwise (O-E)/(O+E)
        for n in np.arange(0,len(upedges),2):
            
            # Determine 1d polynomial interpolations over median c- and d-scapes
            poly1d_lst, spline1d_lst = [], []
            for scape, pltsavetitle, plttitle in zip([cscape_med, dscape_med], 
                                      #[maskind_c, maskind_d],
                                      ["c-scapeNGC+WD", "d-scapeNGC+WD"],
                                      [r"$\delta_x + c$", r"$\delta_y + d$"]):
                                      
                # Extract the x- and y-coordinates corresponding to points 
                # in c- and d-scapes
                xycoord = np.argwhere(~np.isnan(scape))
                points = np.dstack(xycoord)[0]
                x, y = points[1,:], points[0,:]
                # Compute gridpoints for evaluation
                scapex = np.arange(0,scape.shape[1],1) 
                scapey = np.arange(0,scape.shape[0],1)
                scapexarcs, scapeyarcs = (scapex - np.median(scapex))*.126, scapey*.126
                scapexarcs_grid, scapeyarcs_grid = np.meshgrid(scapexarcs, scapeyarcs)  
                # Determine the values of the points within c- and d-scape
                val = scape[y,x]
                
                
                # Select the x- and y-coordinates of c- and d-values on the current slit
                slitmask1 = (y>Oylow)*(y<Oyup)
                slitmask2 = (scapey>Oylow)*(scapey<Oyup)
                xslit, yslit, valslit = x[slitmask1], y[slitmask1], val[slitmask1]
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
                new_scapexslitarcs = np.arange(xslitarcs[np.argmin(xslitarcs)],
                                               xslitarcs[np.argmax(xslitarcs)],0.05)
                cubspl1d_v = cubspl1d_f(new_scapexslitarcs)
                cubspl1d_val = np.tile(cubspl1d_v, [len(scapeyslitarcs),1])
                
                # Contour the gridded data, plotting dots at the randomly spaced data points.
                levels = np.arange(-2,2.1,0.1)
                CS = plt.contour(new_scapexslitarcs,scapeyslitarcs,cubspl1d_val,
                                 levels,linewidths=0.5)
                CS = plt.contourf(new_scapexslitarcs,scapeyslitarcs,cubspl1d_val,
                                  levels,vmin=-2,vmax=2)
                # Plot data points
                plt.scatter(xslitarcs, yslitarcs, marker='o', s=50, c=valslit, 
                            cmap=CS.cmap, norm=CS.norm)
                plt.colorbar() # draw colorbar
                plt.xlabel(r"X [arcsec]", fontsize=20), plt.ylabel(r"Y [arcsec]", fontsize=20)
                plt.title(r"{}".format(plttitle), fontsize=26)
                plt.savefig(plotdir+"/splines/univ/{}slitpair{}.png".format(pltsavetitle, 
                                                                            int(n/2)))
                #plt.show()  
                plt.close()   
                
                                
                
                
                
                ########### SINGLE-SLIT BIVARIATE cubic spline ###########
                # Bivariate cubic splines using only slit datapoints
                scape_i = interpolate.griddata((xslitarcs, yslitarcs), valslit, 
                                               (scapexarcs[None,:], 
                                                scapeyarcs[slitmask2][:,None]),
                                                method='cubic')
                
                
                # Contour the gridded data, plotting dots at the randomly 
                # spaced data points.
                CS = plt.contour(scapexarcs,scapeyarcs[slitmask2],scape_i,15,
                                 linewidths=0.5, colors='k')
                CS = plt.contourf(scapexarcs,scapeyarcs[slitmask2],scape_i,15,vmin=-2, vmax=2)
                # Plot data points
                plt.scatter(xslitarcs, yslitarcs, marker='o', s=50, 
                            c=valslit, cmap=CS.cmap, norm=CS.norm)
                plt.colorbar() # draw colorbar
                plt.xlabel(r"X [arcsec]", fontsize=20)
                plt.ylabel(r"Y [arcsec]", fontsize=20)
                plt.title(r"{}".format(plttitle), fontsize=26)
                plt.savefig(plotdir+"/splines/biv/{}slitpair{}.png".format(pltsavetitle, 
                                                                            int(n/2)))
                #plt.show()  
                plt.close()     
                
                
                
                ########### SINGLE-SLIT BIVARIATE 3rd ORDER POLYNOMIAL ########### 
                # Third order bivariate polynomial fit
                polynom = polfun.polyfit2d(xslitarcs, yslitarcs, valslit, order=3)
                # Evalutate fitted polynomial at gridpoints
                polyfitdata = polfun.polyval2d(scapexslitarcs_grid, scapeyslitarcs_grid, polynom)
                
                # Contour the gridded data, plotting dots at the randomly spaced data points.
                levels = np.arange(-2,2.1,0.1)
                CS = plt.contour(scapexslitarcs,scapeyslitarcs,polyfitdata,levels, linewidths=0.5)
                CS = plt.contourf(scapexslitarcs,scapeyslitarcs,polyfitdata,levels, vmin=-2, vmax=2)
                # Plot data points
                plt.scatter(xslitarcs, yslitarcs, marker='o', s=50, c=valslit, 
                            cmap=CS.cmap, norm=CS.norm)
                plt.colorbar() # draw colorbar
                plt.xlabel(r"X [arcsec]", fontsize=20), plt.ylabel(r"Y [arcsec]", fontsize=20)
                plt.title(r"{}".format(plttitle), fontsize=26)
                plt.savefig(plotdir+"/polyfits/biv/{}slitpair{}.png".format(pltsavetitle, 
                                                                            int(n/2)))
                #plt.show()  
                plt.close()   
                
                
                
                
                ########### SINGLE-SLIT UNIVARIATE 3rd ORDER POLYNOMIAL ########### 
                # Third order univariate polynomial fit
                polynom1d = np.polyfit(xslitarcs[valmask], valslit[valmask], deg=3)
                # Evaluate the derived polynomials
                polyval1d = np.polyval(polynom1d, scapexarcs)
                polyfitdata1d = np.tile(polyval1d, [len(scapeyarcs[slitmask2]),1])
                
                
                fig, ax = plt.subplots(1)
                polplot = ax.imshow(polyfitdata1d, origin='lower', 
                                     extent=[scapexarcs[0], 
                                             scapexarcs[-1],
                                             scapeyarcs[slitmask2][0], 
                                             scapeyarcs[slitmask2][-1]],
                                     aspect='auto', vmin=-2, vmax=2)
                ax.scatter(xslitarcs[valmask], yslitarcs[valmask], 
                           marker='o', s=50, c=valslit[valmask], 
                           cmap=polplot.cmap, norm=polplot.norm)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(polplot, cax=cax)
                ax.set_xlabel("X [arcsec]", fontsize=20)
                ax.set_ylabel("Y [arcsec]", fontsize=20)
                ax.set_title(plttitle, fontsize=26)
                plt.savefig(plotdir+"/polyfits/univ/{}slitpair{}.png".format(pltsavetitle, 
                                                                             int(n/2)))
                #plt.show()   
                plt.close()
                
                # Append 1d interpolation to list
                poly1d_lst.append(polyfitdata1d)
                spline1d_lst.append(scape_i)
            
            
            # Save to numpy savefiles
            polfun.savenp(poly1d_lst, npsavedir+"/interps1d", 
                          "slitp{}_polyfits1d".format(int(n/2)))
            polfun.savenp(spline1d_lst, npsavedir+"/interps1d", 
                          "slitp{}_cubicsplines1d".format(int(n/2)))

        




# I)    Apply offsets over data slits (using TESTdata)
    #i) First wholepixel offset
    #i) Second subpixel offset
# II)   Determine resulting normalized flux difference (apersum_old)
            # - What to do with backgrounds
Nxmin, Nymin = 1685, 80
for i, objdir in enumerate([std_dirs[0], std_dirs[1], sci_dirs[0]]):
    
    # Load 2d splines
    if detsplines2d == False:
        splines = np.load(npsavedir+"/interps2d/cubicsplines.npy")
        polynoms = np.load(npsavedir+"/interps2d/polyfits.npy")
    if i != 2:
        print("ONLY NGC4696!!!")
        continue
    
       
    # Create list with templates
    tpl_dirlst, tpl_flst = polfun.mk_lsts(objdir)
    tpl_dirlst = np.sort(tpl_dirlst)
    
    
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
    for j, tpl_name in enumerate(tpl_dirlst):#tpl_dirlst):
        print("\tTPL:\t{}".format(tpl_name))
        tpl_dir = objdir + '/' + tpl_name     
        
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
        O_E_knplst, O_E_grad_knplst, retan_lst = [], [], []
        for k, fname in enumerate(expfile_lst): 
            print("\n\t\t {}".format(fname))
            
            
            # Extract data
            header, data = polfun.extract_data(tpl_dir +'/'+ fname)
            # Determine the retarder waveplate angle
            retangle = header["HIERARCH ESO INS RETA2 POSANG"]
            retan_lst.append(retangle) #deg
            print("\t\tRetarder angle: \t {} deg".format(retangle))
            # De-biasing and flat-fielding corrections
            data = (data - Mbias) / Mflat_norm 
            
            # Extract the slits
            slits = [data[lowedges[i]:upedges[i],
                     chip_xyranges[0][0]:chip_xyranges[0][0]+np.min(slitshapes[:,1])]
                     for i in range(len(lowedges))]           
            
            
            # Compute slitwise (O-E)/(O+E)
            O_E_nplst, O_E_grad_nplst = [], []
            for n in np.arange(0,len(upedges),2):       
                print("\t\t\tSlitpair:\t{}".format(int(n/2)))
                
                if int(n/2) != 1:
                    print("\t\t\tSelect only NGC4696, skip this slit!\n")
                    continue
                
                # Load 1D splines 
                if not detsplines1d:
                    poly1d_lst = np.load(npsavedir + 
                                   "/interps1d/slitp{}_polyfits1d.npy".format(int(n/2)))
                    spline1d_lst = np.load(npsavedir + 
                                   "/interps1d/slitp{}_cubicsplines1d.npy".format(int(n/2)))
                    print("1D splines and polynomials loaded!")  
                
                
                # Select O and E
                E, O = slits[n], slits[n+1]
                # Adjust shapes to minimum slit shape, for computational ease
                ims_adjusted = []
                for im in [E,O]:
                    if im.shape[0] > Nymin:
                        print("yes!")
                        im = im[0:Nymin]
                        Nymin = im.shape[0]
                    elif im.shape[1] > Nxmin:
                        print("yes!")
                        im = im[:,0:Nxmin]
                        Nxmin = im.shape[1]
                    ims_adjusted.append(im)
                E, O = ims_adjusted
                print("DEBUG:\t\t{}\t{}".format(E.shape, O.shape))
                # Select the O and E slit y-ranges
                Eylow, Eyup = lowedges[n], lowedges[n]+E.shape[0]
                Oylow, Oyup = lowedges[n+1], lowedges[n+1]+O.shape[0]
                # Subtract backgrounds
                imcorr_lst = []
                for im in [E,O]:
                    sigma_clip = SigmaClip(sigma=3., iters=10)
                    bkg_estimator = MedianBackground()
                    bkg = Background2D(im, (20, 20), filter_size=(3, 3),
                                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
                    imcorr_lst.append(im - bkg.background)
                Ecorr, Ocorr = imcorr_lst
                # Ecorr, Ocorr = E, O #TODO WHY ARE THERE NO RESIDUALS ANYMORE AFTER APPLYING
                #                           WHOLE-SLIT OFFSETS TO THE CORRECTED SLITS???
                
                # Diagnostic plots
                '''
                plt.imshow(bkg.background, origin='lower', cmap='afmhot')
                plt.colorbar()
                plt.show()
                plt.close()

                plt.imshow(Ocorr-Ecorr, origin='lower', cmap='afmhot', vmax=18e3)
                plt.colorbar()
                plt.show()
                plt.close()       
                '''         
                
                # Determine overlaps for each individual slitpair via the 1d stacked 
                # offset interps
                O_E_plst, O_E_grad_plst = [], []
                for p, offsets_i in enumerate([polynoms, splines, poly1d_lst, spline1d_lst]):
                    print("\t\t\t\tOffsets:\t{}".format(p+1))  
                    
                    # Extract x- and y-offsets
                    offsx_i, offsy_i = offsets_i
                    
                    # Break if not detoverlaps1d
                    if not detoverlaps1d:
                        break
                    # Crop whole-chip interpolations to slit ranges
                    if (offsx_i.shape != (Oyup-Oylow, np.diff(chip_xyranges, axis=1)[0,0])
                        and (p+1) in [1,2]):
                        
                        offsetx_i = offsx_i[Oylow:Oyup, 
                                            chip_xyranges[0][0]:chip_xyranges[0][1]]
                        offsety_i = offsy_i[Oylow:Oyup, 
                                            chip_xyranges[0][0]:chip_xyranges[0][1]]
                                            
                    
                    O_E, O_E_grad = polfun.detslitdiffnorm([Ecorr,Ocorr], 
                                                           [offsetx_i, offsety_i], savefigs=True, 
                                                            plotdirec=pltsavedir+
                                                            "/tpl{}/exp{}/interp{}".format(j+1,
                                                                                       k+1,p+1),
                                                            imdirec=imsavedir+
                                                            "/tpl{}/exp{}/interp{}".format(j+1,
                                                                                       k+1,p+1),
                                                            slitNR=int(n/2))
                    
                    # Append to lists
                    O_E_plst.append(O_E), O_E_grad_plst.append(O_E_grad)
                O_E_nplst.append(O_E_plst), O_E_grad_nplst.append(O_E_grad_plst)    
            O_E_knplst.append(O_E_nplst), O_E_grad_knplst.append(O_E_grad_nplst)   
        
        # Sort k-level list according to ascending retarder waveplate angle
        O_E_knplstSORT = [i[0] for i in sorted(zip(O_E_knplst, retan_lst), key=lambda l: l[1])]
        O_E_grad_knplstSORT = [i[0] for i in sorted(zip(O_E_grad_knplst, retan_lst), 
                               key=lambda l: l[1])]
        # Compute a list, pollst, with pixel-wise U/I, Q/I, pL and phiL values
        O_E_knparr, O_E_grad_knparr = np.array([O_E_knplstSORT, O_E_grad_knplstSORT])
        pollstO_E_nparr = polfun.detpol(O_E_knparr)
        pollstO_E_grad_nparr = polfun.detpol(O_E_grad_knparr)
        
        plt.imshow(pollstO_E_nparr[0][0,1], origin='lower', cmap='afmhot', vmin=-1, vmax=1)
        plt.colorbar()
        plt.show()
        plt.close()

        plt.imshow(pollstO_E_grad_nparr[0][0,1], origin='lower', cmap='afmhot', vmin=-50, vmax=50)
        plt.colorbar()
        plt.show()
        plt.close()
        
        
        
        
        '''
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        V_min, V_max = -np.max(np.abs(gradxO[~np.isnan(gradxO)])), np.max(np.abs(gradxO[~np.isnan(gradxO)]))
        dat1 = ax1.imshow(gradxO.T, origin='lower', cmap="afmhot", 
                          vmin=V_min, vmax=V_max)
        dat2 = ax2.imshow(gradyO.T, origin='lower', cmap="afmhot", 
                          vmin=V_min, vmax=V_max)
        dat3 = ax3.imshow(O_E.T, origin='lower', cmap="afmhot", 
                          vmin=-np.max(np.abs(O_E[~np.isnan(O_E)])), 
                          vmax=np.max(np.abs(O_E[~np.isnan(O_E)])))
        plt.colorbar(dat1)
        for ax, title in zip([ax1, ax2, ax3], [r"$\nabla_x (O)$", r"$\nabla_y (O)$", r"$O-E$"]):
            ax.set_xlabel("X [pixel]", fontsize=20)
            if ax == ax1: ax.set_ylabel("Y [pixel]", fontsize=20)
            ax.set_title('{}'.format(title), fontsize=24)
            
        plt.show()
        plt.close()
        '''

            
            
            
            
            
            
            

