import numpy as np
import funct as polfun
from astropy.io import fits
from itertools import product as carthprod
import shutil
import os
import re

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
# Boolean variable for switchin on polarization computations of selected stars
compute_anew = False
calc_well, calc_cd = True, True
# Boolean for determining whether the jkl_lsts are already known
jkl_loaded = False



# Compute fluxes and polarizations for selected stars in testdata and carry out slit appenditure
if compute_anew == True:
    compute_fluxlsts(sci_dirs, Mbias, Mflat_norm, star_lsts, r_range)



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




# Initialize lists for storing results as well as the offset ranges
dxrange, dyrange, interpf = np.arange(-6,7,1), np.arange(-6,7,1), 5
optpixoffsets, wells, interpslits = [], [], []
# Define aperture and anulus radii
R, anRmin, anRmax = 6, int(1.2*6), int(2*6)
# Lists for storing the optimal Q and c-/d-values
Qopts, opts_cd = [], []





# Compute the offset wells for all stars within all exposures of all templates
# Iterate over all objects (Vela1_95, WD1615_154, NGC4696)
wells_ijkl, offs_ijkl = [], []
Qopt_ijkl, optcd_ijkl = [], []
cscape_ijkl, dscape_ijkl = [], []
for i, objdir in enumerate([std_dirs[0], std_dirs[1], sci_dirs[0]]):
    print("OBJECT:\t", objdir.split("/")[-2])
    
    if i != 2:
        print("\nOnly NGC4696,IPOL used!!!\n")
        continue
    
    # Select slits and starlist corresponding to current object
    star_lst = star_lsts[i]
    
    # Create list with templates
    tpl_dirlst, tpl_flst = polfun.mk_lsts(objdir)
    tplNRlst = [int(re.findall('\d+', temp)[0]) for temp in tpl_dirlst]
    tpl_dirlst = [i[0] for i in sorted(zip(tpl_dirlst, tplNRlst), key=lambda l: l[1])] #Sort list
    
    
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
    wells_jkl, offs_jkl = [], []
    Qopt_jkl, optcd_jkl = [], []
    cscape_jkl, dscape_jkl = [], []
    for j, tpl_name in enumerate(tpl_dirlst):
        print("\tTPL:\t", tpl_name)
        tpl_dir = objdir + '/' + tpl_name
        
        
        print("DEBUG:\t{}".format(offs_jkl))
            
        
        # Create a list with filenames of files stored within tpldir
        expdir_lst, expfile_lst = polfun.mk_lsts(tpl_dir)
        expfile_lst = np.sort(expfile_lst)
        
        
        # Skip non-usable templates (non-usable templates should be put in a folder "skipped" or an equivalent directory which doesn't start with the string "tpl") or incomplete templates.
        if ((len(expfile_lst) != 4) or
            (objdir.split("/")[-2] == "Vela1_95" and tpl_name in ["tpl1", "tpl2", "tpl3"]) or
            (objdir.split("/")[-2] == "NGC4696,IPOL" and tpl_name == "tpl5")): #TODO INCLUDE TPL5
            print("\t skipped")
            continue
            
        if os.path.exists(datasavedir+"/tpl{}".format(j+1) + 
                          "/dscape_i{}j{}.npy".format(i+1,j+1)):
            
            direc = datasavedir+"/tpl{}".format(j+1)
            offs_kl = np.load(direc + "/optpixoffsets_i{}j{}.npy".format(i+1,j+1))
            wells_kl = np.load(direc + "/wells_i{}j{}.npy".format(i+1,j+1))
            Qopt_kl = np.load(direc + "/Qopt_i{}j{}.npy".format(i+1,j+1))
            optcd_kl = np.load(direc + "/optoffs_i{}j{}.npy".format(i+1,j+1))
            cscape_kl = np.load(direc + "/cscape_i{}j{}.npy".format(i+1,j+1))
            dscape_kl = np.load(direc + "/dscape_i{}j{}.npy".format(i+1,j+1))
            print("\n\t\tTemplate offsets and wells loaded!\n")
        else:
            # Iterate over all exposures
            wells_kl, offs_kl = [], []
            Qopt_kl, optcd_kl = [], []
            cscape_kl, dscape_kl = [], []
            for k, fname in enumerate(expfile_lst):
                print("\n\t\t {}".format(fname))
                
                
                # Load the pixel-accurate offsets if available 
                if os.path.exists(datasavedir+"/tpl{}/exp{}".format(j+1,k+1) + 
                                     "/dscape_i{}j{}k{}.npy".format(i+1,j+1,k+1)):            
                    
                    direc = datasavedir+"/tpl{}/exp{}".format(j+1,k+1)
                    offs_l = np.load(direc + "/optpixoffsets_i{}j{}k{}.npy".format(i+1,j+1,k+1))
                    wells_l = np.load(direc + "/wells_i{}j{}k{}.npy".format(i+1,j+1,k+1))
                    Qopt_l = np.load(direc + "/Qopt_i{}j{}k{}.npy".format(i+1,j+1,k+1))
                    optcd_l = np.load(direc + "/optoffs_i{}j{}k{}.npy".format(i+1,j+1,k+1))
                    cscape = np.load(direc + "/cscape_i{}j{}k{}.npy".format(i+1,j+1,k+1))
                    dscape = np.load(direc + "/dscape_i{}j{}k{}.npy".format(i+1,j+1,k+1))
                    print("\n\t\t\t\tExposure offsets and wells loaded!\n")  
                else:
                    # Extract data
                    header, data = polfun.extract_data(tpl_dir +'/'+ fname)
                    # De-biasing and flat-fielding corrections
                    data = (data - Mbias) / Mflat_norm
                    
                    # Extract the slits
                    slits = [data[lowedges[i]:upedges[i],
                             chip_xyranges[0][0]:chip_xyranges[0][0]+np.min(slitshapes[:,1])]
                             for i in range(len(lowedges))]
                    
                    # Iterate over all selected stars
                    wells_l, offs_l = [], []
                    for l, starpar in enumerate(star_lst):                
                        
                        
                        # Check whether to compute cdscapes
                        if calc_well == False:
                            break
                        print("\n\nComputing offset wells...")
                        print("\n\nStarno:\t\t{}".format(l+1))
                        
                        
                        # Extract ordinary and extraordinary slit
                        slitEnr = 2*(starpar[4]-1)
                        slitE, slitO = slits[slitEnr], slits[slitEnr + 1]
                        '''
                        # Adjust shape so that O and E have equal size
                        Nx, Ny = min(slitE.shape[1], slitO.shape[1]), min(slitE.shape[0], slitO.shape[0])
                        slitE, slitO = slitE[0:Ny,0:Nx], slitO[0:Ny,0:Nx]
                        ''' #TODO REMOVE IF POSSIBLE
                        '''
                        # Determine slit interpolations
                        interpE = interp(slitE, interpf*slitE.shape[1], interpf*slitE.shape[0])
                        interpO = interp(slitO, interpf*slitO.shape[1], interpf*slitO.shape[0])
                        interpslits.append(interpE), interpslits.append(interpO)
                        '''
                        # Determine the upper and lower edges of the slits
                        upedgeE, upedgeO = upedges[slitEnr], upedges[slitEnr+1]
                        lowedgeE, lowedgeO = lowedges[slitEnr], lowedges[slitEnr+1]
                        print("Slit pair {}".format(slitEnr/2))
                        
                        
                        # Compute stellar location on O slit
                        slitOcent = polfun.find_center([starpar[0]-chip_xyranges[0][0], starpar[1]-lowedgeO],
                                                        slitO, 15)
                        
                        
                        # Diagnostic plots
                        '''
                        plt.figure()
                        plt.imshow(np.log(slitO), origin='lower')
                        plt.colorbar()
                        plt.scatter(slitOcent[0], slitOcent[1], c='k', s=50)
                        plt.scatter(starpar[0]-chip_xyranges[0][0], starpar[1]-lowedgeO, c='k', s=50)
                        plt.show()
                        plt.close()
                        
                        plt.figure()
                        plt.imshow(np.log(slitE), origin='lower')
                        plt.colorbar()
                        plt.scatter(slitOcent[0], slitOcent[1], c='k', s=50)
                        plt.scatter(starpar[0]-chip_xyranges[0][0], starpar[1]-lowedgeO, c='k', s=50)
                        plt.show()
                        plt.close()  
                        '''              
                        # IS OK #TODO POSSIBLY REMOVE
                        
                        
                        # Compute wells for original slits
                        offsetopt, well, alignedim_well = polfun.offsetopt_well([slitE,slitO], 
                                                                         dxrange, dyrange, 
                                                                         slitOcent, R, saveims=True, 
                                            pltsavedir=pltsavedir+"/tpl{}/exp{}/star{}".format(j+1,k+1,l+1),
                                            imsavedir=imsavedir+"/tpl{}/exp{}/star{}".format(j+1,k+1,l+1))   
                        offs_l.append(offsetopt), wells_l.append(well)
                    
                    
                    
                    
                    
                    # DETERMINE C- AND D-VALUES OF ALL STARS
                    Qopt_l, optcd_l = [], []
                    cscape = np.tile(np.nan, np.array(Mbias.shape))
                    dscape = np.tile(np.nan, np.array(Mbias.shape))
                    for l, starpar in enumerate(star_lst):
                                        
                        # Check whether to compute cdscapes
                        if calc_cd == False:
                            break
                        print("\n\t\t\tComputing c- and d-scapes...")
                        print("\n\t\t\tStarno:\t\t{}".format(l+1))
                        # Define aperture and anulus radii
                        R, anRmin, anRmax = starpar[3], 1.2*starpar[3], 2*starpar[3]    
                        
                        
                        # Load the pixel-accurate offsets
                        if calc_well == False:
                            offs_l = np.load(datasavedir+"/tpl{}/exp{}".format(j,k) + 
                                             "/optpixoffsets_i{}j{}k{}".format(i,j,k))
                            #wells = np.load(datasavedir+"/wells_{}.npy".format(datasavedir.split("/")[-1]))
                        
                        
                        # Extract ordinary and extraordinary slit
                        slitEnr = 2*(starpar[4]-1)
                        slitE, slitO = slits[slitEnr], slits[slitEnr + 1]
                        '''
                        # Adjust shape so that O and E have equal size
                        Nx, Ny = min(slitE.shape[1], slitO.shape[1]), min(slitE.shape[0], slitO.shape[0])
                        slitE, slitO = slitE[0:Ny,0:Nx], slitO[0:Ny,0:Nx]
                        ''' #TODO REMOVE IF POSSIBLE
                        '''
                        # Determine slit interpolations
                        interpE = interp(slitE, interpf*slitE.shape[1], interpf*slitE.shape[0])
                        interpO = interp(slitO, interpf*slitO.shape[1], interpf*slitO.shape[0])
                        interpslits.append(interpE), interpslits.append(interpO)
                        '''
                        # Determine the upper and lower edges of the slits
                        upedgeE, upedgeO = upedges[slitEnr], upedges[slitEnr+1]
                        lowedgeE, lowedgeO = lowedges[slitEnr], lowedges[slitEnr+1]
                        print("Slit pair {}".format(slitEnr/2))
                        
                            
                        
                        # Compute stellar location on O slit and on the template appended slit
                        slitOcent = polfun.find_center([starpar[0]-chip_xyranges[0][0], starpar[1]-lowedgeO],
                                                 slitO, 15)
                        dataOcent = polfun.find_center([starpar[0], starpar[1]],
                                                 data, 15)
                        
                        
                        '''
                        appendedOcent = polfun.find_center([slitOcent[0], 
                                                     slitOcent[1]+np.sum(slitwidths[[m for m in np.arange(0,n)]])],
                                                     aligntemp, 25) #TODO SAVE TO NP FILE
                        print("appendedOcent:\t\t", appendedOcent)
                        
                        plt.figure()
                        plt.imshow(data, origin='lower', cmap='rainbow')
                        plt.colorbar()
                        plt.scatter(appendedOcent[0], appendedOce[1], s=30, c='k')
                        plt.show()
                        plt.close()
                        '''
                        
                        
                        # Create cutout
                        cutxmin, cutxmax = max(0, slitOcent[0]-35), min(slitO.shape[1]-1, slitOcent[0]+35)
                        cutymin, cutymax = max(0, slitOcent[1]-35), min(slitO.shape[0]-1, slitOcent[1]+35)
                        cutoutO = slitO[cutymin:cutymax, cutxmin:cutxmax]
                        cutoutE = slitE[cutymin:cutymax, cutxmin:cutxmax]
                        cutoutOcent = (slitOcent - np.rint([cutxmin,cutymin])).astype(int)
                        # Apply whole pixel-accuracy offset
                        framesize = 2*np.array(cutoutO.shape)
                        lowlcorn = (0.25*framesize).astype(int)
                        embedE = polfun.embed(cutoutE, framesize, offset=offs_l[l], cornerpix=lowlcorn)
                        embedO = polfun.embed(cutoutO, framesize, offset=[0,0], cornerpix=lowlcorn)
                        embedOcent = np.rint(0.25*framesize[[1,0]]).astype(int) + cutoutOcent
                        
                        
                        # Determine the background fluxes in E and O
                        backgrE, backgrEerr = polfun.ansum(embedE, embedOcent[0], embedOcent[1], 
                                                    minAn_r=anRmin, maxAn_r=anRmax)
                        backgrO, backgrOerr = polfun.ansum(embedO, embedOcent[0], embedOcent[1], 
                                                    minAn_r=anRmin, maxAn_r=anRmax)
                        # Correct the embeded images for the background flux
                        embedOcorr = embedO - backgrO[2]
                        embedEcorr = embedE - backgrE[2]
                        
                        
                        # Diagnostic plot
                        '''
                        plt.figure()
                        plt.imshow(embedOcorr, origin='lower', cmap='rainbow')
                        plt.colorbar()
                        plt.scatter(embedOcent[0], embedOcent[1], s=30, c='k')
                        plt.show()
                        plt.close()
                        
                        plt.figure()
                        plt.imshow(embedEcorr, origin='lower', cmap='rainbow')
                        plt.colorbar()
                        plt.scatter(embedOcent[0], embedOcent[1], s=30, c='k')
                        plt.show()
                        plt.close()
                        '''
                        
                        
                        # Save O-E
                        polfun.savefits(embedOcorr-embedEcorr, imsavedir+"/tpl{}/exp{}/star{}".format(j+1,k+1,l+1), 
                                 "slitdiff_star{}".format(l+1)) 
                        polfun.savefits(cutoutO, imsavedir+"/tpl{}/exp{}/star{}".format(j+1,k+1,l+1), 
                                 "O_star{}".format(l+1))       
                        polfun.savefits(cutoutE, imsavedir+"/tpl{}/exp{}/star{}".format(j+1,k+1,l+1), 
                                 "E_star{}".format(l+1)) 
                        #TODO The allocation of the stellar center on the aligned image is OK
                        
                        
                        # Recall previous c and d values for current star
                        cval_prev, dval_prev = 0, 0
                        '''
                        dval_prev = dscape_prev[appendedOcent[1], appendedOcent[0]]
                        ''' # TODO REMOVE
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
                            gradopt, Qopt, opt_cd = polfun.offsetopt_cd(embedOcorr, embedEcorr, crange, drange,
                                                                 embedOcent, starpar[3],
                                                                 savetofits=True, iteration=itno, 
                                                pltsavedir=pltsavedir+"/tpl{}/exp{}/star{}".format(j+1,k+1,l+1), 
                                                imsavedir=imsavedir+"/tpl{}/exp{}/star{}".format(j+1,k+1,l+1))
                            cval_prev, dval_prev = opt_cd
                            print("Qopt, opt_cd:\t\t", Qopt, opt_cd)
                            
                            
                        # Update c- and dscape
                        cscape[dataOcent[1],dataOcent[0]] = opt_cd[0]+offs_l[l][0]
                        dscape[dataOcent[1],dataOcent[0]] = opt_cd[1]+offs_l[l][1]
                        
                        # Append the best (i.e. last) optima parameters to stellar sublist
                        Qopt_l.append(Qopt), optcd_l.append(opt_cd)
                        
                # Append results to exposures sublist
                offs_kl.append(offs_l), wells_kl.append(wells_l)  
                Qopt_kl.append(Qopt_l), optcd_kl.append(optcd_l)
                cscape_kl.append(cscape), dscape_kl.append(dscape)         
                # Backup1
                if calc_well:
                    print("BACKUP1 offsets")
                    polfun.savenp(offs_l, datasavedir+"/tpl{}/exp{}".format(j+1,k+1), 
                           "optpixoffsets_i{}j{}k{}".format(i+1,j+1,k+1))
                    polfun.savenp(wells_l, datasavedir+"/tpl{}/exp{}".format(j+1,k+1), 
                           "wells_i{}j{}k{}".format(i+1,j+1,k+1))
                if calc_cd:
                    print("BACKUPU1 gradfact")
                    polfun.savenp(Qopt_l, datasavedir+"/tpl{}/exp{}".format(j+1,k+1), 
                           "Qopt_i{}j{}k{}".format(i+1,j+1,k+1))
                    polfun.savenp(optcd_l, datasavedir+"/tpl{}/exp{}".format(j+1,k+1), 
                           "optoffs_i{}j{}k{}".format(i+1,j+1,k+1))
                    polfun.savenp(cscape, datasavedir+"/tpl{}/exp{}".format(j+1,k+1), 
                           "cscape_i{}j{}k{}".format(i+1,j+1,k+1))
                    polfun.savenp(dscape, datasavedir+"/tpl{}/exp{}".format(j+1,k+1), 
                           "dscape_i{}j{}k{}".format(i+1,j+1,k+1))
                   
        # Append results to templates sublist
        offs_jkl.append(offs_kl), wells_jkl.append(wells_kl)    
        Qopt_jkl.append(Qopt_kl), optcd_jkl.append(optcd_kl)
        cscape_jkl.append(cscape_kl), dscape_jkl.append(dscape_kl)
        # Backup2
        if calc_well:
            print("BACKUP2 offsets")
            polfun.savenp(offs_kl, datasavedir+"/tpl{}".format(j+1), 
                   "optpixoffsets_i{}j{}".format(i+1,j+1))
            polfun.savenp(wells_kl, datasavedir+"/tpl{}".format(j+1), 
                   "wells_i{}j{}".format(i+1,j+1))
        if calc_cd:
            print("BACKUP2 gradfact")
            polfun.savenp(Qopt_kl, datasavedir+"/tpl{}".format(j+1), 
                   "Qopt_i{}j{}".format(i+1,j+1))
            polfun.savenp(optcd_kl, datasavedir+"/tpl{}".format(j+1), 
                   "optoffs_i{}j{}".format(i+1,j+1))
            polfun.savenp(cscape_kl, datasavedir+"/tpl{}".format(j+1), 
                   "cscape_i{}j{}".format(i+1,j+1))            
            polfun.savenp(dscape_kl, datasavedir+"/tpl{}".format(j+1), 
                   "dscape_i{}j{}".format(i+1,j+1))         
    
    # Append results to objects sublist
    offs_ijkl.append(offs_jkl), wells_ijkl.append(wells_jkl) 
    Qopt_ijkl.append(Qopt_jkl), optcd_ijkl.append(optcd_jkl)
    cscape_ijkl.append(cscape_jkl), dscape_ijkl.append(dscape_jkl)
    # Backup3
    if calc_well:
        print("BACKUP3 offsets")
        polfun.savenp(offs_jkl, datasavedir, "optpixoffsets_i{}".format(i))
        polfun.savenp(wells_jkl, datasavedir, "wells_i{}".format(i))
    if calc_cd:
        print("BACKUP3 gradfact")
        polfun.savenp(Qopt_jkl, datasavedir, "Qopt_i{}".format(i))
        polfun.savenp(optcd_jkl, datasavedir, "optoffs_i{}".format(i))
        polfun.savenp(cscape_jkl, datasavedir, "cscape_i{}".format(i))
        polfun.savenp(dscape_jkl, datasavedir, "dscape_i{}".format(i))    
    
    






# Determine bivariate third order polynomial fit to c- and dscapes if calc_cd==True
if not calc_cd and not calc_well:

    # Load the c- and d-scapes
    cscape_ijkl = np.load(npsavedir+"/cscape.npy")
    dscape_ijkl = np.load(npsavedir+"/dscape.npy")
    
    # ARTIFICIAL POINT
    '''
    tempc, tempd = cscape[266:lowedges[4]], dscape[266:lowedges[4]]
    cscape[367,843] = np.median(tempc[~np.isnan(tempc)])
    dscape[367,843] = np.median(tempd[~np.isnan(tempd)])
    dscape[320,873] = np.median(tempd[~np.isnan(tempd)])
    dscape[320,1032] = np.median(tempd[~np.isnan(tempd)])
    '''
    
    # High influence points
    maskind_c = [38, 27]
    maskind_d = [22, 29, 45, 26, 50, 49, 19, 17, 23, 39, 40, 32, 5]
    
    # Plot cubic splines for both c- and d-
    for scape, maskind, pltsavetitle, plttitle in zip([cscape, dscape], [maskind_c, maskind_d],
                                                      ["c-scape2", "d-scape2"],
                                                      [r"$\delta_x + c$", r"$\delta_y + d$"]):
        
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
        
        # Determine cubic spline to the c- and d-values
        scape_i = interpolate.griddata((xarcs, yarcs), val, 
                                        (scapexarcs[None,:], scapeyarcs[:,None]), method='linear')

        # Contour the gridded data, plotting dots at the randomly spaced data points.
        CS = plt.contour(scapexarcs,scapeyarcs,scape_i,15,linewidths=0.5)
        CS = plt.contourf(scapexarcs,scapeyarcs,scape_i,15)
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
        plt.savefig(plotdir+"/"+pltsavetitle+".png")
        plt.show()
        
        
        '''
        # Mask high residual points
        mval = np.delete(val, maskind)
        m_x, m_y = np.delete(x, maskind), np.delete(y, maskind)
        m_xarcs, m_yarcs = np.delete(xarcs, maskind), np.delete(yarcs, maskind)
        
        # Determine cubic spline to the c- and d-values
        mscape_i = interpolate.griddata((m_xarcs, m_yarcs), mval, 
                                        (scapexarcs[None,:], scapeyarcs[:,None]), method='linear')    
        
        # contour the gridded data, plotting dots at the randomly spaced data points.
        CS = plt.contour(scapexarcs,scapeyarcs,mscape_i,15,linewidths=0.5)
        CS = plt.contourf(scapexarcs,scapeyarcs,mscape_i,15)
        # plot data points.
        plt.scatter(m_xarcs, m_yarcs, marker='o', s=50, c=mval, cmap=CS.cmap, norm=CS.norm)
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
        plt.savefig(plotdir+"/"+pltsavetitle+"v2.png")
        plt.show() 
        '''
    
    
    
    


'''
c_xycoord, d_xycoord = np.argwhere(~np.isnan(cscape)), np.argwhere(~np.isnan(dscape))
cpoints, dpoints = np.dstack(c_xycoord)[0], np.dstack(d_xycoord)[0]
c_x, c_y, d_x, d_y = cpoints[1,:], cpoints[0,:], dpoints[1,:], dpoints[0,:]
cval, dval = cscape[c_y, c_x], dscape[d_y, d_x]

# Compute gridpoints for evaluation
scapex, scapey = np.arange(0,cscape.shape[1],1), np.arange(0,cscape.shape[0],1)
scape_xgrid, scape_ygrid = np.meshgrid(scapex, scapey)
# Compute coordinates in arcseconds
c_xarcs, c_yarcs = np.array([c_x - np.median(scapex), c_y]) * .126
d_xarcs, d_yarcs = np.array([d_x - np.median(scapex), d_y]) * .126
scapexarcs = (scapex - np.median(scapex))*.126
scapeyarcs = scapey*.126
scapexarcs_grid, scapeyarcs_grid = np.meshgrid(scapexarcs, scapeyarcs)

# Third order bivariate polynomial fit
polynom_c = polyfit2d(c_xarcs, c_yarcs, cval, order=3)
polynom_d = polyfit2d(d_xarcs, d_yarcs, dval, order=3)
# Evalutate fitted polynomial at gridpoints
polyfitdata_c = polyval2d(scapexarcs_grid, scapeyarcs_grid, polynom_c)
polyfitdata_d = polyval2d(scapexarcs_grid, scapeyarcs_grid, polynom_d)
'''

'''
# Third order univariate polynomial fit
polynom_c = np.polyfit(c_x[cval>-3.], cval[cval>-3.], 2)
polynom_d = np.polyfit(d_y[cval>-3.], dval[cval>-3.], 2)
# Evaluate the derived polynomials
polyval_c, polyval_d = np.polyval(polynom_c, c_x), np.polyval(polynom_d, d_x)
polyfitdata_c = np.tile(polyval_c, [len(scapey),len(scapex)])
polyfitdata_d = np.tile(polyval_d, [1,len(scapex)])
'''

'''
plt.scatter(c_xarcs, cval, c=dval)
plt.colorbar()
plt.title(r"c-xprofile")
plt.savefig(plotdir+"/xvsc.png")
plt.show()

plt.scatter(c_yarcs, cval, c=dval)
plt.colorbar()
plt.title(r"c-yprofile")
plt.savefig(plotdir+"/yvsc.png")
plt.show()

plt.scatter(d_xarcs, dval, c=dval)
plt.colorbar()
plt.title(r"d-xprofile")
plt.savefig(plotdir+"/xvsd.png")
plt.show()

plt.scatter(d_yarcs, dval, c=dval)
plt.colorbar()
plt.title(r"d-yprofile")
plt.savefig(plotdir+"/yvsd.png")
plt.show()






# Save results
polfun.savefits(polyfitdata_c, imdir, "cscapefitted")
polfun.savefits(polyfitdata_d, imdir, "dscapefitted")

# Save the third order polynomial fits as png images        
saveim_png(polyfitdata_c, plotdir, "cscape", 
          datextent=[scapexarcs[0],scapexarcs[-1],scapeyarcs[0],scapeyarcs[-1]], 
          scatterpoints=[c_xarcs, c_yarcs], scattercol=cval, 
          title="c-scape")

saveim_png(polyfitdata_d, plotdir, "dscape", 
          datextent=[scapexarcs[0],scapexarcs[-1],scapeyarcs[0],scapeyarcs[-1]], 
          scatterpoints=[d_xarcs, d_yarcs], scattercol=dval, 
          title="d-scape")

# Save to fits
polfun.savefits(cscape, imdir+"/cdscapes","cscape")
polfun.savefits(dscape, imdir+"/cdscapes", "dscape")


plt.imshow(polyfitdata_c, origin='lower', 
           extent=[scapexarcs[0],scapexarcs[-1],scapeyarcs[0],scapeyarcs[-1]])
plt.scatter(c_xarcs, c_yarcs, c=cval)
plt.colorbar()
plt.title(r"c-scape")
plt.savefig(plotdir+"/cscape.png")
plt.show()
    
plt.imshow(polyfitdata_d, origin='lower',
           extent=[scapexarcs[0],scapexarcs[-1],scapeyarcs[0],scapeyarcs[-1]])
plt.scatter(d_xarcs, d_yarcs, c=dval)
plt.colorbar()
plt.title(r"d-scape")
plt.savefig(plotdir+"/dscape.png")
plt.show()
    
    
    
    
    
# Mask high residual points
maskind = [22, 29, 45, 26, 50, 49, 19, 17]
mdval = np.delete(dval, maskind)
md_xarcs = np.delete(d_xarcs, maskind)
md_yarcs = np.delete(d_yarcs, maskind)

# Third order bivariate polynomial fit
mpolynom_d = polyfit2d(md_xarcs, md_yarcs-np.median(scapeyarcs), mdval, order=2)
# Evalutate fitted polynomial at gridpoints
mpolyfitdata_d = polyval2d(scapexarcs_grid, scapeyarcs_grid-np.median(scapeyarcs), mpolynom_d)    

plt.imshow(mpolyfitdata_d, origin='lower',
           extent=[scapexarcs[0],scapexarcs[-1],scapeyarcs[0],scapeyarcs[-1]])
plt.scatter(md_xarcs, md_yarcs, c=mdval)
plt.colorbar()
plt.title(r"d-scape")
plt.savefig(plotdir+"/mdscape.png")
plt.show()      
'''


'''
save3Dim_png(scapexarcs_grid, scapeyarcs_grid, dscape, 
             plotdir, "dscape3D", dataplttype='scatter',
             fit=True, fitXgrid=scapeyarcs_grid, fitYgrid=scapeyarcs_grid, 
             fitZdata=polyfitdata_d, fitdataplttype='surface',
             colmap='coolwarm', rowstride=1, colstride=1, lw=0,
             xtag=r'X [arcsec]', ytag=r'Y [arcsec]', ztag='c')
'''

