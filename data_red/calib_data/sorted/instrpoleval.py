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
for i, objdir in enumerate([std_dirs[0], std_dirs[1], sci_dirs[0]]):
    object_ = objdir.split("/")[-2]
    print("OBJECT:\t", object_)
    
    if i != 2:
        print("\nOnly NGC4696,IPOL used!!!\n")
        continue

    # Load offsets for testscidata. #TODO REPLACE WITH GENERIC OFFSETS TOMORROW (26-06-17)
    wholepixoffs_jkl = np.load(npsavedir+"/{}/optpixoffsets_i{}.npy".format(object_,i))
    subpixoffs_jkl = np.load(npsavedir+"/{}/optoffs_i{}.npy".format(object_,i))
    
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
    J = 0
    I_jkl, F_jkl, F_jklGRAD = np.tile(np.nan, [3,len(tpl_dirlst),4,len(star_lst)])
    UQPphi_jl, sigma_UQPphi_jl, UQPphi_jlGRAD, sigma_UQPphi_jlGRAD = np.tile(np.nan, [4,4,len(tpl_dirlst),len(star_lst)])
    for j, tpl_name in enumerate(tpl_dirlst):
        print("\tTPL:\t{}".format(tpl_name))
        tpl_dir = objdir + '/' + tpl_name
        print("DEBUG:\t{}".format(J))
        
        
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
        for k, fname in enumerate(expfile_lst):
            print("\n\t\t {}".format(fname))
            
            
            # Extract data
            header, data = polfun.extract_data(tpl_dir +'/'+ fname)
            filtername = header["HIERARCH ESO INS FILT1 NAME"]
            retangle = header["HIERARCH ESO INS RETA2 POSANG"]
            exptime = header["EXPTIME"]
            # De-biasing and flat-fielding corrections
            data = (data - Mbias) / Mflat_norm
            # Division by exposure time
            data = data / exptime
            # Extract the slits
            slits = [data[lowedges[m]:upedges[m],
                     chip_xyranges[0][0]:chip_xyranges[0][0]+np.min(slitshapes[:,1])]
                     for m in range(len(lowedges))]
            
            
            # Iterate over all selected stars
            for l, starpar in enumerate(star_lst):    
                
                # Set stellar radius, and annuli inner and outer radii
                R, anRmin, anRmax = starpar[3], 1.2*starpar[3], 2*starpar[3]                 
                
                # Extract ordinary and extraordinary slit
                slitEnr = 2*(starpar[4]-1)
                slitE, slitO = slits[slitEnr], slits[slitEnr + 1]
                # Determine the upper and lower edges of the slits
                upedgeE, upedgeO = upedges[slitEnr], upedges[slitEnr+1]
                lowedgeE, lowedgeO = lowedges[slitEnr], lowedges[slitEnr+1]
                #print("Slit pair {}".format(slitEnr/2))
                
                
                # CHECK!
                '''
                offsetopt, well, alignedim_well = polfun.offsetopt_well([slitE,slitO], 
                                                                         np.arange(-6,7), 
                                                                         np.arange(-6,7), 
                                                                         slitOcent, 6, saveims=False)
                print("pixoffscomp:\t{}\n\t\t{}".format(wholepixoffs_jkl[j,k,l],offsetopt))
                '''

                # Subtract backgrounds
                imcorr_lst = []
                for im in [slitE,slitO]:
                    sigma_clip = SigmaClip(sigma=3., iters=10)
                    bkg_estimator = MedianBackground()
                    bkg = Background2D(im, (20, 20), filter_size=(3, 3),
                                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
                    imcorr_lst.append(im - bkg.background)
                Ecorr, Ocorr = imcorr_lst
                # Compute the stellar location on slit O
                slitOcent = polfun.find_center([starpar[0]-chip_xyranges[0][0], starpar[1]-lowedgeO],
                                                Ocorr, 15)
                
                
                # Create cutout
                cutxmin, cutxmax = max(0, slitOcent[0]-35), min(slitO.shape[1]-1, slitOcent[0]+35)
                cutymin, cutymax = max(0, slitOcent[1]-35), min(slitO.shape[0]-1, slitOcent[1]+35)
                cutoutO = Ocorr[cutymin:cutymax, cutxmin:cutxmax]
                cutoutE = Ecorr[cutymin:cutymax, cutxmin:cutxmax]
                cutoutOcent = (slitOcent - np.rint([cutxmin,cutymin])).astype(int)
                # Apply whole pixel-accuracy offset
                framesize = 2*np.array(cutoutO.shape)
                lowlcorn = (0.25*framesize).astype(int)
                embedE_bg = polfun.embed(cutoutE, framesize, offset=wholepixoffs_jkl[J,k,l],
                                         cornerpix=lowlcorn)
                embedO_bg = polfun.embed(cutoutO, framesize, offset=[0,0], cornerpix=lowlcorn)
                embedOcent = np.rint(0.25*framesize[[1,0]]).astype(int) + cutoutOcent

                
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
                
                # Embed O and E in larger array for alignment
                '''
                framesize = 2*np.array(Ocorr.shape)
                lowlcorn = (0.25*framesize).astype(int)
                embedE_bg = polfun.embed(Ocorr, framesize, offset=list(wholepixoffs_jkl[j,k,l]),
                                         cornerpix=lowlcorn)
                embedO_bg = polfun.embed(Ecorr, framesize, cornerpix=lowlcorn)    
                '''
                
                # Compute gradients
                gradOy, gradOx = np.gradient(embedO_bg)                                    
                
                # Determine O - E and O + E
                O_E_bg = embedO_bg - embedE_bg
                OplusE_bg = embedO_bg + embedE_bg
                # Subtract gradient 
                O_E_bg_grad = (O_E_bg - subpixoffs_jkl[J,k,l,0]*gradOx 
                                      - subpixoffs_jkl[J,k,l,1]*gradOy)
                # Determine the normalized slit difference
                slitdiffcorr = O_E_bg_grad / OplusE_bg
                
                
                # Determine background flux
                '''
                backgrE, backgrEerr = polfun.ansum(embedE, embedOcent[0], embedOcent[1], 
                                                   minAn_r=anRmin, maxAn_r=anRmax)
                backgrO, backgrOerr = polfun.ansum(embedO, embedOcent[0], embedOcent[1], 
                                                   minAn_r=anRmin, maxAn_r=anRmax)
                '''
                
                # Compute stellar location on O embed
                '''
                embedOcent = polfun.find_center(lowlcorn[[1,0]]+slitOcent, embedO_bg, 15)
                '''
                
                
                
                # Diagnostic plots
                '''
                plt.imshow(OplusE_bg, origin='lower', cmap='afmhot')
                plt.colorbar()
                plt.scatter(embedOcent[0], embedOcent[1], color='b', s=50)
                plt.show()
                plt.close                 

                plt.imshow(O_E_bg, origin='lower', cmap='afmhot')
                plt.colorbar()    
                plt.scatter(embedOcent[0], embedOcent[1], color='b', s=50)
                plt.show()
                plt.close

                plt.imshow(O_E_bg_grad, origin='lower',cmap='afmhot')
                plt.colorbar()
                plt.scatter(embedOcent[0], embedOcent[1], color='b', s=50)
                plt.show()
                plt.close    
                    
                plt.imshow(slitdiffcorr, origin='lower', cmap='afmhot')
                plt.colorbar()
                plt.scatter(embedOcent[0], embedOcent[1], color='b', s=50)
                plt.show()
                plt.close  
                '''
                  
                                                                
                # Determine the normalized corrected slit difference at the stellar location
                I, I_err = polfun.apersum_old(OplusE_bg, embedOcent[0], embedOcent[1], starpar[3])
                F, F_err = polfun.apersum_old(O_E_bg/OplusE_bg, embedOcent[0], embedOcent[1], 
                                              starpar[3])
                FGRAD, F_errGRAD = polfun.apersum_old(slitdiffcorr, embedOcent[0], embedOcent[1],
                                                      starpar[3])
                # Append to lists
                kind = np.argwhere(np.array([0., 22.5, 45., 67.5]) == retangle)[0,0]
                I_jkl[j,kind,l], F_jkl[j,kind,l], F_jklGRAD[j,kind,l] = I[0], F[0], FGRAD[0]
                
        # Determine the Stokes parameter and polarization degrees
        if filtername == "b_HIGH":
            offsetangle = 2.*1.54
        elif filtername == "v_HIGH":
            offsetangle = 2.*1.8
        temp = polfun.detpol(np.array(F_jkl[j,:,:]), S_N=np.median(np.sqrt(I_jkl[j,:,:]),axis=0),
                             corran=offsetangle)
        UQPphi_jl[:,j] = np.array(temp[0])
        sigma_UQPphi_jl[:,j] = np.array(temp[1])
        
        temp = polfun.detpol(np.array(F_jklGRAD[j,:,:]), S_N=np.median(np.sqrt(I_jkl[j,:,:]),axis=0),
                             corran=offsetangle)
        UQPphi_jlGRAD[:,j] = np.array(temp[0])
        sigma_UQPphi_jlGRAD[:,j] = np.array(temp[1])        
        
        # Add 1 to index
        J += 1
    
    # Save resultant lists
    polfun.savenp(I_jkl, datasavedir, "I_i{}jkl".format(i))
    polfun.savenp(F_jkl, datasavedir, "F_i{}jkl".format(i))
    polfun.savenp(F_jklGRAD, datasavedir, "F_i{}jklGRAD".format(i))
    polfun.savenp(UQPphi_jl, datasavedir, "UQPphi_i{}jl".format(i))        
    polfun.savenp(UQPphi_jlGRAD, datasavedir, "UQPphi_i{}jlGRAD".format(i))        
    polfun.savenp(sigma_UQPphi_jl, datasavedir, "sigma_UQPphi_i{}jl".format(i))        
    polfun.savenp(sigma_UQPphi_jlGRAD, datasavedir, "sigma_UQPphi_i{}jlGRAD".format(i))          
    
    
    
    # Plot polarization degrees as function of FOV distance
    stardists = (.126/60.) * np.sqrt((star_lst[:,0]-1024)**2 + star_lst[:,1]**2)  #arcmin
    f, ax = plt.subplots(1)
    for j, P_l in enumerate(np.array(UQPphi_jl)[2]):
        ax.scatter(stardists, 100.*P_l)

    ax.set_ylim(ymin=0, ymax=100)
    ax.set_xlabel(r"Radial distance [arcmin]", fontsize=20)
    ax.set_ylabel(r"p [%]", fontsize=20)
    ax.set_title(r"Radial polarization profile", fontsize=26)
    plt.show()
    plt.close()
                
                
                

