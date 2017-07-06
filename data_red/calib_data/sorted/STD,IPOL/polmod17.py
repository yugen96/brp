import numpy as np
from astropy.io import fits
import shutil
import os

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset





#################### FUNCTIONS #################################################
#################### Generic functions ####################



# Returns a list will all subdirectories in a specific folder as well as the
# contained files
def mk_lsts(dir_name):

    #dir_lst = [dirs[0] for dirs in os.walk(dir_name)]
    dir_lst = next(os.walk(dir_name))[1]
    file_lst = next(os.walk(dir_name))[2]
    return dir_lst, file_lst



# Checks whether fname is a fits or reg file and extracts the corresonding header and data or coordinates and radius respectively
def extract_data(fname):

    # Put data and header files into two separate lists
    if fname.endswith("fit") or fname.endswith("fits"):
        hdul = fits.open(fname)
        header = hdul[0].header             
        data = hdul[0].data
        hdul.close()
        
    return header, data



#################### END GENERIC FUNCTIONS ####################





#################### FUNCTIONS FOR FLUX LIST COMPUTATIONS ####################

    
    
# Calculates normalized flux/count differences between the ordinary and extraordinary target spectra or fluxes as specified on p.39 of the FORS2 user manual. The corresponding standard deviation is also computed. N.B.: The fluxes must have been derived for the same angle of the retarder waveplate!
def fluxdiff_norm(f_or, f_ex, sigma_or, sigma_ex):
    
    # Compute normalized flux difference
    flux_sum = f_or + f_ex
    fluxdiff_norm = (f_or - f_ex) / flux_sum
    
    # Compute standard deviation
    temp_or = f_ex**2 / flux_sum**4
    temp_ex = f_or**2 / flux_sum**4
    sigma_F = 2 * np.sqrt(temp_or * sigma_or**2 + temp_ex * sigma_ex**2)
        
    return fluxdiff_norm, sigma_F
    
    
    
# Finds a star's center in the neighbourhood of a certain pixel by iterating through a region surrounding this approximate pixel and storing the pixel containing the highest count rate
def find_center(coord, data_array, window_size):
    
    # Read data_array shape and the aproximate pixel's x- and y-coordinates
    aprox_x, aprox_y = coord[0], coord[1]
    Ny, Nx = data_array.shape
    
    # Define window corners
    xmin = max(1, aprox_x - window_size)
    xmax = min(Nx, aprox_x + window_size)
    ymin = max(1, aprox_y - window_size)
    ymax = min(Ny, aprox_y + window_size)
    
    # Iterate through region and find highest pixel value
    countmax = 0
    for x in np.arange(xmin, xmax, 1):
        for y in np.arange(ymin, ymax, 1):
            
            pix_val = data_array[y,x]
            if pix_val > countmax:
                countmax = pix_val
                center = [x,y]
                
    return center
                
            
    
# Function which calculates the aperture count rate for a star centered at pixel coordinates [px, py] for an aperture radius r
def apersum(image, px, py, r):
    
    
    # Determine the aperture limits
    ny, nx = image.shape
    apx_min = max(1, px - r)
    apx_max = min(nx, px + r)
    apy_min = max(1, py - r)
    apy_max = min(ny, py + r)

    
    # Compute the total count rate within the aperture
    apsum = 0.0
    for i in range(apx_min, apx_max+1):
        for j in range(apy_min, apy_max+1):
            
            # Calculate squared distance to central pixel
            dx = i - px
            dy = j - py
            d2 = dx**2 + dy**2
            
            # Store the current pixel's count value
            pixval = image[j-1,i-1]
            
            if d2 <= r**2:
                apsum += pixval
                
    return apsum
                       
    

# Function which computes normalized flux differences as well as the ordinary and extraordinary counts for a preselection regions in various images defined by 'loc_lsts'. 
def compute_fluxlsts(std_dirs, bias, masterflat_norm, loc_lsts, r_range, datasavedir=None):

    # Compute the linear polarization degrees for each template of exposures taken of Vela1_95 and WD1615_154
    obj_names = ["Vela1_95", "WD1615 154"]
    for i, std_dir in enumerate(std_dirs): 
        print("\n\n\n{}".format(std_dir))
            
            
            
        # Create a list with all the template directories within std_dirs
        tpl_dirlst, tpl_flst = mk_lsts(std_dir)
        tpl_dirlst = np.sort(tpl_dirlst)

        # A 1D lst for storing the exposures containing the least amount of stars per template
        least_expnos = np.zeros(len(tpl_dirlst), dtype=np.int64) 
        
           
        
        # Initiate lists containing the ordinary flux, the extraordinary flux and the normalized flux differences, for each template (axis=0), each exposure file (axis 1), each selected star within the exposure (axis2), and each aperture radius (axis 3)
        O_0lst, sigmaO_0lst = [], []
        E_0lst, sigmaE_0lst = [], []
        F_0lst, sigmaF_0lst = [], []
        # List for storing the filters of each exposure as well as the q-index and xcoordinate of the STD star within each template (axis1) and each exposure (axis2)
        filter_lst = []
        for j, tpl_name in enumerate(tpl_dirlst):
            print("\n\n\t {}".format(tpl_name))
            tpl_dir = std_dir + '/' + tpl_name         
            
            
            # Create a list with filenames of files stored within tpldir
            expdir_lst, expfile_lst = mk_lsts(tpl_dir)
            expfile_lst = np.sort(expfile_lst)

            # Initial setting for the least amount of detected stars within the template
            N_stars = 1e18
            
            
            # Skip non-usable templates (non-usable templates should be put in a folder "skipped" or an equivalent directory which doesn't start with the string "tpl") or incomplete templates.
            if ((std_dir.split("/")[-2] == "Vela1_95" and tpl_name in ["tpl1", "tpl2", "tpl3"])
            or len(expfile_lst != 4)):
                print("\t skipped")
                continue 
                        
            
            # Initiate first sublists for distinguishing different exposures
            O_1lst, sigmaO_1lst = [], []
            E_1lst, sigmaE_1lst = [], []
            F_1lst, sigmaF_1lst = [], []
            for k, f in enumerate(expfile_lst):
                print("\n\t\t {}".format(f))


                # Skip non-fits files
                if not f.endswith(".fits"):
                    print("\t\t skipped")
                    continue 
                    
                
                header, data = extract_data(tpl_dir + '/' + f)
                # Subtract bias
                data = data - bias
                data = data / masterflat_norm
                # Save corrected image
                savedir = tpl_dir + "/corrected2" 
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                elif os.path.exists(savedir) and k==0:
                    shutil.rmtree(savedir)
                    os.makedirs(savedir)
                hdu = fits.PrimaryHDU(data)
                hdulist = fits.HDUList([hdu])
                hdulist.writeto(savedir + '/' + f.split(".fits")[0] + "_COR.fits")
                
                
                
                # Specify observation parameters
                expno = header["HIERARCH ESO TPL EXPNO"]
                filt_name = header["HIERARCH ESO INS FILT1 NAME"]
                filt_id = header["HIERARCH ESO INS FILT1 ID"]
                exptime = header["EXPTIME"]
                ret_angle = header["HIERARCH ESO INS RETA2 POSANG"] * np.pi / 180. #rad
                woll_angle = header["HIERARCH ESO INS WOLL POSANG"] * np.pi / 180. #rad
                print("\t\t\t\tFILTER_ID: {A}; \t FILTER_NAME: {B}".format(A=filt_id, B = filt_name))
                print("\t\t\t\tWollangle: {A}; \t Retangle: {B}".format(A=woll_angle, B = np.round(ret_angle, 2)))                
                
                
                # Calibrate for different exposure times
                data = data / exptime

   
                # Initiate second sublist of F for distinguishing between different stars within the current exposure
                O_2lst, sigmaO_2lst = [], []
                E_2lst, sigmaE_2lst = [], []
                F_2lst, sigmaF_2lst = [], []
                for q, coord in enumerate(loc_lsts[i]):

                    # Finds the central pixel of the selected stars within the specific exposure                    
                    coord1, coord2 = coord[0:2], [coord[0],coord[2]]
                    if q not in np.arange(len(loc_lsts[i])-4, len(loc_lsts[i]), 1):
                        center1 = find_center(coord1, data, 15)
                        center2 = find_center(coord2, data, 15)
                        centers = [center1, center2]
                    if q in np.arange(len(loc_lsts[i])-4, len(loc_lsts[i]), 1):
                        centers = [coord1, coord2] # Sky aperture
                    

                    
                    # Initiate third sublist of F for distinguishing between different aperture radii
                    O_3lst, sigmaO_3lst = [], []
                    E_3lst, sigmaE_3lst = [], []
                    F_3lst, sigmaF_3lst = [], [] 
                    for l, R in enumerate(r_range):  

                        # Lists for temporary storage of aperture sum values and corresponding shotnoise levels
                        apsum_lst, shotnoise_lst = [], []
                        for center in centers:
                        
                            # Compute cumulative counts within aperture
                            apsum = apersum(data, center[0], center[1], R)
                            apsum_lst.append(apsum)
                            # Compute photon shot noise within aperture 
                            shotnoise = np.sqrt(apsum)
                            shotnoise_lst.append(shotnoise)
                        
                        # Compute normalised flux differences for current aperture size
                        F, sigmaF = fluxdiff_norm(apsum_lst[1], apsum_lst[0], shotnoise_lst[1], shotnoise_lst[0]) 
                        
                        
                        
                        # Append results to third sublist
                        O_3lst.append(apsum_lst[1]), sigmaO_3lst.append(shotnoise_lst[1])
                        E_3lst.append(apsum_lst[0]), sigmaE_3lst.append(shotnoise_lst[0])
                        F_3lst.append(F), sigmaF_3lst.append(sigmaF)
                    # Append the third sublist to the second sublist
                    O_2lst.append(O_3lst), sigmaO_2lst.append(sigmaO_3lst)
                    E_2lst.append(E_3lst), sigmaE_2lst.append(sigmaE_3lst)
                    F_2lst.append(F_3lst), sigmaF_2lst.append(sigmaF_3lst)
                # Append second sublist to first sublist
                O_1lst.append(O_2lst), sigmaO_1lst.append(sigmaO_2lst)
                E_1lst.append(E_2lst), sigmaE_1lst.append(sigmaE_2lst)
                F_1lst.append(F_2lst), sigmaF_1lst.append(sigmaF_2lst)
            # Append first sublist to main list 
            O_0lst.append(O_1lst), sigmaO_0lst.append(sigmaO_1lst)
            E_0lst.append(E_1lst), sigmaE_0lst.append(sigmaE_1lst)
            F_0lst.append(F_1lst), sigmaF_0lst.append(sigmaF_1lst)      
            # Append filter name to filter_lst
            filter_lst.append(filt_name)
        # Transform into arrays for future computations
        O_0lst, sigmaO_0lst = np.array(O_0lst), np.array(sigmaO_0lst)
        E_0lst, sigmaE_0lst = np.array(E_0lst), np.array(sigmaE_0lst)
        F_0lst, sigmaF_0lst = np.array(F_0lst), np.array(sigmaF_0lst) 
        
        # Save the flux arrays
        if datasavedir is None:
            continue
        else:
            savedir = datasavedir +'/'+ std_dir.rsplit("/",2)[1] + "/fluxlsts"
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        np.save(savedir + "/O_0lst.npy", O_0lst), np.save(savedir + "/sigmaO_0lst.npy", sigmaO_0lst)
        np.save(savedir + "/E_0lst.npy", E_0lst), np.save(savedir + "/sigmaE_0lst.npy", sigmaE_0lst)
        np.save(savedir + "/F_0lst.npy", F_0lst), np.save(savedir + "/sigmaF_0lst.npy", sigmaF_0lst)
        # Save filt_lst
        np.save(savedir + "/filter_lst.npy", filter_lst)
# END COMPUTE_FLUXLSTS



#################### END FUNCTIONS FOR FLUX LIST COMPUTATIONS ####################





#################### FUNCTIONS FOR COMPUTING POLARIZATION ####################



# Function for loading the lists specifying the ordinary, extraordinary fluxes and the normalized flux differences for all templates, exposures, selected regions and aperture radii, as well as for loading the list specifying the used filter used for each template. 'Loaddir' specifies the directory where the list structures are stored.
def load_lsts(loaddir):
    # Store current directory
    currdir = os.getcwd()
    
    # Move to directory 
    os.chdir(loaddir)
    O_jkqr, sigmaO_jkqr = np.load("O_0lst.npy"), np.load("sigmaO_0lst.npy")
    E_jkqr, sigmaE_jkqr = np.load("E_0lst.npy"), np.load("sigmaE_0lst.npy")
    F_jkqr, sigmaF_jkqr = np.load("F_0lst.npy"), np.load("sigmaF_0lst.npy")
    print("List structures loaded...")
    # Load filter_lst
    filter_lst = np.load("filter_lst.npy")
    os.chdir(currdir)
    
    return np.array([O_jkqr, E_jkqr, F_jkqr]), np.array([sigmaO_jkqr, sigmaE_jkqr, sigmaF_jkqr]), filter_lst



# Compute Stokes Q and U as well as the degree angle of linear polarization and all corresponding error margins for all templates, exposures, selected regions and aperture radii
def compute_pol(F_lst, sigmaF_lst):
    # Compute Stokes variables
    Q_jqr = 0.5 * F_lst[:,0,:,:] - 0.5 * F_lst[:,2,:,:]
    U_jqr = 0.5 * F_lst[:,1,:,:] - 0.5 * F_lst[:,3,:,:]
    # Compute standard deviations
    sigmaQ_jqr = 0.5 * np.sqrt(sigmaF_lst[:,0,:,:]**2 + sigmaF_lst[:,2,:,:]**2)
    sigmaU_jqr = 0.5 * np.sqrt(sigmaF_lst[:,1,:,:]**2 + sigmaF_lst[:,3,:,:]**2)
    
    # Compute degree of linear polarization and polarization angle
    fracUQ_jqr = np.divide(U_jqr, Q_jqr) # --
    P_jqr = np.sqrt(Q_jqr**2 + U_jqr**2) # --
    Phi_jqr = 0.5 * np.arctan(fracUQ_jqr) * (180. / np.pi) # deg
    # Compute standard deviation on degree of linear polarization
    tempP = np.sqrt( (Q_jqr * sigmaQ_jqr)**2 + (U_jqr * sigmaU_jqr)**2 )
    sigmaP_jqr = np.divide(tempP, P_jqr)
    # Compute standard deviation on angle of linear polarization
    temp1Phi = 1. / (np.ones(fracUQ_jqr.shape) + fracUQ_jqr)
    temp2Phi = temp1Phi * np.sqrt(sigmaU_jqr**2 + fracUQ_jqr**2 * sigmaQ_jqr**2)
    sigmaPhi_jqr = 0.5 * np.divide(temp2Phi, Q_jqr)
    
    return np.array([Q_jqr, U_jqr, P_jqr, Phi_jqr]), np.array([sigmaQ_jqr, sigmaU_jqr, 
                                                               sigmaP_jqr, sigmaPhi_jqr])



# Function for selection the correct aperture radius for each region specified in the 4D list outputs of compute_fluxlsts
def select_r(jkqr_lst, regindex, radindex):
    
    jkq_lst = jkqr_lst[:,:,regindex,radindex]
    
    return jkq_lst 
    
    

# Function for computing arry normalizations over the average along a specific axis 'ax', as well as the corresponding errors
def compute_norm(arr, sigma_arr=None, ax=0):
    # Store the input array shape and define the new array shape for arithmetic purposes
    arrshape = np.array(arr.shape)
    arrnewshape = np.where(arrshape!=arrshape[ax], arrshape, 1)
    
    # Initialize sigma_arr
    if sigma_arr==None:
        sigma_arr = np.zeros(arrshape)
    
    # Compute the normalization factors
    norm = np.average(arr, weights = 1. / sigma_arr**2, axis=ax)
    normerr = np.sqrt(1. / np.sum(1./sigma_arr**2, axis=ax))
    
    # Transform shape for arithmetical purposes, such that the normalization factor is equal for all elements along 'ax'
    norm2, normerr2 = norm.reshape(arrnewshape), normerr.reshape(arrnewshape)
    
    # Normalization of input array and computation of corresponding errors
    arr_norm = arr / norm2
    tempErr = sigma_arr**2 + (arr**2 * normerr2**2) / norm2**2
    arr_normErr = (1./norm2) * np.sqrt(tempErr)
    
    return arr_norm, arr_normErr
    
       
    
    
    
#################### END FUNCTIONS FOR COMPUTING POLARIZATION ####################





#################### PLOT FUNCTIONS ####################



# Aperture radius vs linear polarization              
def apRvsPl(ax, r_range, pixscale, P, sigmaP, esoP=0., esosigmaP=0., colour='b', tag=None):
    
    # Plot lines defining given polarization degrees
    tempr = np.linspace(0., r_range[-1]*pixscale, 100)
    esoLOW = np.tile(esoP-esosigmaP, len(tempr))
    ax.plot(tempr, esoLOW, color='0.2', linestyle = '--')
    esoHIGH = np.tile(esoP+esosigmaP, len(tempr))
    ax.plot(tempr, esoHIGH, color='0.2', linestyle = '--')
        
     # Plot the computed polarization degrees as function of aperture radius
    ax.errorbar(r_range*pixscale, P, yerr = sigmaP, marker='o', color=colour, label=tag)
    


# Function for plotting a circle
def plot_circle(ax, radius, xc, yc, colour='b', tag=None):

    X = np.linspace(-1.*radius, radius, 1000) + xc
    R = np.tile(radius, 1000)
    Y = np.sqrt(R**2 - X**2) - yc
    
    ax.plot(X, Y, color=colour, label=tag)
    ax.plot(X, -1.*Y, color=colour, label=tag)
    


# Function for plotting a line
def plot_line(ax, x_range, xc, yc, a, colour='b', tag=None):
    
        y_range = a*(x_range - xc) + yc
        
        ax.plot(x_range, y_range, color=colour, label=tag)

    
    
# Function for plotting Q as a function of U. Parameters are as follows:
# - QU:             1D list specifying Q, sigmaQ, U and sigmaU respectively
# - PlPhi:          1D list specifying P, sigmaP, Phi and sigmaPhi respectively
# - colour (opt):   character specifying the colour which should be used within the plot
# - PLPHI (opt):    1D list specifying known P, sigmaP, Phi and sigmaPhi respectively
# - Checks (opt):   Boolean for determining whether to plot the given polarizations
def QvsU(fig, ax, QU, offsetangle=0., PLPHI=np.zeros(4), checkPphi=[False,False], plot_inset=False, inset_ax=None, colour='b', tag=None):

    # Decompose list
    [Q,sigmaQ,U,sigmaU] = QU
    # Offset corrections
    offsetangle = offsetangle * np.pi / 180. #rad
    Qcorr = Q*np.cos(offsetangle) - U*np.sin(offsetangle)
    Ucorr = Q*np.sin(offsetangle) + U*np.cos(offsetangle)
    
    
    # Plot initial QvsU point
    if tag == "b_HIGH" and offsetangle!=0.:
        uncorrtag = "uncorrected"
    else:
        uncorrtag = None
    ax.errorbar(Q, U, xerr = sigmaQ, yerr = sigmaU, fmt='.', markersize = 16., color = "0.3", label=uncorrtag) # STD star
    
    # Plot QvsU point after offset correction
    ax.errorbar(Qcorr, Ucorr, xerr = sigmaQ, yerr = sigmaU, fmt='.', markersize = 16., color = colour, label = tag) # STD star
    
    
    '''
    # Plot polarization degree circles and angle lines
    plotline = np.linspace(-Pl*1.5, Pl*1.5, 100)
    x_plotline = plotline * np.cos(Phi/180.*np.pi)
    ylow_plotline = plotline * np.sin(Phi/180.*np.pi) 
    yhigh_plotline = plotline * np.sin(Phi/180.*np.pi)
    plt.plot(x_plotline, y_plotline, color='k') # Line indicating calculated polarization angle
    '''
    # Plot given polarization degrees and angles
    [checkP, checkphi] = checkPphi
    print(checkP, checkphi)
    [PL,sigmaPL,PHI,sigmaPHI] = PLPHI # [-], [deg]
    if checkP == True:
        plot_circle(ax, PL-sigmaPL, xc=0., yc=0., colour=colour) # ESO documentation inner polarization circle
        plot_circle(ax, PL+sigmaPL, xc=0., yc=0., colour=colour) # ESO documentation outer polarization circle
    if checkphi == True:
        PHI, sigmaPHI = PHI/180.*np.pi, sigmaPHI/180.*np.pi # rad
        plotline = np.linspace(-PL*1.5, PL*1.5, 100)                 
        x_plotline2 = plotline * np.cos(PHI)
        ylow_plotline2 = plotline * np.sin(PHI-sigmaPHI) 
        yhigh_plotline2 = plotline * np.sin(PHI+sigmaPHI)
        ax.plot(x_plotline2, ylow_plotline2, color=colour) # ESO doc lower 
        ax.plot(x_plotline2, yhigh_plotline2, color=colour) # ESO doc lower 
        

    if plot_inset == True:
        inset_ax.errorbar(Q, U, xerr = sigmaQ, yerr = sigmaU, fmt='.', markersize = 16., color = '0.3') # STD star
        inset_ax.errorbar(Qcorr, Ucorr, xerr = sigmaQ, yerr = sigmaU, fmt='.', markersize = 16., color = colour) # STD star
        plot_circle(inset_ax, PL-sigmaPL, xc=0., yc=0., colour=colour) # ESO documentation inner polarization circle
        plot_circle(inset_ax, PL+sigmaPL, xc=0., yc=0., colour=colour) # ESO documentation outer polarization circle
        inset_ax.plot(x_plotline2, ylow_plotline2, color=colour)
        inset_ax.plot(x_plotline2, yhigh_plotline2, color=colour)
        inset_ax.axis(xmin=-0.095, xmax=-0.06, ymin=0.0, ymax=0.03)
        # Set inset_ax axes limits
        inset_ax.set_xlim(-0.085, -.065) #TODO REMOVE HARDCODE
        inset_ax.set_ylim(0.001, 0.021) #TODO REMOVE HARDCODE
        # fix the number of ticks on the inset axes
        inset_ax.yaxis.get_major_locator().set_params(nbins=1)
        inset_ax.xaxis.get_major_locator().set_params(nbins=1)
        # draw a bbox of the region of the inset axes in the parent axes and connecting lines between the bbox and the inset axes area
        mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="0.7")
    
    
    # Addition 26-06-17: Compute difference angle between corrected and non-corrected
    if PHI is not None:
        phiL_uncorr, phiL_corr = 0.5*np.arctan(U/Q), 0.5*np.arctan(Ucorr/Qcorr) #rad
        phiLdiff = PHI - (180./np.pi)*phiL_corr #deg
        print("26-06-17:\t phiL_corrEXTRA = {}".format(phiLdiff))
    
        
        
        
# Function for fitting a sine wave
def sine_fit(t, w, phi, amp, b):

    sine = amp*np.sin(w*t + phi) + b
    
    return sine
    
    
    
# Function for fitting a parabola
def parabola_fit(x, a, b):
    
    y = a * x**2 + b
    
    return y
    


# TAKEN FROM http://stackoverflow.com/questions/8850142/matplotlib-overlapping-annotations
def get_text_positions(x_data, y_data, txt_width, txt_height):
    a = list(zip(y_data, x_data))
    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height) 
                            and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height: #True == collision
                differ = np.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    #j is the vertical distance between words
                    if j > txt_height * 2: #if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions
    
def text_plotter(x_data, y_data, text_positions, axis,txt_width,txt_height):
    i = 0
    for x,y,t in zip(x_data, y_data, text_positions):
        axis.text(x - txt_width, 1.01*t, str(i),rotation=0, color='blue')
        i += 1
        if y != t:
            axis.arrow(x, t,0,y-t, color='red',alpha=0.3, width=txt_width*0.1, 
                       head_width=txt_width, head_length=txt_height*0.5, 
                       zorder=0,length_includes_head=True)   
                       



    


#################### END PLOT FUNCTIONS ####################
#################### END FUNCTIONS #############################################




# Specify necessary directories
datadir = "/home/bjung/Documents/Leiden_University/brp/data_red/calib_data"
stddatadir = datadir + "/sorted/STD,IPOL"
plotdir = "/home/bjung/Documents/Leiden_University/brp/data_red/plots"
npsavedir = "/home/bjung/Documents/Leiden_University/brp/data_red/npsaves"
# Create list of directories and files within veladir
std_dirs = [stddatadir + "/Vela1_95/CHIP1", stddatadir + "/WD1615_154/CHIP1"]

# Load bias frame
bias_header, bias = extract_data(datadir + "/masterbias.fits")
mflat_normheader, mflat_norm = extract_data(datadir + "/masterflats/masterflat_norm_FLAT,LAM_IPOL_CHIP1.fits")

# Boolean variable 
compute_anew = False

# Pixel scale (same for all exposures)
pixscale = 0.126 #[arcsec/pixel]
# Conversion from ADUs to electrons
conad = 2.18 # e / ADU

# Range of aperture radii for plotting polarisation degree against aperture radius
r_range = np.arange(1, 21, 1) #[pixel]
# Range of retarder waveplate angles
ret_angles = np.arange(0.0, 90.0, 22.5) #[degrees]

# Tracks whether check circles and lines or an inset plot have to be drawn for the QvsU
plotinset, CHECKpphi = False, [[True, True],[True, False]]



# Aproximate coordinates of selection of stars within CHIP1 of 'Vela1_95' and 'WD1615_154'. Axis 0 specifiec the different std_dirs; axis 1 specifies the different stars within the std_dir; axis 2 specifies the x, y1, y2 coordinate of the specific star (with y1 specifying the y coordinate on the upper slit and y2 indicating the y coordinate on the lower slit) and the aproximate stellar radius. NOTE: THE LAST LIST WITHIN AXIS1 IS A SKY APERTURE!!!
star_lsts = [[[1034, 347, 251, 15], [1177, 368, 273, 8], [319, 345, 250, 5], [281, 499, 403, 6], [414, 139, 45, 12], [531, 706, 609, 5], [1583, 322, 229, 3], [1779, 321, 224, 4], [1294, 725, 627, 4], [1501, 719, 622, 7], [1040, 890, 791, 15], [1679, 150, 58, 15], [923, 513, 423, 15], [259, 157, 63, 15]],

            [[1039, 347, 253, 12], [248, 195, 103, 8], [240, 380, 286, 8], [599, 541, 446, 5], [365, 700, 604, 5], [702, 903, 806, 6], [801, 136, 43, 4], [1055, 133, 43, 4], [1186, 130, 37, 4], [1132, 685, 592, 3], [1222, 685, 592, 4], [1395, 679, 587, 4], [1413, 912, 816, 5], [1655, 542, 449, 5], [1643, 512, 417, 5], [1632, 190, 97, 6], [1608, 178, 85, 4], [1437,336,240, 15], [602, 700, 608, 15], [502, 152, 60, 15], [1303, 886, 790, 15]]] #[pixel]



# ESO given polarizations
VelaBV_PlPhi = [[7.55e-2, 173.8],[8.24e-2, 172.1]] # [-], [deg]
VelaBV_sigmaPlPhi = [[5e-4, 0.2],[3e-4, 0.1]] # [-], [deg]
WD1615BV_PlPhi, WD1615BV_sigmaPlPhi = [[0., None], [0., None]], [[2e-3, None], [2e-3, None]]
ESObvPLPHI = np.array([VelaBV_PlPhi, WD1615BV_PlPhi]) 
sigmaESObvPLPHI = np.array([VelaBV_sigmaPlPhi, WD1615BV_sigmaPlPhi])

# Calculate lists containing the ordinary, extraordinary and normalized flux differences 
if compute_anew == True:
    compute_fluxlsts(std_dirs, bias, mflat_norm, star_lsts, r_range)



# Create plots
for i, std_dir in enumerate(std_dirs):
    print(std_dir.split("/")[-2])
    
    loaddir = npsavedir +'/'+ std_dir.rsplit("/",2)[1] + "/fluxlsts"
    regions = np.array(star_lsts[i])
    ESO_BV_PlPhi, ESO_BV_sigmaPlPhi = ESObvPLPHI[i], sigmaESObvPLPHI[i]
    
    
    
    # Indices for selecting the aperture radii corresponding to each star
    reg_ind, rad_ind = np.arange(0, len(regions), 1), regions[:,3]-1
    std_rad = rad_ind[0]+1
    # List of colors for distinguishing the B and V filter in the plots
    Bcolor_lst, b = ["#3333ff", "#3399ff", "#33ffff", "#e6ac00"], 0
    Vcolor_lst, v = ["#336600", "#66cc00", "#cccc00", "#ccff33"], 0
    # Initiate indices for first encountered B and V exposures
    B1ind, V1ind = None, None
    
    
    
    # Load flux lists and filter list for plots
    OEF_jkqr, sigmaOEF_jkqr, filter_lst = load_lsts(loaddir)
    '''
    # Set NAN values to 0
    OEF_jkqr = np.where(OEF_jkqr >= 0., OEF_jkqr, 0.)
    sigmaOEF_jkqr = np.where(sigmaOEF_jkqr >= 0., sigmaOEF_jkqr, 0.)
    '''
    # Select correct aperture radii for each region
    OEF_jkq = np.array([ jkqr[:,:,reg_ind,rad_ind] for jkqr in OEF_jkqr])
    sigmaOEF_jkq = np.array([ sigma_jkqr[:,:,reg_ind,rad_ind] for sigma_jkqr in sigmaOEF_jkqr])
    
        
    # Compute Stokes Q and U as well as the linear polarization degree and angle and all corresponding error margins
    QUPphi_jqr, sigmaQUPphi_jqr = compute_pol(OEF_jkqr[2], sigmaOEF_jkqr[2])
    
        
    # Select correct aperture radii for each region
    QUPphi_jq = np.array([ jkqr[:,reg_ind,rad_ind] for jkqr in QUPphi_jqr])
    sigmaQUPphi_jq = np.array([ sigma_jq[:,reg_ind,rad_ind] for sigma_jq in sigmaQUPphi_jqr])
    
    
    
    # Initiate QUPphi0 where phi is going to be corrected for instrumental offsets
    QUPphi0_jqr, sigmaQUPphi0_jqr = QUPphi_jqr, sigmaQUPphi_jqr
    QUPphi0_jq, sigmaQUPphi0_jq = QUPphi_jq, sigmaQUPphi_jq
    
    
    
    # Initiation of the FOV distances for figure 4
    listshape = OEF_jkqr[0].shape
    regsdistOE, normfluxOE_jkq, normfluxerrOE_jkq = [], [], []
    
    
    
    # Tracks the number of skipped templates
    skips = 0
    # Boolean for checking when the first v_HIGH and b_HIGH filter are iterated through
    seenB, seenV = False, False
    
    
    # Create a list with all the template directories within std_dirs
    tpl_dirlst, tpl_flst = mk_lsts(std_dir)
    tpl_dirlst = np.sort(tpl_dirlst)
    
      
    #### Plots for all templates ####  
    quivims = []
    for J, tpl_name in enumerate(tpl_dirlst): 
        
        # Skip non-usable templates (non-usable templates should be put in a folder "skipped" or an equivalent directory which doesn't start with the string "tpl").
        if std_dir.split("/")[-2] == "Vela1_95" and tpl_name in ["tpl1", "tpl2", "tpl3"]:
            skips += 1
            continue 
        j = J-skips
        print("Jj:\t\t", J, j)
        
        
        # Define plot parameters 
        filtermask = (filter_lst == filter_lst[j])  
        mark_lst = ['o','v']
        
        print("\t", tpl_name)
        # Select the right plot colour and correct the polarization angle for instrumental offset
        if filter_lst[j] == "b_HIGH":
            # Check whether a label should be included in the plots
            if seenB == False:
                B1ind = j
                lbl = "b_HIGH"
            else:
                lbl = None
            # Select plot colour
            plotcolour = 'b'
            colour2 = Bcolor_lst[b]
            # Instrumental offset angle
            if i==0:
                offset = 1.54*2. # deg
            elif i==1:
                offset = 0. # deg
            # Select correct ESO given polarizations
            ESO_PL, ESO_PHI = ESO_BV_PlPhi[0,0], ESO_BV_PlPhi[0,1]
            ESO_sigmaPL, ESO_sigmaPHI = ESO_BV_sigmaPlPhi[0,0], ESO_BV_sigmaPlPhi[0,1]
            
            seenB = True
            b += 1
        
        elif filter_lst[j] == "v_HIGH":
            # Check whether a label should be included in the plots
            if seenV == False:
                V1ind = j
                lbl = "v_HIGH"
            else:
                lbl = None
            # Select plot colour
            plotcolour = 'g'
            colour2 = Vcolor_lst[v]
            # Instrumental offset angle
            if i==0:
                offset = 1.8*2. # deg
            elif i==1:
                offset = 0. #deg
            # Select correct ESO given polarizations
            ESO_PL, ESO_PHI = ESO_BV_PlPhi[1,0], ESO_BV_PlPhi[1,1]
            ESO_sigmaPL, ESO_sigmaPHI = ESO_BV_sigmaPlPhi[1,0], ESO_BV_sigmaPlPhi[1,1]
            
            seenV = True
            v += 1
        
        
        
        # Correct for instrumental offset
        QUPphi0_jq = np.where(QUPphi_jq != QUPphi_jq[3,j,0], QUPphi0_jq, QUPphi0_jq + offset)
        QUPphi0_jqr = np.where(QUPphi_jqr != QUPphi_jqr[3,j,0,:], QUPphi0_jqr, QUPphi0_jqr + offset)
        
        
        '''
        tempB = ESO_PHI - QUPphi_jq[3,[0,2],0]
        if QUPphi0_jq[0,0,0] < 0:
           tempB = tempB - 2*1.54 - 180.

        tempV = ESO_PHI - QUPphi_jq[3,[1,3,4],0]
        if QUPphi0_jq[0,1,0] < 0:
           tempV = tempV - 2*1.8 - 180.
        
        #if temp < 0:
        #    temp -= 180.
        print("26-06-17:\t{}".format(temp)) 
        break    
        '''   
        
        
        
        ############## PLOT 0 ##############     
        # Plot of aperture radius vs degree of linear polarization for the standard star
        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(111)
        apRvsPl(ax0, r_range, pixscale, QUPphi0_jqr[2,j,0], sigmaQUPphi0_jqr[2,j,0], esoP=ESO_PL, esosigmaP=ESO_sigmaPL, colour=plotcolour, tag=lbl)
        ############## PLOT 0 ##############     
        
        
        
        
        ############## PLOT 1 ##############  
        # Figure 1 for the normalized ordinary and extraordinary flux rates as function of retarder waveplate angle for the standard star.  
        for n in range(2):             
            
            # Plot the waveplate-angular normflux profile 2 in O and E for standard star
            fig1 = plt.figure(1)
            # Set labels to none after first iteration    
            if j != B1ind and j != V1ind:
                lblfig1 = None   
            # Only plot retarder waveplate angles vs normalized flux for first useable template
            if j in [B1ind, V1ind]:
                # Assign correct colour
                if j == B1ind:
                    colorlstfig1 = ['b', 'c']
                if j == V1ind:
                    colorlstfig1 = ['g', 'y']
                colfig1 = colorlstfig1[n]
                
                # Assign correct label
                if n == 0:
                        lblfig1 = lbl + " O"
                if n == 1:
                        lblfig1 = lbl + " E"
                
                # Initiate subplots
                ax1 = fig1.add_subplot(111)  
            
                # Compute distances to FOV centers for both the ordinary and extraordinary slits
                regsdistOE.append( np.sqrt( (regions[:,0]-1034)**2 + (regions[:,n+1]-1034)**2 ) )
                
                
                                                             
                # Normalize flux rates and errors using the mean between O and E to show the flux differences between O and E
                normfluxOE_k, normfluxerrOE_k = compute_norm(OEF_jkq[0:2,j,:,0], 
                                                             sigmaOEF_jkq[0:2,j,:,0], 
                                                             ax=0)
                
                
                                           
                # Plot sine waves
                if i == 0:
                    m = n+1
                if i == 1:
                    m = n
                theta = np.linspace(0,(80./90.)*np.pi,100)
                avline = (np.mean(normfluxOE_k[n]) + (-1)**(m) * 
                                (max(abs(normfluxOE_k[n])) - np.mean(normfluxOE_k[n])) * 
                                np.cos(theta*2.))
                ax1.plot(theta*180/np.pi/2., avline, 
                           color=colfig1, label="")
                           
                           
                # Plot mean ordinary and extraordinary lines
                ax1.plot(theta*180/np.pi/2., np.tile(np.mean(normfluxOE_k[n]), len(theta)),
                           color=colfig1, linestyle='--')
                
                # Plot flux points normalized using the mean between O and E to show the flux differences between O and E
                ax1.errorbar(ret_angles, normfluxOE_k[n], yerr=normfluxerrOE_k[n], 
                               color=colfig1, linestyle="", 
                               marker=mark_lst[n], markersize=10., 
                               label=lblfig1)             
            ############## PLOT 1 ##############                      
                    
        
        
        
        
        ############## PLOT 2 ##############     
        # Import cumulative counts
        cumc = np.sum(OEF_jkqr[0:2,j,0,0,:],axis=0)
        sigma_cumc = np.sum(sigmaOEF_jkqr[0:2,j,0,0,:],axis=0)
        skycumcs = np.sum(OEF_jkqr[0:2,j,0,-4::,:],axis=0) 
        sigma_skycumcs = np.sum(sigmaOEF_jkqr[0:2,j,0,-4::,:],axis=0) 
        
        # Correct for background flux
        skycumcAVE = np.average(skycumcs, weights=1./sigma_skycumcs**2, axis=0)
        sigma_skycumcAVE = np.sqrt(1. / np.sum(1./sigma_skycumcs**2, axis=0))
        # Convert nan to zero values
        skycumcAVE, sigma_skycumcAVE = np.nan_to_num(skycumcAVE), np.nan_to_num(sigma_skycumcAVE)
        cumccorr, sigma_cumccorr = cumc - skycumcAVE, np.sqrt(sigma_cumc**2 - sigma_skycumcAVE**2)
        
        # Plot cumulative counts vs aperture radius in O and E for standard star in exposure 1
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)
        ax2.errorbar(r_range*pixscale, cumc, yerr=sigma_cumc, 
                     color=plotcolour, linestyle='--')
        ax2.errorbar(r_range*pixscale, cumccorr, yerr=sigma_cumccorr, 
                     marker='o', color=plotcolour, label=lbl)  
        plot_line(ax2, [0., pixscale*r_range[0], pixscale*r_range[1]], 
                  pixscale*r_range[0], cumccorr[0], 
                  a=(cumccorr[1] - cumccorr[0])/(pixscale*(r_range[1]-r_range[0])),
                  colour=plotcolour)
        ############## PLOT 2 ##############     
        
        
        
        
        
        ############## PLOT 3 ##############     
        # U vs Q for the standard stars
        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot(111)
        if i == 0:
            if j == 0:
                axins = zoomed_inset_axes(ax3, 7, loc=1)
            plotinset = True
        else:
            plotinset = False
            axins = None
        
        print(np.amax(QUPphi_jq[0], axis=1), np.amax(QUPphi0_jq[0], axis=1))
        
        CHECKS = CHECKpphi[i]
        PlPhi_lst = [QUPphi_jq[2,j,0], sigmaQUPphi_jq[2,j,0], 
                     QUPphi_jq[3,j,0], sigmaQUPphi_jq[3,j,0]]  
        QU_lst = [QUPphi0_jq[0,j,0], sigmaQUPphi0_jq[0,j,0], 
                  QUPphi0_jq[1,j,0], sigmaQUPphi0_jq[1,j,0]]
                  
        QvsU(fig3, ax3, QU=QU_lst, offsetangle=offset, 
             colour=plotcolour, PLPHI=[ESO_PL,ESO_sigmaPL,
             ESO_PHI,ESO_sigmaPHI], checkPphi=CHECKS, 
             plot_inset=plotinset, inset_ax = axins, tag=lbl)
        axlim = PlPhi_lst[0]*3.
        ax3.set_xlim(-axlim, axlim), ax3.set_ylim(-(2./3.)*axlim, (2./3.)*axlim)
        ############## PLOT 3 ##############     
        
        
        
        
        
        ############## PLOT 4 ##############     
        # Plot linear polarization degree versus distance from FOV center for B-filter
        fig4 = plt.figure(4)
        ax4 = fig4.add_subplot(111)
        plt.errorbar(regsdistOE[0]*pixscale, QUPphi0_jq[2,j,:], yerr = sigmaQUPphi0_jq[2,j,:], fmt='o', color=plotcolour, label = lbl)
        #set the bbox for the text. Increase txt_width for wider text.
        txt_height = 0.04*(plt.ylim()[1] - plt.ylim()[0])
        txt_width = 0.02*(plt.xlim()[1] - plt.xlim()[0])
        #Get the corrected text positions, then write the text
        text_positions = get_text_positions(regsdistOE[0]*pixscale, QUPphi0_jq[2,j,:], txt_width, txt_height)
        text_plotter(regsdistOE[0]*pixscale, QUPphi0_jq[2,j,:], text_positions, ax4, txt_width, txt_height)
        ############## PLOT 4 ##############  
        
        
        
        
        
        ############## PLOT 5 ##############     
        # Vector plot showing polarization degrees of all regions (Only executed for the first exposure of the last template)
        
        #with sns.axes_style("whitegrid",{'axes.grid' : False}):
        fig5 = plt.figure(5)
        ax5 = fig5.add_subplot(111)
        # Load image
        vecplotdir = std_dir + '/' + tpl_name + '/corrected2/'
        datadirs, datafiles = mk_lsts(vecplotdir)
        header, datacor = extract_data(vecplotdir + datafiles[0])
        # Compute vectors
        vectorX, vectorY = regions[0:-4,0], regions[0:-4,1]
        polangles = np.where(QUPphi0_jq[3,j,0:-4] >= 0,QUPphi0_jq[3,j,0:-4],QUPphi0_jq[3,j,0:-4]+180)
        vectorU = QUPphi0_jq[2,j,0:-4] * np.cos(polangles*np.pi/180.)
        vectorV = QUPphi0_jq[2,j,0:-4] * np.sin(polangles*np.pi/180.)
        # Plot image
        norm = ImageNormalize(stretch=SqrtStretch())
        image = ax5.imshow(datacor, cmap='afmhot', origin='lower', norm=norm, vmin=0, vmax=1500)
        ##########plt.colorbar(image)
        print("DEBUG vectorU, vectorV:\t{}\n\t\t{}".format(vectorU, vectorV))
        print("DEBUG quiverlength\t{}".format(vectorU**2 + vectorV**2))
        # Plot vectors
        Q = ax5.quiver(vectorX, vectorY, vectorU, vectorV, units='x',  
                       scale=5e-4, pivot='mid', color=colour2) #0.1% polarization per pixel
        quivims.append(Q)
                   #color = '#666699')
    '''
    qk = ax5.quiverkey(quivims[0], 0.15, 0.9, 1, r'$5\%$ polarization', 
                       labelpos='N', coordinates='figure', fontproperties={"size": 15}) #0.5 inches = 5%
    '''
    plt.colorbar(image)
    ############## PLOT 5 ##############  
    
    
       

    
    # Check if the savefile directory is present
    savedir = plotdir + '/' + std_dir.split("/")[-2]
    if not os.path.exists(savedir):
        os.chdir(plotdir)
        os.makedirs(savedir)
        os.chdir(stddatadir)
    
    
    plt.figure(0)
    plt.grid()
    plt.xlabel(r'$\mathrm{Radius \ [arcsec]}$', fontsize=20)
    plt.ylabel(r'$\mathrm{Degree \ of \ linear \ polarization \ [-]}$', fontsize=20)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'RvsPl_alltplsV2.png')
    
    
    plt.figure(1)
    # Set axes labels
    ax1.set_xlabel(r'$\alpha \mathrm{\ [^{\circ}]}$', fontsize=16)
    ax1.set_ylabel(r'Normalized flux $\mathrm{\ [-]}$', fontsize=16)
    # Set grods and x ranges
    ax1.set_xlim(xmin=0, xmax=109)
    ax1.xaxis.grid(True), ax1.yaxis.grid(True)
    # Set legends
    ax1.legend(loc="right")
    # Set titles
    fig1.suptitle("Angular profile of normalized flux rates", fontsize=24)
    # Save plot 
    fig1.tight_layout()
    fig1.subplots_adjust(top=0.88)
    plt.savefig(savedir + '/' + 'alpha_F.png') 
    
        
    plt.figure(2)      
    plt.grid()
    plt.xlabel(r'Aperture Radius [arcsec]', fontsize=20)
    plt.ylabel(r'Counts [ADU]', fontsize=20)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'CumCounts.png')
    
    
    plt.figure(3)
    plt.grid()
    plt.xlabel(r'$\frac{Q}{I} \mathrm{\ [-]}$', fontsize=20)
    plt.ylabel(r'$\frac{U}{I} \mathrm{\ [-]}$', fontsize=20)
    plt.legend(loc = 'upper left')
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'UvsQ')
    
    
    plt.figure(4)  
    plt.grid()
    plt.xlabel(r'Radial distance [arcsec]', fontsize=20)
    plt.ylabel(r'Degree of linear polarization [-]', fontsize=20)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'radialprofile.png')
    
    
    plt.figure(5)
    x_tickrange, y_tickrange = np.arange(0,2038,200), np.arange(1000,-1,-100)
    plt.xticks(x_tickrange, (x_tickrange-1000)*pixscale), plt.yticks(y_tickrange,  (y_tickrange)*pixscale)
    plt.xlabel(r'X [arcsec]', fontsize=20)
    plt.ylabel(r'Y [arcsec]', fontsize=20)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig(savedir + '/' + 'polprofile.png')
    
    
    # Show and close figures
    plt.show()
    plt.close()
    
    
    
    # Create tables
    savedir = std_dir.split("/CHIP1")[0] + "/tables"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    os.chdir(savedir)                
    savefile1 = open("tables1_{}".format(std_dir.split("/")[-2]),'w')
    savefile1.write("ID \t&\t X \t&\t Y1 \t&\t Y2 \t&\t PixDist \t&\t AperRad \\\\ \n")
    savefile1.write("\\hline \n")    
    
    savefile2 = open("tables2_{}".format(std_dir.split("/")[-2]),'w')
    savefile2.write("ID \t&\t Counts \t&\t Noise \\\\ \n")
    savefile2.write("\\hline \n")
    
    savefile3 = open("tables3_{}".format(std_dir.split("/")[-2]),'w')
    savefile3.write("ID \t&\t Q/I \t&\t $Q_{err}$ \\\\ \n")
    savefile3.write("\\hline \n")
    
    savefile4 = open("tables4_{}".format(std_dir.split("/")[-2]),'w')
    savefile4.write("ID \t&\t U/I \t&\t $U_{err}$ \\\\ \n")
    savefile4.write("\\hline \n")
    
    savefile5 = open("tables5_{}".format(std_dir.split("/")[-2]),'w')
    savefile5.write("ID \t&\t $P_l$ \t&\t $P_{l_{err}}$ \\\\ \n")
    savefile5.write("\\hline \n")
    
    savefile6 = open("tables6_{}".format(std_dir.split("/")[-2]),'w')
    savefile6.write("ID \t&\t $\phi_0$ \t&\t $\phi_{0_{err}}$ \\\\ \n")
    savefile6.write("\\hline \n")
    
    for q, coords in enumerate(regions):
        if not ((q==0) or (q in np.arange(len(regions)-4,len(regions),1))):
            continue
        savefile1.write("{A} \t&\t {B1} \t&\t {B2} \t&\t {B3} \t&\t {C} \t&\t {D} \\\\ \n".format(A=q+1, B1=coords[0], B2=coords[1], B3=coords[2], C=[np.sum(coords[0:2]**2),coords[0]**2+coords[2]**2], D=coords[3]))
        
        savefile2.write("{A1} \t&\t {E1} \t&\t {E2} \\\\ \n {A2}.1 \t&\t {E11} \t&\t {E21}\\\\ \n".format(A1=q+1, A2=q+1, E1=np.round(list(OEF_jkq[0,:,0,q]),2), E11=np.round(list(OEF_jkq[1,:,0,q]),2), E2=np.round(list(sigmaOEF_jkq[0,:,0,q]),2), E21=np.round(list(sigmaOEF_jkq[1,:,0,q]),2)))
        
        savefile3.write("{A} \t&\t {F1} \t&\t {F2} \\\\ \n".format(A=q+1, F1=np.round(QUPphi0_jq[0,:,q],3), F2=np.round(sigmaQUPphi0_jq[0,:,q],3)))
        
        savefile4.write("{A} \t&\t {G1} \t&\t {G2} \\\\ \n".format(A=q+1, G1=np.round(QUPphi0_jq[1,:,q],3), G2=np.round(sigmaQUPphi0_jq[1,:,q],3)))
        
        savefile5.write("{A} \t&\t {H1} \t&\t {H2} \\\\ \n".format(A=q+1, H1=np.round(QUPphi0_jq[2,:,q],3), H2=np.round(sigmaQUPphi0_jq[2,:,q],3)))
    
        savefile6.write("{A} \t&\t {I1} \t&\t {I2} \\\\ \n".format(A=q+1, I1=np.round(QUPphi0_jq[3,:,q],3), I2=np.round(sigmaQUPphi0_jq[3,:,q],3)))
    os.chdir(stddatadir)        
        
    
    
    










