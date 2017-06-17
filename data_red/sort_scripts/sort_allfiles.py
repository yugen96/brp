from astropy.io import fits
import os
import shutil
import numpy as np



#################### FUNCTIONS ####################



# Function for reading info file. Returns a list containing the filename, celestial coordinates and exposure time for each image taken during the two observing nights. Looks like, e.g.: {'FLAT,SKY': [obs1, obs2, obs3, ...], 'FLAT,LAM': [obs1, obs2, obs3, ...]} with obs_j = [filename, RA, DEC, t_exp]
def read_infofile(infofilename):
    i = -1
    info_dict = {}
    
    with open(infofilename, 'r') as f:
    
        for line in f:
            splittedline = line.split()
            #print(splittedline)
            
            if len(splittedline) > 8 and line.startswith("FORS2"):
                # Reading file information from line
                obj_type = splittedline[3]
                
                filename = splittedline[0] + '.fits'
                RA = splittedline[5]
                DEC = splittedline[6]
                exp_time = splittedline[7]
                
                # Creating list with filename, celestial coordinates and exposure time of image corresponding to line
                file_info = [filename, RA, DEC, exp_time]
                
                # Ads file_info to the dictionary under the right key
                if obj_type in info_dict.keys():
                    info_dict[obj_type].append(file_info)
                else:
                    info_dict[obj_type] = []
                    info_dict[obj_type].append(file_info)
    
    # Transforms the lists into arrays for further use
    for key in info_dict.keys():
        info_dict[key] = np.array(info_dict[key])
                                
    return info_dict
    


# Returns a list with all subdirectories and files in a specific folder
def mk_lsts(dir_name):

    dir_lst = next(os.walk(dir_name))[1]
    file_lst = next(os.walk(dir_name))[2]
    return dir_lst, file_lst
    


# Extracts the data from a specified file and puts these into a list together with the header
def extract_data_from_file(fname):
    data_lst = []

    # Put data and header files into two separate lists
    if fname.endswith("fit") or fname.endswith("fits"):
        hdul = fits.open(fname)
        header = hdul[0].header             
        data = hdul[0].data

        data_lst.append(header, data)

        hdul.close()
        
    return data_lst
        
        

# Checks whether files specified in fname_lst are contained within the current folder and, if so, puts the corresponding header and data in an array. Return value looks like: [[header1, data1], [header2, data2], [etc.]] where header_i is the header of the i-th file in fname_lst that is located in the current folder
def extract_datas(file_lst):
    data_lst = []
    
    for x in file_lst:
        print(x)
        
        data_lst.append(extract_data_from_file(x))
            
    return np.array(data_lst)
    



# Checks whether file 'fname' is contained within the specified folder 'filedir' and, if so, copies them to a new folder 'movedir' corresponding to the file's object type
def move_file(fname, datadir, movedir):   
    initdir = os.getcwd()
    os.chdir(datadir)
    
    if os.path.isfile(fname):
        
        if os.path.exists(movedir):
            shutil.copy2(os.getcwd() + '/' + fname     ,       movedir + '/' + fname)
        else:
            os.makedirs(movedir)
            shutil.copy2(os.getcwd() + '/' + fname     ,       movedir + '/' + fname)
    
    os.chdir(initdir)
            
                





''' 
###########NOT USED########################   
# Extracts the data stored within the files that are contained in a specified directory.
def extract_data_from_dir(files_dir):
    
    # Go to specified directory
    orig_path = os.getcwd()
    os.chdir(files_dir)
    
    # Get dirs and files and contained data
    d_lst, f_lst = mk_lsts(os.getcwd())
    info_lst, data_lst = extract_datas(f_lst)
    
    # return to original directory
    os.chdir(orig_path)
        
    return [info_lst, data_lst]




        




# Makes a dict containing all headers and data arrays contained within each of the subdirectories of "path". Resulting dict looks like: {'subdir1': [header_lst1, data_lst1]   ,    'subdir2': [header_lst2, data_lst2]    ,   etc.  ], where header_lsti and data_lsti respectively stand for a list of headers and 2D data arrays corresponding to each image in the i-th subdirectory. E.g.: data_arr['CALIB'][1,5] calls the data array corresponding to the sixth image contained within the third subdirectory of "path".
def gather_subdirdata(path):
    
    # Create list of subdirectories and initial data array
    dlst, flst = mk_lsts(path)
    data_dict = {}
    
    # Store all image data and headers of each subdir in data_arr
    for direc in dlst:
        print(direc)

        subpath = path + '/' + direc
        data_dict[direc] = extract_data_from_dir(subpath)
        
    # Return the created data dictionary 
    return data_dict
    
###########NOT USED########################    
'''

   
    
  




        
        

#################### END FUNCTIONS ####################    
    
    
    


# Specificies general path to original data 
datapath = "/home/bjung/Documents/Leiden_University/brp/data_red/orig_data/"
# Define subdirectories of datapath
sub_dirs = {"2011-05-03/": ["CALIB/", "SCIENCE/"]      ,     "2011-05-04/": ["CALIB"]}


# Read in data info
data_info = read_infofile(datapath + "087.B-0767A001_DP-SUMMARY.txt")



# Move files of each object type to its own directory
for obj_type in data_info.keys():
    for filename in data_info[obj_type][:,0]:
        for night in sub_dirs.keys():
            for img_type in sub_dirs[night]:
                
                move_file(filename, datapath + '087.B-0767A/' + night + img_type, datapath + '087.B-0767A/' + obj_type)
            
            





















'''
# Create array containing all data within the subdirectories of datapath. Save array.
if os.path.isfile("all_data.npy"):
    data_arr = np.load("all_data.npy")
else:
    data_arr = gather_subdirdata(datapath)
    np.save("all_data.npy", data_arr)

print(data_arr[0,1])
'''






"""
while 
# Creates an individual masterflat for each of the different filters
for x in flat_dlst:
    print("Subdirectory: \t", x)
    
    # Goes to subdirectory x
    print(os.getcwd() + "/" + x)
    os.chdir(os.getcwd() + "/" + x)
"""













