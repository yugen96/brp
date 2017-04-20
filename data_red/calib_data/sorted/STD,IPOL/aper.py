import numpy as np


'''
# Function which searches for pixel regions within 'image' where the count value is higher than 'hlim' and returns the x and y positions of these regions
def find(image, hlim):
    
    coords = np.argwhere(image > hlim)
    
    return coords
'''



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
    for i in range(apx_min, apx_max):
        for j in range(apy_min, apy_max):
            
            # Calculate squared distance to central pixel
            dx = i - px
            dy = j - py
            d2 = dx**2 + dy**2
            
            # Store the current pixel's count value
            pixval = image[j,i]
            
            if d2 <= r**2:
                apsum += pixval
                
    return apsum          
                
            
            
            
            
