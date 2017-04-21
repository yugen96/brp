import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


# Import retarder chromaticity data
labdaWP, eps_theta = np.loadtxt("ret_chrom.txt", comments="--", delimiter = "   ", skiprows=4, unpack=True)
# Import b_HIGH transmission data
labdaB, transmB = np.loadtxt("b_HIGH.txt", delimiter = "   ", skiprows=2, unpack=True) 
# Import v_HIGH transmission data
labdaV, transmV = np.loadtxt("v_HIGH.txt", delimiter = "   ", skiprows=2, unpack=True)


# Select data below 1 micrometer
eps_theta, transmB, transmV = eps_theta[labdaWP<1e4], transmB[labdaB<1000.]/100., transmV[labdaV<1000.]/100.
labdaWP, labdaB, labdaV = labdaWP[labdaWP<1e4]/10., labdaB[labdaB<1000.], labdaV[labdaB<1000.]
# Flip transmission wavelengths and data # TODO FLIP OR DON'T FLIP
labdaB, labdaV = np.flipud(labdaB), np.flipud(labdaV)
transmB, transmV = np.flipud(transmB), np.flipud(transmV)



# Interpolation to create same wavelength selection as labdaWP for labdaB and labdaV
convs1, convs2 = [], []
transm_INTERPs, eps_thetaINTERPs = [], []
for [labda, transm] in [[labdaB,transmB],[labdaV, transmV]]:
    for i,zipth in enumerate(zip(labda, transm)):
        print(zipth)
    print("\n\n")
        
    transm_INTERP = np.interp(labdaWP, labda, transm) 
    eps_thetaINTERP = np.interp(labda, labdaWP, eps_theta) 
    transm_INTERPs.append(transm_INTERP), eps_thetaINTERPs.append(eps_thetaINTERP)
    
    convs1.append( np.convolve(eps_thetaINTERP, transm, mode='same') )
    convs2.append( np.convolve(eps_theta, transm_INTERP, mode='same') )


'''
# Compute convolutions
convB1 = np.convolve(eps_thetaINTERP, np.flipud(transmB), mode='same') #TODO same or full
convV1 = np.convolve(eps_thetaINTERP, np.flipud(transmV), mode='same')
convB2 = np.convolve(eps_theta, np.flipud(transmB_INTERP), mode='same') 
convV2 = np.convolve(eps_theta, np.flipud(transmV_INTERP), mode='same')
'''


# Plot transmissions and convolutions
fig0 = plt.figure(0)
ax0 = fig0.add_subplot(111)
ax0.plot(labdaB, transmB*100., color='b', label="b_HIGH")
ax0.plot(labdaV, transmV*100., color='g', label="v_HIGH")
ax0.plot(labdaWP, transm_INTERPs[0]*100., color='0.8', marker='v')
ax0.plot(labdaWP, transm_INTERPs[1]*100., color='0.5', marker='v')
ax0.grid()
ax0.legend(loc='best')
ax0.set_xlabel(r'$\lambda$ [nm]', fontsize=20)
ax0.set_ylabel(r'$T [\%]$', fontsize=20)
ax0.legend(loc = 'best')
plt.tight_layout()
plt.savefig('transmissions')


fig1_1 = plt.figure(11)
ax1_1 = fig1_1.add_subplot(111)
ax1_1.plot(labdaB, convs1[0], color='b', label="b_HIGH")
ax1_1.plot(labdaV, convs1[1], color='g', label="v_HIGH")
ax1_1.grid()
ax1_1.legend(loc='best')
ax1_1.set_xlabel(r'$\lambda$ [nm]', fontsize=20)
ax1_1.set_ylabel(r'$T * \epsilon_{\theta}$', fontsize=20)
plt.tight_layout()
plt.savefig('convolutions1') 


fig1_2 = plt.figure(12)
ax1_2 = fig1_2.add_subplot(111)
ax1_2.plot(labdaWP, convs2[0], color='b', label="b_HIGH")
ax1_2.plot(labdaWP, convs2[1], color='g', label="v_HIGH")
ax1_2.grid()
ax1_2.legend(loc='best')
ax1_2.set_xlabel(r'$\lambda$ [nm]', fontsize=20)
ax1_2.set_ylabel(r'$T * \epsilon_{\theta}$', fontsize=20)
plt.tight_layout()
plt.savefig('convolutions2') 


fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
ax3.scatter(labdaB, eps_thetaINTERPs[0], color='r', label="Interpolated")
ax3.plot(labdaWP, eps_theta, color='b', label="Original")
ax3.grid()
ax3.legend(loc='best')
ax3.set_xlabel(r'$\lambda$ [nm]', fontsize=20)
ax3.set_ylabel(r'$\epsilon_{\theta} [^{\circ}]$', fontsize=20)
plt.tight_layout()
plt.savefig('chrom')
plt.show()








'''
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
ax1.plot(labdaB, transmB*100., color='b', label="b_HIGH")
ax2.plot(labdaV, transmV*100., color='g', label="v_HIGH")
plt.grid()
ax1.set_ylabel(r'$T [\%]$', fontsize=20)
ax2.set_xlabel(r'$\lambda$ [nm]', fontsize=20)
ax2.set_ylabel(r'$T [\%]$', fontsize=20)
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig('transmissions')



f, ax = plt.subplots(2, sharex=True, sharey=True)
ax[0].plot(convB, color='b', label="b_HIGH")
ax[0].grid()
ax[0].legend(loc='best')
ax[1].plot(convV, color='g', label="v_HIGH")
ax[1].grid()
ax[1].legend(loc='best')
ax[0].set_ylabel(r'$T * \epsilon_{\theta}$', fontsize=20)
ax[1].set_xlabel(r'$\lambda$ [nm]', fontsize=20)
ax[1].set_ylabel(r'$T * \epsilon_{\theta}$', fontsize=20)
plt.tight_layout()
plt.savefig('convolutions') 
plt.show()
'''


