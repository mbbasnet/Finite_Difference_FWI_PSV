from scipy.io import loadmat
import numpy as np
from matplotlib import cm
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# Load the .mat file
data = loadmat('ext_data/Displacement_Field_1.mat')

# View the keys in the loaded data
print(data.keys())

# Access a specific variable
disp = data['Displacement_Field']
print(disp.shape)


# Plotting of the displacement field
    
# Plot the rtf first
nt = disp.shape[0]
nrec = disp.shape[1]
nx = 9
nz = 9
print("NREC: ", nrec)

rtf_ux = disp[:,:,0]
rtf_uz = disp[:,:,1]


# Plotting the RTF functions
plt.figure(1)
plt.subplot(211)
for ii in range(0, nrec,1):   
    plt.plot(rtf_uz[:,ii])
plt.grid()
plt.subplot(212)
for ii in range(0, nrec,1):
    plt.plot(rtf_ux[:,ii])
plt.grid()
plt.show()
plt.savefig('./rtf_signals.png', format='png', bbox_inches='tight')






clip_nt = nt
clip_pz = np.amax(disp[:,:,1])
clip_mz = np.amin(disp[:,:,1])
clipz = 0.3*max([clip_pz, np.abs(clip_mz)])

clip_px = np.amax(disp[:,:,0])
clip_mx = np.amin(disp[:,:,0])
clipx = 0.3*max([clip_px, np.abs(clip_mx)])
zoom_factor = 10

for ii in range(0,clip_nt, 2):
    vx_dat = disp[ii,:,0]
    vz_dat = disp[ii,:,1]
    # reading data from csv file
    vz = vz_dat.reshape(9,9)
    vx = vx_dat.reshape(9,9)
    vz= zoom(vz, zoom_factor, order=1)
    vx= zoom(vx, zoom_factor, order=1)
    plt.figure(1, figsize=(10,6))
    plt.subplot(121)
    plt.imshow(vz, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clipz, vmax=clipz)
    plt.colorbar()
    plt.title('Ux [Time snap '+str(ii)+']', y=-0.2)
    plt.xlabel('X [no. of grids]'+str(ii))
    plt.ylabel('Z [no. of grids]')
    #pyplot.gca().invert_yaxis()
    #pyplot.axis('equal')
    plt.grid()
    plt.subplot(122)
    plt.imshow(vx, animated=True, cmap=cm.seismic, interpolation='nearest', vmin=-clipx, vmax=clipx)
    plt.colorbar()
    plt.title('Vx [Time snap '+str(ii)+']', y=-0.2)
    plt.xlabel('X [no. of grids]'+str(ii))
    plt.ylabel('Z [no. of grids]')
    #pyplot.gca().invert_yaxis()
    #pyplot.axis('equal')
    plt.grid()
    #pyplot.savefig('../io/vz_snap'+numpy.str(ii)+'.pdf', format='pdf',figsize=(10,7), dpi=1000)
    #plt.show()
    #plt.draw()
    plt.savefig('./mat_fwd.png', format='png', bbox_inches='tight')
    plt.pause(0.01)
    plt.clf()
    