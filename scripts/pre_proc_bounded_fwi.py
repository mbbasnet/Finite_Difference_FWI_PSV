# preprocessing of the files
import numpy as np
from seismic_def import read_tensor
import matplotlib.pyplot as plt
from seismic_def import read_tensor, e_lami, v_lami, w_vel

ftype = np.float64

##---------------------------------------------------------------------
# COMPUTATION IN GPU OR CPU
#---------------------------------------------------------------------

cuda_computation = False # True: computation in GPU, False: in CPU

#---------------------------------------------------------------------
# Defining geometry

x = 1.0 # x length in meter (only cube)
z = 1.0 # z length in meter (onlx cube part)


#---------------------------------------------------------------------
# GRID PARAMETERS
#--------------------------------------------------------------------


# Geometric data
dt = 0.1e-4; dz = 0.005; dx = 0.005 # grid intervals
nt = 9000; nz = int(z/dz)+1; nx = int(x/dx)+1 # grid numbers (adding for PMLs as well)


# Number of PMLs in each direction
pml_z = True; pml_x = True # PML exist in both direction
npml = 10
npml_top = npml; npml_bottom = npml; npml_left = npml; npml_right = npml

# Updating the nz and nx for pml layers 
nz += npml_top+npml_bottom
nx += npml_left+npml_right

# Adding air layers all around (10 layers)
num_air_grid = 10
nz+=2*num_air_grid
nx+=2*num_air_grid

# Surface grid index in each direction (0 = no surface)
surf = False # surface exists
isurf_top = 0; isurf_bottom = 0; isurf_left = 0; isurf_right = 0


snap_t1 = 0; snap_t2 = nt-1 # snap in time steps
snap_z1 = npml; snap_z2 = nz-npml  # snap boundaries z
snap_x1 = npml; snap_x2 = nx -npml# snap boundaries x
snap_dt = 3; snap_dz = 1; snap_dx = 1; # the snap intervals


# Taper position
nz_snap = snap_z2 - snap_z1
nx_snap = snap_x2 - snap_x1

# taper relative to the total grid
# t: top, b: bottom, l: left, r: right
#taper_t1 = snap_z1 + np.int32(nz_snap*0.05); taper_t2 = taper_t1 + np.int32(nz_snap*0.1)
#taper_b1 = snap_z2 - np.int32(nz_snap*0.05); taper_b2 = taper_b1 - np.int32(nz_snap*0.1)
#taper_l1 = snap_x1 + np.int32(nx_snap*0.05); taper_l2 = taper_l1 + np.int32(nx_snap*0.1)
#taper_r1 = snap_x2 - np.int32(nx_snap*0.05); taper_r2 = taper_r1 - np.int32(nx_snap*0.1)

taper_t1 = npml+ num_air_grid+5
taper_t2 = taper_t1+5

taper_b1 = taper_t1; taper_b2 = taper_t2
taper_l1 = taper_t1; taper_l2 = taper_t2
taper_r1 = taper_t1; taper_r2 = taper_t2
#------------------------------------------------------------------------------


# -------------------------------------------------------------------------
# FINITE DIFFERENCE PARAMETERS
# --------------------------------------------------------------------------

fdorder = 2 # finite difference order 
fpad = 1 # number of additional grids for finite difference computation

#forward only or fWI?
fwinv = False # True: FWI, False: Forward only

# Internal parameters for different cases 
if (fwinv):
    accu_save = False; seismo_save=True
    mat_save_interval = 1; rtf_meas_true = True # RTF field measurement exists
else:
    accu_save = True; seismo_save=True
    mat_save_interval = -1; rtf_meas_true = False # RTF field measurement exists

# ---------------------------------------------------------------------------------
    

#------------------------------------------------------------------
# MEDIUM (MATERIAL) PARAMETERS
#-----------------------------------------------------------------

# Air parameters
rho_air = 1.25
lam_air, mu_air = v_lami(0.0, 0.0, rho_air)

# Concrete parameters
rho_c = 2400 
C1_c = 3200
C2_c = 2000
Cp = 500.0
Cs = 300.0
rho_c = 1500.0
mu_c = C2_c*C2_c*rho_c
lam_c = C1_c*C1_c*rho_c - 2.0*mu_c


# --------------------------------------------------
# preparing  the starting material arrays
lam = np.full((nz, nx), lam_air)
mu = np.full((nz, nx), mu_air)
rho = np.full((nz, nx), rho_air)

mat_grid = 1 # 0 for scalar and 1 for grid
scalar_lam=0; scalar_mu=0; scalar_rho=0




# Creating a square grid of concrete
for iz in range(0, nz):
    for ix in range(0, nx):
        if iz>npml+num_air_grid and iz<nz-npml-num_air_grid:
            if ix>npml+num_air_grid and ix<nx-npml-num_air_grid:
                mu[iz][ix] = mu_c
                lam[iz][ix] = lam_c
                rho[iz][ix] = rho_c

# modifying density parameter (in original layers)
if (fwinv==False):
    for iz in range(0, nz):
        for ix in range(0, nx):
            #if (((nx/2-ix)**2+(nz/2-iz)**2)<(nx*nx/49)):
            if ix>npml+num_air_grid+30 and ix<npml+num_air_grid+70:
                if ix>iz/2 and ix<iz/2+3:
                    #rho[iz][ix] = 1.5 * rho[iz][ix]
                    mu[iz][ix] = mu_air
                    lam[iz][ix] = lam_air
                    rho[iz][ix] = rho_air

#------------------------------------------------------------



# -----------------------------------------------------
# PML VALUES TO BE USED FOR COMPUTATION
# -----------------------------------------------------

# PML factors
pml_npower_pml = 2.0
damp_v_pml = Cp
rcoef = 0.001
k_max_pml = 1.0
freq_pml = 250.0# PML frequency in Hz

# -----------------------------------------------------




#-----------------------------------------------------
# SOURCES AND RECIEVERS
#--------------------------------------------------------

# source and reciever time functions type
stf_type = 1; rtf_type = 0 # 1:velocity, 2:displacement

# Creating source locations
zsrc = np.array([nz/4, nz/2, 3*nz/4], dtype=np.int32)
xsrc = np.full((zsrc.size,), npml+num_air_grid+1, dtype=np.int32)
nsrc = zsrc.size # counting number of sources from the source location data


# Creating source to fire arrays
src_shot_to_fire = np.arange(0,nsrc,1, dtype=np.int32)
#src_shot_to_fire = np.zeros((nsrc,), dtype=np.int32)

nshot = nsrc # fire each shot separately

# Creating reciever locations
zrec = np.arange(20, nz-20, 2, dtype=np.int32)
xrec = np.full((zrec.size,), nx-20, dtype=np.int32)
nrec = zrec.size



# -----------------------------------------------------
# PLOTTING INPUTS
#---------------------------------------------------

print('Plotting initial materials')
plt.figure(1)
plt.subplot(221)
plt.imshow(lam) # lamda parameter
plt.plot(xsrc,zsrc, ls = '', marker= 'x', markersize=4) # source positions
plt.plot(xrec,zrec, ls = '', marker= '+', markersize=3) # reciever positions
plt.plot([snap_x1, snap_x2, snap_x2, snap_x1, snap_x1], [snap_z1, snap_z1, snap_z2, snap_z2, snap_z1], ls = '--')
plt.plot([taper_l1, taper_r1, taper_r1, taper_l1, taper_l1], [taper_t1, taper_t1, taper_b1, taper_b1, taper_t1], ls = '--')
plt.plot([taper_l2, taper_r2, taper_r2, taper_l2, taper_l2], [taper_t2, taper_t2, taper_b2, taper_b2, taper_t2], ls = '--')
plt.subplot(222)
plt.imshow(mu)
plt.subplot(223)
plt.imshow(rho)
plt.show()

#--------------------------------------------------------




# -------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# PROCESSING TO PREPARE THE ARRAYS (DO NOT MODIFY)
# -------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

# -----------------------------------------------------
# CREATING BINARY INPUT METADATA
# ---------------------------------------------------

# Creating boolen arrays
metabool = np.array([cuda_computation, surf, pml_z, pml_x, accu_save, seismo_save, fwinv, rtf_meas_true], dtype=np.bool_)

# Creating integer arrays and subsequent concatenation of the related fields
metaint = np.array([nt, nz, nx, snap_t1, snap_t2, snap_z1, snap_z2, snap_x1, snap_x2, snap_dt, snap_dz, snap_dx, \
                    taper_t1, taper_t2, taper_b1, taper_b2, taper_l1, taper_l2, taper_r1, taper_r2,\
                    nsrc, nrec, nshot, stf_type, rtf_type, fdorder, fpad, mat_save_interval, mat_grid], dtype=np.int32)
metaint = np.concatenate((metabool, metaint), axis=None) # concatination of boolen and integer as integers

# additional concatenation of int arrays
intarray = np.array([npml_top, npml_bottom, npml_left, npml_right, isurf_top, isurf_bottom, isurf_left, isurf_right], dtype=np.int32)
intarray  = np.concatenate((intarray, zsrc), axis=None)
intarray  = np.concatenate((intarray, xsrc), axis=None)
intarray  = np.concatenate((intarray, src_shot_to_fire), axis=None)
intarray  = np.concatenate((intarray, zrec), axis=None)
intarray  = np.concatenate((intarray, xrec), axis=None)

print("Metaint: ", metaint)

# Creating float arrays and subsequent concatenation
metafloat = np.array([dt, dz, dx, pml_npower_pml, damp_v_pml, rcoef, k_max_pml, freq_pml, \
                     scalar_lam, scalar_mu, scalar_rho], dtype=np.float64)
print("Metafloat: ", metafloat)

# Creating material arrays
material_inp = np.concatenate((lam, mu), axis = None)
material_inp = np.concatenate((material_inp, rho), axis = None)

# ---------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------
# WRITING ARRAYS TO BINARY FILE, READABLE IN C++ KERNELS
# -------------------------------------------------------------------------
metaint.tofile('./bin/metaint.bin')
intarray.tofile('./bin/intarray.bin')
metafloat.tofile('./bin/metafloat.bin')
material_inp.tofile('./bin/mat.bin')
#--------------------------------------------------------

#--------------------------------------------------------
#-------------------------------------------------------


