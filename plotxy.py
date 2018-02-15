%matplotlib inline
import matplotlib.pyplot as plt

def cart2pol(x, y):
    rho_radius = np.sqrt(np.square(x) + np.square(y))
    phi_angles = np.arctan2(x, y) # swapping axes
    return rho_radius, phi_angles

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

r_rho_radius, r_phi_angles = cart2pol(asl.df['right-x'], asl.df['right-y'])
l_rho_radius, l_phi_angles = cart2pol(asl.df['left-x'], asl.df['left-y'])

ax = plt.subplot(121, projection='polar')
plt.plot(r_phi_angles, r_rho_radius)
plt.tight_layout()
ax = plt.subplot(122, projection='polar')
plt.plot(l_phi_angles, l_rho_radius)
plt.suptitle('Before normalizing by the nose');