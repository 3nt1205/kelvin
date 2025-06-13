import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # world setting
    g = 9.8 # m/sec**2
    KELVIN_ANGLE = np.arcsin(1/3)

    # simulation setting    
    d_angle = np.deg2rad(0.1)
    N = 15 # num of interference waves
    U = 10.0 # m/sec

    # Divergent Wave 
    ret1 = []
    for alpha in np.arange(-KELVIN_ANGLE,KELVIN_ANGLE,d_angle):
        tan_th = -(1+np.sqrt(1-8*np.tan(alpha)**2))/(4*np.tan(alpha))
        amp = 2*np.pi*U**2*np.sqrt(1+4*tan_th**2)/(g*(1+tan_th**2))
        x = amp*np.cos(alpha)
        y = amp*np.sin(alpha)
        for ii in range(1,N):
            ret1.append([ii*x,ii*y])

    # Transverse waves
    ret2=[]
    for alpha in np.arange(-KELVIN_ANGLE,KELVIN_ANGLE,d_angle):
        tan_th = -(1-np.sqrt(1-8*np.tan(alpha)**2))/(4*np.tan(alpha))
        amp = 2*np.pi*U**2*np.sqrt(1+4*tan_th**2)/(g*(1+tan_th**2))
        x = amp*np.cos(alpha)
        y = amp*np.sin(alpha)
        for ii in range(1,N):
            ret2.append([ii*x,ii*y])


    ret1 = np.array(ret1) 
    ret2 = np.array(ret2) 
    plt.title("U = {:.1f}m/sec".format(U))
    plt.scatter(ret1[:,0],ret1[:,1],c="blue",s=1,label="Divergent Wave")
    plt.scatter(ret2[:,0],ret2[:,1],c="orange",s=1,label="Transverse waves")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x(m)")
    plt.ylabel("y(m)")
    plt.show()


