import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import RegularGridInterpolator
import cv2
from scipy.optimize import fsolve

class HavelockModel:
    # クラスのコンストラクタ
    def __init__(self, MIN_ORDER=3, MAX_ORDER=15, delt_th =1.0, NOISE_AMP_STD_RATIO=0.2, NOISE_FAI_STD_RATIO=0.10, h=10000): # 観測系の設定
        self.g = 9.8 # 重力加速度：m/sec**2
        self.kelvin_angle = np.arcsin(1/3)  # ≒ 19.47度
        self.MIN_ORDER = MIN_ORDER  # 計算するKelvin波の次数、最小
        self.MAX_ORDER = MAX_ORDER # 計算するKelvin波の次数、最大
        self.th_list = np.arange(np.deg2rad(91),np.deg2rad(270),np.deg2rad(delt_th)) # 要素波方向の積分間隔
        self.NOISE_AMP_STD_RATIO = NOISE_AMP_STD_RATIO # 要素波の振幅に加えるノイズの標準偏差比 (1.0)
        self.NOISE_FAI_STD_RATIO = NOISE_FAI_STD_RATIO # 要素波の位相に加えるノイズの標準偏差比(2*np.pi)
        self.FLG2D = "none"
        self.FLG_RSMPL = False
        self.h = h

    def CreateKelvinImage(self, U, N2D=512): 
        """
        2次元Kelvin波のシミュレーション
        Args:
            U(float):船速(m/sec)
        Returns:
            ndarray:シミュレーション領域meshのx値
            ndarray:シミュレーション領域meshのy値
            ndarray:シミュレーション領域における波高
        """
        Apx_Lambda = 12/16*U**2 # 経験則による
        SIZE = Apx_Lambda * self.MAX_ORDER

        x = np.linspace(-SIZE, 0, N2D)
        y = np.linspace(SIZE/2, -SIZE/2, N2D)
        xx, yy = np.meshgrid(x, y)
        pos2D = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)        

        dist = np.sqrt((pos2D[:, 0])**2 + (pos2D[:, 1])**2)
        eps = 1e-6 
        amp = 1 / np.sqrt(dist + eps)

        w = np.zeros(len(pos2D))
        for order in range(self.MIN_ORDER, self.MAX_ORDER):
            min_th = order*2*np.pi
            for th in self.th_list:
                #ramda = 2 * np.pi * U**2 * np.cos(th) * np.cos(th) / self.g
                ramda = self.compute_wavelength(U,th,self.h)
                w = w + amp * self.PlaneWave2D(pos2D, th, ramda, min_th, min_th+2*np.pi)

        #w = w - np.min(w.reshape(-1))
        #print("ave = ",np.average(w.reshape(-1)))
        KelvinImage = w.reshape(xx.shape)
        
        self.U = U
        self.x = x
        self.y = y
        self.xx = xx
        self.yy = yy
        self.KelvinImage = KelvinImage
        self.dmesh = self.x[1]-x[0]
        self.FLG2D = "sim"

        return self.xx, self.yy, KelvinImage

    def CreateKelvinWavePeakImage(self, U, N2D=512): 
        """
        2次元Kelvin波のシミュレーション
        Args:
            U(float):船速(m/sec)
        Returns:
            ndarray:シミュレーション領域meshのx値
            ndarray:シミュレーション領域meshのy値
            ndarray:シミュレーション領域における波高
        """
        Apx_Lambda = 12/16*U**2 # 経験則による
        SIZE = Apx_Lambda * self.MAX_ORDER

        x = np.linspace(0, SIZE, N2D)
        y = np.linspace(SIZE/2, -SIZE/2, N2D)
        w = np.zeros([N2D,N2D])
        xx, yy = np.meshgrid(x, y)
        dmesh = x[1]-x[0]
        #pos2D = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)        

        d = 2*self.kelvin_angle/N2D
        alpha_arr = np.arange(-self.kelvin_angle, self.kelvin_angle, d)
        #alpha_arr = np.array([-self.kelvin_angle, -self.kelvin_angle+d, self.kelvin_angle])
        _, r = self.PhaseStagnation(alpha_arr, U=U)
        idx = np.where(~np.isnan(r))[0]
        wp_x_base = r[idx]*np.cos(alpha_arr[idx])
        wp_y_base = r[idx]*np.sin(alpha_arr[idx])

        for scale in range(self.MIN_ORDER, self.MAX_ORDER):
        #for scale in [3,5,6,7,9,10,12,13]:
            wp_x = scale*wp_x_base
            wp_y = scale*wp_y_base

            t_r = r[idx]*scale
            t_alpha = alpha_arr[idx]
            z = np.cos(t_r*self.g/(U**2*np.cos(t_alpha)))/np.sqrt(t_r)

            #print(wp_x,wp_y)
            in_bounds = (0 <= wp_x) & (wp_x < SIZE) & (-SIZE/2 <= wp_y) & (wp_y < SIZE/2)

            #i = np.full_like(wp_y, -1, dtype=int)  # 初期値は -1（無効）
            #j = np.full_like(wp_x, -1, dtype=int)

            ii_arr = ((wp_y[in_bounds] + SIZE / 2) // dmesh).astype(int)
            jj_arr = (wp_x[in_bounds] // dmesh).astype(int)
            z_arr = z[in_bounds]
            #print(ii_arr,jj_arr)

            for kk in range(len(ii_arr)):
                w[ii_arr[kk],jj_arr[kk]] = 1

        x = x[::-1]
        xx , yy = np.meshgrid(x,y)
        w = w[:, ::-1]

        self.U = U
        self.x = x
        self.y = y
        self.xx = xx
        self.yy = yy
        self.KelvinImage = w
        self.dmesh = dmesh
        self.FLG2D = "sim"

        return x,y,w

    def SetKelvinImage(self, img, dmesh):
        YSIZE = img.shape[0]
        XSIZE = img.shape[1]
        self.x = np.arange(0, XSIZE)*dmesh
        self.y = np.arange(0,YSIZE)[::-1]*dmesh
        self.xx, self.yy = np.meshgrid(self.x,self.y)

        mean = np.mean(img.reshape(-1))
        std = np.std(img.reshape(-1))
        norm_img = (img - mean) / std
        self.KelvinImage = norm_img
        self.FLG2D = "img"
        self.dmesh = dmesh
        #print("ave = ",np.average(norm_img.reshape(-1)))

    def DrawKelvinImage(self, ax, KELVIN_FLG=True):
        if self.FLG2D == "sim":
            ax.set_title("Kelvin wave:Single Source, No Shadow \nU={:.1f}m/s, h={:.1f}m, Order={}~{}, Noise STD:AMP[{:.1f}],FAI[{:.1f}]".format(
                self.U, self.h, self.MIN_ORDER, self.MAX_ORDER, self.NOISE_AMP_STD_RATIO, self.NOISE_FAI_STD_RATIO))
        elif self.FLG2D == "img":
            ax.set_title("Kelvin wave: from Image\nx[{}]y[{}] d={:.1f}m".format(
                len(self.x),len(self.y),self.dmesh
            ))
        else:
            return
        sc = ax.imshow(self.KelvinImage, extent=[self.x[0], self.x[-1], self.y[-1], self.y[0]])
        if KELVIN_FLG:
            ax.plot([0, self.x[0]],[0,  self.x[0]*np.tan(self.kelvin_angle)], c="red")
            ax.plot([0, self.x[0]],[0, -self.x[0]*np.tan(self.kelvin_angle)], c="red")
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        fig = ax.figure
        fig.colorbar(sc,label='Amplitude')
        ax.grid(True)

    def KelvinWaveAnalysis(self, angle=np.nan, AMP_DW = 5, dWL=0.1):
        if self.FLG2D == "none":
            return -1
        if np.isnan(angle):
            angle = model.kelvin_angle

        dl = model.dmesh
        dw = AMP_DW*model.dmesh

        pos1, prof1 = model.Resampling(angle, dl, dw)
        pos2, prof2 = model.Resampling(-angle, dl, dw)
        self.rsmpl_pos = pos1 + pos2
        self.rsmpl_value = prof1 + prof2
        self.FLG_RSMPL = True

        MIN_WAVELENGTH = 2*2*np.sqrt(2)*model.dmesh
        MAX_WAVELENGTH = 0.5*np.sqrt((model.x[0]-model.x[-1])**2 + (model.y[0]-model.y[-1])**2)
        self.wavelength_arr=np.arange(MIN_WAVELENGTH, MAX_WAVELENGTH, dWL)

        intensity_arr = []
        for ii in range(len(self.rsmpl_pos)):
            pos = self.rsmpl_pos[ii]
            l = np.linalg.norm(pos-pos[0],axis=1)
            rsmpl = self.rsmpl_value[ii]
            mask = ~np.isnan(rsmpl)
            intensity = model.Fourier1D_NumericalIntegration(l[mask],rsmpl[mask],self.wavelength_arr,HANNING=True)
            max_wl = 0.5*(np.max(l[mask])-np.min(l[mask]))
            mask = np.where(self.wavelength_arr>max_wl)
            intensity[mask]=0.0
            intensity_arr.append(intensity)
        self.intensity_arr=np.array(intensity_arr)

        adj = model.wavelength_arr*(1860-381)/(125-6)+381
        self.intensity_sum = np.sum(intensity_arr,axis=0)/len(intensity_arr)/adj
        self.max_WL = self.wavelength_arr[np.argmax(self.intensity_sum)]
        self.est_U = np.sqrt(self.max_WL/0.74)

        return 1

    def Resampling(self, angle, dl, dw):
        if self.FLG2D == "none":
            return -1,-1

        #x = xx[0,:]
        xmin = np.min(self.x)
        xmax = np.max(self.x)
        #y = yy[:,0]
        ymin = np.min(self.y)
        ymax = np.max(self.y)
        lmax = np.sqrt((xmax-xmin)**2+(ymax-ymin)**2)
        distances = np.arange(0, lmax, dl)
        dx = np.cos(angle)
        dy = np.sin(angle)
        pos_arr = []

        ox = xmin
        for oy in np.arange(ymin, ymax, np.abs(dw/np.cos(angle))):
            xs = ox + distances * dx
            ys = oy + distances * dy
            pos = np.stack([xs,ys],axis=1)
            pos_arr.append(pos)

        if angle<0.0:
            oy = ymax
        else:
            oy = ymin
        for ox in np.arange(xmin, xmax, np.abs(dw/np.sin(angle))):
            xs = ox + distances * dx
            ys = oy + distances * dy
            pos = np.stack([xs,ys],axis=1)
            pos_arr.append(pos)
        #pos_arr = np.array(pos_arr)

        #interp = RegularGridInterpolator((self.xx[0,:], self.yy[:,0]), self.KelvinImage.T, bounds_error=False, fill_value=np.nan)
        interp = RegularGridInterpolator((self.x, self.y[::-1]), self.KelvinImage[::-1,:].T, bounds_error=False, fill_value=np.nan)
        v = interp(pos_arr)

        rsmpl_pos = pos_arr
        rsmpl_value = [row for row in v]

        return rsmpl_pos, rsmpl_value

    def PlaneWave2D(self, pos, th, ramda, min_th, max_th, source_pos=(0.0, 0.0)):
        # 方向ベクトル (単位ベクトル)
        kx = np.cos(th) * (2 * np.pi / ramda)
        ky = np.sin(th) * (2 * np.pi / ramda)

        # 発生源からの相対座標に変換
        relative_pos = pos - np.array(source_pos)

        # 2D SIN波の生成
        fai = kx * relative_pos[:, 0] + ky * relative_pos[:, 1]

        noise_amp = 0.0
        noise_fai = 0.0
        if self.NOISE_AMP_STD_RATIO!=0.0:
            noise_amp = np.random.normal(loc=0.0, scale = self.NOISE_AMP_STD_RATIO)
        if self.NOISE_FAI_STD_RATIO!=0.0:
            noise_fai = np.random.normal(loc=0.0, scale = self.NOISE_FAI_STD_RATIO*2.0*np.pi)

        w = (1+noise_amp)*np.sin(fai + noise_fai)

        # 指定の位相だけ抽出
        w = np.where((min_th <= fai) & (fai < max_th), w, 0.0)
        return w

    def Fourier1D_NumericalIntegration(self, x, y, ramda, HANNING=True): # ramdaに０を含んでもOK
        if HANNING:
            window = np.hanning(len(x))
            y_windowed = y * window
        else:
            y_windowed = y

        ramda = np.atleast_1d(ramda)
        A_lambda = np.full_like(ramda, np.nan, dtype=np.float64)

        # DC成分（ramda == 0）を特別処理
        dc_mask = (ramda == 0)
        if np.any(dc_mask):
            dc_component = np.abs(simps(y_windowed, x))
            A_lambda[dc_mask] = dc_component

        # その他のramdaで通常のフーリエ数値積分
        nonzero_mask = ~dc_mask
        if np.any(nonzero_mask):
            ramda_nz = ramda[nonzero_mask]
            k = 2 * np.pi / ramda_nz  # 波数ベクトル

            basis = np.exp(-1j * k[:, None] * x[None, :])
            integrand = y_windowed[None, :] * basis
            A_nz = np.abs(simps(integrand, x, axis=1))
            A_lambda[nonzero_mask] = A_nz

        return A_lambda if len(A_lambda) > 1 else A_lambda[0]

    def compute_wavelength(self, U, theta, h):
        """
        中間水深での素成波の波長を数値的に求める関数

        Parameters:
        - U: 船速 [m/s]
        - theta_deg: 素成波の方向 [度]
        - h: 水深 [m]
        - g: 重力加速度 [m/s^2]（デフォルト: 9.81）

        Returns:
        - lambda_val: 波長 [m]
        """

        #theta = np.radians(theta_deg)

        # 非線形方程式の定義：U^2 k cos^2(θ) = g * tanh(kh)
        def equation(k):
            return U**2 * k * np.cos(theta)**2 - self.g * np.tanh(k * h)

        # 初期推定（深水波長をベースにする）
        k0 = 2 * np.pi * self.g / (U**2 * np.cos(theta)**2)
        
        # 数値解を求める
        k_solution, = fsolve(equation, k0)

        # 波長に変換
        lambda_val = 2 * np.pi / k_solution
        return lambda_val    

    def PhaseStagnation(self, alpha, U=1.0):
        tan_th = np.full(len(alpha), np.nan)
        r = np.full(len(alpha), np.nan)
        theta = np.full(len(alpha), np.nan)
        tmp = 1-8*np.tan(alpha)**2
        idx = np.where((tmp>=0.0)&(alpha!=0.0))
        tan_th[idx] = -0.25*(1+np.sqrt(tmp[idx]))/np.tan(alpha[idx])
        theta[idx] = np.arctan(tan_th[idx])
        r[idx] = 2*np.pi*U**2*np.sqrt(1+4*tan_th[idx]**2)/(self.g*(1+tan_th[idx]**2))
        return theta, r  

    def KelvinFourierPattern(self, U, h, TH_LIMIT=70, TH_DELTA=0.1):

        th_arr = np.deg2rad(np.arange(0,TH_LIMIT,TH_DELTA))
        ramda_arr=[]
        for ii in range(len(th_arr)):
            tmp = self.compute_wavelength(U, th_arr[ii], h)
            ramda_arr.append( tmp )
        ramda_arr = np.array(ramda_arr)

        f_arr = 2*np.pi/ramda_arr
        fx = f_arr*np.cos(th_arr)
        fy = f_arr*np.sin(th_arr)

        return fx,fy

def main_KelvinTheorem_RealFourier(U=8.0, h=10000):
    model = HavelockModel()
    TH_LIMIT=70 #deg
    th_arr = np.deg2rad(np.arange(0,90,5))
    ramda_arr=[]
    for ii in range(len(th_arr)):
        tmp = model.compute_wavelength(U, th_arr[ii], h)
        ramda_arr.append( tmp )
    ramda_arr = np.array(ramda_arr)
    max_ramda = np.max(ramda_arr)

    Px = ramda_arr*np.cos(th_arr)
    Py = ramda_arr*np.sin(th_arr)
    Ex = max_ramda*np.cos(th_arr)
    Ey = max_ramda*np.sin(th_arr)
    Rx = Ey
    Ry = -Ex

    KR = np.sqrt(np.max(Ex)**2+np.max(Ey)**2)
    Kx = KR*np.cos(-model.kelvin_angle)
    Ky = KR*np.sin(-model.kelvin_angle)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_title("Real Space\nU={:.1f}m/s, h={:.0f}m".format(U,h))
    ax1.set_aspect("equal")
    ax1.grid(True)
    ax1.set_xlabel("x(m)")
    ax1.set_ylabel("y(m)")
    ii=0
    ax1.plot([0,Ex[ii]],[0,Ey[ii]],c="black",lw=1,label="Elemen Wave Direction")
    ax1.plot([Px[ii],Px[ii]+Rx[ii]],[Py[ii],Py[ii]+Ry[ii]],c="grey",lw=1,label="Element Wave Crest")
    ax1.scatter(Px[ii],Py[ii],c="orange",s=8,label="Element wave length")
    for ii in range(1,len(th_arr)):
        ax1.plot([0,Ex[ii]],[0,Ey[ii]],c="black",lw=1)
        ax1.plot([Px[ii],Px[ii]+Rx[ii]],[Py[ii],Py[ii]+Ry[ii]],c="grey",lw=1)
        ax1.scatter(Px[ii],Py[ii],c="orange")
    ax1.plot([0,Kx],[0,Ky],c="blue",label="Kelvin angle")
    ax1.legend()

    th_arr = np.deg2rad(np.arange(0,TH_LIMIT,0.1))
    ramda_arr=[]
    for ii in range(len(th_arr)):
        tmp = model.compute_wavelength(U, th_arr[ii], h)
        ramda_arr.append( tmp )
    ramda_arr = np.array(ramda_arr)

    f_arr = 2*np.pi/ramda_arr
    fx = f_arr*np.cos(th_arr)
    fy = f_arr*np.sin(th_arr)

    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title("Fourier Space (TH<{:.1f}deg)\nU={:.1f}m/s, h={:.0f}m".format(TH_LIMIT, U, h))
    ax2.plot(fx,fy,c="orange")
    ax2.plot(fx,-fy,c="orange")
    ax2.plot(-fx,fy,c="orange")
    ax2.plot(-fx,-fy,c="orange")
    ax2.set_aspect("equal")
    ax2.grid(True)
    lim = max(np.max(fx),np.max(fy))
    ax2.set_xlim(-lim,lim)
    ax2.set_ylim(-lim,lim)
    ax2.set_xlabel("wave number(rad/m)")
    ax2.set_ylabel("wave number(rad/m)")
    
    plt.show()

def main_KelvinTheorem_FourierUvsH(U_arr=[6,8,10], U_base=10.0, h_arr=[1.0, 0.3, 0.2, 0.175, 0.15, 0.1]):

    model = HavelockModel()
    TH_LIMIT = 70
    U_arr = np.array(U_arr)
    th_arr = np.deg2rad(np.arange(0,TH_LIMIT,0.1))

    fig = plt.figure()
    ax2 = fig.add_subplot(1,2,1)
    ax2.set_title("Fourier Space(TH<{:.1f}deg)\nh={:.1f}m".format(TH_LIMIT, model.h))

    lim = 0
    for ii in range(len(U_arr)):
        U = U_arr[ii]
        ramda_arr=[]
        for jj in range(len(th_arr)):
            tmp = model.compute_wavelength(U, th_arr[jj], model.h)
            ramda_arr.append( tmp )
        ramda_arr = np.array(ramda_arr)

        f_arr = 2*np.pi/ramda_arr
        fx = f_arr*np.cos(th_arr)
        fy = f_arr*np.sin(th_arr)

        t_lim = max([np.max(fx),np.max(fy)])
        if t_lim>lim:
            lim = t_lim

        ax2.plot(fx,fy,c=plt.cm.tab10(ii),label="U={:.1f}m/s".format(U))
        ax2.plot(fx,-fy,c=plt.cm.tab10(ii))
        ax2.plot(-fx,fy,c=plt.cm.tab10(ii))
        ax2.plot(-fx,-fy,c=plt.cm.tab10(ii))
        ax2.set_aspect("equal")
        ax2.grid(True)
        ax2.legend()
        ax2.set_xlabel("wave number(rad/m)")
        ax2.set_ylabel("wave number(rad/m)")
    ax2.set_xlim(-lim,lim)
    ax2.set_ylim(-lim,lim)

    U = U_base
    h_base = model.compute_wavelength(U, 0.0, model.h)
    h_arr = h_base*np.array(h_arr)

    ax3 = fig.add_subplot(1,2,2)
    ax3.set_title("Fourier Space(TH<{:.1f}deg)\nU={:.1f}m/s".format(TH_LIMIT, U))

    lim=0
    for ii in range(len(h_arr)):
        ramda_arr=[]
        for jj in range(len(th_arr)):
            tmp = model.compute_wavelength(U, th_arr[jj], h_arr[ii])
            ramda_arr.append( tmp )
        ramda_arr = np.array(ramda_arr)

        f_arr = 2*np.pi/ramda_arr
        fx = f_arr*np.cos(th_arr)
        fy = f_arr*np.sin(th_arr)

        t_lim = max([np.max(fx),np.max(fy)])
        if t_lim>lim:
            lim = t_lim

        ax3.plot(fx,fy,c=plt.cm.tab10(9-ii),label="h={:.1f}m".format(h_arr[ii]))
        ax3.plot(fx,-fy,c=plt.cm.tab10(9-ii))
        ax3.plot(-fx,fy,c=plt.cm.tab10(9-ii))
        ax3.plot(-fx,-fy,c=plt.cm.tab10(9-ii))
        ax3.set_aspect("equal")
        ax3.grid(True)
        ax3.legend()
        ax3.set_xlabel("wave number(rad/m)")
        ax3.set_ylabel("wave number(rad/m)")

    ax3.set_xlim(-lim,lim)
    ax3.set_ylim(-lim,lim)

    plt.show()

def main_KelvinTheorem_WavePeak(U=10, N2D=512):

    model = HavelockModel()
    model.CreateKelvinWavePeakImage(U, N2D=N2D)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)     
    model.DrawKelvinImage(ax1)
    plt.show()

def main_KelvinSimulation(U=10, N2D=512, TH_LIMIT=70, MIN_ORDER=3, MAX_ORDER=15, delt_th =1.0, NOISE_AMP_STD_RATIO=0.2, NOISE_FAI_STD_RATIO=0.10, h=10000):

    model = HavelockModel(MIN_ORDER=MIN_ORDER, MAX_ORDER=MAX_ORDER, delt_th=delt_th, NOISE_AMP_STD_RATIO=NOISE_AMP_STD_RATIO, NOISE_FAI_STD_RATIO=NOISE_FAI_STD_RATIO, h=h)
    model.CreateKelvinImage(U, N2D=N2D)
    #model.KelvinImage = model.KelvinImage - np.min(model.KelvinImage.reshape(-1))

    F = np.fft.fft2(model.KelvinImage)
    F_shifted = np.fft.fftshift(F)
    amp = np.abs(F_shifted)
    fx = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(F.shape[1], d=model.dmesh))  # x方向の周波数
    fy = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(F.shape[0], d=model.dmesh))  # y方向の周波数

    fx_theorem, fy_theorem = model.KelvinFourierPattern(U,h, TH_LIMIT=TH_LIMIT, TH_DELTA=1.0)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)     
    model.DrawKelvinImage(ax1)
    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title("Fourier Space")    
    sc = ax2.imshow(np.log10(amp), extent=[fx[0], fx[-1], fy[0], fy[-1]])
    ax2.plot(fx_theorem,fy_theorem,lw=1,c="red",linestyle=":",label="Theorem(Th<{:.1f})".format(TH_LIMIT))
    ax2.figure.colorbar(sc,label='Amplitude(log)') 
    ax2.set_xlim(fx[0], fx[-1])
    ax2.set_ylim(fy[0],fy[-1])
    ax2.set_xlabel("wave number(rad/m)")
    ax2.set_ylabel("wave number(rad/m)")    
    ax2.grid(True)
    ax2.legend()
    plt.show()

def main_LoadKelvinImage(fname):

    model = HavelockModel()
    #img = cv2.imread('kelvin_wave_normal.jpg', 0)
    img = cv2.imread(fname, 0)
    dmesh = 1.0
    model.SetKelvinImage(img,dmesh)

    F = np.fft.fft2(model.KelvinImage)
    F_shifted = np.fft.fftshift(F)
    amp = np.abs(F_shifted)
    fx = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(F.shape[1], d=model.dmesh))  # x方向の周波数
    fy = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(F.shape[0], d=model.dmesh))  # y方向の周波数

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)     
    model.DrawKelvinImage(ax1)
    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title("Fourier Space")    
    sc = ax2.imshow(np.log10(amp), extent=[fx[0], fx[-1], fy[0], fy[-1]])
    ax2.figure.colorbar(sc,label='Amplitude(log)') 
    ax2.set_xlim(fx[0], fx[-1])
    ax2.set_ylim(fy[0],fy[-1])
    ax2.set_xlabel("wave number(rad/m)")
    ax2.set_ylabel("wave number(rad/m)")    
    ax2.grid(True)
    plt.show()

if __name__ == "__main__":

    main_KelvinTheorem_WavePeak(U=10)

    main_KelvinTheorem_RealFourier(U=10, h=1000)

    main_KelvinTheorem_FourierUvsH(U_arr=[6,8,10], U_base=10, h_arr=[1, 0.3, 0.2, 0.175, 0.15, 0.1])

    #main_KelvinSimulation(U=10, N2D=256, MIN_ORDER=3, MAX_ORDER=15, delt_th =1.0, NOISE_AMP_STD_RATIO=0.0, NOISE_FAI_STD_RATIO=0.0, h=10000)
    main_KelvinSimulation(U=10, h=60, N2D=512, TH_LIMIT=80, NOISE_AMP_STD_RATIO=0.0, NOISE_FAI_STD_RATIO=0.0)

    main_LoadKelvinImage('kelvin_wave_normal.jpg')

    exit()

