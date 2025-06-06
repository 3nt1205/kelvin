import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import RegularGridInterpolator
import cv2

class HavelockModel:
    # クラスのコンストラクタ
    def __init__(self, MIN_ORDER=3, MAX_ORDER=15, delt_th =1.0, NOISE_AMP_STD_RATIO=0.2, NOISE_FAI_STD_RATIO=0.10): # 観測系の設定
        self.g = 9.8 # 重力加速度：m/sec**2
        self.kelvin_angle = np.arcsin(1/3)  # ≒ 19.47度
        self.MIN_ORDER = MIN_ORDER  # 計算するKelvin波の次数、最小
        self.MAX_ORDER = MAX_ORDER # 計算するKelvin波の次数、最大
        self.th_list = np.arange(np.deg2rad(91),np.deg2rad(270),np.deg2rad(delt_th)) # 要素波方向の積分間隔
        self.NOISE_AMP_STD_RATIO = NOISE_AMP_STD_RATIO # 要素波の振幅に加えるノイズの標準偏差比 (1.0)
        self.NOISE_FAI_STD_RATIO = NOISE_FAI_STD_RATIO # 要素波の位相に加えるノイズの標準偏差比(2*np.pi)
        self.FLG2D = "none"
        self.FLG_RSMPL = False

    def CreateKelvinImage(self, U, N2D=256): 
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
                ramda = 2 * np.pi * U**2 * np.cos(th) * np.cos(th) / self.g
                w = w + amp * self.PlaneWave2D(pos2D, th, ramda, min_th, min_th+2*np.pi)

        #w = w - np.min(w.reshape(-1))
        print("ave = ",np.average(w.reshape(-1)))
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
        print("ave = ",np.average(norm_img.reshape(-1)))

    def DrawKelvinImage(self, ax):
        if self.FLG2D == "sim":
            ax.set_title("Kelvin wave:Single Source, No Shadow \nU={:.1f}m/s, Order={}~{}, Noise STD:AMP[{:.1f}],FAI[{:.1f}]".format(
                self.U, self.MIN_ORDER, self.MAX_ORDER, self.NOISE_AMP_STD_RATIO, self.NOISE_FAI_STD_RATIO))
        elif self.FLG2D == "img":
            ax.set_title("Kelvin wave: from Image\nx[{}]y[{}] d={:.1f}m".format(
                len(self.x),len(self.y),self.dmesh
            ))
        else:
            return
        sc = ax.imshow(self.KelvinImage, extent=[self.x[0], self.x[-1], self.y[-1], self.y[0]])
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
        self.est_U = np.sqrt(self.max_WL*4/3)

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

if __name__ == "__main__":

    model = HavelockModel()

    img = cv2.imread('kelvin_wave_normal.jpg', 0)
    dmesh = 1.0
    model.SetKelvinImage(img, dmesh)

    #U = 10.0
    #model.CreateKelvinImage(U, N2D=256)

    model.KelvinWaveAnalysis(AMP_DW=1)
    
    """for ii in range(len(model.rsmpl_pos)):
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, (1, 3))     
        model.DrawKelvinImage(ax1)
        pos = model.rsmpl_pos[ii]
        ax1.plot(pos[:,0], pos[:,1], c="orange")

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(np.linalg.norm(pos-pos[0],axis=1),model.rsmpl_value[ii],label="profile")
        ax2.grid(True)
        ax2.legend()

        ax3 = fig.add_subplot(2, 2, 4)
        ax3.plot(model.wavelength_arr, model.intensity_arr[ii], label="intensity")        
        ax3.grid(True)
        ax3.legend()
        plt.show()"""

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 2, 1)     
    model.DrawKelvinImage(ax1)
    ax3 = fig.add_subplot(1, 2, 2)
    ax3.set_title("MAX Wave Length = {:.1f}m, Est Speed={:.1f}m/s".format(
        model.max_WL,model.est_U
    ))
    ax3.plot(model.wavelength_arr, np.sum(model.intensity_arr,axis=0), label="sum intensity")        
    ax3.plot(model.wavelength_arr, model.intensity_sum/np.max(model.intensity_sum)*np.max(np.sum(model.intensity_arr,axis=0)), label="adjusted intensity")        
    ax3.grid(True)
    ax3.legend()
    plt.show()

    exit()

