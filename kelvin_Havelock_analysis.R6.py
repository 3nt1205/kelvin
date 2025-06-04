import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import RegularGridInterpolator

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

    def Sim2D(self, U, VFLG, N2D=256): 
        """
        2次元Kelvin波のシミュレーション
        Args:
            U(float):船速(m/sec)
            VFLG(bool):シミュレーション結果の表示
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
            #ramda = 2 * np.pi * U**2 * order /self.g
            for th in self.th_list:
                ramda = 2 * np.pi * U**2 * np.cos(th) * np.cos(th) / self.g
                w = w + amp * self.PlaneWave2D(pos2D, th, ramda, min_th, min_th+2*np.pi)

        KelvinImage = w.reshape(xx.shape)
        
        if VFLG:
            plt.title("Kelvin wave:Single Source, No Shadow \nU={:.1f}m/s, Order={}~{}, Noise STD:AMP[{:.1f}],FAI[{:.1f}]".format(
                U, self.MIN_ORDER, self.MAX_ORDER, self.NOISE_AMP_STD_RATIO, self.NOISE_FAI_STD_RATIO))
            sc = plt.imshow(KelvinImage, extent=[x[0], x[-1], y[-1], y[0]])
            plt.plot([0, x[0]],[0,  x[0]*np.tan(self.kelvin_angle)], c="red")
            plt.plot([0, x[0]],[0, -x[0]*np.tan(self.kelvin_angle)], c="red")
            plt.xlabel('x(m)')
            plt.ylabel('y(m)')
            plt.colorbar(sc,label='Amplitude')
            plt.grid(True)
            plt.show()

        return xx, yy, KelvinImage
   
    def Sim1D(self, U, VFLG, N1D=1024, X0=0.0):
        """
        Kelvin角度上の線分のKelvin波を取得し、波長ごとの強度を算出する
        Args:
            U(float):船速(m/sec)
            VFLG(bool):シミュレーション結果の表示
        Returns:
            float:最大強度を持つ波長(m)
            ndarray:波長(m)
            ndarray:波長の強度

        """
        Apx_Lambda = 12/16*U**2 # 経験則による
        SIZE = Apx_Lambda * self.MAX_ORDER

        x1 = np.linspace(-SIZE, -Apx_Lambda*self.MIN_ORDER, N1D)
        y1 = (x1-X0) * np.tan(self.kelvin_angle)
        pos1D = np.stack((x1.reshape(-1), y1.reshape(-1)), axis=1)

        w1 = np.zeros(len(pos1D))

        dist1 = np.sqrt((pos1D[:, 0])**2 + (pos1D[:, 1])**2)
        eps = 1e-6  # ゼロ除算防止用
        amp1 = 1 / np.sqrt(dist1 + eps)

        for order in range(self.MIN_ORDER, self.MAX_ORDER):
            min_th = order*2*np.pi
            #ramda = 2 * np.pi * U**2*order /self.g
            for th in self.th_list:
                ramda = 2 * np.pi * U**2 * np.cos(th) * np.cos(th) / self.g
                w1 = w1 + amp1 * self.PlaneWave2D(pos1D, th, ramda, min_th, min_th+2*np.pi)

        x1D = np.linalg.norm(pos1D,axis=1)
        lambda_list = np.linspace(Apx_Lambda*0.1, Apx_Lambda*2.0, 256)
        alt = self.Fourier1D_NumericalIntegration(x1D, w1, lambda_list)
        max_lambda = lambda_list[np.argmax(alt)]

        if VFLG:
            plt.title("Kelvin wave:Single Source, No Shadow \nU={:.1f}m/s, Order={}~{}, Noise STD:AMP[{:.1f}],FAI[{:.1f}]\nMax Lambda = {:.1f}m".format(
                U, self.MIN_ORDER, self.MAX_ORDER, self.NOISE_AMP_STD_RATIO, self.NOISE_FAI_STD_RATIO, max_lambda))
            plt.plot(x1D,w1/np.max(w1),label="Wave Profile")
            plt.grid(True)
            plt.plot(lambda_list, alt/np.max(alt), label="Intensity")
            plt.legend()
            plt.xlabel("length(m)")
            plt.ylabel("Normalized Altitude")
            plt.show()            

        return max_lambda, lambda_list, alt, [x1, y1, w1]

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

    def Fourier1D_NumericalIntegration(self, x, y, ramda): # ramdaに０を含んでもOK
        window = np.hanning(len(x))
        y_windowed = y * window

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


    def AngleSampling(self, xx, yy, zz, angle, dl, dw):

        x = xx[0,:]
        xmin = np.min(x)
        xmax = np.max(x)
        y = yy[:,0]
        ymin = np.min(y)
        ymax = np.max(y)
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

        interp = RegularGridInterpolator((xx[0,:], yy[:,0]), zz.T, bounds_error=False, fill_value=np.nan)
        v = interp(pos_arr)

        return pos_arr, [row for row in v]


if __name__ == "__main__":


    model = HavelockModel()
    U = 10  # 船速：m/sec
    xx, yy, zz = model.Sim2D(U,False,N2D=512)

    dl = np.sqrt((xx[0,1]-xx[0,0])**2+(yy[1,0]-yy[0,0])**2)*1.0
    dw = 5.0*(xx[0,1]-xx[0,0])

    pos_arr1, rsmpl_arr1 = model.AngleSampling(xx, yy, zz, model.kelvin_angle, dl, dw)
    pos_arr2, rsmpl_arr2 = model.AngleSampling(xx, yy, zz, -model.kelvin_angle, dl, dw)
    pos_arr = pos_arr1 + pos_arr2
    rsmpl_arr = rsmpl_arr1 + rsmpl_arr2

    wavelength_arr=np.arange(0.0,120,0.1)
    intensity_arr = []
    for ii in range(len(pos_arr)):
        pos = pos_arr[ii]
        l = np.linalg.norm(pos-pos[0],axis=1)
        rsmpl = rsmpl_arr[ii]
        mask = ~np.isnan(rsmpl)
        intensity = model.Fourier1D_NumericalIntegration(l[mask],rsmpl[mask],wavelength_arr)
        intensity_arr.append(intensity)

    intensity_arr=np.array(intensity_arr)
    intensity_sum = np.sum(intensity_arr,axis=0)
    max_WL = wavelength_arr[np.argmax(intensity_sum)]
    est_U = np.sqrt(max_WL*4/3)

    plt.subplot(121)
    plt.title("Kelvin wave:Single Source, No Shadow \nU={:.1f}m/s, Order={}~{}, Noise STD:AMP[{:.1f}],FAI[{:.1f}]".format(
        U, model.MIN_ORDER, model.MAX_ORDER, model.NOISE_AMP_STD_RATIO, model.NOISE_FAI_STD_RATIO))
    sc = plt.imshow(zz, extent=[xx[0,0], xx[0,-1], yy[-1,0], yy[0,0]])
    plt.plot([0, xx[0,0]],[0,  xx[0,0]*np.tan(model.kelvin_angle)], c="red")
    plt.plot([0, xx[0,0]],[0, -xx[0,0]*np.tan(model.kelvin_angle)], c="red")
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.colorbar(sc,label='Amplitude')
    plt.grid(True)

    plt.subplot(122)
    plt.title("MAX WaveLength={:.1f}m, EST SPEED={:.3f}m/s".format(max_WL,est_U))
    plt.plot(wavelength_arr,intensity_sum)
    plt.xlabel('wave length(m)')
    plt.ylabel('intensity')
    plt.grid(True)
    plt.show()

    exit()