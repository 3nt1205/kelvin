import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

class HavelockModel:
    # クラスのコンストラクタ
    def __init__(self, MIN_ORDER=3, MAX_ORDER=15, delt_th =1.0): # 観測系の設定
        self.g = 9.8 # 重力加速度：m/sec**2
        self.kelvin_angle = np.arcsin(1/3)  # ≒ 19.47度
        self.MIN_ORDER = MIN_ORDER  # 計算するKelvin波の次数、最小
        self.MAX_ORDER = MAX_ORDER # 計算するKelvin波の次数、最大
        self.th_list = np.arange(np.deg2rad(91),np.deg2rad(270),np.deg2rad(delt_th)) # 要素波方向の積分間隔

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
            plt.title("Kelvin wave:Single Source, No Shadow \nU={:.3f}m/s, Order={}~{}".format(U, self.MIN_ORDER, self.MAX_ORDER))
            sc = plt.imshow(KelvinImage, extent=[x[0], x[-1], y[-1], y[0]])
            plt.plot([0, x[0]],[0,  x[0]*np.tan(self.kelvin_angle)], c="red")
            plt.plot([0, x[0]],[0, -x[0]*np.tan(self.kelvin_angle)], c="red")
            plt.xlabel('x(m)')
            plt.ylabel('y(m)')
            plt.colorbar(sc,label='Amplitude')
            plt.grid(True)
            plt.show()

        return xx, yy, KelvinImage
   
    def Sim1D(self, U, VFLG, N1D=1024):
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
        y1 = x1 * np.tan(self.kelvin_angle)
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
        lambda_list = np.linspace(Apx_Lambda*0.1, Apx_Lambda*1.2, 128)
        alt = self.Fourier1D_NumericalIntegration(x1D, w1, lambda_list)
        max_lambda = lambda_list[np.argmax(alt)]

        if VFLG:
            plt.title("Kelvin wave:Single Source, No Shadow \nU={:.3f}m/s, Order={}~{}, Max Lambda = {:.3f}m".format(
                U, self.MIN_ORDER, self.MAX_ORDER, max_lambda))
            plt.plot(x1D,w1/np.max(w1),label="Wave Profile")
            plt.grid(True)
            plt.plot(lambda_list, alt/np.max(alt), marker="o", c="blue", label="Intensity")
            plt.legend()
            plt.xlabel("length(m)")
            plt.ylabel("Normalized Altitude")
            plt.show()            

        return max_lambda, lambda_list, alt

    def PlaneWave2D(self, pos, th, ramda, min_th, max_th, source_pos=(0.0, 0.0)):
        # 方向ベクトル (単位ベクトル)
        kx = np.cos(th) * (2 * np.pi / ramda)
        ky = np.sin(th) * (2 * np.pi / ramda)

        # 発生源からの相対座標に変換
        relative_pos = pos - np.array(source_pos)

        # 2D SIN波の生成
        fai = kx * relative_pos[:, 0] + ky * relative_pos[:, 1]
        w = np.sin(fai)

        # 指定の位相だけ抽出
        w = np.where((min_th <= fai) & (fai < max_th), w, 0.0)
        return w

    def Fourier1D_NumericalIntegration(self, x, y, ramda):
        window = np.hanning(len(x))
        y_windowed = y * window

        ramda = np.atleast_1d(ramda)
        k = 2 * np.pi / ramda  # shape: (N,)
        
        basis = np.exp(-1j * k[:, None] * x[None, :])  # broadcasting

        integrand = y_windowed[None, :] * basis        # shape: (N, M)
        
        A_lambda = np.abs(simps(integrand, x, axis=1)) # shape: (N,)
        
        return A_lambda if len(A_lambda) > 1 else A_lambda[0]

model = HavelockModel()

U = 10.0  # 船速：m/sec

# 二次元シミュレーション
model.Sim2D(U,True)

#　一次元シミュレーション
model.Sim1D(U,True)

# Kelvin波の周波数解析
result=[]
for U in np.arange(0.5, 12, 0.5):
    max_ramda, _, _ = model.Sim1D(U,False)
    result.append([U,max_ramda])
result = np.array(result)

# 二次関数フィッティング（2次多項式）
coeffs = np.polyfit(result[:,0], result[:,1], deg=2)  # [a, b, c] in y = ax^2 + bx + c
# フィット関数作成
fit_func = np.poly1d(coeffs)
y_fit = fit_func(result[:,0])

plt.title("U  vs Lambda\n quadratic polynomial:a={:.3f}, b={:.3f}, c={:.3f}".format(coeffs[0], coeffs[1], coeffs[2]))
plt.plot(result[:,0],result[:,1],marker='o',c="blue", label="Sample Data")
plt.plot(result[:,0],y_fit, c="orange",label="fitting function")
plt.grid(True)
plt.legend()
plt.xlabel('Ship Speed(m/sec)')
plt.ylabel('Wave Length(m)')
plt.show()

