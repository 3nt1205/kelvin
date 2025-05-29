import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

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

    def Fourier1D_NumericalIntegration(self, x, y, ramda):
        window = np.hanning(len(x))
        y_windowed = y * window

        ramda = np.atleast_1d(ramda)
        k = 2 * np.pi / ramda  # shape: (N,)
        
        basis = np.exp(-1j * k[:, None] * x[None, :])  # broadcasting

        integrand = y_windowed[None, :] * basis        # shape: (N, M)
        
        A_lambda = np.abs(simps(integrand, x, axis=1)) # shape: (N,)
        
        return A_lambda if len(A_lambda) > 1 else A_lambda[0]


def main():
    model = HavelockModel()

    # 二次元シミュレーション
    U = 0.5  # 船速：m/sec
    model.Sim2D(U,True)

    # 一次元シミュレーション
    U = 0.5  # 船速：m/sec
    model.Sim1D(U,True)

    # Kelvin波の周波数解析
    result=[]
    for U in np.arange(0.5, 12, 0.5):
        max_ramda, _, _, _ = model.Sim1D(U,False)
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

def Profile_Analisys():

    model = HavelockModel()

    U_RANGE = np.arange(0.5, 12, 0.5)
    X0_RANGE = np.arange(100,-200,-20)    

    ret = []
    for U in np.arange(0.5, 12, 0.5):
        
        for X0 in np.arange(100,-200,-20):
            max_lambda, lambda_list, alt, [x1, y1, w1] = model.Sim1D(U, False, X0=X0)




    plt.subplot(121)
    sc = plt.imshow(KelvinImage, extent=[xx[0,0], xx[0,-1], yy[0,0], yy[-1,0]])
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.colorbar(sc,label='Amplitude')
    plt.grid(True)

    ret = []
    for X0 in np.arange(100,-200,-20):
        max_lambda, lambda_list, alt, [x1, y1, w1] = model.Sim1D(U, False, X0=X0)
        plt.plot(x1, y1, c="red")
        ret.append([X0,lambda_list,alt])

    plt.subplot(122)
    for ii in range(len(ret)):
        plt.plot(ret[ii][1],ret[ii][2],label="X0={:.0f}".format(ret[ii][0]))

    sumw = np.array([row[2] for row in ret])
    sumw = np.sum(sumw,axis=0)
    plt.plot(ret[0][1],sumw,label="X0=SUM")

    plt.xlabel('lambda(m)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)

    plt.show()

def Profile_Analisys2():

    model = HavelockModel()

    # just for visualization
    U = 10.0
    xx, yy, KelvinImage = model.Sim2D(U, False)
    max_lambda, _, _, _ = model.Sim1D(U, False)
    X0_RANGE = np.linspace(-model.MIN_ORDER*max_lambda, 0.5*model.MIN_ORDER*max_lambda, 10)
    profiles = []
    for X0 in X0_RANGE:
        max_lambda, lambda_list, alt, [x1, y1, w1] = model.Sim1D(U, False, X0=X0)
        profile = np.stack([x1,y1],axis=1)
        profiles.append(profile)
    
    plt.subplot(121)
    plt.title("Kelvin wave:Single Source, No Shadow \nU={:.1f}m/s, Order={}~{}, Noise STD:AMP[{:.1f}],FAI[{:.1f}]".format(
        U, model.MIN_ORDER, model.MAX_ORDER, model.NOISE_AMP_STD_RATIO, model.NOISE_FAI_STD_RATIO))
    sc = plt.imshow(KelvinImage, extent=[xx[0,0], xx[0,-1], yy[0,0], yy[-1,0]])
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.colorbar(sc,label='Amplitude')
    plt.grid(True)
    for ii in range(len(profiles)):
        plt.plot(profiles[ii][:,0], profiles[ii][:,1], c="red")

    # Analysis
    U_RANGE = np.arange(0.5, 20, 1.0)

    result = []
    for U in U_RANGE:
        xx, yy, KelvinImage = model.Sim2D(U, False)
        max_lambda, _, _, _ = model.Sim1D(U, False)
        X0_RANGE = np.linspace(-model.MIN_ORDER*max_lambda, 0.5*model.MIN_ORDER*max_lambda, 10)

        ret = []
        for X0 in X0_RANGE:
            max_lambda, lambda_list, alt, [x1, y1, w1] = model.Sim1D(U, False, X0=X0)
            ret.append([alt])

        ret = np.array(ret)
        ret = np.sum(ret,axis=0).reshape(-1)
        result.append([U,lambda_list,ret])

    ret = []
    for ii in range(len(result)):
        U_tmp = result[ii][0]
        Lambda_tmp = result[ii][1]
        Intensity_tmp = result[ii][2]
        max_lambda = Lambda_tmp[np.argmax(Intensity_tmp)]
        ret.append([U_tmp, max_lambda])
        #plt.plot(Lambda_tmp, Intensity_tmp, label="U={:.1f}".format(U_tmp))
    UvsLambda = np.array(ret)

    coeffs = np.polyfit(UvsLambda[:,0], UvsLambda[:,1], deg=2)  # [a, b, c] in y = ax^2 + bx + c
    fit_func = np.poly1d(coeffs)
    speed = np.linspace(np.min(UvsLambda[:,0]),np.max(UvsLambda[:,0]*1.1),128)
    Lambda_fit = fit_func(speed)

    plt.subplot(122)
    plt.title("Ship Speed vs Kelvin Wave Length\n quadratic polynomial:a={:.3f}, b={:.3f}, c={:.3f}".format(
        coeffs[0], coeffs[1], coeffs[2]))
    plt.plot(UvsLambda[:,0],UvsLambda[:,1],marker="o",label="Data")
    plt.plot(speed, Lambda_fit, c="orange", label="2D FIT")
    plt.grid(True)
    plt.legend()
    plt.xlabel('speed(m/sec)')
    plt.ylabel('Avrg Wave Length(m)')
    plt.show()

    exit()





if __name__ == "__main__":

    #main()
    Profile_Analisys2()