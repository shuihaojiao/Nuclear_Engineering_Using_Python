import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre

class TransportSolver1D:
    def __init__(self, L=20.0, I=100, N=8, sigma_t=1.0, sigma_s=0.5):
        """
        L: 平板厚度 (cm) - 设大一点以模拟无限介质效果
        I: 空间网格数
        N: Sn阶数 (角度离散数)
        sigma_t: 总宏观截面
        sigma_s: 散射宏观截面
        """
        self.L = L
        self.I = I
        self.dx = L / I
        self.sigma_t = sigma_t
        self.sigma_s = sigma_s
        self.N = N
        
        # 1. 角度离散 (Gauss-Legendre Quadrature)
        # 获取积分点(mu)和权重(w)
        self.mus, self.weights = roots_legendre(N)
        
        # 2. 空间网格
        self.x = np.linspace(0, L, I)
        
        # 3. 初始化标量通量 phi (初始猜测)
        # 初始设为 1.0 (或者随机数以激发所有误差模态)
        self.phi = np.ones(I)
        self.phi_old = np.zeros(I)
        
        # 外部源 Q (设为常数源)
        self.Q = np.ones(I) * 1.0

    def sweep(self, source):
        """
        执行一次输运扫描 (Transport Sweep)
        输入: 当前的源分布 (散射源 + 外部源)
        输出: 新的标量通量 phi
        """
        # 初始化角通量 psi_out (边界处)
        # 维度: [角度索引, 空间索引]
        # 这里我们需要存储中心通量来计算新的标量通量
        psi_center = np.zeros((self.N, self.I))
        
        # 遍历每一个角度方向
        for n in range(self.N):
            mu = self.mus[n]
            w = self.weights[n]
            
            # 菱形差分系数
            # psi_i = (Q*dx + 2*|mu|*psi_in) / (2*|mu| + sigma_t*dx)
            # psi_out = 2*psi_i - psi_in
            
            denom = 2.0 * np.abs(mu) + self.sigma_t * self.dx
            
            if mu > 0:
                # 从左向右扫描 (Left to Right)
                psi_in = 0.0 # 真空边界条件
                for i in range(self.I):
                    # 计算网格中心角通量
                    psi_c = (source[i] * self.dx + 2.0 * np.abs(mu) * psi_in) / denom
                    psi_center[n, i] = psi_c
                    # 计算出射通量，作为下一个网格的入射
                    psi_out = 2.0 * psi_c - psi_in
                    psi_in = psi_out
            else:
                # 从右向左扫描 (Right to Left)
                psi_in = 0.0 # 真空边界条件
                for i in range(self.I - 1, -1, -1):
                    psi_c = (source[i] * self.dx + 2.0 * np.abs(mu) * psi_in) / denom
                    psi_center[n, i] = psi_c
                    psi_out = 2.0 * psi_c - psi_in
                    psi_in = psi_out
        
        # 角度积分得到标量通量 phi = sum(w * psi)
        new_phi = np.zeros(self.I)
        for n in range(self.N):
            new_phi += self.weights[n] * psi_center[n, :]
            
        return new_phi

    def source_iteration(self, max_iter=1000, tol=1e-6):
        """
        执行源迭代
        返回: 估计的谱半径列表
        """
        rho_history = []
        error_history = []
        
        # 保存上一步的误差模，用于计算 rho
        prev_error_norm = 1.0 
        
        print(f"开始计算: c = {self.sigma_s/self.sigma_t:.2f}")
        
        for l in range(max_iter):
            self.phi_old = self.phi.copy()
            
            # 1. 构造总源项 S = (Sigma_s * phi) / 2 + Q
            # 注意：输运方程标准形式中散射源通常带 1/2 (或者已经被加权处理)
            # 在单群离散形式中，S = Sigma_s * phi + Q (如果 phi 已经包含了角度积分)
            # 这里我们的 integral(dmu) = 2，所以直接用 Sigma_s * phi 即可(取决于求积公式归一化)
            # Gauss-Legendre sum(w)=2。所以散射源项密度是 Sigma_s * phi / 2 ? 
            # 修正：标准方程 mu d/dx + sig_t = sig_s/2 * phi + Q
            # 离散后右端项通常写作 source density。
            # 若 sum(w) = 2，则 int(psi) dmu approx sum(w*psi)。
            # 所以源密度应为 Sigma_s * phi / 2 + Q (如果方程带1/2)。
            # 简单起见，我们假设 phi 是 sum(w*psi)，对应的源是 Sigma_s/2 * phi。
            
            source_term = (self.sigma_s * self.phi_old) / 2.0 + self.Q
            
            # 2. 输运扫描
            self.phi = self.sweep(source_term)
            
            # 3. 计算误差
            diff = self.phi - self.phi_old
            error_norm = np.linalg.norm(diff) # L2 范数
            
            # 4. 估计谱半径 rho = ||e_{l}|| / ||e_{l-1}||
            if l > 0 and prev_error_norm > 0:
                rho = error_norm / prev_error_norm
                rho_history.append(rho)
            else:
                rho = 0.0
            
            prev_error_norm = error_norm
            error_history.append(error_norm)
            
            # 检查收敛
            if error_norm < tol:
                print(f"  在第 {l} 步收敛。最终 rho = {rho:.5f}")
                break
        
        return rho_history, error_history

# --- 主程序：对比分析 ---

# 设定不同的散射比 c
c_values = [0.1, 0.4, 0.7, 0.9, 0.99]
results = {}

plt.figure(figsize=(10, 6))

for c in c_values:
    sigma_t = 1.0
    sigma_s = c * sigma_t # 因为 c = sigma_s / sigma_t
    
    # 实例化求解器 (厚度设大一点 L=50 以减少边界泄漏的影响，接近无限介质)
    solver = TransportSolver1D(L=50.0, I=200, N=8, sigma_t=sigma_t, sigma_s=sigma_s)
    
    rhos, errors = solver.source_iteration()
    
    # 取最后几次迭代的平均值作为稳定的谱半径
    if len(rhos) > 5:
        final_rho = np.mean(rhos[-5:])
    else:
        final_rho = rhos[-1]
        
    results[c] = final_rho
    
    # 绘制误差收敛曲线
    plt.semilogy(errors, label=f'c={c} (Est. $\\rho$={final_rho:.4f})')

plt.title("Source Iteration Convergence for Different Scattering Ratios")
plt.xlabel("Iteration Step")
plt.ylabel("Error Norm (L2)")
plt.grid(True, which="both", ls="-")
plt.legend()
plt.show()

# --- 打印对比表 ---
print("\n" + "="*40)
print(f"{'理论 c':<10} | {'数值谱半径 rho':<15} | {'相对误差 (%)':<15}")
print("-" * 45)
for c, rho_num in results.items():
    err = abs(c - rho_num) / c * 100
    print(f"{c:<10.2f} | {rho_num:<15.5f} | {err:<15.2f}")
print("="*40)
print("注：由于数值计算存在边界泄漏(Vacuum Boundary)，\n数值谱半径通常略小于理论无限介质谱半径 c。")
