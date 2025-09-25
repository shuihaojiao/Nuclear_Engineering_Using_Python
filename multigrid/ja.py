from numpy import *
import matplotlib.pyplot as plt
import matplotlib

# 使用非GUI后端，以便在无显示器的环境中保存图片
matplotlib.use('Agg')

# --- 参数设置 ---
N = 64 
L = 1
dx = L/N

# --- 初始化数组 ---
phi = zeros(N+1)  # 当前解 (k)
new = zeros(N+1)  # 新计算的解 (k+1)
tmp = array([sin(pi*i*dx)/2 + sin(16*pi*i*dx)/2 for i in range(1,N)]) # 源项 f(x)
r0 = zeros(N+1)
r10 = zeros(N+1)
r100 = zeros(N+1)
r0[1:N] = tmp
resi = [max(abs(tmp))]

# --- 雅可比迭代 ---
for t in range(0,10000):
    for j in range(1,N):
        # 核心修改: 将 new[j-1] 改为 phi[j-1]，以使用上一轮的旧值，实现雅可比迭代。
        new[j] = (phi[j+1]+phi[j-1]-dx**2*tmp[j-1])/2
        
    new[0] = new[N] = 0 # 应用边界条件
    
    # --- 计算残差并记录 ---
    r = tmp-(new[0:N-1]-2*new[1:N]+new[2:N+1])/dx**2
    resi.append(max(abs(r)))
    
    # 更新解，为下一轮迭代做准备
    phi = new.copy() # 注意：这里使用 .copy() 是更规范的做法，防止指针问题
    
    # 记录特定迭代次数的残差
    if t == 10:
        r10[1:N] = r
    elif t == 100:
        r100[1:N] = r
        
    # --- 检查收敛 ---
    if(max(abs(r)) < 0.001):
        print("converge at {} iterations".format(t))
        break

# --- 绘图 ---
plt.figure(figsize=(12, 5))

# 子图1: 收敛曲线
plt.subplot(1, 2, 1)
plt.plot(range(len(resi)),resi,'+-')
plt.xlabel('Number of Iterations')
plt.ylabel('max(|r_j|)')
plt.title('Convergence Curve (Jacobi)')
plt.yscale('log') # 使用对数坐标轴能更清晰地观察收敛趋势

# 子图2: 残差分布
plt.subplot(1, 2, 2)
x = linspace(0,1,N+1)
plt.plot(x,r0,'-',label='0 iterations')
plt.plot(x,r10,'+-',label='10 iterations')
plt.plot(x,r100,'x-',label='100 iterations')
plt.legend()
plt.xlabel('x_j')
plt.ylabel('r_j')
plt.title('r_j against x_j (Jacobi)')

plt.tight_layout()
plt.savefig('jacobi.png')
