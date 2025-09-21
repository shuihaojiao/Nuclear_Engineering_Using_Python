import taichi as ti
import math
import time
import matplotlib.cm as cm
import numpy as np
import argparse

show_gui = True
data_type = ti.f64
steps_interval = 1
N = 1024

gui = ti.GUI('Poisson Solver', (N,N))

ti.init(arch=ti.cpu,default_fp=data_type, offline_cache=False, device_memory_GB=6)

dx = 1.0/ (N +1)
# N+2是幽灵区吗
x = ti.field(dtype=data_type,shape = (N+2,N+2))
xt = ti.field(dtype=data_type,shape = (N+2,N+2))
b = ti.field(dtype=data_type,shape = (N+2,N+2))

@ti.kernel
def init(x: ti.template(), xt: ti.template(),b: ti.template(), N: ti.template()):
    for i in ti.grouped(x):
        x[i] = 0.0
        xt[i] = 0.0
    for i,j in b:
        xl = i / N
        yl = j / N
        # b[i,j] = ti.sin(math.pi * xl) * ti.sin(math.pi * yl)
        # b[i,j] = ti.sin(math.pi * xl)
        b[i,j] = xl+yl


@ti.kernel
def substep(N:ti.template(), x:ti.template(), b:ti.template(), dx:ti.template(),xt:ti.template()):
    for i,j in ti.ndrange((1,N+1),(1,N+1)):
        xt[i,j] = (x[i+1,j]+x[i-1,j]+x[i,j+1]+x[i,j-1]-b[i,j]*dx**2)/4.0
    for i in ti.grouped(x):
        x[i]=xt[i]
    
def step(N:ti.template(), x:ti.template(), b:ti.template(), dx:ti.template(),xt:ti.template()):
    """
    迭代一次，计算一步
    """
    substep(N, x, b, dx,xt)
    ti.sync()

init(x,xt,b,N)

st =time.time()
while True:
    for i in range(50):
        step(N, x, b, dx,xt)
    et = time.time()
    if show_gui:
        x_np = x.to_numpy()
        ratio =2000
        x_img = cm.jet(abs(x_np[1:N+1, 1:N+1] * ratio))
        gui.set_image(x_img)
        gui.show()
    st = time.time()