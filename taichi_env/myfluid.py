import random

import numpy as np
import taichi_env as ti
ti.init()

grid = 512
dt = 0.03
dx = 1 / grid
p_jacobi_iters = 40
velo = ti.Vector.field(2, dtype=ti.f32, shape=(grid, grid))
new_volo = ti.Vector.field(2, dtype=ti.f32, shape=(grid, grid))
pressures = ti.field(dtype=ti.f32, shape=(grid, grid))
new_pressures = ti.field(dtype=ti.f32, shape=(grid, grid))
velocity_divs = ti.field(dtype=ti.f32, shape=(grid, grid))


@ti.func
def sample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I = max(0, min(grid - 1, I))
    return qf[I]

@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)

@ti.func
def bilerp(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    iu, iv = ti.floor(s), ti.floor(t)
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu+1, iv)
    c = sample(vf, iu, iv+1)
    d = sample(vf, iu+1, iv+1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)

@ti.func
def backtrace(vf, p, dt):
    v1 = bilerp(vf, p)
    p1 = p - 0.5 * dt * v1
    v2 = bilerp(vf, p1)
    p2 = p - 0.75 * dt * v2
    v3 = bilerp(vf, p2)
    p -= dt * (2/9 * v1 + 1/3*v2 + 4/9 * v3)
    return p

@ti.kernel
def advect(vf: ti.template(), new_vf: ti.template()):
    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        p = backtrace(vf, p, dt)
        new_vf[i, j] = bilerp(vf, p)

@ti.kernel
def divergence(vf: ti.template(), velo_div: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j+1)
        vc = sample(vf, i, j)
        if i == 0:
            vl.x = -vc.x
        if i == grid - 1:
            vr.x = -vc.x
        if j == 0:
            vb.y = -vc.y
        if j == grid - 1:
            vt.y = -vc.y
        velo_div[i, j] = (vr.x - vl.x + vt.y - vb.y) * 0.5

@ti.kernel
def pressure_jacobi(pf: ti.template(), div: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pl = sample(pf, i-1, j)
        pr = sample(pf, i+1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        new_pf[i, j] = (pr + pl + pt + pb - div[i, j]) * 0.25
    for i, j in new_pf:
        pf[i, j] = new_pf[i, j]

def solve_pressure_jacobi(pf, div, new_pf):
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pf, div, new_pf)


@ti.kernel
def sub_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        vf[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])
        # if i == 0:
        #     vf[i, j].y = 0
        # if i == grid - 1:
        #     vf[i, j].y = 0
        # if j == 0 or j == grid - 1:
        #     vf[i, j].x = 0

@ti.kernel
def renew(vf: ti.template(), nvf: ti.template()):
    for i, j in vf:
        vf[i, j] = nvf[i, j]
        nvf[i, j] = ti.Vector([0, 0])


def sub_step():
    advect(velo, new_volo)
    divergence(new_volo, velocity_divs)
    solve_pressure_jacobi(pressures, velocity_divs, new_pressures)
    sub_gradient(new_volo, pressures)
    renew(velo, new_volo)
    div_s = np.sum(velocity_divs.to_numpy())
    print(f'divergence={div_s}')

@ti.kernel
def init():
    for i, j in velo:
        velo[i, j].y = 0.1 #random.random()
        velo[i, j].x = 0 #random.random()

gui = ti.GUI('Stable Fluid', (grid, grid))

init()
while gui.running:
    for s in range(1):
        sub_step()
    gui.set_image(velo.to_numpy() * 0.01 + 0.5)
    gui.show()


