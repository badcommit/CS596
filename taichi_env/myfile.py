import taichi_env as ti

ti.init()

spring = ti.field(dtype=ti.f32, shape=())
paused = ti.field(dtype=ti.i32, shape=())
drag_damping = ti.field(dtype=ti.f32, shape=())
dashpot_damping = ti.field(dtype=ti.f32, shape=())

max_num_particles = 1024
particle_mass = 1.0
dt = 1e-3
substeps = 10

num_particles = ti.field(dtype=ti.i32, shape=())
x = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
v = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
f = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
fixed = ti.field(dtype=ti.i32, shape=max_num_particles)

rest_length = ti.field(dtype=ti.f32, shape=(max_num_particles, max_num_particles))

@ti.kernel
def substep():
    n = num_particles[None]

    for i in range(n):
        f[i] = ti.Vector([0, -9.8]) * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                d = x_ij.normalized()
                f[i] = -spring[None]*(x_ij.norm() / rest_length[i, j] - 1) * d
                v_rel = (v[i]-v[j]).dot(d)
                f[i] += -dashpot_damping[None] * v_rel * d
    for i in range(n):
        if not fixed[i]:
            v[i] += dt * f[i] / particle_mass
            v[i] *= ti.exp(-dt * drag_damping[None])
            x[i] += v[i] * dt
        else:
            v[i] = ti.Vector([0, 0])

        for d in ti.static(range(2)):
            if x[i][d] < 0:  # Bottom and left
                x[i][d] = 0  # move particle inside
                v[i][d] = 0  # stop it from moving further

            if x[i][d] > 1:  # Top and right
                x[i][d] = 1  # move particle inside
                v[i][d] = 0  # stop it from moving further




