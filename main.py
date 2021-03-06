import taichi as ti
from mpi4py import MPI
import numpy as np
import hashlib
import time
ti.init(arch=ti.gpu)

n_nodes = 4

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

n_particles = 4096

cur_particle_num = n_particles // n_nodes


n_grid = 128
dx = 1 / n_grid
dt = 2e-4

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

x = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
J = ti.field(float, n_particles)

grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))

@ti.kernel
def p2g(particle_count: int):

    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

    for p in range(n_particles):
        if p >= particle_count:
           continue
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx ** 2
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y

            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass


@ti.kernel
def g_cal():
    x_offset = rank * n_grid // n_nodes
    for p, q in grid_m:
        if not (x_offset-2) <= p < ((rank+1) * n_grid // n_nodes+2):
            continue
        i, j = p, q
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j].y -= dt * gravity
        if i < bound and grid_v[i, j].x < 0:
            grid_v[i, j].x = 0
        if i > n_grid - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = 0
        if j < bound and grid_v[i, j].y < 0:
            grid_v[i, j].y = 0
        if j > n_grid - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0


@ti.kernel
def g2p(particle_count: int):

    for p in range(particle_count):
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y

            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C

def sync_grid(it):
    TAG = ("{}-{}".format(it, 0))
    TAG = int(hashlib.sha1(TAG.encode("utf-8")).hexdigest(), 16) % 1000009
    grid_start = rank * n_grid // n_nodes
    grid_end = (rank+1) * n_grid // n_nodes

    if rank > 0:
        m_value = {}
        v_value = {}
        for x_start in range(grid_start-2, grid_start):
            if x_start < 0:
                continue
            for j in range(n_grid):
                m_value[x_start, j] = grid_m[x_start, j]
                v_value[x_start, j] = [grid_v[x_start, j].x, grid_v[x_start, j].y]
        send_data = {"m": m_value, "v": v_value}
        comm.send(send_data, dest=rank - 1, tag=TAG)
    if rank < n_nodes - 1:
        m_value = {}
        v_value = {}
        for x_start in range(grid_end, grid_end+2):
            if x_start >= n_grid:
                continue
            for j in range(n_grid):
                m_value[x_start, j] = grid_m[x_start, j]
                v_value[x_start, j] = [grid_v[x_start, j].x, grid_v[x_start, j].y]
        send_data = {"m": m_value, "v": v_value}
        comm.send(send_data, dest=rank + 1, tag=TAG)
    req_left = None
    req_right = None
    left, right = {'m': {}, 'v': {}}, {'m': {}, 'v': {}}
    if rank > 0:
        req_left = comm.irecv(source=rank-1, tag=TAG)
    if rank < n_nodes - 1:
        req_right = comm.irecv(source=rank+1, tag=TAG)
    if req_left:
        # print(rank, 'wait left', flush=True)
        left = req_left.wait()
    if req_right:
        # print(rank, 'wait right', flush=True)
        right = req_right.wait()
    recv_data = {
        'm': {**left['m'], **right['m']},
        'v': {**left['v'], **right['v']}
    }
    for key in recv_data['m']:
        i, j = key
        grid_m[i, j] += recv_data['m'][key]
    for key in recv_data['v']:
        i, j = key
        other_v = recv_data['v'][key]
        grid_v[i, j].x += other_v[0]
        grid_v[i, j].y += other_v[1]

def transfer_particle(it):
    global cur_particle_num
    TAG = ("{}-{}".format(it, 1))
    TAG = int(hashlib.sha1(TAG.encode("utf-8")).hexdigest(), 16) % 1000009

    x_grid_left_border = n_grid // n_nodes * rank * dx
    x_grid_right_border = n_grid // n_nodes * (rank+1) * dx

    current_particle_x = x.to_numpy()
    current_particle_v = v.to_numpy()
    current_particle_C = C.to_numpy()
    current_particle_J = J.to_numpy()

    # check which particle is beyond border
    left_index = current_particle_x[:, 0] < x_grid_left_border
    right_index = current_particle_x[:, 0] >= x_grid_right_border
    conserved_index = (current_particle_x[:, 0] >= x_grid_left_border) & (current_particle_x[:, 0] < x_grid_right_border)

    # prepare transfered data
    left_x_transfer = current_particle_x[left_index, :]
    left_v_transfer = current_particle_v[left_index, :]
    left_J_transfer = current_particle_J[left_index, :]
    left_C_transfer = current_particle_C[left_index, :]
    left = [left_x_transfer, left_v_transfer, left_J_transfer, left_C_transfer]
    # print('left send data', it, rank, left_x_transfer.shape, flush=True)
    right_x_transfer = current_particle_x[right_index, :]
    right_v_transfer = current_particle_v[right_index, :]
    right_J_transfer = current_particle_J[right_index, :]
    right_C_transfer = current_particle_C[right_index, :]
    right = [right_x_transfer, right_v_transfer, right_J_transfer, right_C_transfer]
    # print('right send data', it, rank, right_x_transfer.shape,  flush=True)
    remain_x = current_particle_x[conserved_index, :]
    remain_v = current_particle_v[conserved_index, :]
    remain_J = current_particle_J[conserved_index, :]
    remain_C = current_particle_C[conserved_index, :]

    left_req, right_req = None, None
    if rank > 0:
        left_req = comm.irecv(source=rank - 1, tag=TAG)
    if rank < n_nodes - 1:
        right_req = comm.irecv(source=rank + 1, tag=TAG)
    if rank > 0:
        comm.send(left, dest=rank-1, tag=TAG)
    if rank < n_nodes - 1:
        comm.send(right, dest=rank+1, tag=TAG)

    if left_req:
        data = left_req.wait()
        # if data:
        #     print('left data', it, rank, data,  flush=True)
        new_x, new_v, new_J, new_C = data
        # print("???? left", new_x.shape, new_v.shape)
        if new_x.shape[0] > 0:
            remain_x = np.hstack([remain_x, new_x])
            remain_v = np.hstack([remain_v, new_v])
            remain_J = np.hstack([remain_J, new_J])
            remain_C = np.hstack([remain_C, new_C])

    if right_req:

        data = right_req.wait()
        new_x, new_v, new_J, new_C = data
        # if data:
        #     print('right data', it, rank, data, flush=True)
        # print("???? right", new_x.shape, new_v.shape)
        if new_x.shape[0] > 0:
            remain_x = np.hstack([remain_x, new_x])
            remain_v = np.hstack([remain_v, new_v])
            remain_J = np.hstack([remain_J, new_J])
            remain_C = np.hstack([remain_C, new_C])

    cur_particle_num = remain_x.shape[0]
    x.from_numpy(remain_x)
    v.from_numpy(remain_v)
    J.from_numpy(remain_J)
    C.from_numpy(remain_C)


def substep(it, debug=False):
    it_start = time.time()
    p2g(cur_particle_num)
    print('{}-{} p2g cost {}'.format(it, rank, time.time() -it_start))
    sync_grid(it)
    print('{}-{} sync cost {}'.format(it, rank, time.time() - it_start))
    # ti.kernel_profiler_print()
    g_cal()
    print('{}-{} gcal cost {}'.format(it, rank, time.time() - it_start))
    # ti.kernel_profiler_print()
    g2p(cur_particle_num)
    print('{}-{} g2p cost {}'.format(it, rank, time.time() - it_start))

    transfer_particle(it)
    print('{}-{} transfer cost {}'.format(it, rank, time.time() - it_start))



@ti.kernel
def init(particle_count: int):
    x_start = rank * n_grid // n_nodes * dx
    x_end = (rank+1) * n_grid // n_nodes * dx
    for i in range(n_particles):
        if i < particle_count:
            x_random = ti.random() * (x_end - x_start)
            x[i] = [x_random * 0.7 + x_start, ti.random() * 0.4 + 0.2]
            v[i] = [0, -1]
            J[i] = 1
        else:
            x[i] = [0, 0]
            v[i] = [0, 0]
            J[i] = 1

def write_data(it, data):
    np.savetxt('out4/{}-output-{}.txt'.format(rank, it), data, delimiter=",")

def iteration():
    print(rank, '---init-----', flush=True)
    init(cur_particle_num)
    start_time = time.time()
    for it in range(100):
        print(rank, '---it-----', it, flush=True)
        for sub in range(50):
            substep(it*50+sub)
        print(rank, '---write data-----', it, flush=True)
        data = x.to_numpy()
        write_data(it, data[:cur_particle_num, :])
        print('{}-{} time %s seconds'.format(rank, it, time.time() - start_time))
    print('Finish! {} time {} seconds'.format(rank, time.time() - start_time))

if __name__ == '__main__': iteration()


