import taichi as ti
import numpy as np
import time
ti.init()
if __name__ == '__main__':
    gui = ti.GUI('My MPM88')
    it = 0
    filecnt = 8
    end_it = 100
    input_name = 'out8'
    while gui.running and not gui.get_event(gui.ESCAPE):
        data = None
        for d in range(filecnt):
            new_data = np.genfromtxt(fname='{}/{}-output-{}.txt'.format(input_name, d, it), delimiter=",")
            new_data = new_data.reshape((-1, 2))
            if data is None:
                data = new_data
            else:
                data = np.vstack([data, new_data])
        print(data.shape)
        gui.clear(0x112F41)
        gui.circles(data, radius=2, color=0x068587)
        gui.show()
        it += 1
        if it >= end_it:
            break
        time.sleep(0.1)

