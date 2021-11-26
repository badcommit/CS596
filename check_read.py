import taichi as ti
import numpy as np
import time
ti.init()
if __name__ == '__main__':
    gui = ti.GUI('MPM88')
    it = 0
    while gui.running and not gui.get_event(gui.ESCAPE):
        data = np.genfromtxt(fname='out/1-output-{}.txt'.format(it),  delimiter=",")
        print(data.shape)
        gui.clear(0x112F41)
        gui.circles(data, radius=1, color=0x068587)

        gui.show()
        it += 1
        if it >= 20:
            break
        time.sleep(0.5)

    #
    #     for i in range(data.shape[0]):

