import taichi_env as ti
ti.init(arch=ti.gpu)
with open("output.txt", 'w') as f:
    f.write("hello world")
