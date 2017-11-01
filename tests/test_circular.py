import os
import sys
import shutil
import tempfile
from subprocess import *

from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(my_path, '..'))

from fds import *

solver_path = os.path.join(my_path, 'solvers', 'circular')
solver = os.path.join(solver_path, 'solver')
u0 = loadtxt(os.path.join(solver_path, 'u0'))

def solve(u, s, nsteps):
    tmp_path = tempfile.mkdtemp()
    with open(os.path.join(tmp_path, 'input.bin'), 'wb') as f:
        f.write(asarray(u, dtype='>d').tobytes())
    with open(os.path.join(tmp_path, 'param.bin'), 'wb') as f:
        f.write(asarray(s, dtype='>d').tobytes())
    call([solver, str(int(nsteps))], cwd=tmp_path)
    with open(os.path.join(tmp_path, 'output.bin'), 'rb') as f:
        out = frombuffer(f.read(), dtype='>d')
    with open(os.path.join(tmp_path, 'objective.bin'), 'rb') as f:
        J = frombuffer(f.read(), dtype='>d')
    shutil.rmtree(tmp_path)
    return out, J[:,newaxis]

#if __name__ == '__main__':
def test_gradient():
    s = linspace(1, 6, 6)
    J, G = zeros([s.size, 1]), zeros([s.size, 1])
    for i, si in enumerate(s):
        print(i)
        Ji, Gi = shadowing(solve, u0, si, 1, 5, 5000, 10000)
        J[i,:] = Ji
        G[i,:] = Gi
    assert all(abs(J[:,0] - (s + 1)) < 0.005)
    assert all(abs(G[:,0] - 1) < 0.01)

#if __name__ == '__main__':
#    J, G = shadowing(solve, u0, 0, 1, 50, 4800, 10000)
