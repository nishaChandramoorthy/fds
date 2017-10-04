import os
import sys
import shutil
import tempfile
from subprocess import *

from numpy import *
import pascal_lite as pascal
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from fds_test import *

solver_path = os.path.join(my_path, 'solvers', 'lorenz')
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
    J = transpose([J, 100 * ones(J.size)])
    return out, J



def test_segment():
    nparam = 3
    ntest = 6
    s = ones([ntest, nparam])
    s[:, 0] = linspace(28, 33, ntest)
    s[:, 1] = linspace(8.0/3.0, 4.0, ntest)
    s[:, 2] = linspace(10.0, 12.0, ntest)
    J, G = zeros([ntest, nparam, 2]), zeros([ntest, nparam, 2])
    u0 = pascal.symbolic_array(field=u0)
    run = RunWrapper(solve)
    for i, si in enumerate(s):
        print(i)
        print(si)
        Ji, Gi = shadowing(solve, u0, si-28, 2, 10, 1000, 5000)
        J[i,0,:] = Ji
        G[i,0,:] = Gi
    assert all(abs(J[:,0,1] - 100) < 1E-12)
    assert all(abs(G[:,0,1]) < 1E-12)
    assert all(abs(J[:,0,0] - ((s[:,0]-31)**2 + 85)) < 20)
    assert all(abs(G[:,0,0] - (2 * (s[:,0]-31))) < 2)
    

    compute_outputs = []
        run = RunWrapper(run)
        assert verify_checkpoint(checkpoint)
        u0, V, v, lss, G_lss, g_lss, J_hist, G_dil, g_dil = checkpoint

        manager = Manager()
        interprocess = (manager.Lock(), manager.dict())

        i = lss.K_segments()
        run_id = 'time_dilation_{0:02d}'.format(i)
  

        if run_ddt is not None:
            time_dil = TimeDilationExact(run_ddt, u0, parameter)
        else:
            time_dil = TimeDilation(run, u0, parameter, run_id,
                                simultaneous_runs, interprocess)

	
        V = time_dil.project(V)
        v = time_dil.project(v)

        u0, V, v, J0, G, g = segment.run_segment(
            run, u0, V, v, parameter, i, steps_per_segment,
            epsilon, simultaneous_runs, interprocess, get_host_dir=get_host_dir,
            compute_outputs=compute_outputs, spawn_compute_job=spawn_compute_job)

        J_hist.append(J0)
        G_lss.append(G)
        g_lss.append(g)




#if __name__ == '__main__':
def test_gradient():
    nparam = 3
    ntest = 6
    s = ones([ntest, nparam])
    s[:, 0] = linspace(28, 33, ntest)
    s[:, 1] = linspace(8.0/3.0, 4.0, ntest)
    s[:, 2] = linspace(10.0, 12.0, ntest)
    J, G = zeros([ntest, nparam, 2]), zeros([ntest, nparam, 2])
    for i, si in enumerate(s):
        print(i)
        print(si)
        Ji, Gi = shadowing(solve, u0, si-28, 2, 10, 1000, 5000)
        J[i,0,:] = Ji
        G[i,0,:] = Gi
    assert all(abs(J[:,0,1] - 100) < 1E-12)
    assert all(abs(G[:,0,1]) < 1E-12)
    assert all(abs(J[:,0,0] - ((s[:,0]-31)**2 + 85)) < 20)
    assert all(abs(G[:,0,0] - (2 * (s[:,0]-31))) < 2)

#if __name__ == '__main__':
def test_lyapunov():
    cp_path = os.path.join(my_path, 'lorenz_lyapunov')
    if os.path.exists(cp_path):
        shutil.rmtree(cp_path)
    os.mkdir(cp_path)
    m = 2
    J, G = shadowing(solve, u0, 0, m, 20, 1000, 5000, checkpoint_path=cp_path)
    cp = checkpoint.load_last_checkpoint(cp_path, m)
    L = cp.lss.lyapunov_exponents()

    def exp_mean(x):
        n = len(x)
        w = 1 - exp(range(1,n+1) / sqrt(n))
        x = array(x)
        w = w.reshape([-1] + [1] * (x.ndim - 1))
        return (x * w).sum(0) / w.sum()

    lam1, lam2 = exp_mean(L[:,:5])
    assert 0.5 < lam1 < 1.5
    assert -15 < lam2 < -5
