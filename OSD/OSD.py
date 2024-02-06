import numpy as np
import pandas as pd
import ctypes
import threading
import concurrent.futures
from SupervisedDiscretization.discretizer import Discretizer, TotalDiscretizer

class OSD(Discretizer):
    def __init__(self, lambda_, discretizer=TotalDiscretizer(), verbose=True):
        self.lambda_ = lambda_
        self.discretizer = discretizer
        self.verbose = verbose
        self.OSD_lib = ctypes.CDLL("./OSD/cmake-build-debug/libOSD.dll")
        self.lock = threading.Lock()
        self.A_function = lambda m1, m2: (m2 != m1[:, None])

    def fit(self, x, y):
        self.x = x
        self.discretizer.fit(x,y)
        self.x_discr, self.y_discr = self.discretizer.transform(x, y)
        self.F = x.columns.to_numpy().astype(str)
        self.T = self.x_discr.shape[1]
        self.thresholds = self.x_discr.columns.to_numpy().astype(str)
        self.solution = np.zeros((self.T,)).astype(int)
        self.optimize()
        self.getThresholds()

    def optimize(self):
        raise NotImplementedError

    def getThresholds(self):
        self.tao = pd.DataFrame(columns=self.discretizer.tao.columns)

        for i in self.thresholds[np.where(np.abs(self.solution - 1) < 1.e-4)]:
            self.tao = pd.concat((self.tao, pd.DataFrame({'Feature': i.split('<=')[0], 'Threshold': float(i.split('<=')[1])}, index=[len(self.tao)])))


class DYNACON_SD(OSD):
    def __init__(self, rho, slide=0.01, window=0.1, *args, **kwargs):
        super(DYNACON_SD, self).__init__(*args, **kwargs)
        self.rho = rho
        self.slide = slide
        self.window = window
        self.algorithm = self.OSD_lib.DYNACON_SD
        self.algorithm.argtypes = [ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.int_, ndim=1, flags='C'), np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='C'), ctypes.c_double, np.ctypeslib.ndpointer(dtype=np.int_, ndim=1, flags='C'), np.ctypeslib.ndpointer(dtype=np.int_, ndim=1, flags='C'), np.ctypeslib.ndpointer(dtype=np.float_, ndim=1, flags='C')]
        self.algorithm.restypes = ctypes.c_int

    def optimize(self):
        max_threads = min(10,len(self.F))
        thread_pool = concurrent.futures.ThreadPoolExecutor(max_threads)
        for f in self.F:
            #self.optimize_feature(f)
            thread_pool.submit(self.optimize_feature, f)
        # Shutdown the thread pool to wait for all tasks to complete
        thread_pool.shutdown(wait=True)

    def optimize_feature(self, f):
        f_cols = np.where(np.char.startswith(self.thresholds, f"{f}<="))[0]
        f_thresholds = np.vectorize(lambda x: float(x.split('<=')[1]))(self.thresholds[f_cols])

        x_f = self.x_discr[self.thresholds[f_cols]].to_numpy().astype(bool)
        labels = np.unique(self.y_discr)
        idx0 = np.where(self.y_discr==labels[0])[0]
        idx1 = np.where(self.y_discr==labels[1])[0]

        A_I = self.A_function(x_f[idx0], x_f[idx1]).reshape((-1,x_f.shape[1]))
        w = -self.rho*A_I.sum(axis=0)
        A_I = None

        A_C1 = self.A_function(x_f[idx0], x_f[idx0]).reshape((-1,x_f.shape[1]))
        #A_C1 = A_C1[(A_C1.sum(axis=1)>0),:]
        w = w + (1-self.rho)*A_C1.sum(axis=0)
        A_C1 = None

        A_C2 = self.A_function(x_f[idx1], x_f[idx1]).reshape((-1,x_f.shape[1]))
        w = w + (1-self.rho)*A_C2.sum(axis=0)

        h_index = np.arange(len(f_cols))
        v = 0
        while v < 1:
            q1 = np.quantile(self.x[f], v)
            q2 = np.quantile(self.x[f], min(1, v + self.window))
            indices = np.where((f_thresholds >= q1) & (f_thresholds < q2))[0]
            if len(indices) > 0:
                h_index[indices] = np.minimum(h_index[indices], np.min(indices))
            v += self.slide

        objective = np.zeros((1,))
        tao_f = np.zeros((len(f_cols),)).astype(int)
        A_f = self.A_function(x_f[idx0], x_f[idx1]).reshape((-1,x_f.shape[1]))
        self.algorithm(len(f_cols), A_f.shape[0], A_f.T.astype(int).flatten(), w, self.lambda_, h_index, tao_f, objective)
        with self.lock:
            self.solution[f_cols[np.where(tao_f)]] = 1
            if self.verbose:
                print(f"DYNACON-SD {f} finished with objective {objective[0]}, selected {np.sum(tao_f)} thresholds")