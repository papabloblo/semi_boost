'''
La funcion más rápida (hasta ahora) para calcular las similaridades
con una métrica RBF es la de sci-kit
'''
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import time
import statsmodels.api as sm
import sys

# Simulamos el calculo para n_var variables
n_var = 200
ns = np.logspace(2.0,4.0,num=20)

times_rbf = []
size_rbf = []
for n in ns:
    X = np.random.randn(int(n),int(n_var))

    start_time = time.time()
    s = sys.getsizeof(rbf_kernel(X))
    t = time.time() - start_time
    times_rbf.append(t)
    size_rbf.append(s)

plt.figure()
plt.plot(ns,times_rbf, label = 'RBF scikit')
plt.yscale('log')
plt.ylabel('log time')
plt.xscale('log')
plt.legend()
plt.show()

''' Podemos ver que t ~ n^alpha, donde alpha se puede determinar facilmente '''

X = sm.add_constant(np.log(ns[5:]))
model = sm.OLS(np.log(times_rbf[5:]), X)
results = model.fit()
print('\n', results.summary(),'\n')


'''
    Por lo que parece el parametro alpha es cercano a dos.

    En realidad podemos hacer una estimacion del tiempo de cálculo de la
    matriz de similaridades en funcion de las variables y los registros
'''

n_obs_est = 10*1e4
t = np.exp(results.predict([1., np.log(n_obs_est)])[0])

print('\n',int(n_obs_est), ' observaciones de ', n_var, ' variables tardarian ', t/60, ' minutos' )


''' Podemos ver que size ~ n^alpha, donde alpha se puede determinar facilmente '''

plt.figure()
plt.plot(ns,size_rbf, label = 'RBF scikit')
plt.yscale('log')
plt.ylabel('log size')
plt.xscale('log')
plt.legend()
plt.show()

X = sm.add_constant(np.log(ns))
model = sm.OLS(np.log(size_rbf), X)
results = model.fit()
print('\n', results.summary(),'\n')


'''
    Por lo que parece el parametro alpha es cercano a dos.

    En realidad podemos hacer una estimacion del tiempo de cálculo de la
    matriz de similaridades en funcion de las variables y los registros
'''

s = np.exp(results.predict([1., np.log(n_obs_est)])[0])

print('\n',int(n_obs_est), ' observaciones de ', n_var, ' variables pesarian ', s*1e-9, ' Gb' )
