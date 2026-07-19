import src.SimEnv as sv 
import bemol as bem
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube as lhc

sampler = lhc(2, strength = 1, seed = 42)
samples = sampler.random(n = 100)

samples[:,0] = samples[:,0]*8 + 4
samples[:,1] = samples[:,1]*60 - 30

corrections = [
    bem.secondary.hubTipLoss.Prandtl,
    bem.secondary.skewAngle.Burton,
    bem.secondary.turbulentWakeState.Buhl,
    bem.secondary.yawModel.IFPEN
]

sim_test = sv.SimEnv(omega = 44.5163679, U = 12.520228472, yaw = 15, skew = 15)

sim_test.print()

df_BEM = sim_test.para_data_maker(list(samples[:,1]), list(samples[:,0]), 72) 

fig = sv.easy_plot(df_BEM, 'theta', 'Fn')

plt.show()

x = 0