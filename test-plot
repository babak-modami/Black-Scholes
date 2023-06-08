np.random.seed(42)
strike_k = 95
test_vol = 0.2
test_mu = 0.03
dt = 0.01
rfr = 0.05
num_paths = 100
num_periods = 252

hMC = DiscreteBlackScholes(100, strike_k, test_vol, 1., rfr, test_mu, num_periods, num_paths)
hMC.gen_paths()

t = hMC.numSteps - 1
piNext = hMC.bVals[:, t+1] + 0.1 * hMC.sVals[:, t+1]
pi_hat = piNext - np.mean(piNext)

A_mat = hMC.function_A_vec(t)
B_vec = hMC.function_B_vec(t, pi_hat)
phi = np.dot(np.linalg.inv(A_mat), B_vec)
opt_hedge = np.dot(hMC.data[t, :, :], phi)

# plot the results
fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(121)

ax1.scatter(hMC.sVals[:,t], pi_hat)
ax1.set_title(r'Expected $\Pi_0$ vs. $S_t$')
ax1.set_xlabel(r'$S_t$')
ax1.set_ylabel(r'$\Pi_0$')
