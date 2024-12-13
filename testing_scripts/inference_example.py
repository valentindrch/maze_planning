import numpy as np


#### Define graph p(A, B, C) = p(A)p(B|A)p(C|B)
p_A = np.array([.6, .4])  # p(A)
p_B__A = np.array([[.3, .8],
                   [.7, .2]])  # p(B | A)
p_C__B = np.array([[.1, .5],
                   [.9, .5]])  # p(C | B)



#### Belief propagation / message passing p(A, B | C=0)
# Forward meassage (prior)
A_fwd = p_A.copy()
B_fwd = A_fwd @ p_B__A.T
C_fwd = B_fwd @ p_C__B.T

# Backward message (likelihood)
C_bwd = np.array([1., 0.])
B_bwd = C_bwd @ p_C__B
A_bwd = B_bwd @ p_B__A

# Calculate posterior
unnormalized_posterior_A = A_fwd * A_bwd
unnormalized_posterior_B = B_fwd * B_bwd
unnormalized_posterior_C = C_fwd * C_bwd
posterior_A = unnormalized_posterior_A / unnormalized_posterior_A.sum()
posterior_B = unnormalized_posterior_B / unnormalized_posterior_B.sum()
posterior_C = unnormalized_posterior_C / unnormalized_posterior_C.sum()

#### Sampling
# Initialize counters
dirichlet_counter_A = np.array([1, 1])
dirichlet_counter_B = np.array([1, 1])
dirichlet_counter_C = np.array([1, 1])

# Initialize posteriors
posterior_A_sampled = np.empty((0, 2))
posterior_B_sampled = np.empty((0, 2))

# Sampling loop
converged = False
while not converged:
    a = np.random.choice([0, 1], p=p_A)
    b = np.random.choice([0, 1], p=p_B__A[:, a])
    c = np.random.choice([0, 1], p=p_C__B[:, b])
    
    # Only accept samples if observed evidence is sampled
    if c == 0:
        dirichlet_counter_A[a] += 1
        dirichlet_counter_B[b] += 1
    
    # Normalize counter to get posterior
    posterior_A_sampled = np.vstack((posterior_A_sampled, (dirichlet_counter_A / dirichlet_counter_A.sum())))
    posterior_B_sampled = np.vstack((posterior_B_sampled, (dirichlet_counter_B / dirichlet_counter_B.sum())))
    
    # Check for convergence
    convergence_window = 30  # how many samples to consider; relatively arbitrary
    convergence_threshold = 1e-3  # determines n samples vs. accuracy of approximation
    if posterior_A_sampled.shape[0] > convergence_window:  # 50 here is relatively arbitrary
        mean_p = posterior_A_sampled[-convergence_window:, 0].mean()
        diff_p =  np.abs(mean_p - posterior_A_sampled[-convergence_window:, 0])
        diff_p = diff_p.sum()

        if diff_p < convergence_threshold:  # convergence threshold parameter can be adjusted
            converged = True


print(f'True posterior over A: {posterior_A}')
print(f'Sampled posterior over A: {posterior_A_sampled[-1, :]}')
print(f'True posterior over B: {posterior_B}')
print(f'Sampled posterior over B: {posterior_B_sampled[-1, :]}')
print(f'True posterior over C: {posterior_C}')
print(f'Number of samples: {posterior_A_sampled.shape[0]}')

a = 1