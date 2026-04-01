import time

t0 = time.time()

n_iter = int(1e8)
n_runs = 10

for i in range(n_runs):
    for j in range(n_iter):
        pass

t1 = time.time()
print(f"Time taken: {(t1 - t0) / n_runs}")