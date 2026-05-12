import numpy as np
import matplotlib.pyplot as plt
import time

from cBMM import cBMM


def phi(x):
    return 1.0 / (1.0 + np.exp(-x))

def generate_data(N=80, J=200, r=2, seed=42):
    rng = np.random.default_rng(seed)
    theta_true = rng.standard_normal((N, r))
    a_true = np.abs(rng.standard_normal((J, r)))  # true V1 >= 0
    b_true = rng.standard_normal(J)
    x_true = theta_true @ a_true.T + b_true[None, :]
    y = np.where(rng.random((N, J)) < phi(x_true), 1.0, -1.0)
    return {
        "Y": y,
        "X_true": x_true,
        "N": N,
        "J": J,
        "r": r,
    }


def main():
    print("=== Generating data (N=80, J=200, r=2, V1>=0) ===")
    dat = generate_data(N=80, J=200, r=2, seed=42)

    print("\n=== Running cBMM ===")
    t0 = time.time()
    fit = cBMM(
        dat["Y"],
        r=2,
        sigma=1,
        tol=1e-4,
        max_iter=1000,
        verbose=10,
        num_threads="auto",
    )
    runtime_sec = time.time() - t0

    final_loss = float(fit["loss_history"][-1]) if fit["n_iter"] > 0 else float("nan")
    recon_error = float(np.linalg.norm(fit["X_hat"] - dat["X_true"], ord="fro"))
    print(f"\nFinal loss             : {final_loss:.4f}")
    print(f"Total iterations       : {fit['n_iter']}")
    print(f"X reconstruction error : {recon_error:.4f}")
    print(f"Runtime (sec)          : {runtime_sec:.4f}")

    plt.figure(figsize=(7, 4))
    plt.plot(fit["loss_history"], color="tomato", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Negative log-likelihood")
    plt.title(
        f"cBMM convergence (N={dat['N']}, J={dat['J']}, r={dat['r']})"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve_cBMM.pdf")
    plt.close()
    print("Loss curve saved to loss_curve_cBMM.pdf")


if __name__ == "__main__":
    main()
