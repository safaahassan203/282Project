import numpy as np
import torch


# Step 1: Configuration + setup


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


# dataset sizes
N_train = 20000
N_val = 5000
N_test = 5000

# noise levels for the two mechanisms
noise_AB = 0.05
noise_BC = 0.05

# set the seed at import time
set_seed(0)


# Step 2: Mechanism 1 (A → B)
def mechanism_AB(A, noise_std=noise_AB):
    """
    A: numpy array of shape (N, 4)
    returns B: numpy array of shape (N, 3)
    """
    A1 = A[:, 0]
    A2 = A[:, 1]
    A3 = A[:, 2]
    A4 = A[:, 3]

    eps = np.random.normal(0, noise_std, size=(A.shape[0], 3))

    B1 = np.sin(A1) + 0.5 * (A2**2) + eps[:, 0]
    B2 = A3 * (1.0 / (1.0 + np.exp(-A4))) + eps[:, 1]  # sigmoid(A4)
    B3 = np.exp(-np.abs(A2)) + 0.1 * A1 + eps[:, 2]

    B = np.column_stack([B1, B2, B3])
    return B


# Step 3: Mechanism 1 under intervention (A → B′)
def mechanism_AB_intervened(A, noise_std=noise_AB):
    """
    A: numpy array of shape (N, 4)
    returns B': numpy array of shape (N, 3)
    This uses modified functional forms for the intervention.
    """
    A1 = A[:, 0]
    A2 = A[:, 1]
    A3 = A[:, 2]
    A4 = A[:, 3]

    eps = np.random.normal(0, noise_std, size=(A.shape[0], 3))

    # Example intervention:
    B1p = np.cos(A1) + 0.5 * (A2**3) + eps[:, 0]
    B2p = -A3 * (1.0 / (1.0 + np.exp(-A4))) + eps[:, 1]
    B3p = np.exp(-np.abs(A2)) - 0.2 * A1 + eps[:, 2]

    Bp = np.column_stack([B1p, B2p, B3p])
    return Bp


# Step 4: Mechanism 2 (B → C)


# fixed weights for Mechanism 2
w1 = np.random.randn(3)
w2 = np.random.randn(3)


def mechanism_BC(B, noise_std=noise_BC):
    """
    B: numpy array of shape (N, 3)
    returns C: numpy array of shape (N, 3)
    """
    eps = np.random.normal(0, noise_std, size=(B.shape[0], 3))

    # dot products
    z1 = B @ w1
    z2 = B @ w2

    C1 = np.tanh(z1) + np.maximum(z2, 0) + eps[:, 0]  # ReLU(z2)
    C2 = (B[:, 0] * B[:, 1]) + np.cos(B[:, 2]) + eps[:, 1]
    C3 = 1.0 / (1.0 + np.exp(-(B[:, 0] + 0.3 * B[:, 2]))) + eps[:, 2]  # sigmoid

    C = np.column_stack([C1, C2, C3])
    return C

# DATA GENERATION PIPELINE 

# Step 1: Sample A from N(0, I)
def sample_A(n_samples):
    """
    Draws A ~ N(0, I), shape (n, 4)
    """
    return np.random.randn(n_samples, 4)


# Step 2: Generate observational dataset (A → B → C)
def generate_dataset(n_samples):
    """
    Builds the full observational dataset:
       A → B → C
    Returns dict with numpy arrays.
    """
    A = sample_A(n_samples)
    B = mechanism_AB(A)
    C = mechanism_BC(B)

    return {
        "A": A,
        "B": B,
        "C": C,
    }


# Step 3: Generate interventional dataset (A → B' → C')
def generate_interventional_dataset(n_samples):
    """
    Builds the interventional dataset:
        A → B' → C'
    using the modified A→B' mechanism.
    """
    A = sample_A(n_samples)
    Bp = mechanism_AB_intervened(A)
    Cp = mechanism_BC(Bp)

    return {
        "A": A,
        "B": Bp,
        "C": Cp,
    }


# Step 4: Split into train/val/test
def create_splits(data, n_train, n_val, n_test):
    """
    data: dict {"A":..., "B":..., "C":...}
    Returns a dict with split datasets.
    """
    N = data["A"].shape[0]
    assert N == (n_train + n_val + n_test)

    idx = np.arange(N)
    np.random.shuffle(idx)

    def take(indices):
        return {
            "A": data["A"][indices],
            "B": data["B"][indices],
            "C": data["C"][indices],
        }

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    return {
        "train": take(train_idx),
        "val": take(val_idx),
        "test": take(test_idx),
    }


# Step 5: Package everything (main entry point)
def package_all_datasets(
    N_train=N_train,
    N_val=N_val,
    N_test=N_test,
    N_interv=5000,
    N_few_shot=1000,
    include_few_shot=True,
):
    """
    Generates:
      - observational train/val/test
      - interventional test set
      - optional few-shot adaptation set

    Returns a dict with all components.
    """

    # 1. Full observational dataset
    total_obs = N_train + N_val + N_test
    full_obs_data = generate_dataset(total_obs)

    # 2. Splits
    splits = create_splits(full_obs_data, N_train, N_val, N_test)

    # 3. Interventional test
    interv_test = generate_interventional_dataset(N_interv)

    # 4. Few-shot adaptation data (optional)
    if include_few_shot:
        few_shot = generate_interventional_dataset(N_few_shot)
    else:
        few_shot = None

    return {
        "observational": splits,
        "interventional_test": interv_test,
        "few_shot": few_shot,
    }
