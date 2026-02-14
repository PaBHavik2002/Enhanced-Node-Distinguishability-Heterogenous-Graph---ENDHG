""""MODEL UTILITY FUNCTIONS"""
def coo_sparse(indices: torch.Tensor, values: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return torch.sparse_coo_tensor(indices, values, size=size).coalesce()

def row_norm_adj(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Nor(A) = D^{-1} A  (row-normalization)
    """
    A = A.coalesce()
    idx = A.indices()
    val = A.values()
    rows = idx[0]
    row_sum = torch.zeros(A.size(0), device=val.device, dtype=val.dtype)
    row_sum.scatter_add_(0, rows, val)
    inv = 1.0 / (row_sum + eps)
    val = val * inv[rows]
    return coo_sparse(idx, val, A.size())

def delta_set_diag_to_one(A: torch.Tensor) -> torch.Tensor:
    """
    δ(·): after normalization, reset diagonal self-loop weight to 1.
    Paper intent: prevent self-feature dilution in deep aggregation.
    """
    A = A.coalesce()
    n = A.size(0)
    if A.size(0) != A.size(1):
        raise ValueError("delta_set_diag_to_one expects square adjacency (N×N).")

    idx = A.indices()
    val = A.values()

    keep = idx[0] != idx[1]
    idx2 = idx[:, keep]
    val2 = val[keep]

    diag = torch.arange(n, device=idx.device, dtype=idx.dtype)
    diag_idx = torch.stack([diag, diag], dim=0)
    diag_val = torch.ones(n, device=val.device, dtype=val.dtype)

    idx_new = torch.cat([idx2, diag_idx], dim=1)
    val_new = torch.cat([val2, diag_val], dim=0)
    return coo_sparse(idx_new, val_new, A.size())

def preprocess_A_tilde(A: torch.Tensor) -> torch.Tensor:
    """
    Eq. (17): Ã = δ(Nor(A))
    Here Nor(A) = row_norm_adj (D^{-1}A), then δ sets diag to 1.
    """
    A_norm = row_norm_adj(A)
    A_tilde = delta_set_diag_to_one(A_norm)
    return A_tilde

def preprocess_meta_adjs(meta_adjs: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    CHANGED: Preprocess meta adjacencies ONCE (paper says Eq. 17 can be preprocessed).
    Input: list of M_i (each N×N sparse)
    Output: list of Ã_i (each N×N sparse), cached for training.
    """
    out = []
    for i, A in enumerate(meta_adjs):
        if A.size(0) != A.size(1):
            raise ValueError(f"meta_adjs[{i}] must be square (N×N). Got {tuple(A.size())}. "
                             f"Did you pass bipartite (T→X) instead of meta adjacency (T–X–T)?")
        out.append(preprocess_A_tilde(A))
    return out


# Same info dataclass
@dataclass
class HeteroBuildInfo:
    attr_types: List[str]
    attr_offsets: Dict[str, int]
    attr_sizes: Dict[str, int]
    edge_types: Dict[str, Tuple[str, str, str]]
