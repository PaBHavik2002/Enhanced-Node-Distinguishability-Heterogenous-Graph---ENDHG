class ENDHG(nn.Module):
    def __init__(
        self,
        d_tx: int,
        attr_input_dims: Dict[str, int],
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
        use_layer_norm: bool = True,
        numlayers = 1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm

        # Eq. (18): project each node type into a common space
        self.tx_proj = nn.Linear(d_tx, hidden_dim)
        self.attr_proj = nn.ModuleDict({a: nn.Linear(attr_input_dims[a], hidden_dim) for a in attr_input_dims})

        if use_layer_norm:
            self.ln_tx = nn.LayerNorm(hidden_dim)
            self.ln_attr = nn.ModuleDict({a: nn.LayerNorm(hidden_dim) for a in attr_input_dims})

        # Number of Layers
        self.numLayer = numlayers

        # Eq. (16): meta-path conv weight W^(l)
        self.W_meta = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(numlayers)
        ])

        # Eq. (22): hetero conv weight W_He
        self.W_he = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(numlayers)
        ])

        # Eqs. (20)-(21): semantic attention params
        self.W_attn = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(numlayers)
        ])

        self.a_attn = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, 1) * 0.01)
            for _ in range(numlayers)
        ])

        # Eq. (24): final projection into task space
        self.W_fin = nn.Linear(hidden_dim, num_classes)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def _compute_metapath_attention(self, M: torch.Tensor, W_attn_layer, a_attn_layer) -> torch.Tensor:
        """
        Eqs. (20)-(21):
          w_Mi = (1/|V_T|) Σ_{v in V_T} a^T σ(W h_v^(Mi) + b)
          γ = softmax(w)
        Here |V_T| is the number of target nodes (N).
        """
        N, P, d = M.shape
        w_scores = []
        for i in range(P):
            h = M[:, i, :]                      # (N, d)
            s = torch.tanh(W_attn_layer(h))      # σ(.)
            scores = (s @ a_attn_layer).squeeze(-1)  # (N,)
            w_scores.append(scores.mean())      # average over target nodes
        w = torch.stack(w_scores, dim=0)        # (P,)
        return F.softmax(w, dim=0)              # (P,)
    

    """FORWARD PASS"""
    def forward(
        self,
        X_tx: torch.Tensor,
        X_attrs: Dict[str, torch.Tensor],
        meta_adjs: List[torch.Tensor],   # CHANGED: expects Ã_i (each N×N sparse)
        A_He: torch.Tensor,                    # (N x total_attr) sparse
        info: HeteroBuildInfo,
        return_gamma: bool = False,
        use_meta: bool = False,
        use_hetero: bool = True
    ):
        # --------------------
        # Eq. (18) projections
        # --------------------
        H_tx = self.tx_proj(X_tx)
        if self.use_layer_norm:
            H_tx = self.ln_tx(H_tx)
        H_tx = self._act(H_tx)                 # H_T

        total_attr = A_He.size(1)
        H_attr_all = torch.zeros(
            total_attr,
            self.hidden_dim,
            device=H_tx.device,
            dtype=H_tx.dtype
        )

        for a in info.attr_types:
          off = info.attr_offsets[a]
          n = info.attr_sizes[a]

          Ha = self.attr_proj[a](X_attrs[a])
          if self.use_layer_norm:
              Ha = self.ln_attr[a](Ha)
          Ha = self._act(Ha)

          H_attr_all[off : off + n] = Ha

        
        # For each meta-path:
        #   Eq. (17): Ã_i  (preprocessed outside, or optional inside)
        #   Eq. (16): H(M_i) = σ( Ã_i · H_tx · W_meta )
        
        N = H_tx.size(0)
        for layer in range(self.numLayer):

          if use_meta and len(meta_adjs) > 0:

              msgs = []

              for i, A_tilde in enumerate(meta_adjs):

                  if A_tilde.size(0) != N or A_tilde.size(1) != N:
                    raise ValueError(
                        f"meta_adjs[{i}] must be (N×N) with N={N}, "
                        f"got {tuple(A_tilde.size())}. "
                        f"Ensure this is a T–X–T meta adjacency, not bipartite."
                    )

                  # --- SAFETY CHECK 2: graph must be fixed ---
                  assert not A_tilde.requires_grad, (
                      "meta adjacency must be non-trainable. "
                      "Preprocess it once outside the model."
                  )

                  out = torch.sparse.mm(A_tilde.coalesce(), H_tx)
                  out = self.W_meta[layer](out) # Use layer specific weight
                  out = F.relu(out)
                  msgs.append(out)

              M = torch.stack(msgs, dim=1)
              gamma = self._compute_metapath_attention(
                  M,
                  self.W_attn[layer], # Use layer specific weight
                  self.a_attn[layer]  # Use layer specific param
              )
              H_M = (M * gamma.view(1, -1, 1)).sum(dim=1)

          else:

              H_M = torch.zeros_like(H_tx)
              gamma = None

        
        # Eq. (22) hetero aggregation
        #   H^He = Â_He · H_He · W_He
        
        if A_He.size(0) != N:
            raise ValueError(f"A_He first dim must be N={N}. Got {tuple(A_He.size())}.")

        # A_He is supposed to be row-normalized outside the model
        A_He_hat = A_He.coalesce()

        if self.training:
          row_deg_all = torch.sparse.sum(A_He_hat, dim=1).to_dense()  # (N,)

          # Sample rows from the dense vector (safe)
          rows = torch.randint(
              0, row_deg_all.size(0),
              (2048,),
              device=row_deg_all.device
          )

          row_deg = row_deg_all[rows]
          nz = row_deg > 0

          if nz.any():
              assert torch.allclose(
                  row_deg[nz],
                  torch.ones_like(row_deg[nz]),
                  atol=1e-3
              ), "A_He must be row-normalized"


        if A_He.size(1) != H_attr_all.size(0):
          raise RuntimeError(
              f"Shape mismatch in Eq.(22): "
              f"A_He is (N×{A_He.size(1)}), "
              f"but H_attr_all is ({H_attr_all.size(0)}×d). "
              f"This means attr_offsets / attr_sizes do not match A_He construction."
          )

        if use_hetero:
            H_He = torch.sparse.mm(A_He_hat, H_attr_all)
            # Use the last layer's weight since this block is outside the loop
            H_He = self.W_he[self.numLayer - 1](H_He)
            H_He = F.relu(H_He)
        else:
            H_He = torch.zeros_like(H_tx)

        
        # Eq. (23) sum fusion
        
        Z = torch.zeros_like(H_tx)

        if use_meta:
            Z = Z + H_M
        if use_hetero:
            Z = Z + H_He

        Z = F.dropout(Z, p=self.dropout, training=self.training)

        
        # Eq. (24) final projection
        
        logits = self.W_fin(Z)

        return (logits, Z, gamma) if return_gamma else (logits, Z, None)
