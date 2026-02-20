# Additional Unilities
def predict_with_threshold(probs, threshold):
    return (probs >= threshold).long()

# PREPROCESS (ONCE)
def preprocess_graph_once(
    meta_adjs: List[torch.Tensor],
    A_He: torch.Tensor,
):
    """
    Paper alignment:
      - Meta-path adjacencies (N×N): Eq. (17)  Ã = δ(Nor(A))
      - Hetero adjacency (N×total_attr): row-normalization only (no δ possible)
    """
    meta_adjs_tilde = [preprocess_A_tilde(A) for A in meta_adjs]   # Eq. (17)
    A_He_hat = row_norm_adj(A_He.coalesce())                       # row-norm only
    return meta_adjs_tilde, A_He_hat



# TRAIN ONE EPOCH (UNCHANGED)


def train_one_epoch(
    model, optimizer,
    X_tx, X_attrs,
    meta_adjs, A_He,
    info, y, train_mask, class_weight=None, use_meta = True, use_hetero = True # --------------> Change (v2)
):
    model.train()
    optimizer.zero_grad()

    logits, _, _ = model(
      X_tx=X_tx, X_attrs=X_attrs,
      meta_adjs=meta_adjs, A_He=A_He, info=info,
      return_gamma=False,
      use_meta=use_meta,
      use_hetero=use_hetero,
    )

    loss = F.cross_entropy(
    logits[train_mask],
    y[train_mask],
    weight=class_weight
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return loss.item()

def predict_with_threshold(probs, threshold):
    return (probs >= threshold).long()



# METRICS
def compute_eval_metrics(
    logits: torch.Tensor,
    Z: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    gamma: torch.Tensor = None,
):
    
    # Mask
    
    logits_m = logits[mask]
    Z_m = Z[mask]
    y_m = y[mask]

    
    # Loss
    
    loss = F.cross_entropy(logits_m, y_m).item()

    
    # Probabilities (masked)
    
    probs = F.softmax(logits_m, dim=1)[:, 1]

    p_np = probs.detach().cpu().numpy()
    y_np = y_m.detach().cpu().numpy()

    
    # FINAL predictions
    
    FIXED_THRESHOLD = 0.3   # or 0.5 if you want stricter behavior
    pred_np = (p_np >= FIXED_THRESHOLD).astype(int)

    
    # Confusion matrix
    
    tn, fp, fn, tp = confusion_matrix(
        y_np, pred_np, labels=[0, 1]
    ).ravel()
    # Metrics
    
    metrics = {
        "loss": loss,
        "accuracy": accuracy_score(y_np, pred_np),
        "precision": precision_score(y_np, pred_np, zero_division=0),
        "recall": recall_score(y_np, pred_np, zero_division=0),
        "f1": f1_score(y_np, pred_np, zero_division=0),
        "roc_auc": roc_auc_score(y_np, p_np) if len(numpy.unique(y_np)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_np, p_np),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "fraud_score_mean": float(p_np[y_np == 1].mean()) if (y_np == 1).any() else 0.0,
        "normal_score_mean": float(p_np[y_np == 0].mean()) if (y_np == 0).any() else 0.0,
        "embedding_norm_mean": float(Z_m.norm(dim=1).mean()),
        "embedding_norm_std": float(Z_m.norm(dim=1).std()),
        "gamma": gamma.detach().cpu().numpy() if gamma is not None else None
    }

    # Balanced accuracy
    metrics["balanced_accuracy"] = 0.5 * (
        metrics["recall"] + metrics["specificity"]
    )

    return metrics

"""Evaluate Utility For Training, Test and Validation"""

@torch.no_grad()
def evaluate_model(
    model,
    X_tx, X_attrs,
    meta_adjs, A_He,
    info, y, mask, use_meta = True, use_hetero = True
):
    model.eval()

    logits, Z, gamma = model(
        X_tx=X_tx,
        X_attrs=X_attrs,
        meta_adjs=meta_adjs,   # expects Ã_i
        A_He=A_He,             # expects Â_He
        info=info,
        return_gamma=True,
        use_meta = use_meta,
        use_hetero = use_hetero
    )

    return compute_eval_metrics(logits=logits, Z=Z, y=y, mask=mask, gamma=gamma)



# TRAIN MODEL
"""Training Epoch"""
def train_model(
    model,
    X_tx,
    X_attrs,
    meta_adjs,
    A_He,
    info,
    y,
    train_mask,
    val_mask,
    test_mask,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-3,
    device="cpu",
    patience=10,        # early stopping
    loss_delta=1e-4,
    auc_delta=1e-4,
    inputs_are_preprocessed=False,  # <--- IMPORTANT: prevents double-preprocessing
    use_meta=True,  # Ablation 1
    use_hetero=True # Ablation 2
):

     
    # Preprocess adjacencies ONCE
     
    if inputs_are_preprocessed:
        meta_adjs_tilde, A_He_hat = meta_adjs, A_He
    else:
        meta_adjs_tilde, A_He_hat = preprocess_graph_once(meta_adjs, A_He)

     
    # Move to device
     
    model.to(device)
    X_tx = X_tx.to(device)
    y = y.to(device)

    """Soft Weighting Established"""
    pos = (y[train_mask] == 1).sum().item()
    neg = (y[train_mask] == 0).sum().item()

    ratio = neg / max(pos, 1)
    class_weights = torch.tensor(
        [1.0, ratio ** 0.5],
        device=device,
        dtype=torch.float32
    )

    meta_adjs_tilde = [A.to(device) for A in meta_adjs_tilde]
    A_He_hat = A_He_hat.to(device)

    X_attrs = {k: v.to(device) for k, v in X_attrs.items()}
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

     
    # Early stopping state (MUST be local)
     
    best_val_loss = float("inf")
    best_pr_auc = -float("inf")
    best_state = None
    counter = 0

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model, optimizer, X_tx, X_attrs,
            meta_adjs_tilde, A_He_hat, info, y, train_mask, class_weight = class_weights
        )

        val_stats = evaluate_model(
            model, X_tx, X_attrs,
            meta_adjs_tilde, A_He_hat, info, y, val_mask
        )

        # improvement checks
        loss_improved = val_stats["loss"] < best_val_loss - loss_delta
        pr_auc = val_stats["pr_auc"]
        auc_improved = (pr_auc == pr_auc) and (pr_auc > best_pr_auc + auc_delta)  # NaN-safe: NaN != NaN

        if loss_improved or auc_improved:
            best_val_loss = min(best_val_loss, val_stats["loss"])
            best_pr_auc = max(best_pr_auc, val_stats["pr_auc"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_stats['loss']:.4f} | "
                f"Val PR-AUC: {val_stats['pr_auc']:.4f} | "
                f"Patience: {counter}/{patience}"
            )

        if counter >= patience:
            print(
                f"Early stopping at epoch {epoch} | "
                f"Best Val Loss: {best_val_loss:.4f} | "
                f"Best PR-AUC: {best_pr_auc:.4f}"
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No best_state was saved. Check that val_mask has at least one node of each class "
            "and that your model forward returns valid logits."
        )

     
    # Load best model
     
    model.load_state_dict(best_state)
     
    # Final test evaluation
     
    test_stats = evaluate_model(
        model,
        X_tx,
        X_attrs,
        meta_adjs_tilde,   # FIX: use processed
        A_He_hat,          # FIX: use processed
        info,
        y,
        test_mask,
    )

    print(f"TEST | Loss: {test_stats['loss']:.4f} | Acc: {test_stats['accuracy']:.4f}")
    return model, test_stats, (meta_adjs_tilde, A_He_hat)
