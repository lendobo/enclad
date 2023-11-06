def optimize_graph(data, prior_matrix, lambda_val):
    """
    Optimizes the objective function using the entire data set and the estimated lambda.

    Parameters
    ----------
    data : array-like, shape (n, p)
        The data matrix.
    prior_matrix : array-like, shape (p, p)
        The prior matrix.
    lambda_val : float
        The regularization parameter for the edges.

    Returns
    -------
    opt_precision_mat : array-like, shape (p, p)
        The optimized precision matrix.
    """
    # Use GraphicalLasso to estimate the precision matrix
    model = GraphicalLasso(alpha=lambda_val, mode='cd', max_iter=100)
    try:
        model.fit(data)
        return model.precision_
    except Exception as e:
        print(f"Optimization did not succeed due to {str(e)}")
        return np.zeros((data.shape[1], data.shape[1]))



def evaluate_reconstruction(adj_matrix, opt_precision_mat, threshold=1e-5):
    """
    Evaluate the accuracy of the reconstructed adjacency matrix.

    Parameters
    ----------
    adj_matrix : array-like, shape (p, p)
        The original adjacency matrix.
    opt_precision_mat : array-like, shape (p, p)
        The optimized precision matrix.
    threshold : float, optional
        The threshold for considering an edge in the precision matrix. Default is 1e-5.

    Returns
    -------
    metrics : dict
        Dictionary containing precision, recall, f1_score, and jaccard_similarity.
    """
    # Convert the optimized precision matrix to binary form
    reconstructed_adj = (np.abs(opt_precision_mat) > threshold).astype(int)
    np.fill_diagonal(reconstructed_adj, 0)

    # True positives, false positives, etc.
    tp = np.sum((reconstructed_adj == 1) & (adj_matrix == 1))
    fp = np.sum((reconstructed_adj == 1) & (adj_matrix == 0))
    fn = np.sum((reconstructed_adj == 0) & (adj_matrix == 1))
    tn = np.sum((reconstructed_adj == 0) & (adj_matrix == 0))

    # Precision, recall, F1 score
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # Jaccard similarity
    jaccard_similarity = tp / (tp + fp + fn)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'jaccard_similarity': jaccard_similarity
    }

    return metrics
