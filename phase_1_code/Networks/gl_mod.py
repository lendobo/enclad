
################## Excerpt from sklearn.covariance.graph_lasso_ ##################

def _objective(mle, precision_, P):
    """Evaluation of the graphical-lasso objective function

    The objective function is made of a shifted scaled version of the
    normalized log-likelihood (i.e. its empirical mean over the samples) and a
    penalization term to promote sparsity.
    """
    p = precision_.shape[0]
    cost = -2.0 * log_likelihood(mle, precision_) + p * np.log(2 * np.pi)
    cost += (np.abs(precision_) * P).sum() - (np.abs(np.diag(precision_)) * np.diag(P)).sum()
    return cost

def _dual_gap(emp_cov, precision_, P):
    """Expression of the dual gap convergence criterion

    The specific definition is given in Duchi "Projected Subgradient Methods
    for Learning Sparse Gaussians".
    """
    gap = np.sum(emp_cov * precision_)
    gap -= precision_.shape[0]
    gap += (np.abs(precision_) * P).sum() - (np.abs(np.diag(precision_)) * np.diag(P)).sum()
    return gap


def _graphical_lasso(
    emp_cov,
    P,  # replacing alpha with matrix P for regularization
    *,
    cov_init=None,
    mode="cd",
    tol=1e-4,
    enet_tol=1e-4,
    max_iter=100,
    verbose=False,
    eps=np.finfo(np.float64).eps,
):
    _, n_features = emp_cov.shape
    if np.all(P == 0):
        # Early return without regularization
        precision_ = linalg.inv(emp_cov)
        cost = -2.0 * log_likelihood(emp_cov, precision_)
        cost += n_features * np.log(2 * np.pi)
        d_gap = np.sum(emp_cov * precision_) - n_features
        return emp_cov, precision_, (cost, d_gap), 0

    if cov_init is None:
        covariance_ = emp_cov.copy()
    else:
        covariance_ = cov_init.copy()

    covariance_ *= 0.95
    diagonal = emp_cov.flat[:: n_features + 1]
    covariance_.flat[:: n_features + 1] = diagonal
    precision_ = linalg.pinvh(covariance_)

    indices = np.arange(n_features)
    i = 0
    costs = list()

    if mode == "cd":
        errors = dict(over="raise", invalid="ignore")
    else:
        errors = dict(invalid="raise")

    try:
        d_gap = np.inf
        sub_covariance = np.copy(covariance_[1:, 1:], order="C")
        for i in range(max_iter):
            for idx in range(n_features):
                if idx > 0:
                    di = idx - 1
                    sub_covariance[di] = covariance_[di][indices != idx]
                    sub_covariance[:, di] = covariance_[:, di][indices != idx]
                else:
                    sub_covariance[:] = covariance_[1:, 1:]

                row = emp_cov[idx, indices != idx]
                with np.errstate(**errors):
                    if mode == "cd":
                        coefs = -(
                            precision_[indices != idx, idx]
                            / (precision_[idx, idx] + 1000 * eps)
                        )
                        # Modify the regularization strength using P matrix
                        rho_values = P[idx, indices != idx]
                        coefs, _, _, _ = cd_fast.enet_coordinate_descent_gram(
                            coefs,
                            rho_values,  # Passing vector of regularization strengths
                            0,
                            sub_covariance,
                            row,
                            row,
                            max_iter,
                            enet_tol,
                            check_random_state(None),
                            False,
                        )
                    else:  # mode == "lars"
                        rho_values = P[idx, indices != idx]
                        _, _, coefs = lars_path_gram(
                            Xy=row,
                            Gram=sub_covariance,
                            n_samples=row.size,
                            alpha_min=np.max(rho_values / (n_features - 1)),
                            copy_Gram=True,
                            eps=eps,
                            method="lars",
                            return_path=False,
                        )
                precision_[idx, idx] = 1.0 / (
                    covariance_[idx, idx]
                    - np.dot(covariance_[indices != idx, idx], coefs)
                )
                precision_[indices != idx, idx] = -precision_[idx, idx] * coefs
                precision_[idx, indices != idx] = -precision_[idx, idx] * coefs
                coefs = np.dot(sub_covariance, coefs)
                covariance_[idx, indices != idx] = coefs
                covariance_[indices != idx, idx] = coefs

            if not np.isfinite(precision_.sum()):
                raise FloatingPointError(
                    "The system is too ill-conditioned for this solver"
                )
            d_gap = _dual_gap(emp_cov, precision_, P)  # Using mean of P for d_gap
            cost = _objective(emp_cov, precision_, P)  # Using mean of P for cost
            if verbose:
                print(
                    "[graphical_lasso] Iteration % 3i, cost % 3.2e, dual gap %.3e"
                    % (i, cost, d_gap)
                )
            costs.append((cost, d_gap))
            if np.abs(d_gap) < tol:
                break
            if not np.isfinite(cost) and i > 0:
                raise FloatingPointError(
                    "Non SPD result: the system is too ill-conditioned for this solver"
                )
        else:
            warnings.warn(
                "graphical_lasso: did not converge after %i iteration: dual gap: %.3e"
                % (max_iter, d_gap),
                ConvergenceWarning,
            )
    except FloatingPointError as e:
        e.args = (e.args[0] + ". The system is too ill-conditioned for this solver",)
        raise e

    return covariance_, precision_, costs, i + 1
