import numpy as np
import threading
from joblib import Parallel

from sklearn.utils.sparsefuncs import count_nonzero
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_consistent_length
from sklearn.metrics._classification import _check_targets
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.fixes import delayed

class WeightForestClassifier(RandomForestClassifier):
    """
    Parameters
    ----------
    Basically, it has the same parameters as RandomForestClassifier

    init_estimators_weight : float, default=100.0
        Default setting for estimators_weight_.

    reward : float, default=0.01
        Reward value to be used for additional fitting.

    punishment : float, default=1.0
        Punishment value to be used for additional fitting.

    Attributes
    ----------
    estimators_weight_ : list of float
        Pair with estimators_ to perform a prediction
    """
    
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        init_estimators_weight=100.0,
        reward=0.01,
        punishment=1.0
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )
        self.init_estimators_weight = init_estimators_weight
        self.reward = reward
        self.punishment = punishment
        self.estimators_weight_ = np.zeros(n_estimators) + init_estimators_weight

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.estimators_weight_ = np.zeros(self.n_estimators) + self.init_estimators_weight
        return super().fit(X, y, sample_weight)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        The predicted class probability of the input sample is calculated as
        values weighted to the predicted class probability of the tree in
        the forest.
        A tree with "weight <= 0" is not calculated
        The class probability of a single tree is the fraction of samples of
        the same class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes), or a list of such arrays
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [
            np.zeros((X.shape[0], j), dtype=np.float64)
            for j in np.atleast_1d(self.n_classes_)
        ]
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_weight_prediction)(e.predict_proba, X, w, all_proba, lock)
            for e, w in zip(self.estimators_, self.estimators_weight_)
        )

        for proba in all_proba:
            proba /= len(self.estimators_)

        if len(all_proba) == 1:
            return all_proba[0]
        else:
            return all_proba
    
    def weight_fit(self, X, y, reward=None, punishment=None):
        """
        Training weight.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression)

        reward : float, default=self.reward
            Incremental value to be used for weight training.

        punishment : float, default=self.punishment
            Reduction value to be used for weight training

        Returns
        -------
        weight : 
            Fitted weight.
        """
        check_is_fitted(self)
        if reward == None:
            reward = self.reward
        if punishment == None:
            punishment = self.punishment
        X = self._validate_X_predict(X)
        if y.ndim != 1 or (self.n_outputs_ != 1 and y.shape[1] != self.n_outputs_):
            raise ValueError(
                "There is a problem with the format of the target values."
            )
        elif X.shape[0] != y.shape[0]:
            raise ValueError(
                "The number of input samples and target values is different."
            )

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        all_proba = [
            np.zeros((X.shape[0], j), dtype=np.float64)
            for j in np.atleast_1d(self.n_classes_)
        ]
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_fit_weight)(e.predict, X, y, self.estimators_weight_, i, self.reward, self.punishment, lock)
            for e, i in zip(self.estimators_, range(self.n_estimators))
        )
        
        return self.estimators_weight_

def _accumulate_weight_prediction(predict, X, weight, out, lock):
    """
    This is a utility function for joblib's Parallel.
    """
    if weight > 0.0:
        prediction = predict(X, check_input=False) * weight
        with lock:
            if len(out) == 1:
                out[0] += prediction
            else:
                for i in range(len(out)):
                    out[i] += prediction[i]

def _fit_weight(predict, X, y_true, weight, idx, reward, punishment, lock):
    """
    This is a utility function for joblib's Parallel.
    """
    y_pred = predict(X, check_input=False)
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    check_consistent_length(y_true, y_pred)
    correct = np.count_nonzero(y_true == y_pred)
    update = correct * reward - (y_true.shape[0] - correct) * punishment
    with lock:
        weight[idx] += update