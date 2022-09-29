import numpy as np
from scipy.optimize import fmin_cg   
           
class PrejudiceRemoverAlgorithm:
    """Two class LogisticRegression with Prejudice Remover"""

    def __init__(self, 
                 estimator,
                 objetive_fn,
                 logger):
        """
        Parameters
        ----------
        estimator : estimator object
            Prejudice Remover Model
            
        objetive_fn : object
            Objective class with loss and grad_loss function
            
        logger : object
            Support for print information
        """
        
        self.estimator = estimator
        self.logger = logger
        self.objetive_fn = objetive_fn
        
    def fit(self, 
            X : np.ndarray, 
            y : np.ndarray, 
            sensitive_features: np.ndarray):
        """
        Description
        -----------
        Optimize the model paramters to reduce the loss function
        
        Parameters
        ----------
        X : matrix-like
            Input matrix

        y_true : numpy array
            Target vector

        sensitive_features : numpy array
            Matrix where each columns is a sensitive feature e.g. [col_1=group_a, col_2=group_b]
        """
        
        groups_num = self.gutil.create_groups(sensitive_features, convert_numeric=True)
        self.estimator.init_params(X, y, groups_num)
        self.logger.set_log_fn(loss = lambda coef:self.obj.loss(coef, X, y, groups_num), type=float)
                
        self.coef = fmin_cg(
            self.objetive_fn.loss, 
            self.estimator.coef , 
            fprime=self.objetive_fn.grad_loss, 
            args=(X, y, groups_num), 
            maxiter=self.maxiter,
            disp=False,
            callback=self.logger.callback
        )
        self.estimator.set_params(self.coef)
        
        self.f_loss_ = self.objetive_fn.loss(self.coef, X, y, groups_num)
        self.logger.info(f"Best Loss : {self.f_loss_:.4f}")
        return self
         
    def _preprocess_groups(self, sensitive_features):
        group_ids = self.gutil.merge_columns(sensitive_features)
        group_num = group_ids.apply(lambda x: self.gutil.group2num[x])
        return group_num
    
    def predict(self, X, sensitive_features):
        """
        predict classes
        
        Parameters
        ----------
        X : array, shape=(n_samples, n_features)
            feature vectors of samples
            
        sensitive_features : numpy array
            Matrix where each columns is a sensitive feature e.g. [col_1=group_a, col_2=group_b]
        Returns
        -------
        y : array, shape=(n_samples), dtype=int
            array of predicted class
        """
        group_num = self._preprocess_groups(sensitive_features)
        return self.estimator.predict(X, group_num)

    def predict_proba(self, X, sensitive_features):
        """
        Predict probabilities
        
        Parameters
        ----------
        X : array, shape=(n_samples, n_features)
            feature vectors of samples
        
        sensitive_features : numpy array
            Matrix where each columns is a sensitive feature e.g. [col_1=group_a, col_2=group_b]
            
        Returns
        -------
        y_proba : array, shape=(n_samples, n_classes), dtype=float
            array of predicted class
        """
        group_num = self._preprocess_groups(sensitive_features)
        return self.estimator.predict_proba(X, group_num)