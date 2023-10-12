import sys

import utils.metrics as metrics
from logginig_example import generate_experiment_name
import numpy as np
from configs.linear_regression_cfg import cfg
from utils.enums import TrainType
from logs.Logger import Logger
import cloudpickle

class LinearRegression():

    def __init__(self, base_functions: list, learning_rate: float=0.1, reg_coefficient: float=0):
        self.weights = np.random.randn(len(base_functions))
        self.base_functions = base_functions
        self.learning_rate = learning_rate
        self.reg_coefficient = reg_coefficient
        self.logger = None

    # Methods related to the Normal Equation

    def _pseudoinverse_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Compute the pseudoinverse of a matrix using SVD.

        The pseudoinverse (Φ^+) of the design matrix Φ can be computed using the formula:

        Φ^+ = V * Σ^+ * U^T

        Where:
        - U, Σ, and V are the matrices resulting from the SVD of Φ.

        The Σ^+ is computed as:

        Σ'_{i,j} =
        | 1/Σ_{i,j}, if Σ_{i,j} > ε * max(N, M+1) * max(Σ)
        | 0, otherwise

        and then:
        Σ^+ = Σ'^T

        where:
        - ε is the machine epsilon, which can be obtained in Python using:
            ε = sys.float_info.epsilon
        - N is the number of rows in the design matrix.
        - M is the number of base functions (without φ_0(x_i)=1).

        For regularisation

        Σ'_{i,j} =
        | Σ_{i,j}/(Σ_{i,j}ˆ2 + λ) , if Σ_{i,j} > ε * max(N, M+1) * max(Σ)
        | 0, otherwise

        Note that Σ'_[0,0] = 1/Σ_{i,j}

        TODO: Add regularisation
        """
        u, z, v = np.linalg.svd(matrix)
        n = matrix.shape[0]
        m = len(self.base_functions)
        b = z > sys.float_info.epsilon * max(n, m) * max(z)
        ps_z = np.power(z,-2)*b*z+self.reg_coefficient
        ps_z[0]=b[0]*(z[0]**(-1))
        vz = v.T[:,:min(v.shape[0],len(ps_z))]*ps_z
        pseudo = np.dot(vz,u.T[:min(vz.shape[1],(u.T).shape[0]),:] )
        return pseudo
        pass

    def _calculate_weights(self, pseudoinverse_plan_matrix: np.ndarray, targets: np.ndarray) -> None:
        """Calculate the optimal weights using the normal equation.

            The weights (w) can be computed using the formula:

            w = Φ^+ * t

            Where:
            - Φ^+ is the pseudoinverse of the design matrix and can be defined as:
                Φ^+ = (Φ^T * Φ)^(-1) * Φ^T

            - t is the target vector.
        """
        self.weights = pseudoinverse_plan_matrix@targets
        pass

    # General methods
    def _plan_matrix(self, inputs: np.ndarray) -> np.ndarray:
        """Construct the design matrix (Φ) using base functions.

            The structure of the matrix Φ is as follows:

            Φ = [ [ φ_0(x_1), φ_1(x_1), ..., φ_M(x_1) ],
                  [ φ_0(x_2), φ_1(x_2), ..., φ_M(x_2) ],
                  ...
                  [ φ_0(x_N), φ_1(x_N), ..., φ_M(x_N) ] ]

            where:
            - x_i denotes the i-th input vector.
            - φ_j(x_i) represents the j-th base function applied to the i-th input vector.
            - M is the total number of base functions (without φ_0(x_i)=1).
            - N is the total number of input vectors.


        """
        n= len(inputs)
        m = len(self.base_functions)
        i = 0
        res = np.ndarray(shape=(n,m))
        while i<m:
            i_=i
            res[:,i_]=np.array(list(map(self.base_functions[i_],inputs)))
            i=i+1
        return res
        pass

    def calculate_model_prediction(self, plan_matrix: np.ndarray) -> np.ndarray:
        """Calculate the predictions of the model.

            The prediction (y_pred) can be computed using the formula:

            y_pred = Φ * w^T

            Where:
            - Φ is the design matrix.
            - w^T is the transpose of the weight vector.

            To compute multiplication in Python using numpy, you can use:
            - `numpy.dot(a, b)`
            OR
            - `a @ b`

        """
        y_pred = np.dot(plan_matrix,self.weights)
        return y_pred
        pass

    # Methods related to Gradient Descent
    def _calculate_gradient(self, plan_matrix: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the cost function with respect to the weights.

            The gradient of the error with respect to the weights (∆w E) can be computed using the formula:

            ∆w E = (2/N) * Φ^T * (Φ * w - t)

            Where:
            - Φ is the design matrix.
            - w is the weight vector.
            - t is the vector of target values.
            - N is the number of data points.

            This formula represents the partial derivative of the mean squared error with respect to the weights.

            For regularisation
            ∆w E = (2/N) * Φ^T * (Φ * w - t)  + λ * w

            """
        k = np.ones(len(self.weights))
        k[0] = 0
        return (plan_matrix.transpose())@(plan_matrix@self.weights - targets)*2/len(targets) + k*self.reg_coefficient*self.weights
        pass

    def calculate_cost_function(self, plan_matrix, targets):
        """Calculate the cost function value for the current weights.

        The cost function E(w) represents the mean squared error and is given by:

        E(w) = (1/N) * ∑(t - Φ * w^T)^2

        Where:
        - Φ is the design matrix.
        - w is the weight vector.
        - t is the vector of target values.
        - N is the number of data points.

        For regularisation
        E(w) = (1/N) * ∑(t - Φ * w^T)^2 + λ * w^T * w



        """
        a = (targets - plan_matrix@(self.weights.transpose()))
        k = np.ones(len(self.weights))
        k[0]=0
        return (1/len(targets)) * np.dot(a,a.T) + self.reg_coefficient * (sum((k*self.weights)**2))
        pass
    def create_logger(self,valid_inputs: np.ndarray, valid_targets: np.ndarray) -> Logger:
        self.experiment_name, self.base_function_str = generate_experiment_name(self.base_functions,
                                                                                self.reg_coefficient,
                                                                                self.learning_rate)
        logger = Logger(cfg.env_path, cfg.project_name, self.experiment_name)
        logger.set_valid(valid_inputs,valid_targets)
        self.log(logger)
        return logger

    def train(self, inputs: np.ndarray, targets: np.ndarray,logger : Logger =None) -> None:
        """Train the model using either the normal equation or gradient descent based on the configuration.

        """
        if(logger!=None):
            valid_plan_matrix = self._plan_matrix(logger.inputs)
        plan_matrix = self._plan_matrix(inputs)
        if cfg.train_type.value == TrainType.normal_equation.value:
            pseudoinverse_plan_matrix = self._pseudoinverse_matrix(plan_matrix)
            # train process
            self._calculate_weights(pseudoinverse_plan_matrix, targets)
        else:
            """
            At each iteration of gradient descent, the weights are updated using the formula:
        
            w_{k+1} = w_k - γ * ∇_w E(w_k)
        
            Where:
            - w_k is the current weight vector at iteration k.
            - γ is the learning rate, determining the step size in the direction of the negative gradient.
            - ∇_w E(w_k) is the gradient of the cost function E with respect to the weights w at iteration k.
        
            This iterative process aims to find the weights that minimize the cost function E(w).
        """
            for e in np.arange(cfg.epoch):
                gradient = self._calculate_gradient(plan_matrix,targets)
                # update weights w_{k+1} = w_k - γ * ∇_w E(w_k)

                self.weights = self.weights - self.learning_rate*gradient
                if (e % 100 == 0) and logger!=None:
                    logger.save_param('metrics','train_loss',self.calculate_cost_function(plan_matrix,targets))
                    logger.save_param('metrics','train_mse',metrics.MSE(self(inputs),targets))
                    logger.save_param('metrics','valid_loss',self.calculate_cost_function(valid_plan_matrix,logger.targets))
                    logger.save_param('metrics','valid_mse',metrics.MSE(self(logger.inputs),logger.targets))
                    pass
            if(logger!= None):
                logger.log_final_val_mse(metrics.MSE(self(logger.inputs),logger.targets))


    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """return prediction of the model"""
        plan_matrix = self._plan_matrix(inputs)
        predictions = self.calculate_model_prediction(plan_matrix)

        return predictions

    def log(self,logger:Logger):
        logger.log_hyperparameters(params={
            'base_function': self.base_function_str,
            'regularisation_coefficient': self.reg_coefficient,
            'learning_rate': self.learning_rate
        })
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return cloudpickle.load(f)