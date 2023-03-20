import numpy as np
import utils
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] in the range (-1, 1)

    Note: Bias-trick: Include biases in weight-matrix
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    ones = np.ones(X.shape[0])     # Initializes biases to ones, with size equal X 
    X = np.interp(X,[0,255],[-1,1])# Remaps X from [0,255] to [-1,1] ??Maybe move above stack??
    X = np.column_stack((X,ones))  # Bias-trick, adds a column with ones to the matrix 
    #X = np.column_stack((ones,X))
    #print("**X-shape: ** ",X.shape )
    
    return X

i = 0
def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, 1]
        outputs: outputs of model of shape: [batch size, 1]
    Returns:
        Cross entropy error (float)
    """
    # TODO implement this function (Task 2a)
    # @ Elementwise multiplication
    # targets/labels = y,  outputs = y_hat
    # Error/Loss/Cost function:

    #with open('_CEL_targets.txt', 'a') as f:
    #        print('targets.T:', targets.T, file=f)  # Python 3.x
    
    #with open('_CEL_outputs.txt', 'a') as f:
    #        print('outputs.T:', outputs.T, file=f)  # Python 3.x

    #Cn = -(targets.T@np.log(outputs)+(1-targets).T@np.log(1-outputs)) #Eq 3
     #Cn = -(targets.T@np.log2(outputs)+(1-targets).T@np.log2(1-outputs)) #Eq 3
    #Cn = -(targets.T.dot(np.log(outputs)) ) #+(1-targets).T@np.log(1-outputs)) #Eq 3
     #C = np.sum(Cn)


    cost = - (np.dot(targets.T, np.log(outputs)) + np.dot((1-targets).T,np.log(1-outputs)))

  
    #print("**Targes:**",targets )               #[[1],[0],...[1],[0]] 
    #print("**Targes.shape:**",targets.shape )   #(100,1) | (128,1) x6 (2042,1) |
    #print("**output:**",outputs )               #[[1],[0],...[1],[0]] 
    #print("**output.shape:**",outputs.shape )   #(100,1)

    #print("**Cn:**", Cn)                        #[[69.313....]] Fungerer?? -> Blir matrise feil i task2.py
    #print("**Cn.shape:**", Cn.shape)            #(1,1)
    #print("**C:**",C)                            #69.315...
    #print("**C.shape:**", C.shape)               #()            
   
    #print("**cost:**",cost)                     #[[69.301...]]                   
    #print("**cost.shape:**",cost.shape)         #(1,1)
    #with open('_Cross_Enropy.txt', 'a') as f:
    #    print('Cn:', C, file=f)  # Python 3.x

    #global i
    #print("Cross_entropy(2a): ",i)
    #i=i+1
    
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"


    return np.squeeze(cost)#C#C  #


class BinaryModel:

    def __init__(self):
        # Define number of input nodes
        self.I = 785 #None #Whats is I? 
        self.w = np.zeros((self.I, 1)) #Weights 
        self.grad = None #WHAT HERE? Should prob be initialized to None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, 1]
        """
        # TODO implement this function (Task 2a)
        z = X.dot(self.w)       # Eq 1.2
        y = 1/(1+np.exp(-z))    # Eq 1.1 Sigmoid function/ Logistic function
       
        #print("y.shape:", y.shape) #(100,1)
        #print("w.shape: ",self.w.shape)  #(785,1)
        #print("X.shape: ",X.shape)       #(100,785)

        return y #None

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, 1]
            targets: labels/targets of each image of shape: [batch size, 1]
        """
        # TODO implement this function (Task 2a)
        # Computes dC/dw

        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"

        #self.grad = np.zeros_like(self.w)    # Placeholder
        #self.grad = X.T@(outputs - targets)   # Eq. 6 y=target, y_hat=output
        self.grad = -X.T@(targets - outputs)  # dC/dw = dC/dz dy/dz dz/dw 

        #print("grad.shape: ",self.grad.shape) #(785,1)
        #with open('_grad.txt', 'a') as f:
        #    print('grad.T:', self.grad.T, file=f)  # Python 3.x



        assert self.grad.shape == self.w.shape,\
            f"Grad shape: {self.grad.shape}, w: {self.w.shape}"

    def zero_grad(self) -> None:
        self.grad = None


def gradient_approximation_test(model: BinaryModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    w_orig = np.random.normal(
        loc=0, scale=1/model.w.shape[0]**2, size=model.w.shape)
    epsilon = 1e-3
    for i in range(w_orig.shape[0]):
        model.w = w_orig.copy()
        orig = w_orig[i].copy()
        model.w[i] = orig + epsilon
        logits = model.forward(X)
        cost1 = cross_entropy_loss(Y, logits)
        model.w[i] = orig - epsilon
        logits = model.forward(X)
        cost2 = cross_entropy_loss(Y, logits)
        gradient_approximation = (cost1 - cost2) / (2 * epsilon)
        model.w[i] = orig
        # Actual gradient
        logits = model.forward(X)
        model.backward(X, logits, Y)
        difference = gradient_approximation - model.grad[i, 0]
        assert abs(difference) <= epsilon**2,\
            f"Calculated gradient is incorrect. " \
            f"Approximation: {gradient_approximation}, actual gradient: {model.grad[i,0]}\n" \
            f"If this test fails there could be errors in your cross entropy loss function, " \
            f"forward function or backward function"


def main():
    category1, category2 = 2, 3
    X_train, Y_train, *_ = utils.load_binary_dataset(category1, category2)
    X_train = pre_process_images(X_train)
    assert X_train.max(
    ) <= 1.0, f"The images (X_train) should be normalized to the range [-1, 1]"
    assert X_train.min() < 0 and X_train.min() >= - \
        1, f"The images (X_train) should be normalized to the range [-1, 1]"
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    # Simple test for forward pass. Note that this does not cover all errors!
    model = BinaryModel()
    logits = model.forward(X_train)
    np.testing.assert_almost_equal(
        logits.mean(), .5,
        err_msg="Since the weights are all 0's, the sigmoid activation should be 0.5")

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for i in range(2):
        gradient_approximation_test(model, X_train, Y_train)
        model.w = np.random.randn(*model.w.shape)


if __name__ == "__main__":
    main()
