import numpy as np

class LogisticRegression:

    def __init__(self):
        self.parameters = np.array([])
    
    def fit(self,x, y):
        # convert from pandas series to numpy arrays
        x = np.array(x)
        y = np.array(y)

        # initialize parameters
        self.parameters = np.array([0.0,0.0])


        self.gradient_descent(x,y)

    def predict(self,x):
        value = self.parameters[0] + self.parameters[1] * x
        return 1 if value >= 0 else 0

    def gradient_descent(self,x,y):

        learning_rate = 0.0001

        for i in range(200000):
            u0 = 0.0
            u1 = 0.0
            for m in range(len(x)):
                residual = self.hypothesis([1,x[m]]) - y[m]
                u0 += residual * 1
                u1 += residual * x[m] 

            self.parameters[0] -= learning_rate * u0 / len(x)
            self.parameters[1] -= learning_rate * u1 / len(x)

            if i % 10000 == 0:
                pass#print('cost:',self.cost(x,y),'parameters:',self.parameters)


        print('parameters:',self.parameters)
        
        pass

    def hypothesis(self, x):
        return self.sigmoid(np.dot(self.parameters,x))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def cost(self,x,y):
        a = np.sum([(self.hypothesis([1,x[m]])-y[m])**2 for m in range(len(y))])
        return a
        