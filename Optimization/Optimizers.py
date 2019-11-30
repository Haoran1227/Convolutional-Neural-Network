class Sgd:
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate
        pass

    def calculate_update(self, weight_tensor, gradient_tensor):
        #update weight_tensor by SGD
        updated_weight_tensor = weight_tensor - self.learning_rate*gradient_tensor
        return updated_weight_tensor
