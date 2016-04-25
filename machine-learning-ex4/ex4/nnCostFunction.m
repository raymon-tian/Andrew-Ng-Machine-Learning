function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
size_theta1 = size(Theta1);
size_theta2 = size(Theta2);
size_y = size(y);
for i=1:m
	a_1 = X(i,:)';
	a_1 = [1;a_1];
	a_2 = sigmoid(Theta1*a_1);
	a_2 = [1;a_2];
	a_3 = sigmoid(Theta2*a_2);
	h = a_3;
	y_vector = zeros(num_labels,1);
	y_vector(y(i)) = 1;
	temp_k = (-y_vector).*log(h)-(1-y_vector).*log(1-h);
	J = J+sum(temp_k);
end

J = J/m;

% regularized cost function
s1 = sum(sum(Theta1.^2));
s2 = sum(sum(Theta2.^2));
s3 = sum(sum(Theta1(:,1).^2));
s4 = sum(sum(Theta2(:,1).^2));
sumForAllTheta = s1+s2-s3-s4;
J = J+lambda/(2*m)*(sumForAllTheta);

%backpropagation

deTa1 = 0;
deTa2 = 0;

for i=1:m
	a_1 = X(i,:)';
	a_1 = [1;a_1];
	a_2 = sigmoid(Theta1*a_1);
	a_2 = [1;a_2];
	a_3 = sigmoid(Theta2*a_2);
	y_vector = zeros(num_labels,1);
	y_vector(y(i)) = 1;
	ipxl_3 = a_3-y_vector;
	ipxl_2 = Theta2'*ipxl_3.*sigmoidGradient([1;Theta1*a_1]);
	ipxl_2 = ipxl_2(2:end);
	deTa1 = deTa1+ipxl_2*a_1';
	deTa2 = deTa2+ipxl_3*a_2';
	Theta1_grad = deTa1/m;
	Theta2_grad = deTa2/m; 
end

for i = 1:size(Theta1,1)
	for j = 2:size(Theta1,2)
		Theta1_grad(i,j) = Theta1_grad(i,j)+lambda/m*Theta1(i,j);
	end
end

for i = 1:size(Theta2,1)
	for j = 2:size(Theta2,2)
		Theta2_grad(i,j) = Theta2_grad(i,j)+lambda/m*Theta2(i,j);
	end
end










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
