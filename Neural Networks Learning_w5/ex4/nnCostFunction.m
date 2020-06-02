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

X=[ones(m,1) X];
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

a1=X;
% a1= [ones(m,1) a1]; PROSOXI BIAS
a2=sigmoid(a1*Theta1');
a2=[ones(m,1) a2];
h=sigmoid(a2*Theta2'); % a(3) = h(?)

VecY=zeros(m,num_labels);
VecJ=zeros(m,num_labels);

for i=1:m
    %change y data nubers to complete vectors with 1s and 0s 
    VecY(i,:) =1:num_labels;
    VecY(i,:) = VecY(i,:) == y(i);

    for j=1: num_labels
        temp1=-VecY(i,j)*log(h(i,j));
        temp2=(1-VecY(i,j))*log(1-h(i,j));
        VecJ(i,j)=temp1-temp2;
    end
    VecJ(i)=sum(VecJ(i,:));
end
J=(1/m)*sum(VecJ(:,1));


%---------------REGULARIZATION--------------
temp1=0;
for i=1:(size(Theta2,2)-1) %25
    for j=2:size(Theta1,2)%400
        temp1=temp1+Theta1(i,j).^2;
    end     
end

temp2=0;
for i=1:num_labels %10
    for j=2:size(Theta2,2) %25
        temp2=temp2+Theta2(i,j).^2;
    end
end

J=J+((lambda/(2*m))*(temp1+temp2));
% ------------------------------------------------------




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


DELTA1=zeros(size(Theta1));
DELTA2=zeros(size(Theta2));

for t=1:m
    
    %===========STEP 1 - forward propagation===========
    a1=X(t,:); %exo vali ones se olon ton matrix X stin arxi tou programmatos
    a1=a1';
    z2=Theta1*a1;
    a2=sigmoid(z2);
    a2=[1; a2];
    z3=Theta2*a2;
    h=sigmoid(z3);

    %===========STEP 2 -errors delta===========
    %______For Layer 3 (Output Layer)______
    VecY=zeros(num_labels,1);
    VecY(y(t))=1; %Etsi ftiaxno to y se vector opos to xriazome
    
    delta3=h-VecY;
    
    %______For Layer 2 (Hidden Layer)______
    z2=[1;z2];
    delta2=(delta3'*Theta2)'.*sigmoidGradient(z2);
    delta2= delta2(2:end);
    
    %===========STEP 3 -accumulate gradient===========
    DELTA1=DELTA1+ delta2*a1';
    DELTA2=DELTA2+ delta3*a2';

end

%====STEP 4 -Obtain the unregularized & regularized gradient======
Theta1_grad = (1/m).* DELTA1; %unregularized
Theta2_grad = (1/m).* DELTA2; %unregularized

%--------------------------------------REGULARIZATION-----------------------------------
Theta1_grad(:,2:end) = (1/m).* DELTA1(:,2:end)+ (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = (1/m).* DELTA2(:,2:end) + (lambda/m)*Theta2(:,2:end);



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%TO EKANA META APO KATHE UNREGULARIZED CODE FOR COST AND GRADIENT
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
