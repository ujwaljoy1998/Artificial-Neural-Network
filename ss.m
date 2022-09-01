load data.txt;
A = input('Enter the number of training pattern(Maximum 55)');      %Number of training pattern
B = input('Enter the number of testing pattern');       %Number of testing pattern
L = input('Enter the number of inputs(Maximum 6)');      %Number of inputs
N = input('Enter the number of outputs(Maximum 3)');     %Number of outputs
M = 10;             %Number of hidden layers
eta = 0.7;          %Learning rate
mom = 0.6;          %Momemtum term

I1 = data;
I1 = I1(randperm(size(I1,1)),:);     %Randomizing the data

T = I1(1:A,end-N+1:end);        %Target values
l = min(T);
m = max(T);
Ts1 = I1(A+1:A+B,end-N+1:end);

minim = min(I1,[],1);       %Normalization of input
maxim = max(I1,[],1);
[a,b] = size(I1);      
for i=1:a
    for j=1:b
        I1(i,j) = 0.1 + 0.8*((I1(i,j) - minim(j))/(maxim(j) - minim(j)));
    end
end

I(2:L+1,1:A) = I1(1:A,1:L)';        %Input for training data
for i=1:A
    I(1,i) = 1;
end

Itest(2:L+1,1:B) = I1(A+1:A+B,1:L)';    %Input for testing data
for i=1:B
    Itest(1,i) = 1;
end

%for M=1:15
V = rand(L+1,M);        %Finding weights between input and hidden layer
for i=1:M
    V(1,i) = 1;
end

W = rand(M+1,N);        %Finding weights between hidden and output layer
for i=1:N
    W(1,i) = 1;
end

T1 = I1(1:A,end-N+1:end);
Ts = I1(A+1:A+B,end-N+1:end);
deltaV1 = zeros(L+1,M);
deltaW1 = zeros(M+1,N);
n = 1;
MSE =1;

while n<10000
    Oo = zeros(N,A);
    Ih = zeros(M,A);
    Oh = zeros(M,A);
    Io = zeros(N,A);
    for p = 1:A             %Forward Propagation
        for j = 1:M
            for i = 1:L+1
                Ih(j,p) = Ih(j,p) + ((I(i,p))*V(i,j));
            end
        end
    end
    Oh = 1./(1+exp(-1*Ih));
    Oh = [ ones(1,A) ; Oh ];
    for p=1:A
       for j=1:N 
           for i=1:M+1
               Io(j,p) = Io(j,p) + (Oh(i,p)*W(i,j));
           end
       end
    end
    Oo = 1./(1+exp(-1*Io));
    Oo = Oo';

    Er = 0.5*((T1 - Oo).^2);        %Calculation of MSE for training data
    Error = sum(Er,2);
    MSE = sum(Error)/A;

    deltaV = zeros(L+1,M);          %Back propagation of error
    deltaW = zeros(M+1,N);

    for i=1:M+1                 %Back Propagation of error
        for j=1:N
            for p=1:A
               deltaW(i,j) = deltaW(i,j) + ((T1(p,j)-Oo(p,j))*Oo(p,j)*(1-Oo(p,j))*Oh(i,p));
            end
            deltaW(i,j) = deltaW(i,j)*(eta/A);
        end
    end

    for i=1:L+1                 
        for j=1:M
            for p=1:A
                for k=1:N
                    deltaV(i,j) = deltaV(i,j) + ((T1(p,k)-Oo(p,k))*Oo(p,k)*(1-Oo(p,k))*W(j+1,k)*Oh(j+1,p)*(1-Oh(j+1,p))*I(i,p) );
                end
                deltaV(i,j) = deltaV(i,j)*(1/N);
            end
            deltaV(i,j) = deltaV(i,j)*(1/A)*(eta);
        end
    end

    W = W + deltaW + (mom*deltaW1);         %Updating weights
    V = V + deltaV + (mom*deltaV1);       
    deltaV1 = deltaV;
    deltaW1 = deltaW;
    MSE1(n) = MSE;
    n = n + 1;
end
% MSE_arr(M) = MSE;
% end

for p=1:A               %Denormalization of output
    for q=1:N
        Oo(p,q) = ((Oo(p,q)-0.1)*(m(q)-l(q))/0.8) + l(q);
    end
end

Oo_test = zeros(N,B);           
Ih = zeros(M,B);
Oh = zeros(M,B);
Io = zeros(N,B);
for p=1:B               %Forward Propagation for test data
    for j=1:M 
        for i=1:L+1
            Ih(j,p) = Ih(j,p) + (Itest(i,p)*V(i,j));
        end
    end
end
Oh = 1./(1+exp(-1*Ih));
Oh = [ones(1,B);Oh];

for p=1:B
    for j=1:N 
        for i=1:M+1
            Io(j,p) = Io(j,p) + (Oh(i,p)*W(i,j));
        end
    end
end

Oo_test = 1./(1 + exp(-1*Io));
Oo_test = Oo_test';

Ert = 0.5*((Ts - Oo_test).^2);      %Calculation of MSE for test set
Errort = sum(Ert,2);
MSEt = sum(Errort)/B;

foutput = fopen('output','w');
fprintf('\nTotal number of iterations is %d\n',n);
fprintf('Mean Square Error for training set is %f\n',MSE);
fprintf('Mean Square Error for the test set is %f\n',MSEt);

fprintf(foutput,'\nTotal number of iterations is %d\n',n);
fprintf(foutput,'Mean Square Error for training set is %f\n',MSE);
fprintf(foutput,'Mean Square Error for the test set is %f\n\n',MSEt);

for p=1:B               %Denormalization of output
    for q=1:N
        Oo_test(p,q) = ((Oo_test(p,q)-0.1)*(m(q)-l(q))/0.8) + l(q);
    end
end

fprintf(foutput,'\nThe output of the network is\n');
for i=1:B
    for j=1:N
        fprintf(foutput,'%f\t',Oo_test(i,j));
    end
    fprintf(foutput,'\n');
end

E = abs(Ts1 - Oo_test);
fprintf(foutput,'\nThe absolute error is \n');
for i=1:B
    for j=1:N
        fprintf(foutput,'%f\t',E(i,j));
    end
    fprintf(foutput,'\n');
end

x = [1:(n-1)];                              %% Plot of Iterations vs MSE
plot(x,MSE1,'m:s');
xlabel('ITERATION');
ylabel('MSE');

% Y = [1:15];                                  %%  plot of no. hidden neurons vs
% plot(Y,MSE_arr,'g:s');                       
% xlabel('NO. OF HIDDEN NEURONS');
% ylabel('MSE');
    


