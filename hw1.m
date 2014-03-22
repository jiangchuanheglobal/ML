%CSE555_HW1
%author: Jiangchuan He 50096978
%clear;
%%================================================
%file path
src_1 = '/Users/jiangchuan/Desktop/hw1/data1.txt';
src_2 = '/Users/jiangchuan/Desktop/hw1/data2.txt';
src_3 = '/Users/jiangchuan/Desktop/hw1/data3.txt';
%%
%=================================================
%read in data
%w_1 = textread(src_1);
%w_2=textread(src_2);
%w_3=textread(src_3);
%------
load dataset;
%%
%=================================================
%display data points
plot(w_1(1:1000,1),w_1(1:1000,2),'r+');
hold on;
plot(w_2(1:1000,1),w_2(1:1000,2),'bo');
hold on;
plot(w_3(1:1000,1),w_3(1:1000,2),'kx');
hold on;
legend('class 1','class 2', 'class 3')
xlabel('X1');
ylabel('X2');
%%
%=================================================
%get the first 1000 points from each src
w_1_train=w_1(1:1000,:);
w_2_train=w_2(1:1000,:);
w_3_train=w_3(1:1000,:);
%%
%=================================================
w_1_x1=w_1_train(:,1);
w_1_x2=w_1_train(:,2);

w_2_x1=w_2_train(:,1);
w_2_x2=w_2_train(:,2);

w_3_x1=w_3_train(:,1);
w_3_x2=w_3_train(:,2);
%%
%=======================================
%get mu and cov matrix from each catgory
w_1_mu = mean(w_1_train,1);
w_2_mu = mean(w_2_train,1);
w_3_mu = mean(w_3_train,1);

w_1_cov= cov(w_1_train);
w_2_cov= cov(w_2_train);
w_3_cov= cov(w_3_train);
%%
%=======================================
%
[X1,X2]=meshgrid(linspace(-10,10,100)',linspace(-10,10,100)');
Z_1 = mvnpdf([X1(:) X2(:)],w_1_mu,w_1_cov);
Z_2 = mvnpdf([X1(:) X2(:)],w_2_mu,w_1_cov);
Z_3 = mvnpdf([X1(:) X2(:)],w_3_mu,w_1_cov);

%P = reshape(Z,size(Z));
%%
%=======================================
%draw graph
figure;
surf(X1,X2,reshape(Z_1,100,100));
hold on;
surf(X1,X2,reshape(Z_2,100,100));
hold on;
surf(X1,X2,reshape(Z_3,100,100));
%meshc(X1,X2,reshape(Z_3,100,100));
%axis tight;
%%
%=======================================
%predict rest data points labels
w_1_test=w_1(1001:2000,:);
w_2_test=w_2(1001:2000,:);
w_3_test=w_3(1001:2000,:);
%=========evaluate error rate for catgory 1
w_1_predict_1 = mvnpdf(w_1_test, w_1_mu, w_1_cov);
w_1_predict_2 = mvnpdf(w_1_test, w_2_mu, w_2_cov);
w_1_predict_3 = mvnpdf(w_1_test, w_3_mu, w_3_cov);
w_1_predict = [w_1_predict_1, w_1_predict_2, w_1_predict_3]';
[max_v, predict_label]= max(w_1_predict);
w_1_err_num = length(find(predict_label~=1));
%=========evaluate error rate for catgory 2
w_2_predict_1 = mvnpdf(w_2_test, w_1_mu, w_1_cov);
w_2_predict_2 = mvnpdf(w_2_test, w_2_mu, w_2_cov);
w_2_predict_3 = mvnpdf(w_2_test, w_3_mu, w_3_cov);
w_2_predict = [w_2_predict_1, w_2_predict_2, w_2_predict_3]';
[max_v, predict_label]= max(w_2_predict);
w_2_err_num = length(find(predict_label~=2));
%=========evaluate error rate for catgory 3
w_3_predict_1 = mvnpdf(w_3_test, w_1_mu, w_1_cov);
w_3_predict_2 = mvnpdf(w_3_test, w_2_mu, w_2_cov);
w_3_predict_3 = mvnpdf(w_3_test, w_3_mu, w_3_cov);
w_3_predict = [w_3_predict_1, w_3_predict_2, w_3_predict_3]';
[max_v, predict_label]= max(w_3_predict);
w_3_err_num = length(find(predict_label~=3));
%======
fprintf('---------using 1000 data points from each catgory----------\n');
fprintf('error rate of catgory_1 is:%3.2f\n',100* w_1_err_num/1000);
fprintf('error rate of catgory_2 is:%3.2f\n',100* w_2_err_num/1000);
fprintf('error rate of catgory_3 is:%3.2f\n',100* w_3_err_num/1000);
err_rate_mat=[w_1_err_num/1000, w_2_err_num/1000, w_3_err_num/1000];
figure;
bar(err_rate_mat);
%%
%===================================================================
%use 500 data points to train classifers
w_1_train=w_1(1:500,:);
w_2_train=w_2(1:500,:);
w_3_train=w_3(1:500,:);
%===
w_1_x1=w_1_train(:,1);
w_1_x2=w_1_train(:,2);

w_2_x1=w_2_train(:,1);
w_2_x2=w_2_train(:,2);

w_3_x1=w_3_train(:,1);
w_3_x2=w_3_train(:,2);
%===
w_1_mu = mean(w_1_train,1);
w_2_mu = mean(w_2_train,1);
w_3_mu = mean(w_3_train,1);

w_1_cov= cov(w_1_train);
w_2_cov= cov(w_2_train);
w_3_cov= cov(w_3_train);
%=======================================
%predict rest data points labels
w_1_test=w_1(501:2000,:);
w_2_test=w_2(501:2000,:);
w_3_test=w_3(501:2000,:);
%=========evaluate error rate for catgory 1
w_1_predict_1 = mvnpdf(w_1_test, w_1_mu, w_1_cov);
w_1_predict_2 = mvnpdf(w_1_test, w_2_mu, w_2_cov);
w_1_predict_3 = mvnpdf(w_1_test, w_3_mu, w_3_cov);
w_1_predict = [w_1_predict_1, w_1_predict_2, w_1_predict_3]';
[max_v, predict_label]= max(w_1_predict);
w_1_err_num = length(find(predict_label~=1));
%=========evaluate error rate for catgory 2
w_2_predict_1 = mvnpdf(w_2_test, w_1_mu, w_1_cov);
w_2_predict_2 = mvnpdf(w_2_test, w_2_mu, w_2_cov);
w_2_predict_3 = mvnpdf(w_2_test, w_3_mu, w_3_cov);
w_2_predict = [w_2_predict_1, w_2_predict_2, w_2_predict_3]';
[max_v, predict_label]= max(w_2_predict);
w_2_err_num = length(find(predict_label~=2));
%=========evaluate error rate for catgory 3
w_3_predict_1 = mvnpdf(w_3_test, w_1_mu, w_1_cov);
w_3_predict_2 = mvnpdf(w_3_test, w_2_mu, w_2_cov);
w_3_predict_3 = mvnpdf(w_3_test, w_3_mu, w_3_cov);
w_3_predict = [w_3_predict_1, w_3_predict_2, w_3_predict_3]';
[max_v, predict_label]= max(w_3_predict);
w_3_err_num = length(find(predict_label~=3));
%======
fprintf('---------using 500 data points from each catgory----------\n');
fprintf('error rate of catgory_1 is:%3.2f\n',100* w_1_err_num/1500);
fprintf('error rate of catgory_2 is:%3.2f\n',100* w_2_err_num/1500);
fprintf('error rate of catgory_3 is:%3.2f\n',100* w_3_err_num/1500);
err_rate_mat=[w_1_err_num/1500, w_2_err_num/1500, w_3_err_num/1500];
figure;
bar(err_rate_mat);