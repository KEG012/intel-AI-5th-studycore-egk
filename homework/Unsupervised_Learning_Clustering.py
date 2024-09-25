#%%
# K-Means 기법

import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

# ---------- 데이터 생성 ----------
np.random.seed(1)
N = 100
K = 3
T3 = np.zeros((N,3), dtype=np.uint8)
X = np.zeros((N,2))
X_range0 = [-3, 3]
X_range1 = [-3, 3]
X_col = ['cornflowerblue', 'black', 'white']
Mu = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]])    # 분포의 중심
Sig = np.array([[.7, .7], [.8, .3], [.3, .8]])        # 분포의 분산
Pi = np.array([0.4, 0.8, 1])    # 누적 확률
for n in range(N):
    wk = np.random.rand()
    for k in range(K):
        if wk < Pi[k]:
            T3[n, k] = 1
            break
    for k in range(2):
        X[n, k] = (np.random.randn() * Sig[T3[n, :] == 1, k] + Mu[T3[n, :] == 1, k])
            
# ---------- 데이터 그리기 ----------
def show_data(x):
    plt.plot(x[:, 0], x[:, 1], linestyle = 'none',
             marker = 'o', markersize = 6,
             markeredgecolor = 'black', color = 'gray', alpha = 0.8)
    plt.grid(True)


# ---------- 그래프 출력 ----------
plt.figure(1, figsize=(4,4))
show_data(X)
plt.xlim(X_range0)
plt.ylim(X_range1)
plt.show()
np.savez('data_ch9.npz', X=X, X_range0=X_range0, X_range1=X_range1)

# ---------- Mu 및 R 초기화 ----------
Mu = np.array([[-2, 1], [-2, 0], [-2, -1]]) # 임의의 중심 벡터 설정
R = np.c_[np.ones((N, 1), dtype=int), np.zeros((N, 2), dtype=int)] # 모든 데이터가 클래스 0에 속하도록 R을 초기화

# ---------- 갱신된 데이터 그리기 ----------
def show_prm(x, r, mu, col):
    for k in range(K):
        plt.plot(x[r[:,k] == 1,0], x[r[:,k] == 1,1],
                 marker = 'o', markerfacecolor=col[k],
                 markeredgecolor='k', markersize=6,
                 alpha=0.5, linestyle='none')
        plt.plot(mu[k,0], mu[k,1], marker='*',
                 markerfacecolor=col[k], markersize=15,
                 markeredgecolor='k', markeredgewidth=1)
        plt.xlim(X_range0)
        plt.ylim(X_range1)
        plt.grid(True)
        
# ---------- 그래프 출력 ----------
plt.figure(figsize=(4,4))
R = np.c_[np.ones((N,1)), np.zeros((N,2))]
show_prm(X,R,Mu,X_col)
plt.title('initial Mu and R')
plt.show()

# 각 데이터 점을 가장 중심이 각까운 클러스터에 넣는 과정을 수행하기 위해 R을 갱신
# |X_n - u_k|^2 = (x_n0 - u_k0)^2 + (x_n1 - u_k1)^2 ==> 거리가 알고 싶은것이 아니라서 제곱으로 진행.
# ---------- r 값을 결정 ----------
def step1_kmeans(x0, x1, mu):
    N = len(x0)
    r = np.zeros((N,K))

    for n in range(N):
        wk = np.zeros(K)
        for k in range(K):
            wk[k] = (x0[n] - mu[k,0])**2 + (x1[n] - mu[k,1])**2
        r[n, np.argmin(wk)] = 1

    return r

# ---------- 그래프 출력 ----------
plt.figure(figsize=(4,4))
R = step1_kmeans(X[:,0], X[:,1], Mu)
show_prm(X,R,Mu,X_col)
plt.title('Step1')
plt.show()

# 각 클러스터에 속하는 데이터 점의 중심을 새로운 u로 결정
# 각 점의 평균을 구하여 그것을 중심으로 결정
# ---------- Mu 값을 결정 ----------
def step2_kmeans(x0, x1, r):
    mu = np.zeros((K,2))

    for k in range(K):
        mu[k, 0] = np.sum(r[:,k]*x0)/np.sum(r[:,k])
        mu[k, 1] = np.sum(r[:,k]*x1)/np.sum(r[:,k])
        
    return mu

# ---------- 그래프 출력 ----------
plt.figure(figsize=(4,4))
Mu = step2_kmeans(X[:,0], X[:,1], R)
show_prm(X, R, Mu, X_col)
plt.title('Step2')
plt.show()

# 변화된 값을 다시 입력값에 넣어 클레스터의 클래스 변경 및 클러스터의 중심값을 변경
# ---------- Mu, r 값을 결정하는 것을 반복 ----------
plt.figure(1, figsize=(10,6.5))
Mu = np.array([[-2, 1], [-2, 0], [-2, -1]])
max_it = 6 # 반복 횟수

for it in range(0, max_it):
    plt.subplot(2,3,it+1)
    R = step1_kmeans(X[:,0], X[:,1], Mu)
    show_prm(X, R, Mu, X_col)
    plt.title("{0:d}".format(it+1))
    plt.xticks(range(X_range0[0], X_range0[1]), "")
    plt.yticks(range(X_range1[0], X_range1[1]), "")
    Mu = step2_kmeans(X[:,0], X[:,1], R)

plt.show()

# ---------- 목적 함수 ----------
def distortion_measure(x0, x1, r, mu):
    # 입력은 2차원으로 제한
    N = len(x0)
    J = 0
    for n in range(N):
        for k in range(K):
            J = J + r[n, k]*((x0[n] - mu[k,0])**2 + (x1[n] - mu[k,1])**2)
    
    return J

# test
# ---------- Mu와 R의 초기화 ---------- 
Mu = np.array([[-2, 1], [-2, 0], [-2, -1]])
R = np.c_[np.ones((N, 1), dtype=int), np.zeros((N, 2), dtype=int)]
distortion_measure(X[:,0], X[:,1], R, Mu)

# ---------- Mu와 R의 초기화 ---------- 
N = X.shape[0]
K = 3
Mu = np.array([[-2, 1], [-2, 0], [-2, -1]])
R = np.c_[np.ones((N, 1), dtype=int), np.zeros((N, 2), dtype=int)]
max_it = 10
it = 0
DM = np.zeros(max_it) # 왜곡 척도의 계산 결과를 넣는 변수

for it in range(0, max_it): # K-Means 기법
    R = step1_kmeans(X[:,0], X[:,1], Mu)
    DM[it] = distortion_measure(X[:,0], X[:,1], R, Mu) # 왜곡 척도
    Mu = step2_kmeans(X[:,0], X[:,1], R)

print(np.round(DM,2))
plt.figure(2, figsize=(4,4))
plt.plot(DM, color='black', linestyle='-', marker='o')
plt.ylim(40,80)
plt.grid(True)
plt.show()

# K-Means 기법으로 얻을 수 있는 해는 초기값 의존성이 있음.
# 처음 u에 무엇을 할당하냐에 따라 결과가 달라짐.
# 실제는 다양한 u에서 시작해서 얻을 결과 중 가장 왜곡 척도가 작은 결과를 사용하는 방법이 사용됨.
# R을 먼저 결정하여도 문제가 없음.