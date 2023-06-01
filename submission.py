import pandas as pd
import numpy as np
import math
import scipy as sc

P_daily = pd.read_excel('gold_oil_.xlsx')

np_P=np.array(P_daily)

r_gold=np.log(np.divide(np.subtract(np_P[1:,0], np_P[:-1,0]),np.absolute(np_P[:-1,0]))+1)
r_oil=np.log(np.divide(np.subtract(np_P[1:,1], np_P[:-1,1]),np.absolute(np_P[:-1,1]))+1)

def my_portfolio(np_a,np_b,lookback,pw):
	n=np_a.size
	T=n-lookback
	r_portfolio=np.empty([T])
	for t in range(T):
		R_a=np.sum(np_a[t:t+lookback-1])
		R_b=np.sum(np_b[t:t+lookback-1])
		sigma_a=np.std(np_a[t:t+lookback-1])
		sigma_b=np.std(np_b[t:t+lookback-1])
		S_a=abs((R_a/sigma_a)**pw)
		S_b=abs((R_b/sigma_b)**pw)
		w_a=math.copysign(S_a/(S_a+S_b), R_a)
		w_b=math.copysign(S_b/(S_a+S_b), R_b)
		r_portfolio[t]=w_a*np_a[t+lookback]+w_b*np_b[t+lookback]
		df_r = pd.DataFrame()
		df_r['returns'] = r_portfolio
	return df_r


#df_returns=my_portfolio(r_gold,r_oil,20,2)
#print(type(df_returns))
#print(df_returns)

c=np.percentile([-0.05,0.06,0.01],5)
print(c)


def calculate_historical_var(df_portfolio_returns, alpha):
	np_ret=np.array(df_portfolio_returns)
	pc=(1-alpha)*100
	my_var=float(np.percentile(np_ret,pc))
	return my_var

#v=calculate_historical_var([-0.05,0.06,0.01],0.95)
#print(type(v))
#print(v)

par_lb=np.array([5,10,20,40,60,120])
par_pw=np.array([1,2,3,4,5])

table1=np.empty([par_lb.size,par_pw.size])

for i in range(par_lb.size):
	for j in range(par_pw.size):
		df_returns=my_portfolio(r_gold,r_oil,par_lb[i],par_pw[j])
		table1[i,j]=calculate_historical_var(df_returns,0.95)*100


print(table1)

df_export1 = pd.DataFrame(table1)
df_export1.to_excel('py_results_1.xlsx')

###################################################################################################################

exp_r_gold=np.mean(r_gold)
exp_r_oil=np.mean(r_oil)
vol_gold=np.std(r_gold)
vol_oil=np.std(r_oil)

#real correlation
combined = np.vstack((r_gold, r_oil))
cr=np.corrcoef(combined)
print(cr)


def simulated_returns(expected_return, volatility, correlation, numOfSim):
	T=numOfSim
	expected_return2=exp_r_oil
	volatility2=vol_oil
	corr_M=np.array(([[1, correlation], [correlation, 1]]))
	D_chol=np.linalg.cholesky(corr_M)
	q=np.random.rand(T)
	rnd_z1=sc.stats.norm.ppf(q,loc=0,scale=1)
	rnd_z2=sc.stats.norm.ppf(q,loc=0,scale=1)
	corr_var1=rnd_z1*volatility+expected_return
	corr_var2=(rnd_z1*D_chol[1,0]+rnd_z2*D_chol[1,1])*volatility2+expected_return2
	sim_arrs=np.vstack((corr_var1, corr_var2)).T
	return sim_arrs

arrs=simulated_returns(exp_r_gold, vol_gold, -0.01, 10)
#print(arrs)
#print(np.shape(arrs))
#print(np.corrcoef(arrs.T))

def my_portfolio_2(np_a,np_b,w_a,w_b):
	T=np_a.size
	r_portfolio=np.empty([T])
	for t in range(T):
		r_portfolio[t]=w_a*np_a[t]+w_b*np_b[t]
	return r_portfolio

corrs=np.arange(-0.9, 0.91, 0.1)
w_gold=vol_gold**2/((vol_gold**2)+(vol_oil**2))
w_oil=vol_oil**2/((vol_gold**2)+(vol_oil**2))

abra1=np.empty([corrs.size])
for i in range(corrs.size):
	arrs=simulated_returns(exp_r_gold, vol_gold, corrs[i], 100)
	asset1=np.array(arrs[:,0])
	asset2=np.array(arrs[:,1])
	portfolio_2=my_portfolio_2(asset1,asset2,w_gold,w_oil)
	abra1[i]=calculate_historical_var(portfolio_2,0.95)*100


print(abra1)

###################################################################################################################

P_etf_daily = pd.read_excel('msci_etf.xlsx')
np_P_etf=np.array(P_etf_daily)
r_etf=np.log(np.divide(np.subtract(np_P_etf[1:,0], np_P_etf[:-1,0]),np_P_etf[:-1,0])+1)
df_r_etf = pd.DataFrame(r_etf)

def calculate_ewma_variance(df_etf_returns, decay_factor, window):
	init_p=100
	etf_ret=np.array(df_etf_returns)
	etf_ret1=etf_ret[:init_p]
	etf_ret2=etf_ret[init_p:]
	ewma=np.empty([window+1])
	ewma[0]=np.var(etf_ret1)
	for i in range(window):
		ewma[i+1]=(1-decay_factor)*ewma[i]+decay_factor*etf_ret2[i+1]**2
	df_ewma = pd.DataFrame(ewma)
	return df_ewma

ew=calculate_ewma_variance(df_r_etf, 0.97, 100)
print(type(ew))
print(ew)


