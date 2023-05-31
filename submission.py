import pandas as pd
import numpy as np
import math

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

