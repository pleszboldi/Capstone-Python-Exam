import pandas as pd
import numpy as np
import math
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 

#1. feladat
#Napi arfolyamok beolvasasa excelbol:
P_daily = pd.read_excel('gold_oil_.xlsx')
np_P=np.array(P_daily)

#Napi loghozamok szamolasa:
r_gold=np.log(np.divide(np.subtract(np_P[1:,0], np_P[:-1,0]),np.absolute(np_P[:-1,0]))+1)
r_oil=np.log(np.divide(np.subtract(np_P[1:,1], np_P[:-1,1]),np.absolute(np_P[:-1,1]))+1)

#Portfolio hozam idosort generalo fuggveny 2 hozam idosorbol es 2 parameterbol:
def my_portfolio(np_a,np_b,lookback,pw):
	n=np_a.size
	T=n-lookback #ilyen hosszu lesz az output idosor
	r_portfolio=np.empty([T]) #outputnak ures tomb, ami fel lesz toltve
	for t in range(T): #vegig megyunk a napokon
		R_a=np.sum(np_a[t:t+lookback-1]) #egyenlet (2)
		R_b=np.sum(np_b[t:t+lookback-1])
		sigma_a=np.std(np_a[t:t+lookback-1])
		sigma_b=np.std(np_b[t:t+lookback-1])
		S_a=abs((R_a/sigma_a)**pw)
		S_b=abs((R_b/sigma_b)**pw)
		w_a=math.copysign(S_a/(S_a+S_b), R_a) #egyenlet (3-4)
		w_b=math.copysign(S_b/(S_a+S_b), R_b)
		r_portfolio[t]=w_a*np_a[t+lookback]+w_b*np_b[t+lookback] #a portfolio hozama, egyenlet (5)
		df_r = pd.DataFrame() #dataframe formara transzformalas
		df_r['returns'] = r_portfolio
	return df_r

#VaR szamolo fuggveny:
def calculate_historical_var(df_portfolio_returns, alpha):
	np_ret=np.array(df_portfolio_returns)
	pc=(1-alpha)*100
	my_var=float(np.percentile(np_ret,pc)) #VaR ertek adott percentilisbol
	return my_var

#Ellenorzes:
#v=calculate_historical_var([-0.05,0.06,0.01],0.95)
#print(type(v))
#print(v)

#parameter tombok megadasa
#Ezektol fuggnek a portfolio sulyok
par_lb=np.array([5,10,20,40,60,120])
par_pw=np.array([1,2,3,4,5])

table1=np.empty([par_lb.size,par_pw.size]) #ures tomb, a tablazatnak
for i in range(par_lb.size): #vegig megyunk mindket parameter vektoron
	for j in range(par_pw.size):
		df_returns=my_portfolio(r_gold,r_oil,par_lb[i],par_pw[j]) #portfolio hozam generalas
		table1[i,j]=calculate_historical_var(df_returns,0.95)*100 #VaR szamolas


#print(table1)
#df_export1 = pd.DataFrame(table1) #eredmeny exportalas
#df_export1.to_excel('py_results_1.xlsx')

###################################################################################################################
#2. feladat

#A hozam idosorokhoz tartozo allando atlag es szoras kiszamolasa
exp_r_gold=np.mean(r_gold)
exp_r_oil=np.mean(r_oil)
vol_gold=np.std(r_gold)
vol_oil=np.std(r_oil)

#valodi korrelacio a ket eszkoz kozott (0.11)
combined = np.vstack((r_gold, r_oil))
cr=np.corrcoef(combined)
#print(cr)

#Korrelalt hozamokat szimulalo fuggveny
def simulated_returns(expected_return, volatility, correlation, numOfSim):
	T=numOfSim
	expected_return2=exp_r_oil
	volatility2=vol_oil
	corr_M=np.array(([[1, correlation], [correlation, 1]]))
	D_chol=np.linalg.cholesky(corr_M)
	q=np.random.rand(T) #veletlen valtozok generalasa
	rnd_z1=sc.stats.norm.ppf(q,loc=0,scale=1)
	rnd_z2=sc.stats.norm.ppf(q,loc=0,scale=1)
	corr_var1=rnd_z1*volatility+expected_return #1. hozam idosor
	corr_var2=(rnd_z1*D_chol[1,0]+rnd_z2*D_chol[1,1])*volatility2+expected_return2 #2. hozam idosor, ami a parameter szerint korrelal
	sim_arrs=np.vstack((corr_var1, corr_var2)).T
	return sim_arrs

arrs=simulated_returns(exp_r_gold, vol_gold, -0.01, 10)

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

mycolor=(150/255,5/255,160/255)
figure,axes=plt.subplots()
axes.plot(corrs,abra1,color=mycolor)
axes.set(xlabel='Korreláció',ylabel='VaR értékek',title='Diverzifikáció korrelált hozamokkal')
#figure.savefig('abra1.png')
#plt.show()

###################################################################################################################
#3. feladat
#MSCI World Index ETF betoltes
P_etf_daily = pd.read_excel('msci_etf.xlsx')
np_P_etf=np.array(P_etf_daily)
r_etf=np.log(np.divide(np.subtract(np_P_etf[1:,0], np_P_etf[:-1,0]),np_P_etf[:-1,0])+1)
df_r_etf = pd.DataFrame(r_etf)

#EWMA szamolo fuggveny
def calculate_ewma_variance(df_etf_returns, decay_factor, window):
	#rekurziv szamolas
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
#abra kirajzolas
figure,axes=plt.subplots()
axes.plot(range(ew.size),ew*100,color=mycolor)
axes.set(xlabel='Idő',ylabel='EWMA variancia',title='EWMA variancia alakulása %-ban')
#figure.savefig('abra2.png')
#plt.show()

###################################################################################################################
#4. feladat
#minta 3 reszre osztasa
r2_etf=r_etf**2
S1=r2_etf[:400]
S2=r2_etf[401:801]
S3=r2_etf[802:]

#sajat ar modellt keszito fuggveny
def my_ar(ts,lags):
	T=ts.size-lags
	y=ts[lags:]
	x=np.empty([T,lags])
	for i in range(lags):
		x[:,i]=ts[i:T+i]
	model = LinearRegression().fit(x, y)
	y_pred = model.predict(x)
	residuals=y-y_pred
	return model, residuals

#ar parametereket szamolo fuggveny
def my_model_param(ts, lags):
	m_ar_1 = my_ar(ts,lags)
	model1=m_ar_1[0]
	#coeffs1=model1.coef_
	res1=m_ar_1[1]
	m_ar_2 = my_ar(res1,lags)
	model2=m_ar_2[0]
	#coeffs2=model2.coef_
	#cfs=np.vstack((coeffs1, coeffs2)).T
	return model1, model2

#variancia szamolas
def calculate_ewma_variance_1(df_etf_returns, decay_factor, window,b):
	init_p=b
	etf_ret=np.array(df_etf_returns)
	etf_ret1=etf_ret[:init_p]
	etf_ret2=etf_ret[init_p:]
	ewma=np.empty([window+1])
	ewma[0]=np.var(etf_ret1)
	for i in range(window):
		ewma[i+1]=(1-decay_factor)*ewma[i]+decay_factor*etf_ret2[i+1]**2
	return ewma

#a predikciot es validalast elvegzo fuggveny barmilyen idosorra es kesleltetesre
def my_model_pred(S2, max_lag,y,T):
	#T=S2.size-max_lag
	my_mse=np.empty(max_lag)
	for i in range(1,max_lag+1,1):
		md_i=my_model_param(S2, i)
		md_r=md_i[0]
		ar_int=md_r.intercept_
		ar_cfs=md_r.coef_
		x=np.empty([T,i])
		for j in range(i):
			x[:,j-1]=S2[max_lag-j:S2.size-j]
			#print(np.shape(x))
		r_ar_pred=np.ones((T,1))*ar_int
		#print(np.shape(r_ar_pred))
		for k in range(i):
			for l in range(T-1):
				r_ar_pred[l,0]=r_ar_pred[l,0]+ar_cfs[k]*x[l,k]
			#r_ar_pred=np.add(r_ar_pred,ar_cfs[k]*x[:,k])
		my_mse[i-1]=np.sum(np.square(np.subtract(y,r_ar_pred)))/(T-1)
	residuals=y-r_ar_pred
	return residuals, my_mse, r_ar_pred

#feladat megoldas a fuggvenyekkel
#variancia elorejelzes
max_lag=20
T=S2.size-max_lag
y1=S2[max_lag:]
md_p1=my_model_pred(S2, max_lag,y1,T)
res1=md_p1[0]
y2=calculate_ewma_variance_1(S2, 0.97, T-1,max_lag)
md_p2=my_model_pred(S2, max_lag,y2,T)
var_mse=md_p2[1]
#print(var_mse[6]*100)

mycolor=(150/255,5/255,160/255)
figure,axes=plt.subplots()
axes.plot(range(1,max_lag+1,1),var_mse,color=mycolor)
axes.set(xlabel='Késleltetés',ylabel='MSE',title='Variancia előrejelzési hiba')
#figure.savefig('abra4.png')
#plt.show()

#a 3. mintan a legjobb parameteru (7) modell tesztelese
def my_model_pred_2(S2, lag,y,T):
	#T=S2.size-max_lag
	i=lag
	max_lag=lag
	md_i=my_model_param(S2, i)
	md_r=md_i[0]
	ar_int=md_r.intercept_
	ar_cfs=md_r.coef_
	x=np.empty([T,i])
	for j in range(i):
		x[:,j-1]=S2[max_lag-j:S2.size-j]
	r_ar_pred=np.ones((T,1))*ar_int
	for k in range(i):
		for l in range(T-1):
			r_ar_pred[l,0]=r_ar_pred[l,0]+ar_cfs[k]*x[l,k]
		#r_ar_pred=np.add(r_ar_pred,ar_cfs[k]*x[:,k])
	my_mse=np.sum(np.square(np.subtract(y,r_ar_pred)))/(T-1)
	return  my_mse

T3=S3.size-7
y3=calculate_ewma_variance_1(S3, 0.97, T3-1,7)
md_p3=my_model_pred_2(S3, 7,y3,T3)
#print(md_p3*100)

