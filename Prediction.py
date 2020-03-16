# -*- coding: utf-8 -*-
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import statistics
import skfuzzy.membership as mf

df=pd.read_csv("all.us.txt")

liste=[]
df_open=df['Open']
df_high=df['High']
df_low=df['Low']
df_close=df['Close']
Lavg=[]

date=[]
for x in range(1,219):
    date.append(x);

df_date=pd.DataFrame(date)

for i in range(df_open.size):
    avg=(df_open[i]+df_close[i]+df_low[i]+df_high[i])/4
    Lavg.append(avg)
        #print(avg)
        
df_avg=pd.DataFrame(Lavg)
for x in range(df_open.size-1):
    yuzde=(Lavg[x+1]*100)/Lavg[x]
        #print(yuzde-100)
    liste.append(yuzde-100)
s=statistics.stdev(liste)
print("Standart Sapma:",s*40)


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(df_date, df_avg, test_size=0.005, shuffle=False)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)

plt.scatter(df_date,df_avg,color='red')
plt.plot(df_date,lr.predict(df_date), color = 'blue')
plt.show()

egim=tahmin[1]-tahmin[0]
print('Eğim:',egim)

egim=egim+1
egim100=(100*egim)/2

print(egim100)

#4le çarptıkkkkk
x_std=np.arange(0,101,1)
std_low=mf.trimf(x_std,[0,28,56])
std_med=mf.trimf(x_std,[28,56,84])
std_high=mf.trimf(x_std,[56,84,100000])

x_egim=np.arange(0,101,1)
egim_inc=mf.trimf(x_egim,[0,100,100])
egim_dec=mf.trimf(x_egim,[0,0,100])

x_risk=np.arange(0,101,1)
risk_az=mf.trimf(x_risk,[0,0,50])
risk_orta=mf.trimf(x_risk,[30,50,70])
risk_cok=mf.trimf(x_risk,[50,100,100])


plt.set_title='Egim'
plt.plot(x_egim,egim_inc,'r',linewidth=2,label="Artan")
plt.plot(x_egim,egim_dec,'b',linewidth=2,label="Azalan")
plt.legend()
plt.show()


plt.set_title='Std Sapma'
plt.plot(x_std,std_low,'r',linewidth=2,label="Std.Sapma Az")
plt.plot(x_std,std_med,'g',linewidth=2,label="Std.Sapma Orta")
plt.plot(x_std,std_high,'b',linewidth=2,label="Std.Sapma Çok")
plt.legend()
plt.show()

plt.set_title='Risk'
plt.plot(x_risk,risk_az,'r',linewidth=2,label="Az Risk")
plt.plot(x_risk,risk_orta,'g',linewidth=2,label="Orta Risk")
plt.plot(x_risk,risk_cok,'b',linewidth=2,label="Çok Risk")
plt.legend()
plt.show()

s=s*40
input_std=s
input_egim=egim100

import skfuzzy as fuzz

std_fit_low=fuzz.interp_membership(x_std,std_low,input_std)
std_fit_med = fuzz.interp_membership(x_std,std_med,input_std)
std_fit_high=fuzz.interp_membership(x_std,std_high,input_std)

egim_fit_inc=fuzz.interp_membership(x_egim,egim_inc,input_egim)
egim_fit_dec=fuzz.interp_membership(x_egim,egim_dec,input_egim)

rule1=np.fmin(np.fmin(std_fit_low,egim_fit_inc),risk_az)
rule2=np.fmin(np.fmin(std_fit_low,egim_fit_dec),risk_orta)
rule3=np.fmin(np.fmin(std_fit_med,egim_fit_inc),risk_orta)
rule4=np.fmin(np.fmin(std_fit_med,egim_fit_dec),risk_orta)
rule5=np.fmin(np.fmin(std_fit_high,egim_fit_dec),risk_cok)
rule6=np.fmin(np.fmin(std_fit_high,egim_fit_inc),risk_cok)

out_az=rule1
out_orta=np.fmax(rule2,rule3,rule4)
out_cok=np.fmax(rule5,rule6)


risk0=np.zeros_like(x_risk)
fig,bx0=plt.subplots(figsize=(7,4))
bx0.fill_between(x_risk,risk0,out_az,facecolor='r',alpha=0.7)
bx0.plot(x_risk,risk_az,'r',linestyle='--')

bx0.fill_between(x_risk,risk0,out_orta,facecolor='g',alpha=0.7)
bx0.plot(x_risk,risk_orta,'g',linestyle='--')

bx0.fill_between(x_risk,risk0,out_cok,facecolor='b',alpha=0.7)
bx0.plot(x_risk,risk_cok,'b',linestyle='--')

bx0.set_title('Risk')


out_risk=np.fmax(out_az,out_orta,out_cok)
defuzzified=fuzz.defuzz(x_risk,out_risk,'centroid')
result = fuzz.interp_membership(x_risk,out_risk,defuzzified)

fig, cx0 = plt.subplots(figsize=(8, 4))

cx0.plot(x_risk, risk_az, 'r', linewidth=0.5, linestyle='--', )
cx0.plot(x_risk, risk_orta, 'g', linewidth=0.5, linestyle='--')
cx0.plot(x_risk, risk_cok, 'b', linewidth=0.5, linestyle='--')
cx0.fill_between(x_risk, risk0, out_risk, facecolor='Orange', alpha=0.7)
cx0.plot([defuzzified, defuzzified], [0, result], 'k', linewidth=1.5, alpha=0.9)
cx0.set_title('Ağırlık Merkezi ile Durulaştırma')

# Turn off top/right axes
for cx in (cx0,):
    cx.spines['top'].set_visible(False)
    cx.spines['right'].set_visible(False)
    cx.get_xaxis().tick_bottom()
    cx.get_yaxis().tick_left()

plt.tight_layout()
plt.show()

print("Hesaplanan çıkış değeri:",defuzzified)

