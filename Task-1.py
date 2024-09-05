#####Task 1 #####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
filepath='C:/Users/91960/Desktop/python_ws/API_SP.POP.TOTL_DS2_en_csv_v2_2431709.csv'
dd=pd.read_csv(filepath,skiprows=4,usecols=range(68))
print(dd)
print(dd.head(5))
print(dd.info())
dd.duplicated().sum()
dd_clean=dd.dropna()
print(dd_clean)
total_popu=dd[dd['Indicator Code']=='SP.POP.TOTL']
total_pop=total_popu.sort_values(by='2022',ascending=False)
total_top=total_pop.head(15)
print(total_top[['Country Code']])

plt.figure()
sns.barplot(x="Country Code",y="2022",data=total_top)
plt.ylabel("total population")
plt.xlabel("country")
plt.show()




data=np.array(total_top)
num_bins=int(np.ceil(1+np.log2(len(data))))
plt.hist(data,bins=num_bins,edgecolor='black')
plt.xlabel('Country Code')
plt.ylabel('total_pop')
plt.show()
print(dd.columns)
cols=['1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968',
       '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977',
       '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986',
       '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',
       '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
       '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
       '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022',
       '2023']
dd_cle=dd_clean.drop(['Country Code','Indicator Name','Indicator Code'],axis=1,inplace=False)
print(dd_cle)
print(dd_cle.info)



