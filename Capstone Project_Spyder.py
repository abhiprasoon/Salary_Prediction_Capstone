import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_clust=pd.read_excel('Dataset.xlsx')

df_clust.head()

df_clust.shape

df_clust.isnull().sum()/len(df_clust)*100 

df_describe=df_clust.describe()

df_clust.duplicated().sum()


# there are 339 unique values
df_clust['JobCity'].nunique()

df_clust['JobCity'].unique()

df_clust['JobCity']=df_clust['JobCity'].str.capitalize()

df_clust['JobCity']=df_clust['JobCity'].str.strip()

#236 unique values
df_clust['JobCity'].nunique()

df_clust['JobCity']=df_clust['JobCity'].replace({'-1':np.nan,'haryana':'Haryana','KOTA':'Kota','manesar':'Manesar','MEERUT':'Meerut','Asifabadbanglore':'Bengaluru'})

df_clust['JobCity']=df_clust['JobCity'].replace({'karnal':'Karnal','Banglore ':'Bengaluru','manesar':'Manesar','Bangalore':'Bengaluru','Banglore':'Bengaluru'})

df_clust['JobCity']=df_clust['JobCity'].replace({'shahibabad':'Sahibabad','KANPUR':'Kanpur','pondy':'Puducherry','mohali':'Mohali','noida':'Noida','delhi':'Delhi','HYDERABAD':'Hyderabad','mysore':'Mysore','latur (Maharashtra )':'Latur'})

df_clust['JobCity']=df_clust['JobCity'].replace({'Ambala City':'Ambala','RAE BARELI':'Raebareli','jAipur':'Jaipur','sampla':'Sampla','NEW DELHI':'Delhi'})
# 232 unique values
df_clust['JobCity'].nunique()

df_clust['JobCity']=df_clust['JobCity'].replace({'noida ':'Noida','ranchi':'Ranchi','PUNE':'Pune',' Pune':'Pune','orissa':'Odisha','kala amb ':'Kala Amb','chennai ':'Chennai'})

df_clust['JobCity']=df_clust['JobCity'].replace({'ghaziabad':'Ghaziabad','Panchkula ':'Panchkula','Mettur, Tamil Nadu ':'Mettur','Baddi HP':'Baddi','Pune ':'Pune','Greater NOIDA':'Greater Noida','hyderabad ':'Hyderabad','chandigarh':'Chandigarh','BHUBANESWAR':'Bhubaneswar','Navi mumbai':'Navi Mumbai','hyderabad(bhadurpally)':'Hyderabad','GREATER NOIDA':'Greater Noida'})

df_clust['JobCity'].nunique()


df_clust['JobCity']=df_clust['JobCity'].replace({'Punr':'Pune','Latur (maharashtra )':'Latur',' delhi':'Delhi',' pune':'Pune','Sadulpur,rajgarh,distt-churu,rajasthan':'Sadulpur','Hyderabad(bhadurpally)':'Hyderabad'})


df_clust['JobCity']=df_clust['JobCity'].replace({'Kochi/cochin, chennai and coimbatore':'Chennai','New delhi - jaisalmer':'Delhi','Chennai, bangalore':'Chennai','Navi mumbai , hyderabad':'Hyderabad','A-64,sec-64,noida':'Noida','Rayagada, odisha':'Odisha','Orissa':'Orissa','pune':'Pune','bangalore':'Bengaluru','New delhi':'Delhi','Banaglore':'Bengaluru','Delhi/ncr':'Delhi'})

df_clust['JobCity'].nunique()
#220 unique values

df_clust['JobCity']=df_clust['JobCity'].replace({'Mettur, tamil nadu':'Mettur','Hderabad':'Hyderabad','New dehli':'Delhi','Bellary':'Ballari'})

df_clust['JobCity']=df_clust['JobCity'].replace({'Technopark, trivandrum':'Thiruvananthapuram','Trivandrum':'Thiruvananthapuram','ariyalur':'Ariyalur','Kochi/cochin':'Kochi','Indirapuram, ghaziabad':'Ghaziabad','Gaziabaad':'Ghaziabad','Bhubneshwar':'Bhubaneswar'})

df_clust['JobCity']=df_clust['JobCity'].replace({'Banagalore':'Bengaluru','Bhubneshwar':'Bhubaneswar','Gajiabaad':'Ghaziabad','Kudankulam ,tarapur':'Koodankulam','Ncr':'Delhi','mumbai':'Mumbai'})

df_clust['JobCity'].nunique()
#208 unique values

df_clust['JobCity']=df_clust['JobCity'].replace({'Guragaon':'Gurugram','Gurgaon':'Gurugram','Gurgoan':'Gurugram','Nouda':'Noida','chennai':'Chennai','Kolkata`':'Kolkata','Orissa':'Odisha','Guragaon':'Gurugram','Kalmar, sweden':'Sweden','Chennai & mumbai':'Chennai'})

df_clust['JobCity'].nunique()
#201 unique

df_clust['JobCity']=df_clust['JobCity'].replace({'Al jubail,saudi arabia':'Saudi Arabia','Jeddah saudi arabia':'Saudi Arabia','Nasikcity':'Nashik'})


df_clust['JobCity']=df_clust['JobCity'].replace({'Bhubaneswar':'Bhubaneshwar','Shahibabad':'Sahibabad','Baddi hp':'Baddi','Burdwan':'Bardhaman'})

#197 unique values
df_clust['JobCity'].nunique()

#461 np.nan - 11%
df_clust['JobCity'].isnull().sum()/len(df_clust)*100

df_clust['JobCity'].unique()

df_clust['JobCity']=df_clust['JobCity'].replace({'Sonepat':'Sonipat','Gandhi nagar':'Gandhinagar','Muzzafarpur':'Muzaffarpur',})
#194 unique values

df_clust['JobCity']=df_clust['JobCity'].replace({'Vsakhapttnam':'Visakhapatnam','Vizag':'Visakhapatnam','Hospete':'Hosapete','Trichy':'Tiruchirappalli','Tirunelveli':'Tirunelvelli','Keral':'Kerala'})
#192 unique values

x=df_clust['JobCity'].value_counts()

y=df_clust[df_clust['JobCity']=='India'] #salary is low so i will take it as non metropolitian city

non-metropolitian=['Bengaluru','Hyderabad','Pune','Ahmedabad','Surat','Jaipur','Lucknow','Kanpur','Nagpur','Indore','Thane','Bhopal','Agra','Nashik','Visakhapatnam','Patna','Kannur','Ludhiana','Rajkot','Chandigarh','Amritsar','Meerut','Thiruvananthapuram','Vadodara','Ranchi','Varanasi','Kota','Raipur','Madurai','Jodhpur','Vijayawada','Gwalior','Jabalpur','Coimbatore','Howrah','Allahabad','Dhanbad','Aurangabad','Srinagar','Vassai-Virar','Kalyan-Dombivali']

metropolitian=['Mumbai','Delhi','Chennai','Kolkata','New Delhi','Navi Mumbai','Ghaziabad','Faridabad','Gurugram']

len(metropolitian)

abroad=['Australia','Dubai','Saudi Arabia','Sweden','Johannesburg','London','Ras al khaimah','Dammam']


df_clust[df_clust['JobCity']==metropolitian]

for i in range(0,len(df_clust['JobCity'])):
    if df_clust.loc[i,'JobCity'] in metropolitian:
        df_clust.loc[i,'City_Type']='Metropolitian'
    elif df_clust.loc[i,'JobCity'] in abroad:
        df_clust.loc[i,'City_Type']='Abroad'
    elif df_clust.loc[i,'JobCity'] in non_metro:
        df_clust.loc[i,'City_Type']='Non-Metropolitian'


df_clust['City_Type'].value_counts()

df_clust[df_clust['City_Type']=='Non Metropolitian']['JobCity']




df_clust['JobCity'].isnull().sum()
















