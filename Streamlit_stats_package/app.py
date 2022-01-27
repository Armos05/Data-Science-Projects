import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import style
from matplotlib.pyplot import plot
from PIL import Image
from prettytable import PrettyTable
from random import randrange
import statsmodels.api as sm
import scipy.stats as stats
from sklearn import datasets, linear_model, metrics
from scipy.stats import norm, kurtosis



image = Image.open('stats.jpg')
st.image(image, use_column_width  = True)
st.markdown("<h1 style='text-align: center;'>Statistical Test App</h1>", unsafe_allow_html=True)
st.markdown('A one stop app for all your statistical tests')
st.markdown('-------------------------------------------------------------------')


tests = ["Descriptive Statistics", "Histogram", "Box-plot", "Q-Q plot", "Linear Regression", "Correlation Analysis",
        "T-Test one sample", "T-test independent 2 sample", "Paired T sample", "Chi square test", 
        "ANOVA"]
tags = ["Awesome", "Social"]

page = st.sidebar.radio("Which Statistical Test is to be performed", options=tests)

if page == "Descriptive Statistics":
    st.markdown("<h2 style='text-align: center;'>Descriptive Statistics of the Data</h2>", unsafe_allow_html=True)
    st.markdown("This function tells you about the general properties of your dataset")
    st.session_state.gs_URL = st.sidebar.text_input("Please enter the Google sheet URL having the cleaned data:","https://docs.google.com/spreadsheets/d/1MXlwsbKDWtlHcEuqJo_I2VJwuMH-ngdlMACNt2vu2BY/edit#gid=0") 
    csv_export_url = st.session_state.gs_URL.replace('/edit#gid=', '/export?format=csv&gid=')
    df = pd.read_csv(csv_export_url)
    #st.dataframe(df)
    rows, columns = df.shape
    numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
    total_columns = len(numeric_columns)
    quant_table = df[numeric_columns].describe()
    st.table(quant_table)


if page == "Histogram":
   st.markdown("<h2 style='text-align: center;'>Histogram of the Data</h2>", unsafe_allow_html=True)
   st.markdown("This function tells you about the distribution of your dataset. You can also choose the number of bins you want to keep in this distribution")
   st.session_state.gs_URL = st.sidebar.text_input("Please enter the Google sheet URL having the cleaned data:","https://docs.google.com/spreadsheets/d/1MXlwsbKDWtlHcEuqJo_I2VJwuMH-ngdlMACNt2vu2BY/edit#gid=0") 
   bins = st.sidebar.slider('Choose the number of bins', 4, 20, 1)
   csv_export_url = st.session_state.gs_URL.replace('/edit#gid=', '/export?format=csv&gid=')
   df = pd.read_csv(csv_export_url)
   rows, columns = df.shape
   numeric_columns = list(df.select_dtypes(['float', 'int']).columns)    
   total_columns = len(numeric_columns)
   for i in range(total_columns):
       colors = ['red', 'yellow', 'orange', 'blue', 'green', 'pink', 'cyan']
       x = df[numeric_columns[i]]
       mu = df[numeric_columns[i]].mean()
       sigma = df[numeric_columns[i]].std()
       #y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
       fig = plt.figure()
       c = randrange(7)
       plt.style.use('seaborn-whitegrid') 
       plt.hist(x, bins= bins, facecolor = colors[c], edgecolor='black', density = True)
      # plt.hist(x,bins = bins, alpha = 0.7, color = colors[c], density = True)
       #plt.plot(bins, y, '--', color ='black')
       plt.title("Distribution of  " + numeric_columns[i])
       plt.xlabel(numeric_columns[i])
       plt.ylabel("Proportion")
       st.pyplot(fig)
       #fig = plt.figure(figsize=(10, 4))
       #sns.histplot(data = df , x = numeric_columns[i], kde = True, color = colors[c])
       #st.pyplot(fig)

if page == "Box-plot":
   st.markdown("<h2 style='text-align: center;'>Box plot of the Data</h2>", unsafe_allow_html=True)
   st.markdown("This function gives a box plot of your dataset. Here the orange bar is the median, the circles represent the outliers")
   st.session_state.gs_URL = st.sidebar.text_input("Please enter the Google sheet URL having the cleaned data:","https://docs.google.com/spreadsheets/d/1MXlwsbKDWtlHcEuqJo_I2VJwuMH-ngdlMACNt2vu2BY/edit#gid=0")
   csv_export_url = st.session_state.gs_URL.replace('/edit#gid=', '/export?format=csv&gid=')
   df = pd.read_csv(csv_export_url)
   rows, columns = df.shape
   numeric_columns = list(df.select_dtypes(['float', 'int']).columns)    
   total_columns = len(numeric_columns)
   for i in range(total_columns):
       colors = ['red', 'yellow', 'orange', 'blue', 'green', 'pink', 'cyan']
       x = df[numeric_columns[i]]
       mu = df[numeric_columns[i]].mean()
       sigma = df[numeric_columns[i]].std()
       #y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
       fig = plt.figure()
       c = randrange(7)
       plt.boxplot(x)
       #plt.plot(bins, y, '--', color ='black')
       plt.title("Box Plot of  " + numeric_columns[i])
       plt.xlabel(numeric_columns[i])
       plt.ylabel("Values")
       st.pyplot(fig)

if page == "Q-Q plot":
   st.markdown("<h2 style='text-align: center;'>Normality of the Data</h2>", unsafe_allow_html=True)
   st.markdown("QQ plot gives you a graphical insight to if your data follows a normal distribution. A normal data follows perffectly along the y = x line. It also underlines the kurtosis score for the dataset, a kurtosis of score of 3 corresponds to the normal distribution. ")
   st.session_state.gs_URL = st.sidebar.text_input("Please enter the Google sheet URL having the cleaned data:","https://docs.google.com/spreadsheets/d/1MXlwsbKDWtlHcEuqJo_I2VJwuMH-ngdlMACNt2vu2BY/edit#gid=0") 
   csv_export_url = st.session_state.gs_URL.replace('/edit#gid=', '/export?format=csv&gid=')
   df = pd.read_csv(csv_export_url)
   rows, columns = df.shape
   numeric_columns = list(df.select_dtypes(['float', 'int']).columns)    
   total_columns = len(numeric_columns)
   for i in range(total_columns):
       colors = ['red', 'yellow', 'orange', 'blue', 'green', 'pink', 'cyan']
       x = df[numeric_columns[i]]
       mu = df[numeric_columns[i]].mean()
       sigma = df[numeric_columns[i]].std()
       #y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
       #fig = plt.figure()
       #c = randrange(7)
       fig = sm.qqplot(x,line='45',fit=True,dist=stats.norm)
       plt.title("QQ plot for  " + numeric_columns[i])
       plt.xlabel("Theoretical Quantiles")
       plt.ylabel("Sample Quantiles")
       st.pyplot(fig)
       st.markdown("The kurtosis score of this dataset is " +str(round(kurtosis(x),3)))

   
if page == "Linear Regression":
   st.markdown("<h2 style='text-align: center;'>Linear Regression of the Dataset</h2>", unsafe_allow_html=True)
   st.markdown("This function lets you choose the dependent and the independt variable and then runs a linear regression in it. It returns the equation along with the R square score and p value associated with regression. ")
   st.markdown("The closer the R2 value is to the one the better is your linear model")
   st.session_state.gs_URL = st.sidebar.text_input("Please enter the Google sheet URL having the cleaned data:","https://docs.google.com/spreadsheets/d/1MXlwsbKDWtlHcEuqJo_I2VJwuMH-ngdlMACNt2vu2BY/edit#gid=0") 
   csv_export_url = st.session_state.gs_URL.replace('/edit#gid=', '/export?format=csv&gid=')
   df = pd.read_csv(csv_export_url)
   rows, columns = df.shape
   numeric_columns = list(df.select_dtypes(['float', 'int']).columns)    
   total_columns = len(numeric_columns)
   dependent_variable = st.sidebar.radio("Which is the Dependent Variable?", numeric_columns)
   st.sidebar.write('Select independent variables:')
   independent_variable = st.sidebar.radio("Which is the Independent Variable?", numeric_columns)
   y = df[dependent_variable]
   x = df[independent_variable]
   slope, intercept, r, p, std_err = stats.linregress(x,y)
   def myfunc(x):
      return slope * x + intercept

   mymodel = list(map(myfunc, x))
   fig = plt.figure()
   plt.scatter(x, y)
   plt.plot(x, mymodel)
   plt.xlabel(independent_variable)
   plt.ylabel(dependent_variable)
   st.pyplot(fig)
   
   st.markdown("The Regression Equation is   " + dependent_variable + '   ' + '=' + str(round(slope,3)) + '*' + independent_variable + '    '+  '+' + str(round(intercept,3)))
   st.markdown("The R squred value is    " + str(round(pow(r,2),2)))
   st.markdown("The p squred value is    " + str(p))

if page == "Correlation Analysis":
   st.session_state.gs_URL = st.sidebar.text_input("Please enter the Google sheet URL having the cleaned data:","https://docs.google.com/spreadsheets/d/1MXlwsbKDWtlHcEuqJo_I2VJwuMH-ngdlMACNt2vu2BY/edit#gid=0") 
   csv_export_url = st.session_state.gs_URL.replace('/edit#gid=', '/export?format=csv&gid=')
   df = pd.read_csv(csv_export_url)
   rows, columns = df.shape
   numeric_columns = list(df.select_dtypes(['float', 'int']).columns)    
   total_columns = len(numeric_columns)
   corr_data = df[numeric_columns]
   cormat = corr_data.corr()
   cormat = round(cormat, 2)
   fig = plt.figure(figsize=(10, 4))
   sns.heatmap(cormat)
   st.pyplot(fig)

if page == "T-Test one sample":
   st.markdown("<h2 style='text-align: center;'>One sample t-test</h2>", unsafe_allow_html=True)
   st.markdown("This function lets you perform one sample t-test. All you have to do is type the population mean, significance level and if the test is one sided or two sided")
   st.session_state.gs_URL = st.sidebar.text_input("Please enter the Google sheet URL having the cleaned data:","https://docs.google.com/spreadsheets/d/1MXlwsbKDWtlHcEuqJo_I2VJwuMH-ngdlMACNt2vu2BY/edit#gid=0") 
   csv_export_url = st.session_state.gs_URL.replace('/edit#gid=', '/export?format=csv&gid=')
   df = pd.read_csv(csv_export_url)
   rows, columns = df.shape
   numeric_columns = list(df.select_dtypes(['float', 'int']).columns)    
   total_columns = len(numeric_columns)
   sample = st.sidebar.radio("Which is the Sample to be tested?", numeric_columns)
   pop_mean = st.sidebar.number_input("Enter the population mean")
   signif = st.sidebar.number_input("Enter the significance level")

   alternative = st.sidebar.radio("Check if the test is one sided or two sided", options = ['two-sided', 'less', 'greater'])
   sample = df[sample]
   t, p = stats.ttest_1samp(a=sample, popmean=pop_mean, alternative = alternative)
   st.markdown("The t statitics "+ str(round(t,3)) + " and the p value is " + str(p))
   if p > signif:
       st.markdown("As the p value is greater than that alpha so, We fail to reject the null hypotheis")
   else:
       st.markdown("As the p value is less than alpha we can reject the null hypothesis")

if page == "T-Test independent 2 sample":
    st.markdown("Under Development,if you have any query please mail at: akhilratanm@gmail.com")
if page == "Paired T Test":
    st.markdown("Under Development,if you have any query please mail at: akhilratanm@gmail.com")
if page == "Chi square test":
   st.markdown("Under Development,if you have any query please mail at: akhilratanm@gmail.com")
if page == "ANOVA":
    st.markdown("Under Development,if you have any query please mail at: akhilratanm@gmail.com")

   
