#USER Profiler

#Since we were not able to make an API for collecting the user information. As a substitute we have made a program that collects the Articles read by a user,
#The time taken and updates the user Matrix, and if the information about the new user is added it adds the new user to our database.

#User Profiler serves as a substitute of API. The user profiler is a mannual approach instead of automatic approach as used by API to generate clickstream data

#The code after every session i.e. after recommending 10 articles to users takes the following input from the user:

#    UserId of the Reader
#    Article ID of the articles read by the user
#    Approximate time spent by user to read a particular article


#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

#Example of a Gaussian Mixture Model
#The program helps in generating initial clickstream data which is used to make initial recommendations. 
#The program uses gaussian mixture model to compute the time spent by user to read a particular article on he web
mu = 0
sigma = 1
x1 =np.linspace(mu - 4*sigma, mu + 4*sigma, 100).tolist()
y1 = stats.norm.pdf(x, mu, sigma).tolist()
mu_1 = 2.3
sigma_1 = 1
x2 = [1, 3, 5]
x2 =np.linspace(mu_1 - 4*sigma_1, mu_1 + 4*sigma_1, 100).tolist()
y2 = stats.norm.pdf(x2, mu_1, sigma_1).tolist()
plt.plot(x1, y1)
plt.plot(x2, y2)

#Defining Gaussian Mixture Model
def Gaussian_Mixture(word_length):

  thorough_avg_time = word_length/4
  skimmming_avg_time = thorough_avg_time/5
  mu_2 = thorough_avg_time
  sigma_2 = 40
  x2 =np.linspace(mu_2 - 3*sigma_2, mu_2 + 3*sigma_2, 100).tolist()
  y2 = stats.norm.pdf(x1, mu_2, sigma_2).tolist()
  mu_3 = skimmming_avg_time
  sigma_3 = 20
  x3 =np.linspace(mu_3 - 3*sigma_3, mu_3 + 3*sigma_3, 100).tolist()
  y3 = stats.norm.pdf(x3, mu_3, sigma_3).tolist()
  y4 = 50*y2 + 40*y3
  random_time = random.choice(y4)

 return(random_time)

#Generate a Random Clickstream Data for Session1 of 1000 users


import random

new_users= 1000                                 #ENTER THE NUMBER OF NEW USERS you want to add


NEW_USER_PROFILE=[]
 
# UserId  SessionID  ArticleID  Time_Spent 
for j in range(new_users):
    UserID = []
    SessionID = []
    ArticleID = []
    Time_Spent = []
    Clicked = []
  
    
   
    for i in range(random.randint(1,10)):
        ArticleID.append(random.randint(0, 3035))
        UserID.append(j)
        SessionID.append(1)
        Time_Spent.append(int(10000*Gaussian_Mixture(random.randint(200,1000))))
        Clicked.append(random.choice(["yes", "no"]))

    profile_new=pd.DataFrame()
    profile_new['UserID']= UserID
    profile_new["SessionID"]= SessionID
    profile_new['ArticleID'] = ArticleID
    profile_new['Time_Spent']= Time_Spent
    profile_new['Clicked']= Clicked

    profile_new.sort_values(by=['UserID'], inplace=True,ascending=True)
    NEW_USER_PROFILE.append(profile_new)

#Example of a random Dataframe is against our 1000 users
NEW_USER_PROFILE[1]     # CHeck for any number

#Suppose User has opened an article with an specified "ID" and has spent "t" time on it. Then the User matrix will get updated
#The particular set of code operates after every session(iteration of recommendation) to ask users to manually input the information 
#about the articles read by the user and time spent on reading particular articles. Later the data collected is updated to initial 
#artificial generated clickstream data.

# Funtion to update clickstream database based on every click of the user.
import numbers

if __name__ == "__main__":

  user_ID = int(input('Enter the Id of the user' + "\n"))

 # DocumentID  
  DocumentID = input('Enter the ID of the articles read by the User with spaces'+ "\n")
  print("\n")
  ID_list = DocumentID.split()
  # print list
  print('Article ID list: ', ID_list)

  Time_taken = input('Enter the time taken per each article by the user seperated by spaces, if the user has not clicked on an article Enter 0' + "\n")
  print("\n")
  time_list = Time_taken.split()
  number = len(ID_list)
  if len(time_list) == number:
    # print list
    print('Time taken to read each article list: ', time_list)


    # updating the DataBase:
    all_user_list = [ ]

    for i in range(1000):
      if user_ID == int(NEW_USER_PROFILE[i].iloc[0]["UserID"]):

        all_user_list.append(i)

        print("the Previous User Profile was:" + "\n")
        print(NEW_USER_PROFILE[i])

        #to_append = [ ]
        x, y = (NEW_USER_PROFILE[i].shape)

        for j in range(number):

            if int(time_list[j]) > 0:
              
              new_entry = {'UserID': user_ID , 'SessionID': 2, 'ArticleID' : ID_list[j], 'Time_Spent' : time_list[j], 'Clicked': 'Yes'}

              NEW_USER_PROFILE[i].loc[len(NEW_USER_PROFILE[i])] = new_entry

            else:

              to_append.append([user_ID, 2, ID_list[j], time_list[j], 'No'])
              
              new_entry = {'UserID': user_ID , 'SessionID': 2, 'ArticleID' : ID_list[j], 'Time_Spent' : time_list[j], 'Clicked': 'No'}

              NEW_USER_PROFILE[i].loc[len(NEW_USER_PROFILE[i])] = new_entry

        print("The new profile becomes" + "\n")
        print(NEW_USER_PROFILE[i])
        print("\n" + "The DataFrame is updated")
        

    if user_ID not in all_user_list:

        df_new = pd.DataFrame(columns = ['UserID', 'SessionID', 'ArticleID', 'Time_Spent', 'Clicked'])

        print("This User data did not exist, Thus creating a new Dataframe for it")
        for j in range(number):

            if int(time_list[j]) > 0:
              
              df_new = df_new.append({'UserID' : user_ID, 'SessionID' : 2, 'ArticleID' : ID_list[j], 'Time_Spent': time_list[j], 'Clicked' : 'Yes'}, ignore_index = True)

              
            else:

              df_new = df_new.append({'UserID' : user_ID, 'SessionID' : 2, 'ArticleID' : ID_list[j], 'Time_Spent': time_list[j], 'Clicked' : 'No'}, ignore_index = True)

              
        print("The new is added to our database with profile" + "\n")
        print(df_new)  

        print("\n" + "The DataFrame is updated")
        NEW_USER_PROFILE = NEW_USER_PROFILE + df_new.values.tolist() 

  
  
  else:
    print('Error ! The number of Documents Donot match the number of time list')
    
    
