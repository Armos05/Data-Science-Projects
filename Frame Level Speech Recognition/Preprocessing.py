train_data = np.load("C:/bio/8th Sem/IDC 410 ML/Frame level Speech Recog/DATA/FINAL_train_data.npy", allow_pickle=True,encoding="latin1")
train_labels = np.load("C:/bio/8th Sem/IDC 410 ML/Frame level Speech Recog/DATA/train_labels.npy", allow_pickle=True,encoding="latin1")


dev_data = np.load("C:/bio/8th Sem/IDC 410 ML/Frame level Speech Recog/DATA/FINAL_dev_data.npy", allow_pickle=True,encoding="latin1")
dev_labels = np.load("C:/bio/8th Sem/IDC 410 ML/Frame level Speech Recog/DATA/dev_labels.npy", allow_pickle=True,encoding="latin1")

print("The shape of training data is", train_data.shape, "\n The shape of the train labels is", train_labels.shape, "\n The shape of the dev data is", dev_data.shape, "\n The shape of the Dev Labels is", dev_labels.shape)

train_data.shape, train_labels.shape, dev_data.shape, dev_labels.shape

# Distribution of Original Dataset
from matplotlib import pyplot as plt
  
a = np.array(train_labels)

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = [0, 5, 10, 15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140])
  
# Show plot
plt.show()

#Distribution of Subset Dataset  
k = np.array(train_labels[1:1200000])

# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(k, bins = [0, 5, 10, 15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140])
  
# Show plot
plt.show()

