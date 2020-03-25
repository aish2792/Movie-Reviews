import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import norm

class LogisticRegression:

    def __init__(self,filename):
        """ Initialise the parameters """
        self.data = pd.read_csv(filename)
        self.dataset = self.data.iloc[:, 0:58].values
        np.random.shuffle(self.dataset)
        # np.random.seed(0)
        self.traindata, self.testdata = train_test_split(self.dataset, test_size=0.33, shuffle=False)
        self.X_train = self.traindata[:,0:57]
        self.X_test = self.testdata[:,0:57]
        self.Y_train = self.traindata[:,57]
        self.Y_test = self.testdata[:,57]
        self.X_train_std = self.X_train - np.mean(self.X_train) / np.std(self.X_train)
        self.X_test_std = self.X_test - np.mean(self.X_train) / np.std(self.X_train)
        self.data_std = np.concatenate((self.X_train_std, self.traindata[:,57:58]),axis=1)
        self.spamPriori, self.nonSpamPriori = self.calcPriori(self.data_std)
        self.spam = []
        self.notSpam = []
        self.notSpamArray = np.empty((0))
        self.spamArray = np.empty((0))
        self.train(self.data_std)


    def train(self, filedata):
        """To divide the data into two classes based on the output - 1(Spam) and 0(Not Spam) """

        for i in filedata:
            if i[i.shape[0]-1] == 0:
                self.notSpam.append(i)
            else:
                self.spam.append(i)
        """ Convert the classes into an array """
        self.notSpamArray = np.asarray(self.notSpam)
        self.spamArray = np.asarray(self.spam)

        """ Intialise a list to store the calculated conditional probabilities for each class """
        dist_notSpam = []
        dist_spam = []

        for i,j in zip(range(self.notSpamArray.shape[1]-1),range(1,self.notSpamArray.shape[1])):
            fitdatanotSpam = self.fit_distribution(self.notSpamArray[:,i:j])
            dist_notSpam.append(fitdatanotSpam)

        for i,j in zip(range(self.spamArray.shape[1]-1),range(1,self.spamArray.shape[1])):
            # print(self.spamArray[:,i:j])
            fitdataSpam = self.fit_distribution(self.spamArray[:,i:j])
            dist_spam.append(fitdataSpam)

        """ Total probability of each class with respect to the test data"""
        epsilon = 1e-200
        g_pdf_notSpam = 1
        g_pdf_Spam = 1
        for i,j in zip(range(self.X_test_std.shape[1]),range(1,self.X_test_std.shape[1]+1)):
            g_pdf_notSpam += np.log(self.pdf(self.X_test_std[:,i:j],dist_notSpam[i][0],dist_notSpam[i][1])+epsilon)
        total_notSpam = np.add(g_pdf_notSpam,self.nonSpamPriori)

        for i, j in zip(range(self.X_test_std.shape[1]), range(1, self.X_test_std.shape[1] + 1)):
            g_pdf_Spam += np.log(self.pdf(self.X_test_std[:, i:j], dist_spam[i][0], dist_spam[i][1]) + epsilon)
        total_Spam = np.add(g_pdf_Spam,self.spamPriori)

        """ List to store the predicted values """
        predicted = []
        for i in range(self.X_test_std.shape[0]):
            if total_Spam[i] > total_notSpam[i]:
                predicted.append(1)
            else:
                predicted.append(0)
        print("predicted is :",predicted)
        print("actual is :",self.Y_test)

        """ Variables to store True Positive, True Negative, False Positive and False Negative """
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for i, y in enumerate(self.Y_test):
            if y == 0 and predicted[i] == 0:
                TN += 1
            elif y == 1 and predicted[i] == 1:
                TP += 1
            elif y == 1 and predicted[i] == 0:
                FN += 1
            elif y == 0 and predicted[i] == 1:
                FP += 1

        """ Calculate precision, recall, F1_score and Accuracy """
        precision = (TP) / (TP + FP)
        Recall = (TP) / (TP + FN)
        F1_score = 2 * ((precision * Recall) / precision + Recall)
        Accuracy = (TP + TN) / (TP + TN + FP + FN)

        print(TP)
        print(FP)
        print(TN)
        print(FN)
        print("precision :", precision)
        print("Recall :", Recall)
        print("F1_score :", F1_score)
        print("Accuracy :", Accuracy)

    def calcPriori(self, data):
        """Calculate the apriori probabilities of the documents,
         that is the number of spam and non spam divided by all the data"""

        countSpam, countNonSpam = 0, 0
        fileContent = data
        for i in fileContent:
            if i[i.shape[0]-1] == 1:
                countSpam += 1
            else:
                countNonSpam += 1
        total  = countSpam+countNonSpam
        return np.log(countSpam/total), np.log(countNonSpam/total)

    def fit_distribution(self,data):
        """ Estimate the parameters - mean and the variance """
        mu = np.mean(data)
        sigma = np.std(data)
        variance = sigma**2
        dist = mu, variance
        return dist

    def pdf(self,x,mean, sd):
        """ Calculate the conditional probability using the Gaussian Distribution"""
        return (1 / np.sqrt(2 * 3.14 * sd)) * np.exp(-0.5* pow((x - mean), 2) /sd)


if __name__ == '__main__':
    lr = LogisticRegression("spambase.data")
    print(lr)
