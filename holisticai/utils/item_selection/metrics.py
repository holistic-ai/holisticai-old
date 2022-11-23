from scipy import spatial
import numpy as np

def gini(array):
    """
    Description
    -----------
    Calculate the Gini coefficient of a numpy array.
    
    Parameters:
    ----------
    array : matrix-like
        ranking matrix
    """
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

def get_top_n_list_for_users(P,Q,training,topn):
    eR = np.dot(P,Q)
    Pl1_MF=[0]
    for i,row in enumerate(np.dot(P,P.T)):
        Pl1_MF.append(np.sqrt(row[i]))
    Pl1_MF = Pl1_MF[1:]
        
    Ql1_MF=[0]
    for i,row in enumerate(np.dot(Q.T,Q)):
        Ql1_MF.append(np.sqrt(row[i]))
    Ql1_MF = Ql1_MF[1:]
        
    for i in range(eR.shape[0]):
        for j in range(eR.shape[1]):
            eR[i][j]= eR[i][j]/(Pl1_MF[i]*Ql1_MF[j])
    recommended_items = [0]     
    training_mask = training.values>0
    eR_new = eR*(~training_mask)

    for i in range(training.shape[0]):
            #for j in range(len(eR[0])):
        sorted_ind = np.argsort(eR_new[i])[::-1]  # sorting the calculated ratings in the descending order for user i
        top_5 = sorted_ind[0:topn]  # get the 5 items with highest predicted ratings # can change to top 10, so on if you want
        recommended_items.append(top_5)
    return recommended_items[1:]
    
class EfficacyMetrics:
    def __call__(self, r_mat, pred):
        mask = r_mat>0
        rmse= np.sqrt(np.sum(np.power(pred-r_mat,2)*mask)/np.sum(mask))
        mae= np.sum(np.abs(pred-r_mat)*mask)/np.sum(mask)
        return {'rmse':rmse, 'mae':mae}