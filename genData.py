import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve
import matplotlib.pyplot as plt

def_seed=None

#use defualt seed if no seed was given
def get_seed(seed):
    if seed == None:
        seed = def_seed
    return seed

def generate_points(size=1000,seed=None,_min=-10000,_max=10001,data:any=[]):
    #valid input is an array of shape (2,n) for any positive n or dictionary/dataframe with x and y as keys
    seed = get_seed(seed)
    out = []
    rng = np.random.RandomState(seed)
    if len(data) != 0:
        if isinstance(data,(dict,pd.DataFrame)):
            out = np.array( [data["x"],data["y"]])
        else:
            out = np.array(data[:2])
        #slice out to the correct size
        if size != None and out.shape[1] > size:
            out = out[:,0:size]
        elif size != None and out.shape[1] < size:
            extra = rng.randint(_min,_max,(2,size-out.shape[1]))
            out = np.concatenate((out,extra),axis=1)   
    else:
        out = rng.randint(_min,_max,(2,size))
    return out
    


#generate a DataSet for Part A
def create_A(size=1000,raw_data:np.ndarray=[],seed=None,_min=-10000,_max=10001):
    seed = get_seed(seed)
    raw_data = generate_points(size,seed,_min,_max,raw_data)
    #true iff y>100 (implies y/100 > 1)
    target = np.ma.masked_greater(raw_data[1],100).mask
    #turn change from 0 and 1 to -1 and 1
    return pd.DataFrame({"x":raw_data[0]/100, "y":raw_data[1]/100, "target":target*2-1})

#generate a data set for Part B
def create_B(size=1000,raw_data:np.ndarray=[],seed=None,_min=-10000,_max=10001):
    seed = get_seed(seed)
    raw_data = generate_points(size,seed,_min,_max,raw_data)
    conditionArg = raw_data[0]**2+raw_data[1]**2
    #true iff 40000 <= x^2 + y^2 <= 90000 (implies 4 <= x^2 + y^2 <= 9)
    target = np.ma.masked_where((40000<=conditionArg) & (conditionArg<=90000),conditionArg).mask
    #turn change from 0 and 1 to -1 and 1
    return pd.DataFrame({"x":raw_data[0]/100, "y":raw_data[1]/100, "target":target*2-1})




#helper function to plot model results
def plot_model_results(y_true,y_pred,labels=None,name=None):
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    cm = ConfusionMatrixDisplay.from_predictions(y_true,y_pred,display_labels=labels,values_format="d",ax=ax1)
    
    fig.set_size_inches(20,10)
    num = cm.text_[0][0].get_text()
    txt = "True Negative"+ "\n" + num + "\n" + '{0:.3f}'.format(100*(int(num)/len(y_pred)))+"%"
    cm.text_[0][0].set_text(txt)
    
    num = cm.text_[1][0].get_text()
    txt = "False Negative"+ "\n" + num + "\n" + '{0:.3f}'.format(100*(int(num)/len(y_pred)))+"%"
    cm.text_[1][0].set_text(txt)

    num = cm.text_[0][1].get_text()
    txt = "False Positive"+ "\n" + num + "\n" + '{0:.3f}'.format(100*(int(num)/len(y_pred)))+"%"
    cm.text_[0][1].set_text(txt)

    num = cm.text_[1][1].get_text()
    txt = "True Positive"+ "\n" + num + "\n" + '{0:.3f}'.format(100*(int(num)/len(y_pred)))+"%"
    cm.text_[1][1].set_text(txt)
    
    RocCurveDisplay.from_predictions(y_true,np.random.choice([0,1],size=len(y_true)),ax=ax2,linestyle='--',name="Random classifier")
    
    
    #ax2.plot(falsePosRate, truePosRate, linestyle='--',label="random classifier")
    RocCurveDisplay.from_predictions(y_true,y_pred,ax=ax2)
    if name != None:
        plt.savefig(name)






