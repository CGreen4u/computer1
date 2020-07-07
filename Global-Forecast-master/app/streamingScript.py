import pandas as pd
import torch
import numpy as np
import yaml
import time #won't need this
import hbpostgres

# Load in data

infile = posgres_pull(pull_GF)
data = pd.DataFrame(infile)

#clean data
data = data.replace([99999.9, 9999.99, 999.99, 99999.], np.nan)

dRes = 5        #resolution of the data (fixed)

model = torch.load('curWorkingModel.pt') #Load our model
with open ('curWorkingModel.yml', 'r') as f:
        config = yaml.full_load(f)

TSL = config['TSL']     #Number of elements in data corresponding to time
hist = config['hist']   #Time history model is expecting (# samples)
mRes = config['res']     #Resolution of data model expects (minutes)

# We'll compute a (mRes/dRes)-pt rolling average over our data. This wa
# y the data is treated the same as in model training

data.iloc[:, TSL:] = data.iloc[:, TSL:].rolling(window=int(mRes/dRes)).mean()

# Because of the way THIS model is trained, we can only output our 6Hr
# forecasts once per hour, so we have to decimate this for now (the mod
# el has only ever seen examples with the minute feature=0)

data = data[::int(mRes/dRes)]

# We'll just take a small number of samples (testing data)
data = data[-250000:]   
                       

# Now we're ready to start propagating data through the model
# this loop is here to simulate streaming. In an offline setting, we
# could propagate batches of data.

for i in range(len(data)):
        model_input = torch.unsqueeze(torch.tensor(data[i-hist:i].values), dim=0)
        #Depending on the end system, casting may need to be changed
        model_input = model_input.type(torch.FloatTensor)

        if(torch.isnan(model_input).any()):
            print("Missing Values. Cannot propagate data.")
        else:
            try:
                model_output = model(model_input)
                #do something with model output
                postgres_insert(insert_GF, model_output) #This code has a for loop in it to manage the data
                #print(model_output)
            except:
                print("Error Propagating Data")
        time.sleep(1)

