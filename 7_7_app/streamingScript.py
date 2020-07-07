import time #won't need this
#time.sleep(900)
import pandas as pd
import torch
import numpy as np
import yaml
import os
from hbpostgres import postgres, posgres_pull, postgres_insert, pull_GF, insert_GF, param_dic 
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import sched, time

#part of configuration file

#time.sleep(900) #we need to build up the system first before this even starts I put it on a 10 min delayed start. 

class global_forcast():
    
    def __init__(self, infile):
        df = pd.DataFrame()
        self.data = pd.DataFrame(infile)
        self.year = None
        self.day = None
        self.hour = None 
        self.min = None
        self.counter = 0
        self.get_data_from_database(self.data)
        #self.get_row_clock()
        #self.counting_by_12()
        #self.add_six_hours()

    def get_row_clock(self, data):
        """we are looping over the data frame to find the last date in the data. the model projects up three days. so we need to count from the last point"""
        # data = self.data
        temp_dataframe = data.iloc[[-1]] # we are just looking for the day and year of last column
        for row, column in temp_dataframe.iterrows():
            self.year = column[0]
            self.day = column[1]
            self.hour = column[2]
            self.min = column[3]
        return self.year, self.day, self.hour, self.min

    def concat(self, a, b, c, d):
        return eval(f"{a}{b}{c}{d}")

    def counting_by_12(self, data):
        """This is a complicated but important function it directly counts from the last row up to create the 
        new dates. I placed some year and day changes as contengencies but these may need to be updated for leap years"""
        # data = self.data
        lines_of_data_in_each_section = 12
        counter_end = 0
        #length_of_data = len(data)
        #number_of_times_data_goes_into_each_section = length_of_data/lines_of_data_in_each_section

        counter_end += (self.counter + lines_of_data_in_each_section) #we are looking at a 12 line section 12-24
        section_of_data = data.iloc[self.counter:counter_end,0:3] #For first row and some specific column i.e first three cols
        year = section_of_data.iloc[11,0] #last row and day column
        day = section_of_data.iloc[11,1] #last row and day column
        hour= section_of_data.iloc[11,2] #last row and day column
        #min = section_of_data.iloc[11,3] #last row and day column
        if hour == 0:
            day += 1
        elif day == 365:
            year += 1           #add some contegencies for the dates
                # counter = 0
        self.counter += lines_of_data_in_each_section # add the lines to the gloabal counter
        return year, day, hour


    def add_six_hours(self, df):
        "adding six hours one hour for each line in the dataframe"
        for row, index in df.iterrows():
            for x in range(1, 7):
                df['hour'][x-1] = index[2] + x
        return df

    def get_data_from_database(self, data):
        try:
        

            # min = 12

            #data = pd.read_csv('12T19Data.csv')
            data = data.replace([99999.9, 9999.99, 999.99, 99999.], np.nan)

            dRes = 5        #resolution of the data (fixed)
            os.chdir('/home/chris/Git_heartbeat/CG-Global-Forecast-Testing-master/app')
            model = torch.load('curWorkingModel.pt', map_location=torch.device('cpu')) #Load our model # load under cpu map_location=torch.device('cpu')
            with open ('curWorkingModel.yml', 'r') as f:
                    config = yaml.full_load(f)

            TSL = config['TSL']     #Number of elements in data corresponding to time
            hist = config['hist']   #Time history model is expecting (# samples)
            mRes = config['res']     #Resolution of data model expects (minutes)

            # We'll compute a (mRes/dRes)-pt rolling average over our data. This wa
            # y the data is treated the same as in model training
            # print(mRes)
            # print(dRes)

            data.iloc[:, TSL:] = data.iloc[:, TSL:].rolling(window=int(mRes/dRes)).mean()

            # Because of the way THIS model is trained, we can only output our 6Hr
            # forecasts once per hour, so we have to decimate this for now (the mod
            # el has only ever seen examples with the minute feature=0)

            data = data[::int(mRes/dRes)]

            # We'll just take a small number of samples (testing data)
            data = data[-250000:]   

            #data = data[1:]
                                

            # Now we're ready to start propagating data through the model
            # this loop is here to simulate streaming. In an offline setting, we
            # could propagate batches of data.
            for i in range(len(data)):
                model_input = torch.unsqueeze(torch.tensor(data[i-hist:i].values), dim=0)
                #Depending on the end system, casting may need to be changed
                model_input = model_input.type(torch.FloatTensor)

                if(torch.isnan(model_input).any()):
                    print("Missing Values. Cannot propagate data.")
                if(model_input.shape[1]==2):
                    try:
                        model_output = model(model_input)
                        px = model_output[:][:][:] #stripping tensor
                        px = px.data #pulling out of tensor and takeing offgan_f
                        px = px.numpy()[0] #pulling out of tensor
                        year, day, hour = self.counting_by_12(data)
                        minx = 0
                        #column_names = ['year','day', 'AE','AL','AU', 'SYM_H']
                        #df = pd.DataFrame(columns=column_names)
                        #print(minx)
                        #print(timestamp)
                        df = pd.DataFrame(({'year' : year, 'day': day, 'hour':hour, 'min':minx, 'AE': px[:, 0], 'AL': px[:, 1],'AU': px[:, 2], 'SYM_H': px[:, 3]}))
                        #print(df)
                        df = self.add_six_hours(df)
                        df['timestamp'] = df['year']+'_'+df['day']#self.concat(df['year', 'day', 'hour', 'minx'])
                        print(df)
                        df_np = df.values.astype(float) #converting types
                        
                        df_np = df_np[~np.isnan(df_np).any(axis=1)] #delete any dataframes with NAN
                        postgres_insert(insert_GF,df_np, param_dic) #dumping to database
                    except:
                        pass
            #s.enter(1, 1, global_forcast.get_data_from_database, (sc,)) #this is a timer to loop over full program every 60 seconds
        except IndexError: #exception for the delay in the database data getting pumped thorugh the pipeline.
            time.sleep(60) # we are looking to make sure data is there or the system will break. 
            #global_forcast.get_data_from_database(s) #if no daya is there at the time it will just attempt again in a few seconds
            global_forcast(infile = posgres_pull(pull_GF, param_dic))


# if __name__== "__main__":
#     import sched, time
#     s = sched.scheduler(time.time, time.sleep)
    # global_forcast.get_data_from_database(s)
    # s.enter(1, 1, global_forcast.get_data_from_database, (s,))
    # s.run()
if __name__== "__main__":     
    global_forcast(infile = posgres_pull(pull_GF, param_dic))

# if __name__== "__main__":
#     import sched, time
#     s = sched.scheduler(time.time, time.sleep)
#     # s.enter(1, 1, global_forcast.get_data_from_database, (s,))
#     # s.run()


