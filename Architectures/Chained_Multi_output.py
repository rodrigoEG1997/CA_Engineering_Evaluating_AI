from modelling.modelling import *
from modelling.data_model import *
from embeddings import *
from Config import *


#Predict y2
def predict_y2(df: pd.DataFrame):
    #Add y
    df["y"] = df["y2"]
    #Delete all the rows that have null in y3 to train the model (Random forest cant manage null values)
    df = df.loc[(df["y2"] != '') & (~df["y2"].isna()),]
    
    #Prepare prediction
    X = get_tfidf_embd(df)
    data = Data(X, df)
    
    #Train model
    y2_model = train_model(data, 'Chained Multi-output y2')
    
    # #Make prediction  
    y2_model.predict(X)  
    y_predict = y2_model.predictions     
    df['y_predict'] = y_predict
    
    return df

#Predict y3
def predict_y3(df: pd.DataFrame):
    #Add y
    df["y"] = df["y3"]
    #Delete all the rows that have null in y3 to train the model (Random forest cant manage null values)
    df = df.loc[(df["y2"] != '') & (~df["y2"].isna()),]
    df = df.loc[(df["y3"] != '') & (~df["y3"].isna()),]
    
    #Prepare prediction
    X = get_tfidf_embd(df)
    data = Data(X, df)
    
    #Train model
    y3_model = train_model(data, 'Chained Multi-output y3')
    
    # #Make prediction  
    y3_model.predict(X)  
    y_predict = y3_model.predictions    
    df['y_predict'] = y_predict

    return df

#Predict y4
def predict_y4(df: pd.DataFrame):
    #Add y
    df["y"] = df["y4"]
    #Delete all the rows that have null in y3 to train the model (Random forest cant manage null values)
    df = df.loc[(df["y2"] != '') & (~df["y2"].isna()),]
    df = df.loc[(df["y3"] != '') & (~df["y3"].isna()),]
    df = df.loc[(df["y4"] != '') & (~df["y4"].isna()),]
    
    #Prepare prediction
    X = get_tfidf_embd(df)
    data = Data(X, df)
    
    #Train model
    y4_model = train_model(data, 'Chained Multi-output y4')
    
    #Make prediction  
    y4_model.predict(X)  
    y_predict = y4_model.predictions    
    df['y_predict'] = y_predict
    
    return df
    
#Start prediction in Chained Multi-Output Way
def chained_multioutput_prediction(df: pd.DataFrame):
    
    #Predict DataFrame
    p_df = df.drop(columns=['y2', 'y3', 'y4'])
    
    #Predict y2
    y2_df = p_df.copy(deep=True)
    y2_df['y2'] = df['y2']
    df_y2_prediction = predict_y2(y2_df)
    #Add prediction in DataFrame
    p_df['y2'] = p_df['ID'].map(df_y2_prediction.set_index('ID')['y_predict'])
    
    #Predict y3
    y3_df = p_df.copy(deep=True)
    y3_df['y3'] = df['y3']
    df_y3_prediction = predict_y3(y3_df)
    #Add prediction in DataFrame
    p_df['y3'] = p_df['ID'].map(df_y3_prediction.set_index('ID')['y_predict'])
    
    #Predict y4
    y4_df = p_df.copy(deep=True)
    y4_df['y4'] = df['y4']
    df_y4_prediction = predict_y4(y4_df)
    #Add prediction in DataFrame
    p_df['y4'] = p_df['ID'].map(df_y4_prediction.set_index('ID')['y_predict'])
    
    p_df.to_csv('chained_multioutput_prediction.csv', index=False) #to visualize
    
    
    
    