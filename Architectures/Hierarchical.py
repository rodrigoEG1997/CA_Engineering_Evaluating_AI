from modelling.modelling import *
from modelling.data_model import *
from embeddings import *
from Config import *

#Predict y2
def predict_y2(df: pd.DataFrame, eval_df):
    
    #Put togheter all the predictions
    final_df = pd.DataFrame()
    
    #Add y
    df["y"] = df["y2"]
    #Delete all the rows that have null in y3 to train the model (Random forest cant manage null values)
    df = df.loc[(df["y2"] != '') & (~df["y2"].isna()),]
    
    #Split DataFrame by 'y1'
    grouped_df = df.groupby("y1")
    
    #Predict all groups
    for name, group_df in grouped_df:
        if len(group_df) > 2:
            X = get_tfidf_embd(group_df)
            data = Data(X, group_df)
            model, eval_df = train_hierachical_model(data, eval_df, 'y2', name)
            
            model.predict(X)  
            y_predict = model.predictions    
            group_df['y_predict'] = y_predict
            
            #Add prediction into final DataFrame
            final_df = pd.concat([final_df, group_df], ignore_index=True)
        
    return final_df, eval_df

#Predict y3
def predict_y3(df: pd.DataFrame, eval_df):
    
    #Put togheter all the predictions
    final_df = pd.DataFrame()
    
    #Add y
    df["y"] = df["y3"]
    #Delete all the rows that have null in y3 to train the model (Random forest cant manage null values)
    df = df.loc[(df["y2"] != '') & (~df["y2"].isna()),]
    df = df.loc[(df["y3"] != '') & (~df["y3"].isna()),]
    
    #Split DataFrame by 'y1'
    grouped_df = df.groupby("y2")
    
    #Predict all groups
    for name, group_df in grouped_df:
        if len(group_df) > 2:
            X = get_tfidf_embd(group_df)
            data = Data(X, group_df)
            model, eval_df = train_hierachical_model(data, eval_df, 'y3', name)
            
            model.predict(X)  
            y_predict = model.predictions    
            group_df['y_predict'] = y_predict
            
            #Add prediction into final DataFrame
            final_df = pd.concat([final_df, group_df], ignore_index=True)
        
    return final_df, eval_df

#Predict y4
def predict_y4(df: pd.DataFrame, eval_df):
    #Put togheter all the predictions
    final_df = pd.DataFrame()
    
    #Add y
    df["y"] = df["y4"]
    #Delete all the rows that have null in y3 to train the model (Random forest cant manage null values)
    df = df.loc[(df["y2"] != '') & (~df["y2"].isna()),]
    df = df.loc[(df["y3"] != '') & (~df["y3"].isna()),]
    df = df.loc[(df["y4"] != '') & (~df["y4"].isna()),]
    
    #Split DataFrame by 'y1'
    grouped_df = df.groupby("y3")
    
    #Predict all groups
    for name, group_df in grouped_df:
        if len(group_df) > 2:
            X = get_tfidf_embd(group_df)
            data = Data(X, group_df)
            model, eval_df = train_hierachical_model(data, eval_df, 'y4', name)
            
            model.predict(X)  
            y_predict = model.predictions    
            group_df['y_predict'] = y_predict
            
            #Add prediction into final DataFrame
            final_df = pd.concat([final_df, group_df], ignore_index=True)
        
    return final_df, eval_df

def hierarchical_prediction(df: pd.DataFrame):
    
    #Create a Evaluation table
    eval_df = []
    
    p_df = df.drop(columns=['y2', 'y3', 'y4'])
    
    #Predict y2
    y2_df = p_df.copy(deep=True)
    y2_df['y2'] = df['y2']
    df_y2_prediction, eval_df = predict_y2(y2_df, eval_df)
    #Add prediction in DataFrame
    p_df['y2'] = p_df['ID'].map(df_y2_prediction.set_index('ID')['y_predict'])
    
    #Predict y3
    y3_df = p_df.copy(deep=True)
    y3_df['y3'] = df['y3']
    df_y3_prediction, eval_df = predict_y3(y3_df, eval_df)
    #Add prediction in DataFrame
    p_df['y3'] = p_df['ID'].map(df_y3_prediction.set_index('ID')['y_predict'])
    
    #Predict y4
    y4_df = p_df.copy(deep=True)
    y4_df['y4'] = df['y4']
    df_y4_prediction, eval_df = predict_y4(y4_df, eval_df)
    #Add prediction in DataFrame
    p_df['y4'] = p_df['ID'].map(df_y4_prediction.set_index('ID')['y_predict'])
    
    evaluation_df = pd.DataFrame(eval_df)
    evaluation_df.to_csv('hierarchical_evaluation.csv', index=False)
    p_df.to_csv('hierarchical_prediction.csv', index=False)
    
    