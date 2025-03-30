from modelling.modelling import *
from modelling.data_model import *
from embeddings import *
from Config import *

class Hierarchical():
    def __init__(self, c_grouped, c_types, df: pd.DataFrame) -> None:
        
        self.predic_df = pd.DataFrame()
        self.origin_df = df
        self.grouped = c_grouped
        self.c_types = c_types
        self.eval_df = []
        
    def _predict_y(self, g, y, df, eval_df):

        #Put togheter all the predictions
        final_df = pd.DataFrame()
        
        #Add y
        df["y"] = df[y]
        
        #Delete all the rows that have null 

        for ty in self.c_types:
            df = df.loc[(df[ty] != '') & (~df[ty].isna()),]
            if ty == y:
                break  

        #Split DataFrame 
        grouped_df = df.groupby(g)

        #Predict all groups
        for name, group_df in grouped_df:
            if len(group_df) > 2:
                X = get_tfidf_embd(group_df)
                data = Data(X, group_df)
                model, eval_df = train_hierachical_model(data, eval_df, y, name)
                model.predict(X)  
                y_predict = model.predictions    
                group_df['y_predict'] = y_predict
 
                #Add prediction into final DataFrame
                final_df = pd.concat([final_df, group_df], ignore_index=True)

        return final_df, eval_df
            
        

        
    def prediction(self):
        #Get predict df (Delete y's)
        self.predic_df = self.origin_df.drop(columns=self.c_types)

        #Iterate predictions
        for num in range(len(self.c_types)):
            y = self.c_types[num]
            g = self.grouped[num]
            
            temp_df = self.predic_df.copy(deep=True)
            temp_df[y] = self.origin_df[y]
            temp_df_prediction, self.eval_df = self._predict_y(g, y, temp_df, self.eval_df)
            #Add prediction in DataFrame
            self.predic_df[y] = self.predic_df['ID'].map(temp_df_prediction.set_index('ID')['y_predict'])
    
    def get_prediction_csv(self):
        self.predic_df.to_csv('hierarchical_prediction.csv', index=False)
        
    def get_results_csv(self):
        evaluation_df = pd.DataFrame(self.eval_df)
        evaluation_df.to_csv('hierarchical_results.csv', index=False)