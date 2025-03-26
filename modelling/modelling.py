from model.randomforest import RandomForest
from sklearn.metrics import classification_report
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def train_model(data, title):
    print("Model of " + str(title))
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)
    return model

def train_hierachical_model(data, eval_df, y, name):
    #Train
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    y_predict = model.predictions   
    #Evaluation
    report = classification_report(data.y_test, y_predict, output_dict=True)
    
    eval_df.append({'y': y, 
                    'name': name,
                    'precision': round(report['macro avg']['precision'], 2),
                    'recall': round(report['macro avg']['recall'], 2),
                    'f1-score': round(report['macro avg']['f1-score'], 2),
                    'support': round(report['macro avg']['support'], 2),
                    'accuracy': round(report['accuracy'], 2)})
    
    return model, eval_df


def model_evaluate(model, data):
    model.print_results(data)