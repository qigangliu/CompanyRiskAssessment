import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from catboost import CatBoostClassifier, Pool

# load dataset
df = pd.read_csv("../../data/GCN&RGCN/company_feature.csv")
target=df.iloc[:,-1]
data=df.iloc[:,:-1]

# split the dataset
train_x, test_x, train_y, test_y = train_test_split(data,target,test_size=0.3,random_state=20)

# train the model
model = CatBoostClassifier(
    depth=10, 
    learning_rate=0.5,
    n_estimators = 100,
    custom_loss='CrossEntropy',
    random_seed=42,
    logging_level='Silent'
)
model.fit(
    train_x, train_y,
    plot=True
)
test_data = catboost_pool = Pool(train_x, 
                                 train_y)
test_pool = Pool(test_x)

y_pred = model.predict(test_pool)
preds_proba = [l[1] for l in model.predict_proba(test_pool)]

print ('Accuracy: %.4f' % metrics.accuracy_score(test_y,y_pred))
print ('Precesion: %.4f' %metrics.precision_score(test_y,y_pred))
print ('Recall: %.4f' % metrics.recall_score(test_y,y_pred))
print ('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))