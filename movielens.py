from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly, KNNBasic
from surprise import accuracy
from surprise.model_selection import KFold

# https://www.kaggle.com/jneupane12/movielens/download
reader = Reader(line_format = 'user item rating timestamp', sep = ',', skip_lines = 1)
data = Dataset.load_from_file('./ratings.csv', reader = reader)
train_set = data.build_full_trainset()

# Baseline SGD default reg = 0.02, learning_rate = 0.005, n_epochs = 20
# bsl_options = {'method' : 'sgd', 'n_epochs' : 5}

# Baseline ALS default reg_i = 20, reg_u = 15, n_epochs = 10
bsl_options = {'method' : 'als', 'n_epochs' : 5, 'reg_u' : 12, 'reg_i' : 5}
algo = BaselineOnly(bsl_options =  bsl_options)
kf = KFold(n_splits = 3)

for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    accuracy.rmse(predictions, verbose = True)

uid = str(196)
iid = str(302)
pred = algo.predict(uid, iid, r_ui = 4, verbose = True)