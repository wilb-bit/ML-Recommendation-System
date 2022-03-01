
def parameter_sweep():

    train_df = pd.read_csv('train.tsv', header=None, sep='\t')
    val_df = pd.read_csv('val.tsv', header=None, sep='\t')
    test_df = pd.read_csv('test.tsv', header=None, sep='\t')
    header = pd.read_csv('header.tsv', header=None, sep='\t')
    header = header.values.tolist()
    train_df.columns = header[0]
    val_df.columns = header[0]
    test_df.columns = header[0][0:5]

    # create data frame
    train_df = train_df.rename(columns={'Label': 'rating'})
    reader = Reader(rating_scale=(0, 5))
    train_10 = train_df[:84623]
    data10 = Dataset.load_from_df(train_10[["ReviewerID", "BeerID", "rating"]], reader)
    data = Dataset.load_from_df(train_df[["ReviewerID", "BeerID", "rating"]], reader)

    # val
    val_df = val_df.rename(columns={'Label': 'rating'})
    datav = Dataset.load_from_df(val_df[["ReviewerID", "BeerID", "rating"]], reader)

    trainset = data.build_full_trainset()
    NA, valset = train_test_split(datav, test_size=1.0)

    # kkn grid

    param_grid = {
        "k": [1, 3, 40, 100],
        "sim_options ": [{'name': 'cosine', 'user_based': False},
                         {'name': 'msd', 'user_based': False},
                         {'name': 'pearson', 'user_based': False},
                         {'name': 'pearson_baseline', 'user_based': False},
                         {'name': 'cosine', 'user_based': True},
                         {'name': 'msd', 'user_based': True},
                         {'name': 'pearson', 'user_based': True},
                         {'name': 'pearson_baseline', 'user_based': True}]}

    gs = GridSearchCV(KNNWithMeans, param_grid, measures=["mae"], cv=3)
    gs.fit(data10)
    print("kkn grid")
    print(gs.best_score["mae"])
    print(gs.best_params["mae"])

    # SVD
    param_grid = {
        "n_factors": [100, 300, 10],
        "n_epochs": [5, 20, 50],
        "lr_all": [0.001, 0.005, 0.01],
        "reg_all": [0.4, 0.01, 0.02],
        "random_state": [4]
    }
    gs = GridSearchCV(SVD, param_grid, measures=["mae"], cv=3)

    gs.fit(data10)

    print("SVD gridsearch: ")
    print(gs.best_score["mae"])
    print(gs.best_params["mae"])

    # SVDpp gridsearch
    param_grid = {
        "n_factors": [100, 300, 10],
        "n_epochs": [5, 20, 50],
        "lr_all": [0.001, 0.005, 0.01],
        "reg_all": [0.4, 0.01, 0.02],
        "random_state": [4]
    }
    gs = GridSearchCV(SVDpp, param_grid, measures=["rmse", "mae"], cv=3)

    gs.fit(data10)

    print("SVDpp gridsearch: ")
    print(gs.best_score["mae"])
    print(gs.best_params["mae"])

    ## CoClustering Gridsearch
    param_grid = {
        "n_cltr_u": [3, 2, 10],
        "n_cltr_i": [3, 2, 10],
        "n_epochs": [20, 5, 50],
        "random_state": [4]
    }
    gs = GridSearchCV(CoClustering, param_grid, measures=["rmse", "mae"], cv=3)

    gs.fit(data10)

    print("CoClustering gridsearch: ")
    print(gs.best_score["mae"])
    print(gs.best_params["mae"])

    # NMF gridsearch
    param_grid = {
        "n_factors": [15, 3, 40],
        "n_epochs": [50, 10, 150],
        "random_state": [4]
    }
    gs = GridSearchCV(NMF, param_grid, measures=["rmse", "mae"], cv=3)

    gs.fit(data10)


    print("NMF gridsearch: ")
    print(gs.best_score["mae"])
    print(gs.best_params["mae"])


    parameters = 0
    return parameters


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from surprise import Dataset
    from surprise import Reader

    from surprise import KNNWithMeans
    from surprise import accuracy
    from surprise.model_selection import train_test_split
    from surprise.model_selection import GridSearchCV
    from surprise import SVD
    from surprise import SVDpp
    from surprise import SlopeOne
    from surprise import CoClustering
    from surprise import NMF
    from surprise import AlgoBase



    sweep = False
    if sweep == True:
        parameters = parameter_sweep()

    else:
        train_df = pd.read_csv('train.tsv', header=None, sep='\t')
        val_df = pd.read_csv('val.tsv', header=None, sep='\t')
        test_df = pd.read_csv('test.tsv', header=None, sep='\t')
        header = pd.read_csv('header.tsv', header=None, sep='\t')
        header = header.values.tolist()
        train_df.columns = header[0]
        val_df.columns = header[0]
        test_df.columns = header[0][0:5]

        # create data frame
        train_df = train_df.rename(columns={'Label': 'rating'})
        reader = Reader(rating_scale=(0, 5))
        train_10 = train_df[:84623]
        data10 = Dataset.load_from_df(train_10[["ReviewerID", "BeerID", "rating"]], reader)
        data = Dataset.load_from_df(train_df[["ReviewerID", "BeerID", "rating"]], reader)

        # val
        val_df = val_df.rename(columns={'Label': 'rating'})
        datav = Dataset.load_from_df(val_df[["ReviewerID", "BeerID", "rating"]], reader)

        trainset = data.build_full_trainset()
        NA, valset = train_test_split(datav, test_size=1.0)

        # knn pred on val
        algo = KNNWithMeans(k=100, sim_options={'name': 'cosine', 'user_based': False})
        algo.fit(trainset)

        # run the trained model against the testset
        test_pred = algo.test(valset)
        # get MAE

        print("kkn val predict: ")
        accuracy.mae(test_pred, verbose=True)

        print("write KNNwithMeans to A3-2.tsv")
        test_df2 = test_df
        for index, row in test_df2.iterrows():
            # print(row["BeerID"])
            # print(row["ReviewerID"])
            est = algo.predict(row["ReviewerID"], row["BeerID"]).est
            # print(est)
            test_df2.loc[index, 'rating'] = est
        test_df2.head()
        submit = test_df2.drop(['ReviewerID', 'BeerID', 'BeerName', 'BeerType'], axis=1)
        submit.to_csv('A3-2.tsv', sep='\t', index=False, header=False)
        print("Done")

        # SVD pred on val set
        algo = SVD(
            n_factors=10,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.02,
            random_state=4)
        algo.fit(trainset)

        # run the trained model against the testset
        test_pred = algo.test(valset)
        # get MAE
        print("SVD val predict: ")
        accuracy.mae(test_pred, verbose=True)

        print("write SVD to A3-3.tsv")
        test_df2 = test_df
        for index, row in test_df2.iterrows():
            # print(row["BeerID"])
            # print(row["ReviewerID"])
            est = algo.predict(row["ReviewerID"], row["BeerID"]).est
            # print(est)
            test_df2.loc[index, 'rating'] = est
        test_df2.head()
        submit = test_df2.drop(['ReviewerID', 'BeerID', 'BeerName', 'BeerType'], axis=1)
        submit.to_csv('A3-3.tsv', sep='\t', index=False, header=False)
        print("Done")

        # SVDpp pred on val set
        algo = SVDpp(n_factors=10, n_epochs=20, lr_all=0.005, reg_all=0.02,
                     random_state=4)  # n_epochs = 10, lr_all = 0.005, reg_all = 0.4
        algo.fit(trainset)

        # run the trained model against the testset
        test_pred = algo.test(valset)

        # get MAE
        print("SVDpp val predict: ")
        accuracy.mae(test_pred, verbose=True)

        print("write SVDpp to A3-1.tsv")
        test_df2 = test_df
        for index, row in test_df2.iterrows():
            # print(row["BeerID"])
            # print(row["ReviewerID"])
            est = algo.predict(row["ReviewerID"], row["BeerID"]).est
            # print(est)
            test_df2.loc[index, 'rating'] = est
        test_df2.head()
        submit = test_df2.drop(['ReviewerID', 'BeerID', 'BeerName', 'BeerType'], axis=1)
        submit.to_csv('A3-1.tsv', sep='\t', index=False, header=False)
        print("Done")

        # SlopeOne
        algo = SlopeOne()
        algo.fit(trainset)

        # run the trained model against the testset
        test_pred = algo.test(valset)
        # get MAE
        print("SlopeOne val predict: ")
        accuracy.mae(test_pred, verbose=True)

        # CoClustering on val set
        algo = CoClustering(n_cltr_u=2, n_cltr_i=2, n_epochs=5, random_state=4)
        algo.fit(trainset)

        test_pred = algo.test(valset)
        # get MAE
        print("CoClustering val predict: ")
        accuracy.mae(test_pred, verbose=True)

        # NMF pred on val set
        algo = NMF(n_factors=3, n_epochs=150, random_state=4)
        algo.fit(trainset)

        test_pred = algo.test(valset)
        # get MAE
        print("NMF val predict: ")
        accuracy.mae(test_pred, verbose=True)

        # hybrid
        print("Hybrid: ")


        class HybridRegress(AlgoBase):

            def __init__(self, algorithms, weights, n_epochs, learning_rate, sim_options={}):
                AlgoBase.__init__(self)
                self.algorithms = algorithms
                self.weights = weights
                self.n_epochs = n_epochs
                self.learning_rate = learning_rate
                self.x = weights[0]
                self.y = weights[1]

            def fit(self, trainset):
                trainset, testset = train_test_split(data, test_size=0.3, random_state=10)
                AlgoBase.fit(self, trainset)

                for algorithm in self.algorithms:
                    algorithm.fit(trainset)

                predictions = []
                for algorithm in self.algorithms:
                    pred = algorithm.test(testset)
                    predictions.append(pred)

                a = accuracy.mae(predictions[0], verbose=False)
                b = accuracy.mae(predictions[1], verbose=False)

                print(a)
                print(b)
                print(self.weights)
                print(self.n_epochs)
                print(self.learning_rate)

                for epoch in range(self.n_epochs):
                    maeOld = ((a * self.x) + (b * self.y)) / (self.x + self.y)
                    newx = self.x - self.learning_rate * (self.y * (a - b)) / ((self.x + self.y) ** 2)
                    newy = self.y - self.learning_rate * (self.x * (b - a)) / ((self.x + self.y) ** 2)
                    maeNew = ((a * newx) + (b * newy)) / (newx + newy)
                    print("epoch: ", epoch, " maeOld: ", maeOld, " maeNew: ", maeNew, " newx: ", newx, " newy: ", newy)
                    if (maeOld - maeNew) < 0.00001:
                        break
                    if newx < 0:
                        self.x = 0
                        self.y = 1
                        break
                    if newy < 0:
                        self.x = 1
                        self.y = 0
                        break
                    self.x = newx / (newx + newy)
                    self.y = newy / (newx + newy)

                self.weights = [self.x, self.y]
                return self

            def estimate(self, u, i):

                sumScores = 0
                sumWeights = 0
                a = 0
                for idx in range(len(self.algorithms)):
                    if a == 0:
                        sumScores += self.algorithms[idx].estimate(u, i) * self.weights[idx]
                    else:
                        sumScores += self.algorithms[idx].estimate(u, i)[0] * self.weights[idx]
                    a = 1  # this is required if estimate output for second model isnt a single scalar.
                    # which is the case for KNNwithMeans (has dictionary)
                    sumWeights += self.weights[idx]

                return sumScores / sumWeights


        algoSVDpp = SVDpp(
            n_factors=10,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.02,
            random_state=4)

        algoSVD = SVD(
            n_factors=10,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.02,
            random_state=4)

        algoKNN = KNNWithMeans(k=100, sim_options={'name': 'cosine', 'user_based': False})

        algoSlope = SlopeOne()

        # Combine them
        Hybrid = HybridRegress([algoSVDpp, algoKNN], [0.5, 0.5], 20, 0.8)
        # train
        Hybrid.fit(data)
        # predict
        test_pred = Hybrid.test(valset)
        # get MAE
        print("Hybrid val predict: ")
        accuracy.mae(test_pred, verbose=True)

        # predict hybrid
        test_df['rating'] = 0
        print("write hybrid to A3-4.tsv")
        test_df2 = test_df
        for index, row in test_df2.iterrows():
            # print(row["BeerID"])
            # print(row["ReviewerID"])
            est = Hybrid.predict(row["ReviewerID"], row["BeerID"]).est
            # print(est)
            test_df2.loc[index, 'rating'] = est
        test_df2.head()
        submit = test_df2.drop(['ReviewerID', 'BeerID', 'BeerName', 'BeerType'], axis=1)
        submit.to_csv('A3-4.tsv', sep='\t', index=False, header=False)
        print("Done")



