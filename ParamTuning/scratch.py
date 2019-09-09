
    def run(self):
        self.initialRun()
        while(self.Bused <= self.budget or self.j >= self.nRaces):
            self.updateBJ()
            if (self.j > 2):
                self.grid = self.grid.tolist()
            elitesRanks = np.asarray(self.ranks)[self.eliteIndices]
            worstIndex = scores[eliteIndices].argmin()
            parent = np.asarray(np.std(elites, axis = 0))
            for i in range(self.getSampleSize() - len(elites)):
                self.grid.append(np.absolute(np.random.normal(parent,sds)).tolist())

            for row in range(len(self.grid)):
                for column in range(len(self.params)):
                    if (self.types[column] == True):
                        self.grid[row][column] = math.ceil(self.grid[row][column])
            self.grid = np.vstack(self.grid)

            # first we race the fuckers, getting a new elite
            f = self.features
            L = self.labels
            pars = {}
            results = []
            for row in range(len(self.grid)):
                for column in range(len(self.names)):
                    pars[self.names[column]] = (self.grid[row][column])
                clf = Classifier(self.method, pars)
                self.Bused += 1
                print("calculating:",(row + 1),"out of:", len(self.grid))
                scores = []
                for ids,(train_index, test_index) in enumerate(self.sampler.method.split(f)):
                    X_train, X_test = f[train_index], f[test_index]
                    y_train, y_test = L[train_index], L[test_index]
                    clf.train(features = X_train, labels = y_train)
                    # I would like to get the clf.predict method working
                    preds = clf.predict(X_test)
                    scores.append(self.metric(preds, y_test))
                res = sum(scores)/len(scores)
                print(res)
                results.append((row, res))
            #print(results)
            self.scores = np.asarray([x[1] for x in results])
            self.ranks = ss.rankdata(scores)
            self.eliteIndices = np.where(self.ranks >= np.mean(self.ranks) + np.std(self.ranks))
            self.grid = np.asarray(self.grid)[eliteIndices].tolist()
            self.bestScore = self.scores[self.scores.argmax()]
            self.bestTry =self.scores.argmax()
            for col in range(len(self.names)):
                self.bestPars[self.names[col]] = self.grid[self.bestTry][col]
            self.j += 1
