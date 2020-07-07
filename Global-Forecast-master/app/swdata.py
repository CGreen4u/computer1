import numpy as np
import torch
import torch.utils.data as data
import copy
import pickle
import pdb

class swDataset(data.Dataset):
    """Solar wind Dataset"""

    def __init__(
        self,
        dataFile,
        split="train",
        valType="rand",
        trainPortion=0.7,
        normalize=True,
        portion=1,
        nVal=10,
    ):
        with open(dataFile, "rb") as f:
            x, y, tstamp, feats, outFeats = pickle.load(f)


        x, y = x[: int(portion * len(x))], y[: int(portion * len(y))]
        nsamples = np.shape(x)[0]
        nfeats = np.shape(x)[-1]
        self.tnames = outFeats

        valPortion = 0.15  # valPortion is 15% of the training data
        self.split = split

# Random validation split
        if valType == "rand":  # Decide whether to leave the data as sequential or randomly shuffled
            randidx = torch.randperm(nsamples)
            trainData = x[randidx[: int(trainPortion * len(randidx))]]
            trainLabels = y[randidx[: int(trainPortion * len(randidx))]]

            self.test_timestamp = tstamp[randidx[int(trainPortion * len(randidx)) :]]
            self.train_timestamp = tstamp[:np.int((1 - valPortion) * len(trainLabels))]
            self.val_timestamp = tstamp[np.int((1 - valPortion) * len(trainLabels)):]

            self.testData = x[randidx[int(trainPortion * len(randidx)) :]]
            self.testLabels = y[randidx[int(trainPortion * len(randidx)) :]]
            self.trainData = trainData[:np.int((1 - valPortion) * len(trainLabels))]
            self.trainLabels = trainLabels[:np.int((1 - valPortion) * len(trainLabels))]
            self.valData = trainData[np.int((1 - valPortion) * len(trainLabels)):]
            self.valLabels = trainLabels[np.int((1 - valPortion) * len(trainLabels)):]

# Seasonality - sensitive validation split
        elif valType == "szn":
            trainData = x[: int(trainPortion * nsamples)]
            trainLabels = y[: int(trainPortion * nsamples)]
            self.testData = x[int(trainPortion * nsamples) :]
            self.testLabels = y[int(trainPortion * nsamples) :]

            self.trainData = trainData[: int(nsamples * (1 / nVal) * (1 - valPortion))]
            self.trainLabels = trainLabels[:int(nsamples * (1 / nVal) * (1 - valPortion))]

            self.valData = trainData[int(nsamples * (1 / nVal) * (1 - valPortion)):int(nsamples * (1 / nVal))]

            self.valLabels = trainLabels[
                int(nsamples * (1 / nVal) * (1 - valPortion)) : int(nsamples * (1 / nVal))
            ]

            valDiff = int(nsamples * (1 / nVal)) - int(
                nsamples * (1 / nVal) * (1 - valPortion)
            )
            for i in range(1, nVal):
                start = np.int(i * nsamples * (1 / nVal))
                end = np.int((i + 1) * nsamples * (1 / nVal))
                self.trainData = torch.cat(
                    [self.trainData, trainData[start : end - valDiff]]
                )
                self.trainLabels = torch.cat(
                    [self.trainLabels, trainLabels[start : end - valDiff]]
                )
                self.valData = torch.cat([self.valData, trainData[end - valDiff : end]])
                self.valLabels = torch.cat(
                    [self.valLabels, trainLabels[end - valDiff : end]]
                )

            self.test_timestamp = tstamp[int(trainPortion * nsamples) :]
            self.train_timestamp = tstamp[: int(nsamples * (1 / nVal) * (1 - valPortion))]
            self.val_timestamp = tstamp[int(nsamples * (1 / nVal) * (1 - valPortion)):int(nsamples * (1 / nVal))]

# Sequential validation split
        elif valType == "seq":
            trainData = x[: int(trainPortion * nsamples)]
            trainLabels = y[: int(trainPortion * nsamples)]
            self.testData = x[int(trainPortion * nsamples) :]
            self.testLabels = y[int(trainPortion * nsamples) :]
            self.trainData = trainData[: np.int((1 - valPortion) * len(trainLabels))]
            self.trainLabels = trainLabels[: np.int((1 - valPortion) * len(trainLabels))]
            self.valData = trainData[np.int((1 - valPortion) * len(trainLabels)) :]
            self.valLabels = trainLabels[np.int((1 - valPortion) * len(trainLabels)) :]

        for i in range(nfeats):
            mean = torch.tensor([torch.mean(trainData[:, :, i])])
            std = torch.tensor([torch.std(trainData[:, :, i])])

            self.trainData[:, :, i] -= mean
            self.testData[:, :, i] -= mean
            self.valData[:, :, i] -= mean

            if not (std == 0):
                self.trainData[:, :, i] /= std
                self.testData[:, :, i] /= std
                self.valData[:, :, i] /= std
            
            self.test_timestamp = tstamp[int(trainPortion * nsamples) :]
            self.train_timestamp = tstamp[: np.int((1 - valPortion) * len(trainLabels))]
            self.val_timestamp = tstamp[np.int((1 - valPortion) * len(trainLabels)) :]

# Normalization of data
        if normalize:
            self.factors = torch.zeros(trainLabels.shape[-1])
            self.shifts = torch.zeros(trainLabels.shape[-1])
            for i in range(trainLabels.shape[-1]):
                self.shifts[i] = torch.mean(self.trainLabels[:, :, i])
                self.factors[i] = torch.std(self.trainLabels[:, :, i])

            self.testLabels -= self.shifts
            self.trainLabels -= self.shifts
            self.valLabels -= self.shifts

            self.testLabels /= self.factors
            self.valLabels /= self.factors
            self.trainLabels /= self.factors

# Torch Dataset Specific Functions
    def __getitem__(self, index):
        if self.split == "train":
            data, labels = self.trainData[index], self.trainLabels[index]
        elif self.split == "val":
            data, labels = self.valData[index], self.valLabels[index]
        elif self.split == "test":
            data, labels = self.testData[index], self.testLabels[index]
        return data, labels

    def __len__(self):
        if self.split == "train":
            return len(self.trainData)
        elif self.split == "val":
            return len(self.valData)
        elif self.split == "test":
            return len(self.testData)

    def reconstruct(self, predictions):
        return predictions * self.factors + self.shifts
