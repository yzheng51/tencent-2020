import torch


class SingleIdTrainDataset(torch.utils.data.Dataset):
    def __init__(self, X_id, y):
        self.X = torch.LongTensor(X_id)
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.size(0)


class SingleIdTestDataset(torch.utils.data.Dataset):
    def __init__(self, X_id):
        self.X = torch.LongTensor(X_id)

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.X.size(0)
