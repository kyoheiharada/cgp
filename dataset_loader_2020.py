import os
import numpy as np
import pandas as pd
import json


class DatasetLoader(object):
    def __init__(
            self,
            directory_path,
            x_label=None,
            aux_label=None,
            y_label=None,
            flatten=False,
            row=360):
        self._train, self._test = DatasetLoader.import_dataframe(
            directory_path, x_label, aux_label, y_label, flatten, row)

    @staticmethod
    def import_dataframe(
            directory_path,
            x_label,
            aux_label,
            y_label,
            flatten=False,
            row=720):
        _X_train = []
        _X_test = []
        _Aux_train = []
        _Aux_test = []
        _Y_train = []
        _Y_test = []

        # with open("train_test.json") as f:
        with open("train_test_30.json") as f:
            files = json.load(f)

        train_file = files["train"]
        test_file = files["test"]

        for i in train_file:
            train_x = np.load(
                directory_path + "/MainData/main{0:010d}.npy".format(i))
            auxdata = np.load(
                directory_path + "/AuxData/aux{0:010d}.npy".format(i))
            if len(train_x) != len(auxdata):
                if len(train_x) > len(auxdata):
                    train_x = train_x[:len(auxdata)]
                else:
                    auxdata = auxdata[:len(train_x)]

            if aux_label is not None:
                aux_x = auxdata[:, aux_label, :]
            else:
                aux_x = auxdata

            if y_label is not None:
                train_y = auxdata[:, y_label, :]
            else:
                train_y = auxdata

            for x in train_x:
                _X_train.append(x)
            for aux in aux_x:
                _Aux_train.append(aux)
            for y in train_y:
                _Y_train.append(y)

        for i in test_file:
            test_x = np.load(directory_path +
                             "/MainData/main{0:010d}.npy".format(i))
            auxdata = np.load(
                directory_path + "/AuxData/aux{0:010d}.npy".format(i))
            if len(test_x) != len(auxdata):
                if len(test_x) > len(auxdata):
                    test_x = test_x[:len(auxdata)]
                else:
                    auxdata = auxdata[:len(test_x)]

            if aux_label is not None:
                aux_x = auxdata[:, aux_label, :]
            else:
                aux_x = auxdata

            if y_label is not None:
                test_y = auxdata[:, y_label, :]
            else:
                test_y = auxdata

            for x in test_x:
                _X_test.append(x)
            for aux in aux_x:
                _Aux_test.append(aux)
            for y in test_y:
                _Y_test.append(y)

        # Train
        _X_train = np.asarray(_X_train)
        print(_X_train.shape)

        _Y_train = np.asarray(_Y_train)
        print(_Y_train.shape)

        _Aux_train = np.asarray(_Aux_train)
        print(_Aux_train.shape)

        if aux_label is not None:
            _Aux_train = _Aux_train.reshape((-1, len(aux_label), row))
        else:
            _Aux_train = _Aux_train.reshape((-1, 583, row))

        # Test
        _X_test = np.asarray(_X_test)
        print(_X_test.shape)

        _Y_test = np.asarray(_Y_test)
        print(_Y_test.shape)

        _Aux_test = np.asarray(_Aux_test)
        print(_Aux_test.shape)

        return DataSet(
            _X_train, _Y_train, _Aux_train), DataSet(
            _X_test, _Y_test, _Aux_test)

    @staticmethod
    def import_cross_data(
            directory_path,
            x_label,
            y_label,
            flatten=False,
            row=800,
            test_idx=0):
        _X_train = []
        _X_test = []
        _Aux_train = []
        _Aux_test = []
        _Y_train = []
        _Y_test = []

        cross1_file = pd.read_csv(os.getcwd() + "/cross1.csv", header=None)
        cross1_len = len(cross1_file.index)
        cross1_file = np.array(cross1_file)
        cross1_file = cross1_file.reshape(cross1_len)

        cross2_file = pd.read_csv(os.getcwd() + "/cross2.csv", header=None)
        cross2_len = len(cross2_file.index)
        cross2_file = np.array(cross2_file)
        cross2_file = cross2_file.reshape(cross2_len)

        cross3_file = pd.read_csv(os.getcwd() + "/cross3.csv", header=None)
        cross3_len = len(cross3_file.index)
        cross3_file = np.array(cross3_file)
        cross3_file = cross3_file.reshape(cross3_len)

        cross4_file = pd.read_csv(os.getcwd() + "/cross4.csv", header=None)
        cross4_len = len(cross4_file.index)
        cross4_file = np.array(cross4_file)
        cross4_file = cross4_file.reshape(cross4_len)

        cross5_file = pd.read_csv(os.getcwd() + "/cross5.csv", header=None)
        cross5_len = len(cross5_file.index)
        cross5_file = np.array(cross5_file)
        cross5_file = cross5_file.reshape(cross5_len)

        files = [cross1_file, cross2_file,
                 cross3_file, cross4_file, cross5_file]

        test_file = files[test_idx]
        cross_train = list(set(range(5)) - set([test_idx]))
        train_file = np.concatenate(
            (files[cross_train[0]], files[cross_train[1]], files[cross_train[2]], files[cross_train[3]]))
        print(train_file)

        x_label = list(map(lambda x: x - 1, x_label))
        if y_label is not None:
            y_label = list(map(lambda y: y - 1, y_label))

        for i in train_file:
            if row == 200:
                maindata = np.load(
                    directory_path + "/200slide/main{0:03d}.npy".format(i))
                auxdata = np.load(
                    directory_path + "/200slide/aux{0:03d}.npy".format(i))
            elif row == 800:
                maindata = np.load(
                    directory_path + "/main{0:03d}.npy".format(i))
                auxdata = np.load(directory_path + "/aux{0:03d}.npy".format(i))
            train_x = maindata[:, x_label, :]
            aux_x = auxdata[:, x_label, :]
            if y_label is not None:
                train_y = auxdata[:, y_label, :]
            else:
                train_y = auxdata
            if flatten is False:
                for x in train_x:
                    _X_train.append(x)
                for aux in aux_x:
                    _Aux_train.append(aux)
                for y in train_y:
                    _Y_train.append(y)
            else:
                for x in train_x:
                    _X_train.append(x.flatten())
                for aux in aux_x:
                    _Aux_train.append(aux.flatten())
                for y in train_y:
                    _Y_train.append(y.flatten())

        for i in test_file:
            if row == 200:
                maindata = np.load(
                    directory_path + "/200slide/main{0:03d}.npy".format(i))
                auxdata = np.load(
                    directory_path + "/200slide/aux{0:03d}.npy".format(i))
            elif row == 800:
                maindata = np.load(
                    directory_path + "/main{0:03d}.npy".format(i))
                auxdata = np.load(directory_path + "/aux{0:03d}.npy".format(i))
            test_x = maindata[:, x_label, :]
            aux_x = auxdata[:, x_label, :]
            if y_label is not None:
                test_y = auxdata[:, y_label, :]
            else:
                test_y = auxdata
            if flatten is False:
                for x in test_x:
                    _X_test.append(x)
                for aux in aux_x:
                    _Aux_test.append(aux)
                for y in test_y:
                    _Y_test.append(y)
            else:
                for x in test_x:
                    _X_test.append(x.flatten())
                for aux in aux_x:
                    _Aux_test.append(aux.flatten())
                for y in test_y:
                    _Y_test.append(y.flatten())

        # Train
        _X_train = np.asarray(_X_train)
        _X_train = _X_train.reshape((-1, len(x_label), row))
        _Y_train = np.asarray(_Y_train)

        if y_label is not None:
            _Y_train = _Y_train.reshape((-1, len(y_label), row))
        else:
            _Y_train = _Y_train.reshape((-1, 112, row))

        _Aux_train = np.asarray(_Aux_train)
        _Aux_train = _Aux_train.reshape((-1, len(x_label), row))

        # Test
        _X_test = np.asarray(_X_test)
        _X_test = _X_test.reshape((-1, len(x_label), row))
        _Y_test = np.asarray(_Y_test)

        if y_label is not None:
            _Y_test = _Y_test.reshape((-1, len(y_label), row))
        else:
            _Y_test = _Y_test.reshape((-1, 112, row))

        _Aux_test = np.asarray(_Aux_test)
        _Aux_test = _Aux_test.reshape((-1, len(x_label), row))

        return DataSet(
            _X_train, _Y_train, _Aux_train), DataSet(
            _X_test, _Y_test, _Aux_test)

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test


class DataSet(object):

    def __init__(self, X, Y, Aux=None):
        # print(X.shape)
        self._X = np.asarray(X)
        self._Y = np.asarray(Y)
        if Aux is not None:
            self._Aux = np.asarray(Aux)
        else:
            self._Aux = None

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def Aux(self):
        return self._Aux

    @property
    def length(self):
        return len(self._X)

    def print_information(self):
        print("****** Dataset Information ******")
        print("[Number of value] ", len(self._X))
        print("[Shape of X] ", self._X.shape)
        print("[Shape of Y] ", self._Y.shape)

    def shuffle(self):
        if self._Aux is None:
            _list = list(zip(self._X, self._Y))
            np.random.shuffle(_list)
            _X, _Y = zip(*_list)
            return DataSet(np.asarray(_X), np.asarray(_Y))
        else:
            _list = list(zip(self._X, self._Y, self._Aux))
            np.random.shuffle(_list)
            _X, _Y, _Aux = zip(*_list)
            return DataSet(np.asarray(_X), np.asarray(_Y), np.asarray(_Aux))

    def perm(self, start, end):
        if end > len(self._X):
            end = len(self._X)
        return DataSet(self._X[start:end], self._Y[start:end])

    def __call__(self, batch_size=20, shuffle=True):
        """

        `A generator which yields a batch. The batch is shuffled as default.
         バッチを返すジェネレータです。 デフォルトでバッチはシャッフルされます。

        Args:
            batch_size (int): batch size.
            shuffle (bool): If True, randomize batch datas.

        Yields:
            batch (ndarray[][][]): A batch data.

        """

        if batch_size < 1:
            raise ValueError("batch_size must be more than 1.")
        _data = self.shuffle() if shuffle else self

        for start in range(0, self.length, batch_size):
            permed = _data.perm(start, start + batch_size)
            yield permed


if __name__ == "__main__":
    dataset_loader = DatasetLoader(
        "C:/Users/KyoheiHarada/Desktop/sensor/data_200221")
    train = dataset_loader._train
    test = dataset_loader._test
    train.print_information()
    test.print_information()
