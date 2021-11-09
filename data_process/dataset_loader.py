from fastNLP.io import ConllLoader


class DataSetLoader(ConllLoader):
    def __init__(self):
        headers = [
            'words', 'target',
        ]
        super(DataSetLoader, self).__init__(headers=headers)


class AnnLoader(ConllLoader):
    def __init__(self):
        headers = [
            'ids', 'types', 'start', 'end', 'words'
        ]
        super(AnnLoader, self).__init__(headers=headers)


if __name__ == '__main__':
    train_path = 'data.conll'
    loader = DataSetLoader()
    train_data = loader.load(train_path)
    print(train_data)
