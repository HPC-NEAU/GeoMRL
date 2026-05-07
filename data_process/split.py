import pickle

from torch.utils.data import Subset

__all__ = [
    'ScaffoldSplitter',
    'RandomScaffoldSplitter',
    'RandomSplitter'
]

from utils.userconfig_util import get_split_dir


def create_splitter(split_type, seed):
    """Return a splitter according to the ``split_type``"""
    if split_type == 'scaffold':
        splitter = ScaffoldSplitter()
    elif split_type == 'random_scaffold':
        splitter = RandomScaffoldSplitter(seed)
    elif split_type == 'random':
        splitter = RandomScaffoldSplitter(seed)
    else:
        raise ValueError('%s not supported' % split_type)
    return splitter


class Splitter(object):
    """
    The abstract class of splitters which split up dataset into train/valid/test
    subsets.
    """

    def __init__(self):
        super(Splitter, self).__init__()
 


class ScaffoldSplitter(Splitter):

    def __init__(self):
        super(ScaffoldSplitter, self).__init__()

    @staticmethod
    def split(dataset, task_name):
        split_idx = pickle.load(open(get_split_dir() / f'scaffold/{task_name}.pkl', 'rb'))
        train_dataset = Subset(dataset, split_idx['train_idx'])
        valid_dataset = Subset(dataset, split_idx['valid_idx'])
        test_dataset = Subset(dataset, split_idx['test_idx'])
        return train_dataset, valid_dataset, test_dataset


class RandomScaffoldSplitter(Splitter):

    def __init__(self, seed):
        super(RandomScaffoldSplitter, self).__init__()
        self.seed = seed

    def split(self, dataset, task_name):
        seed_ = self.seed
        split_idx = pickle.load(open(get_split_dir() / f'random_scaffold/{task_name}/{task_name}_{seed_}.pkl', 'rb'))
        train_dataset = Subset(dataset, split_idx['train_idx'])
        valid_dataset = Subset(dataset, split_idx['valid_idx'])
        test_dataset = Subset(dataset, split_idx['test_idx'])
        return train_dataset, valid_dataset, test_dataset
class RandomSplitter(Splitter):
    def __init__(self, seed):
        super(RandomSplitter, self).__init__()
        self.seed = seed

    def split(self, dataset, task_name=None):
        """
        Randomly split the dataset into train, validation, and test subsets.

        Args:
            dataset (Dataset): The dataset to be split.
            task_name (str, optional): The name of the task. Defaults to None.

        Returns:
            tuple: (train_dataset, valid_dataset, test_dataset)
        """
        # 设置随机种子
        np.random.seed(self.seed)

        train_size = 0.8
        val_size = 0.1
        test_size = 0.1

        # 计算各个子集的大小
        total_size = len(dataset)
        train_length = int(train_size * total_size)
        val_length = int(val_size * total_size)
        test_length = total_size - train_length - val_length

        # 随机划分数据集
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset,
            [train_length, val_length, test_length],
            generator=torch.Generator().manual_seed(self.seed)
        )

        return train_dataset, valid_dataset, test_dataset        