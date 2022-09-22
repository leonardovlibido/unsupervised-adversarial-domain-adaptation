from pathlib import Path
import pandas as pd
import shutil


def create_image_folder_dataset(
        images_dir_path: Path,
        labels_path: Path,
        target_image_folder_dir: Path,
        dataset_name: str
):
    target_dir = target_image_folder_dir / dataset_name
    if target_dir.is_dir():
        raise ValueError('target dir already exists')

    df_train = pd.read_csv(labels_path, delim_whitespace=True, header=None, names=['image_name', 'label'])
    df_train = df_train.set_index('image_name')

    for ind, image_path in enumerate(images_dir_path.glob('*')):
        image_name = image_path.name
        label = df_train.loc[image_name, 'label']

        src_path = image_path
        dst_path = target_image_folder_dir / dataset_name / str(label) / image_name
        dst_path.mkdir(parents=True, exist_ok=True)

        shutil.copy(src_path, dst_path)

        if ind % 2000 == 0:
            print(ind)
        #     print(ind, image_path, label)
        #     print('   ', src_path)
        #     print('   ', dst_path)


def main():

    images_dir_path = Path('/home/rdjordjevic/master/repos/domain-adaptation-codebase/datasets/mnistm/mnist_m/mnist_m_train')
    labels_path = Path('/home/rdjordjevic/master/repos/domain-adaptation-codebase/datasets/mnistm/mnist_m/mnist_m_train_labels.txt')
    target_image_folder_dir = Path('/home/rdjordjevic/master/repos/domain-adaptation-codebase/datasets/mnistm/image_folder')
    dataset_name = 'train'

    create_image_folder_dataset(images_dir_path, labels_path, target_image_folder_dir, dataset_name)

    # images_dir_path = Path('/home/rdjordjevic/master/repos/domain-adaptation-codebase/datasets/mnistm/mnist_m/mnist_m_test')
    # labels_path = Path('/home/rdjordjevic/master/repos/domain-adaptation-codebase/datasets/mnistm/mnist_m/mnist_m_test_labels.txt')
    # target_image_folder_dir = Path('/home/rdjordjevic/master/repos/domain-adaptation-codebase/datasets/mnistm/image_folder')
    # dataset_name = 'test'

    # create_image_folder_dataset(images_dir_path, labels_path, target_image_folder_dir, dataset_name)


if __name__ == "__main__":
    main()
