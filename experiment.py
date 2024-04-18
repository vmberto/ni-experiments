from dataset.cifar import get_cifar10_kfold_splits, get_cifar10_dataset, get_cifar10_corrupted
<<<<<<< Updated upstream
from experiments_config import CONFIGS, KFOLD_N_SPLITS, INPUT_SHAPE
from utils.metrics import write_fscore_result
from utils.consts import CORRUPTIONS_TYPES
from keras.callbacks import EarlyStopping
from utils.functions import filter_active
from utils.images import save_img_examples
from utils.logger import print_execution, print_evaluation
=======
from experiments_config import configs as all_experiments
from lib.metrics import write_fscore_result
from lib.consts import CORRUPTIONS_TYPES
from keras.callbacks import EarlyStopping
from lib.functions import filter_active
from sklearn.metrics import f1_score
from tqdm import tqdm
>>>>>>> Stashed changes
import multiprocessing


def experiment():
    x_train, y_train, x_test, y_test, splits = get_cifar10_kfold_splits(KFOLD_N_SPLITS)

    experiments_config = filter_active(CONFIGS)

    for index, config in tqdm(enumerate(experiments_config)):
        approach_name = config['approach_name']
        model_config = config['model']
        data_augmentation_layers = config['data_augmentation_layers']

        for fold_number, (train_index, val_index) in splits:
            model = model_config(input_shape=INPUT_SHAPE)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            print_execution(fold_number, approach_name, model.name)

            x_train_fold, y_train_fold = x_train[train_index], y_train[train_index]
            x_val_fold, y_val_fold = x_train[val_index], y_train[val_index]

            train_ds = get_cifar10_dataset(x_train_fold, y_train_fold, data_augmentation_layers)
            val_ds = get_cifar10_dataset(x_val_fold, y_val_fold)
            test_ds = get_cifar10_dataset(x_test, y_test)

            save_img_examples(train_ds, file_name=approach_name, img_num=5)

            _, training_time = model.fit(
                train_ds,
                val_dataset=val_ds,
                epochs=100,
                callbacks=[EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True, verbose=1)]
            )

            print_evaluation(fold_number, approach_name, model.name, f'in-distribution')

            f_score = model.predict(test_ds)
            write_fscore_result(
                'in-distribution',
                approach_name,
                model.name,
                f_score,
                training_time,
            )

            for corruption in CORRUPTIONS_TYPES:
                print_evaluation(fold_number, approach_name, model.name, f'in {corruption}')

                corrupted_dataset = get_cifar10_corrupted(corruption)
                f_score = model.predict(corrupted_dataset)

                write_fscore_result(
                    corruption,
                    approach_name,
                    model.name,
                    f_score,
                    training_time,
                )


if __name__ == "__main__":
    p = multiprocessing.Process(target=experiment)
    p.start()
    p.join()
    print("finished")
