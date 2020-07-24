from tensorflow.keras.layers import Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from keras.utils.layer_utils import count_params
from sklearn.metrics import classification_report
import numpy as np

from DataGeneratorClass import DataGeneratorClass as DataGenerator
from swish import swish


class Classifier:
    """ Class to standardize the learning of a fine-tuned classifier.
        Allows for gradual defreeze while training and evaluation on
        test set wit sklearn-report.
    """

    def __init__(
        self,
        base_model,
        num_classes,
        batch_size=32,
        reg_dense=0.005,
        reg_out=0.005,
        save_path="models/trashnet",
    ):
        self.base_model = base_model
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.reg_dense = reg_dense
        self.reg_out = reg_out
        self.save_path = save_path

        self.classifier_model = self.build_classifier()

    def build_classifier(self):
        """ Build classifier
        """
        # Design model
        base_model = self.base_model

        # Adding dense layer
        x = base_model.output
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dense(
            32, activation="swish", kernel_regularizer=l1(self.reg_dense)
        )(x)

        classifier_model = Dense(
            self.num_classes,
            activation="softmax",
            kernel_regularizer=l1(self.reg_out),
        )(x)

        # Adamgrad optimizer
        opt = Adam(lr=0.001, amsgrad=True)

        # Combine VGG and extra layers
        classifier_model = Model(base_model.input, classifier_model)
        classifier_model.compile(
            optimizer=opt,
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )

        return classifier_model

    def get_generators(self, dfs, fraction, batch_size, params_generator):
        """ Returns generators used while training the classifier
        """
        # Generators
        data_train = DataGenerator(
            dfs["train"].sample(frac=fraction).reset_index(drop=True),
            batch_size=batch_size,
            shuffle=True,
            **params_generator,
        )
        data_val = DataGenerator(
            dfs["val"].reset_index(drop=True),
            batch_size=batch_size,
            shuffle=True,
            **params_generator,
        )
        data_test = DataGenerator(
            dfs["test"].reset_index(drop=True),
            batch_size=1,
            shuffle=False,
            **params_generator,
        )

        return data_train, data_val, data_test

    def get_callbacks(self, fraction):
        """ Returns callbacks used while training the classifier
        """
        checkpoint = ModelCheckpoint(
            self.save_path
            + "/NL_classifier/classifier_pr_"
            + str(fraction)
            + ".h5",
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
        )
        earlyStopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=0,
            mode="auto",
            restore_best_weights=True,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", patience=5, verbose=0, factor=0.5,
        )
        return checkpoint, earlyStopping, reduce_lr

    def unfreeze(self, num_of_unfrozen_layers, pr=False):
        """ Unfreeze layers of base_model
        """
        for layer in self.classifier_model.layers[:-num_of_unfrozen_layers]:
            layer.trainable = False
        for layer in self.classifier_model.layers[-num_of_unfrozen_layers:]:
            layer.trainable = True

        if pr:
            trainable_count = count_params(
                self.classifier_model.trainable_weights
            )
            non_trainable_count = count_params(
                self.classifier_model.non_trainable_weights
            )

            print(f"trainable parameters: {round(trainable_count/1e6,2)} M.")
            print(
                f"non-trainable parameters: {round(non_trainable_count/1e6,2)} M."
            )

    def train(
        self,
        data_train,
        data_val,
        fraction,
        nums_of_unfrozen_layers,
        lrs,
        epochs,
        verbose_epoch=0,
        verbose_cycle=1,
    ):
        """ Training of classifier.
            Two levels of verbosity:
                1. verbose_epoch: metrics every epoch
                2. verbose_cycle: metrics after every unfreezing cycle
        """
        classifier = self.classifier_model
        # Callbacks
        checkpoint, earlyStopping, reduce_lr = self.get_callbacks(fraction)

        # Train with gradual defreeze
        for i, (num_of_unfrozen_layers, lr, epoch) in enumerate(
            zip(nums_of_unfrozen_layers, lrs, epochs)
        ):
            # Unfreeze
            self.unfreeze(num_of_unfrozen_layers)

            # Change learning rate
            classifier_model = self.classifier_model
            K.set_value(classifier_model.optimizer.learning_rate, lr)

            # Fit
            history = classifier_model.fit(
                data_train,
                epochs=epoch,
                verbose=verbose_epoch,
                validation_data=data_val,
                callbacks=[earlyStopping, reduce_lr]
                # callbacks = [checkpoint, earlyStopping, reduce_lr]
            )
            if verbose_cycle:
                print(
                    f"CYCLE {i}: num_of_unfrozen_layers: {num_of_unfrozen_layers}"
                    + f" - epochs: {epoch} - lr: {lr:.1e}",
                    end=" | ",
                )
                print(
                    f"Training Loss at end of cycle: {history.history['loss'][-1]:.2f}"
                    + f"- Training Acc: {np.max(history.history['categorical_accuracy']):.2f}"
                    + f"- Validation Acc: {np.max(history.history['val_categorical_accuracy']):.2f}"
                )
            if np.isnan(history.history["val_loss"]).any():
                print("Learning diverged, stopped.")
                break

    def evaluate_on_test(self, df_test, data_test, class_labels):
        """ Evaluation of the trained fine-tuned classifier.
            Minimum accuracy of 0.3 is imposed to avoid evaluation on diverged model.
        """
        class_test = self.classifier_model.predict(
            data_test, steps=len(df_test.index)
        )
        # Test
        y_true_test = []
        y_pred_test = []
        for y_t, y_p in zip(data_test, class_test):
            y_true_test.append(np.argmax(y_t[1]))
            y_pred_test.append(np.argmax(y_p))

        acc = np.mean(np.equal(y_true_test, y_pred_test))
        if 0.3 < acc:
            classification_report_test = classification_report(
                y_true_test,
                y_pred_test,
                labels=list(range(0, len(class_labels))),
                target_names=class_labels,
            )
        else:
            classification_report_test = None

        return acc, classification_report_test
