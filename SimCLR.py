from tensorflow.keras.layers import Input, Flatten, Dense
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

from datetime import datetime

from SoftmaxCosineSim import SoftmaxCosineSim
from Classifier import Classifier
from swish import swish


class SimCLR:
    """
    SimCLR-class contains among others a SimCLR keras-model
    The SimCLR_model has
        - (2 * batch_size) inputs of shape = (feat_dim)
        - base_model which is stored independently to evaluate its feature quality
        - flatten_layer
        - projection_head
        - 1 output = matrix of shape (batch_size x 4.batch_size)
    """

    def __init__(
        self,
        base_model,
        input_shape,
        batch_size,
        feat_dim,
        feat_dims_ph,
        num_of_unfrozen_layers,
        ph_regul=0.005,
        lr=1e-4,
        loss="categorical_crossentropy",
        save_path="models/trashnet",
        r=1,
    ):
        self.base_model = base_model
        self.input_sh = input_shape
        self.batch_size = batch_size
        self.feat_dim = feat_dim
        self.feat_dims_ph = feat_dims_ph
        self.num_layers_ph = len(feat_dims_ph)
        self.num_of_unfrozen_layers = num_of_unfrozen_layers
        self.ph_regul = ph_regul
        self.lr = lr
        self.optimizer = Adam(lr, amsgrad=True)
        self.loss = loss
        self.save_path = save_path
        self.r = r

        # Different layers around base_model
        self.flatten_layer = Flatten()
        self.soft_cos_sim = SoftmaxCosineSim(
            batch_size=self.batch_size, feat_dim=self.feat_dim
        )
        # Projection head
        self.ph_l = []
        for j in range(self.num_layers_ph):
            if j < self.num_layers_ph - 1:
                self.ph_l.append(
                    Dense(
                        feat_dims_ph[j],
                        activation="swish",
                        kernel_regularizer=l1(ph_regul),
                    )
                )
            else:
                self.ph_l.append(
                    Dense(feat_dims_ph[j], kernel_regularizer=l1(ph_regul))
                )

        self.SimCLR_model = self.build_model()

    def build_model(self):
        """ Building SimCLR_model
        """

        for layer in self.base_model.layers[: -self.num_of_unfrozen_layers]:
            layer.trainable = False
        for layer in self.base_model.layers[-self.num_of_unfrozen_layers :]:
            layer.trainable = True

        i = []  # Inputs (# = 2 x batch_size)
        f_x = []  # Output base_model
        h = []  # Flattened feature representation
        g = []  # Projection head
        for j in range(self.num_layers_ph):
            g.append([])

        # Getting learnable building blocks
        base_model = self.base_model
        ph_l = []
        for j in range(self.num_layers_ph):
            ph_l.append(self.ph_l[j])

        for index in range(2 * self.batch_size):
            i.append(Input(shape=self.input_sh))
            f_x.append(base_model(i[index]))
            h.append(self.flatten_layer(f_x[index]))
            for j in range(self.num_layers_ph):
                if j == 0:
                    g[j].append(ph_l[j](h[index]))
                else:
                    g[j].append(ph_l[j](g[j - 1][index]))

        o = self.soft_cos_sim(g[-1])  # Output = Last layer of projection head

        # Combine model and compile
        SimCLR_model = Model(inputs=i, outputs=o)
        SimCLR_model.compile(optimizer=self.optimizer, loss=self.loss)
        return SimCLR_model

    def train(self, data_train, data_val, epochs=10, pr=True):
        """ Training the SimCLR model and saving best model with time stamp
            Transfers adapted weights to base_model
        """

        # Callbacks
        checkpoint, earlyStopping, reduce_lr = self.get_callbacks()

        # Fit
        SimCLR_model = self.SimCLR_model
        SimCLR_model.fit(
            data_train,
            epochs=epochs,
            verbose=1,
            validation_data=data_val,
            callbacks=[checkpoint, earlyStopping, reduce_lr],
        )

        # Print number of trainable weights
        if pr:
            self.print_weights()

        # Save
        self.save_base_model()

    def unfreeze_and_train(
        self,
        data_train,
        data_val,
        num_of_unfrozen_layers,
        r,
        lr=1e-4,
        epochs=10,
        pr=True,
    ):
        """ Changes number of unfrozen layers in the base model and rebuilds it
            Training the SimCLR model and saving best model with time stamp
            Transfers adapted weights to base_model
        """
        # Update parameters
        self.num_of_unfrozen_layers = num_of_unfrozen_layers
        self.r = r
        if self.lr != lr:
            self.change_lr(lr)

        # (Un)freeze layers of base_model
        self.SimCLR_model = self.build_model()

        # Print number of trainable weights
        if pr:
            self.print_weights()

        # Train
        self.train(data_train, data_val, epochs)

    def predict(self, data):
        """ SimCLR prediction
        """
        return self.SimCLR_model.predict(data)

    def save_base_model(self):
        """ Save base_model with time stamp
        """
        self.base_model.save(
            self.save_path
            + "/base_model/base_model_round_"
            + str(self.r)
            + ".h5"
        )

    def change_lr(self, lr):
        """ Changing learning rate of SimCLR_model
        """
        self.lr = lr
        K.set_value(self.SimCLR_model.optimizer.learning_rate, self.lr)

    def get_callbacks(self):
        """ Returns callbacks used while training
        """
        # Time stamp for checkpoint
        now = datetime.now()
        dt_string = now.strftime("_%m_%d_%Hh_%M")

        checkpoint = ModelCheckpoint(
            self.save_path + "/SimCLR/SimCLR" + dt_string + ".h5",
            monitor="val_loss",
            verbose=1,
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
            monitor="val_loss", patience=5, verbose=1, factor=0.5,
        )
        return checkpoint, earlyStopping, reduce_lr

    def print_weights(self):
        """ Function to print (non)-learnable weights
            Helps checking unfreezing process
        """
        trainable_count = count_params(self.SimCLR_model.trainable_weights)
        non_trainable_count = count_params(
            self.SimCLR_model.non_trainable_weights
        )

        print(f"trainable parameters: {round(trainable_count/1e6,2)} M.")
        print(
            f"non-trainable parameters: {round(non_trainable_count/1e6,2)} M."
        )

    def train_NL_and_evaluate(
        self,
        dfs,
        batch_size,
        params_generator,
        fraction,
        class_labels,
        reg_dense=0.005,
        reg_out=0.005,
        nums_of_unfrozen_layers=[5, 5, 6, 7],
        lrs=[1e-3, 1e-4, 5e-5, 1e-5],
        epochs=[5, 5, 20, 25],
        verbose_epoch=0,
        verbose_cycle=1,
    ):
        """ Trains and evaluates a nonlinear classifier on top of the base_model
        """
        results = {"acc": 0}
        for i in range(5):
            if verbose_cycle:
                print(f"Learning attempt {i+1}")

            classifier = Classifier(
                base_model=self.base_model,
                num_classes=params_generator["num_classes"],
                reg_dense=reg_dense,
                reg_out=reg_out,
            )

            data_train, data_val, data_test = classifier.get_generators(
                dfs, fraction, batch_size, params_generator
            )

            classifier.train(
                data_train,
                data_val,
                fraction,
                nums_of_unfrozen_layers,
                lrs,
                epochs,
                verbose_epoch,
                verbose_cycle,
            )
            acc, report = classifier.evaluate_on_test(
                dfs["test"], data_test, class_labels
            )

            if results["acc"] < acc:
                results["acc"] = acc
                results["report"] = report
                results["attempt"] = i + 1

        print("Best result from attempt", str(results["attempt"]))
        print(results["report"])
