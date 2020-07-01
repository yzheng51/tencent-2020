import time
import gensim


class MyLossCalculator(gensim.models.callbacks.CallbackAny2Vec):
    def __init__(self):
        self.epoch = 1
        self.losses = []
        self.cumu_losses = []
        self.previous_epoch_time = time.perf_counter()

    def on_epoch_end(self, model):
        cumu_loss = model.get_latest_training_loss()
        loss = cumu_loss if self.epoch <= 1 else cumu_loss - self.cumu_losses[-1]
        epoch_seconds = time.perf_counter() - self.previous_epoch_time
        self.previous_epoch_time = time.perf_counter()
        print(
            f"Loss after epoch {self.epoch}: {loss} / cumulative loss: {cumu_loss}... "
            f"epoch took {epoch_seconds:.2f} s"
        )
        self.epoch += 1
        self.losses.append(loss)
        self.cumu_losses.append(cumu_loss)
