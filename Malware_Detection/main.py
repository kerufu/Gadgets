from autoencoder_worker import autoencoder_woker
from dense_worker import dense_worker
from dataset_worker import dataset_worker

class main_worker():
    
    def __init__(self, enable_discriminator=False, load_from_raw=False) -> None:
        self.dsw = dataset_worker()
        self.dsw.dataset_loader(load_from_raw)
        self.aew = autoencoder_woker(self.dsw, enable_discriminator)
        self.dw = dense_worker(self.dsw)

    def train_dense(self, epoch):
        indexes, _, _, _ = self.aew.anormal_analysis()
        self.dw.train(epoch, indexes)

    def test_dense(self):
        indexes, _, _, _ = self.aew.anormal_analysis()
        self.dw.test(indexes)

    def test_ae(self):
        _, indexes, _, _ = self.aew.anormal_analysis()
        self.aew.test(indexes)

mw = main_worker(True, False)
# mw.aew.train(10)
# mw.train_dense(10)
mw.test_dense()
mw.test_ae()