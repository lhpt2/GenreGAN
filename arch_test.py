from keras.models import Model

class GenreGAN(Model):
    def __init__(self, generator, discriminator, siamese, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generator = generator
        self.discriminator = discriminator
        self.siamese = siamese

    def compile(self, g_opt, d_opt, s_opt, g_loss, d_loss, s_loss, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.s_opt = s_opt
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.s_loss = s_loss
        pass

    def train_step(self, batch):
        sources = batch[1]
        targets = batch[2]
        pass
