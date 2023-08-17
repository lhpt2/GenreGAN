import numpy as np

class LPrint:
    def __init__(self):
        self.g_loss = []
        self.s_loss = []
        self.d_loss = []
        self.dr_loss = []
        self.df_loss = []
        self.id_loss = []
        self.val_loss = []

    def append_all(self, idict: dict):
        self.g_loss.append(idict["gloss"])
        self.d_loss.append(idict["dloss"])
        self.dr_loss.append(idict["drloss"])
        self.df_loss.append(idict["dfloss"])
        self.s_loss.append(idict["sloss"])
        self.id_loss.append(idict["idloss"])
        self.val_loss.append(idict["vloss"])

    def append_disc(self, idict: dict):
        self.d_loss.append(idict["dloss"])
        self.df_loss.append(idict["dfloss"])
        self.dr_loss.append(idict["drloss"])

    def to_str(self, startbackwards: int):
        msgstr = f'[D loss: {np.mean(self.d_loss[-startbackwards:])}, real: {np.mean(self.dr_loss[-startbackwards:])}, fake: {np.mean(self.df_loss[-startbackwards:])}] '
        msgstr += f'[G loss: {np.mean(self.g_loss[-startbackwards:])}] '
        msgstr += f'[S loss: {np.mean(self.s_loss[-startbackwards:])}] '
        msgstr += f'[ID loss: {np.mean(self.id_loss[-startbackwards:])}] '
        msgstr += f'[Val loss: {np.mean(self.val_loss[-startbackwards:])}] '
        return msgstr

    def get_mean_all(self):
        return self.get_mean(len(self.g_loss))

    def to_csv_part(self, back: int):
        msgstr = f'{np.mean(self.d_loss[-back:])},{np.mean(self.dr_loss[-back:])},{np.mean(self.df_loss[-back:])},'
        msgstr += f'{np.mean(self.g_loss[-back:])},'
        msgstr += f'{np.mean(self.s_loss[-back:])},'
        msgstr += f'{np.mean(self.id_loss[-back:])},'
        msgstr += f'{np.mean(self.val_loss[-back:])}'
        return msgstr

    def to_csv(self):
        msgstr = f'{np.mean(self.d_loss)},{np.mean(self.dr_loss)},{np.mean(self.df_loss)}],'
        msgstr += f'{np.mean(self.g_loss)},'
        msgstr += f'{np.mean(self.s_loss)},'
        msgstr += f'{np.mean(self.id_loss)},'
        msgstr += f'{np.mean(self.val_loss)}'
        return msgstr

    def get_mean(self, back: int):
        return {
                    "gloss": np.mean(self.g_loss[-back:]),
                    "dloss": np.mean(self.d_loss[-back:]),
                    "drloss": np.mean(self.dr_loss[-back:]),
                    "dfloss": np.mean(self.df_loss[-back:]),
                    "sloss": np.mean(self.s_loss[-back:]),
                    "idloss": np.mean(self.id_loss[-back:]),
                    "vloss": np.mean(self.val_loss[-back:])
                }

    def __str__(self):
        msgstr = f'[D loss: {np.mean(self.d_loss)} , real: {np.mean(self.dr_loss)}, fake: {np.mean(self.df_loss)}] '
        msgstr += f'[G loss: {np.mean(self.g_loss)}] '
        msgstr += f'[S loss: {np.mean(self.s_loss)}] '
        msgstr += f'[ID loss: {np.mean(self.id_loss)}] '
        msgstr += f'[Val loss: {np.mean(self.val_loss)}] '
        return msgstr