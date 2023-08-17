import numpy as np

class LPrint:
    def __init__(self):
        self.nr = 5
        self.g_loss = []
        self.s_loss = []
        self.d_loss = []
        self.dr_loss = []
        self.df_loss = []
        self.id_loss = []
        self.val_loss = []

    def header(self):
       return 'dloss,drloss,dfloss,gloss,sloss,idloss,vloss'

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
        msgstr = f'[D loss: {str(np.mean(self.d_loss[-startbackwards:]))[:self.nr]}, real: {str(np.mean(self.dr_loss[-startbackwards:]))[:self.nr]}, fake: {str(np.mean(self.df_loss[-startbackwards:]))[:self.nr]}] '
        msgstr += f'[G loss: {str(np.mean(self.g_loss[-startbackwards:]))[:self.nr]}] '
        msgstr += f'[S loss: {str(np.mean(self.s_loss[-startbackwards:]))[:self.nr]}] '
        msgstr += f'[ID loss: {str(np.mean(self.id_loss[-startbackwards:]))[:self.nr]}] '
        msgstr += f'[Val loss: {str(np.mean(self.val_loss[-startbackwards:]))[:self.nr]}] '
        return msgstr

    def get_mean_all(self):
        return self.get_mean(len(self.g_loss))

    def to_csv_part(self, back: int):
        msgstr = f'{str(np.mean(self.d_loss[-back:]))[:self.nr]},{str(np.mean(self.dr_loss[-back:]))[:self.nr]},{str(np.mean(self.df_loss[-back:]))[:self.nr]},'
        msgstr += f'{str(np.mean(self.g_loss[-back:]))[:self.nr]},'
        msgstr += f'{str(np.mean(self.s_loss[-back:]))[:self.nr]},'
        msgstr += f'{str(np.mean(self.id_loss[-back:]))[:self.nr]},'
        msgstr += f'{str(np.mean(self.val_loss[-back:]))[:self.nr]}'
        return msgstr

    def to_csv(self):
        msgstr = f'{str(np.mean(self.d_loss))[:self.nr]},{str(np.mean(self.dr_loss))[:self.nr]},{str(np.mean(self.df_loss))[:self.nr]}],'
        msgstr += f'{str(np.mean(self.g_loss))[:self.nr]},'
        msgstr += f'{str(np.mean(self.s_loss))[:self.nr]},'
        msgstr += f'{str(np.mean(self.id_loss))[:self.nr]},'
        msgstr += f'{str(np.mean(self.val_loss))[:self.nr]}'
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
        msgstr = f'[D loss: {str(np.mean(self.d_loss))[:self.nr]} , real: {str(np.mean(self.dr_loss))[:self.nr]}, fake: {str(np.mean(self.df_loss))[:self.nr]}] '
        msgstr += f'[G loss: {str(np.mean(self.g_loss))[:self.nr]}] '
        msgstr += f'[S loss: {str(np.mean(self.s_loss))[:self.nr]}] '
        msgstr += f'[ID loss: {str(np.mean(self.id_loss))[:self.nr]}] '
        msgstr += f'[Val loss: {str(np.mean(self.val_loss))[:self.nr]}] '
        return msgstr