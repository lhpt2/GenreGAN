from tools.dataset_processing import load_wv_array_splitseconds_checked, construct_save_ds


ids, olist, rlist = load_wv_array_splitseconds_checked("./dsSimple/train_o")
construct_save_ds(ids, olist, rlist, "dsSimple_train")
ids, val_o, val_r = load_wv_array_splitseconds_checked("./dsSimple/val_o")
construct_save_ds(ids, val_o, val_r, "dsSimple_val")
