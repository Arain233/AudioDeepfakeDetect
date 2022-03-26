# customize collate_fn to make different matrix into one batch
def diff_dataset_collate(batch):
    mats = []
    labels = []
    for mat, label in batch:
        mats.append(mat)
        labels.append(label)
    # mats = np.array(mats)
    # mats = torch.from_numpy(mats)
    # labels = np.array(labels)
    # labels = torch.from_numpy(labels)
    return mats, labels
