
def make_weights_for_balanced_classes(sample_names, labels):
    """
    Modified version of:
    https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    """

    sample_names = [int(i) for i in sample_names]

    # this purposely uses all possible labels by using max() as some labels are missed through the pipeline,
    # i.e. cancer type (2) doesn't show up and things quickly get awkward.
    nclasses = int(max(labels)) + 1

    count = [0] * nclasses    # make a list of zeros for each class

    ##################################
    #   For each class, count how
    #   many occurrences there are
    #   in the data.
    ##################################

    for i,_ in enumerate(sample_names):
        L = labels[i]
        count [ int(L) ] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))

    for i in range(nclasses):
        # if/else statement for some classes that don't make it
        # as far as being called by the pipeline.
        if count[i] != 0:
            weight_per_class[i] = N/float(count[i])
        else:
            weight_per_class[i] = 0.

    weight = [0] * len(sample_names)

    for i,_ in enumerate(sample_names):
        L = int(labels[i])
        weight[i] = weight_per_class[L]

    return weight
