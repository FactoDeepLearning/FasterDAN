def ctc_remove_successives_identical_ind(ind):
    res = []
    for i in ind:
        if res and res[-1] == i:
            continue
        res.append(i)
    return res


def LM_str_to_ind(labels, str):
    return [labels.index(c) for c in str]


def LM_ind_to_str(labels, ind, oov_symbol=None):
    if oov_symbol is not None:
        res = []
        for i in ind:
            if i < len(labels):
                res.append(labels[i])
            else:
                res.append(oov_symbol)
    else:
        res = [labels[i] for i in ind]
    return "".join(res)
