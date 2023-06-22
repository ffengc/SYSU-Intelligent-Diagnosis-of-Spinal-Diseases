

def get_acc(pred, label):
    # pred,label都是tensor,而且在cuda上
    pred = pred.cpu().tolist()
    label = label.cpu().tolist()
    assert(len(pred) == len(label))
    bingo = 0
    for i in range(0,len(pred)):
        if pred[i] == label[i]:
            bingo += 1
    return bingo / len(pred)