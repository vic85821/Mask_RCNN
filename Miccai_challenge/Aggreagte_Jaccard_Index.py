# slow version of Aggregated Jaccrd Index
def agg_jc_index(mask, pred):
    """Calculate aggregated jaccard index for prediction & GT mask
    reference paper here: https://www.dropbox.com/s/j3154xgkkpkri9w/IEEE_TMI_NuceliSegmentation.pdf?dl=0

    mask: Ground truth mask, shape = [1000, 1000, instances]
    pred: Prediction mask, shape = [1000,1000], dtype = uint16, each number represent one instance

    Returns: Aggregated Jaccard index for GT & mask 
    """


    c = 0   # intersection
    u = 0   # unio 

    pred_mark_used = []
    for idx_m in tqdm_notebook(range(len(mask[0,0,:]))):
        m = mask[:,:,idx_m]
        intersect_list = []
        union_list = []
        for idx_pred in range(1, int(np.max(pred))):
            p = (pred==idx_pred) * 1
            intersect = np.sum(m.ravel()* p.ravel())
            union = np.sum(m.ravel() + p.ravel() - m.ravel()*p.ravel())
            intersect_list.append(intersect)
            union_list.append(union)
            
        hit_idx = np.argmax(np.array(intersect_list))
        c += intersect_list[hit_idx]
        u += union_list[hit_idx]
        pred_mark_used.append(hit_idx)

    pred_fp = set(np.unique(pred)) - set(pred_mark_used)
    pred_fp_pixel = np.sum([np.sum(pred==i) for i in pred_fp])

    u += pred_fp_pixel
    print(c / u)
    return c / u