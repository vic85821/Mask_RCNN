# slow version of Aggregated Jaccrd Index
def agg_jc_index(mask, pred):
    """Calculate aggregated jaccard index for prediction & GT mask
    reference paper here: https://www.dropbox.com/s/j3154xgkkpkri9w/IEEE_TMI_NuceliSegmentation.pdf?dl=0

    mask: Ground truth mask, shape = [1000, 1000, instances]
    pred: Prediction mask, shape = [1000,1000], dtype = uint16, each number represent one instance

    Returns: Aggregated Jaccard index for GT & mask 
    """
    
    c = 0 # count intersection
    u = 0 # count union
    tqdm.monitor_interval = 0 # disable tqdm monitor to prevent warning message
    pred_instance = pred.max() # predcition instance number
    pred_mark_used = [] # mask used
    pred_mark_isused = np.zeros((pred_instance+1), dtype=bool)
    
    for idx_m in tqdm_notebook(range(len(mask[0,0,:]))):
        m = mask[:,:,idx_m]
        intersect_list = []
        union_list = []
        iou_list = []
        
        for idx_pred in range(1, pred_instance+1):
            p = (pred == idx_pred)

            # replace multiply with bool operation 
            intersect = np.count_nonzero((m!=0) & p)
            union = np.count_nonzero((m!=0) | p)       
            intersect_list.append(intersect)
            union_list.append(union)
            
        iou_list = np.array(intersect_list) / np.array(union_list)    
        hit_idx = np.argmax(np.array(iou_list))
        c += intersect_list[hit_idx]
        u += union_list[hit_idx]
        pred_mark_used.append(hit_idx)
        p = (pred == hit_idx+1)
        pred = (1-p) * pred
        
    pred_mark_used = [x+1 for x in pred_mark_used]
    pred_fp = set(np.unique(pred)) - {0} - set(pred_mark_used)
    pred_fp_pixel = np.sum([np.sum(pred==i) for i in pred_fp])

    u += pred_fp_pixel
    print (c / u)
    return (c / u)