def ex_metrics(ratio_results,real_label_test,ExperimentName):
    #评价参数
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    index_com = 0
    for test_value in ratio_results['ratio']:
        if real_label_test[index_com] == 1:
            if test_value >= 1:
                TP = TP + 1
            if test_value < 1:
                FN = FN + 1
        if real_label_test[index_com] == 0:
            if test_value >= 1:
                FP = FP + 1
            if test_value < 1:
                TN = TN + 1
        index_com = index_com + 1
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    Detection_Rate = TP/(TP+FN)
    FPR = FP/(FP+TN)
    Precision = TP/(TP+FP)
    if ExperimentName == 'EX_3': #EX_3 do not need F1_Score
        external_metrics = [Accuracy,Precision,FPR,Detection_Rate]
        return external_metrics
    else:
        F1_Score = 2 * ((Precision * Detection_Rate)/(Precision + Detection_Rate))
        external_metrics = [Accuracy,Precision,FPR,Detection_Rate,F1_Score]
        return external_metrics