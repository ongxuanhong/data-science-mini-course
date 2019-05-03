import collections

import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import metrics


def get_Xy_from_pdf(pdf_input, ls_features, tvt_code):
    pdf_data = pdf_input[pdf_input["tvt_code"] == tvt_code].copy()

    #
    X = pdf_data[ls_features]
    y = pdf_data["TARGET"]

    return X, y


def visualize_auc(pdf, tvt_code, res_model, target_pred_posval=0):
    # get Xy and predict
    X, y = get_Xy_from_pdf(pdf, res_model["features"], tvt_code)
    y_pred = res_model["model"].predict_proba(X)[:, target_pred_posval]

    # get values
    auc_value = metrics.roc_auc_score(y, y_pred)
    res01 = metrics.roc_curve(y, y_pred)

    # plot
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    lw = 2
    ax1.plot(res01[0], res01[1], color="darkorange", lw=lw, label="ROC")
    ax1.plot([0, 1], [0, 1], color="navy", label="Random", lw=lw, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("AUC = %0.5f" % (auc_value))
    ax1.legend(loc="lower right")

    # prediction probability histogram
    ax2.set_title("{} set (size: {})".format(tvt_code, y.shape[0]))
    ax2.hist(y_pred, bins=200)

    plt.show()


def feature_selection_steps(
        pdf_input,
        ls_features,
        target_name,
        target_posval,
        idx_train,
        idx_test_list,
        xgb_param_init,
        xgb_param_fit,
        options={}
):
    #
    default_options = {
        "nturn": 1,
        "random_state_ori": 1,
        "auc_check_per_turn_n": 1,
        "auc_delta_check_per_turn": -1,
        "ndrop_per_turn": 1
    }

    default_options.update(options)
    options = default_options

    # tracked vars  
    ls_fit_info = []
    auc_last_turn = 1.0
    ls_curr_features = ls_features

    ### start selecting features
    for turn in range(options["nturn"]):

        # store results in each turn check
        ls_xgb_model = [None] * options["auc_check_per_turn_n"]
        ls_tracked_auc = [None] * options["auc_check_per_turn_n"]
        ls_tracked_imp = [None] * options["auc_check_per_turn_n"]

        for j_turn in range(options["auc_check_per_turn_n"]):
            # init model
            xgb_param_init["random_state"] = options["random_state_ori"] + j_turn
            xgb_model = xgb.XGBClassifier(**xgb_param_init)

            # add eval set
            xgb_param_fit["eval_set"] = []
            for i_idx in [idx_train] + idx_test_list:
                X_i = pdf_input.loc[i_idx, ls_curr_features]
                y_i = (pdf_input.loc[i_idx, target_name] == target_posval).astype("i4")
                xgb_param_fit["eval_set"].append((X_i, y_i))

            # model fitting
            xgb_model.fit(X=pdf_input.loc[idx_train, ls_curr_features],
                          y=(pdf_input.loc[idx_train, target_name] == target_posval).astype("i4"),
                          **xgb_param_fit)
            #
            best_iteration = xgb_model.get_booster().best_iteration
            xgb_evals_result = xgb_model.evals_result()

            # get best auc list and sorted importance features
            ls_sub_auc = [xgb_evals_result[val_name]["auc"][best_iteration] for val_name in xgb_evals_result]
            ls_imp_feat = sorted(xgb_model.get_booster().get_score(importance_type="gain").items(), key=lambda x: -x[1])

            # store result
            ls_xgb_model[j_turn] = xgb_model
            ls_tracked_auc[j_turn] = ls_sub_auc
            ls_tracked_imp[j_turn] = ls_imp_feat

            #
            print("Turn {} | j_repeat: {} | auc: {} | len(features): {} | len(importance): {}\n\n".format(turn, j_turn,
                                                                                                          ls_sub_auc,
                                                                                                          len(
                                                                                                              ls_curr_features),
                                                                                                          len(
                                                                                                              ls_imp_feat)))

            # early stop on last validation set
            if ls_sub_auc[-1] >= (auc_last_turn - options["auc_delta_check_per_turn"]):
                break

        # get max turn result on last validation set
        j_turn_max = max(range(j_turn + 1), key=lambda j_val: ls_tracked_auc[j_val][-1])
        selected_xgb_model = ls_xgb_model[j_turn_max]
        selected_auc_list = ls_tracked_auc[j_turn_max]
        selected_imp = ls_tracked_imp[j_turn_max]
        auc_last_turn = selected_auc_list[-1]

        # update info
        info = collections.OrderedDict([
            ("auc", selected_auc_list), ("ls_tracked_auc", ls_tracked_auc),
            ("ls_curr_features", ls_curr_features),
            ("imp", selected_imp), ("ls_tracked_imp", ls_tracked_imp),
            ("model", selected_xgb_model)])
        ls_fit_info.append(info)
        print("Turn {} | j_max: {} | auc: {} | len(features): {} | len(importance): {}\n\n".format(turn, j_turn_max,
                                                                                                   selected_auc_list,
                                                                                                   len(
                                                                                                       ls_curr_features),
                                                                                                   len(selected_imp)))

        # update features list
        ls_curr_features = list(list(zip(*(selected_imp[:-options["ndrop_per_turn"]])))[0])

    return ls_fit_info
