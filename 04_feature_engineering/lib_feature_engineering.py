import numpy as np
import pandas as pd
from IPython.display import display
from sklearn import metrics


def my_auc(y_score, y_true, flexible_sign=True):
    # filter NaN
    idx = np.isfinite(y_score)
    xxx = y_score[idx]
    yyy = y_true[idx]

    # if label not only 1s/0s
    if yyy.std() > 0.0:
        auc = metrics.roc_auc_score(y_score=xxx, y_true=yyy)
    else:
        auc = 0.5

    # for evaluation only
    if (auc < 0.5) & (flexible_sign):
        auc = 1.0 - auc
    return auc


def feature_evaluate(pdf_train, pdf_feat, ls_feat=None):
    out_res = {
        "name": [],
        "auc": [],
        "corr": [],
        "coverage": []
    }
    pdf_eval = pdf_train.merge(pdf_feat, on="SK_ID_CURR")
    if ls_feat is None:
        ls_feat = [cname for cname in pdf_feat.columns if cname != "SK_ID_CURR"]

    # calculate correlation
    pdf_corr = pdf_eval.corr()

    for feat in ls_feat:
        out_res["name"].append(feat)
        out_res["auc"].append(my_auc(pdf_eval[feat], pdf_eval["TARGET"]))
        out_res["corr"].append(pdf_corr.loc[feat, "TARGET"])
        out_res["coverage"].append((~pdf_eval[feat].isna()).mean())

    pdf_res = pd.DataFrame(out_res)
    pdf_res = pdf_res[["name", "auc", "corr", "coverage"]].sort_values(by="auc", ascending=False)
    return pdf_res


def gen_binary_one_hot_feat(pdf_input):
    pdf_data = pdf_input.copy()
    select_features = []
    dict_feat = {
        "binary_default": {
            "NAME_CONTRACT_TYPE": ['Cash loans', 'Revolving loans'],
            "CODE_GENDER": ['M', 'F', 'XNA'],
            "FLAG_OWN_CAR": ['Y', 'N'],
            "FLAG_OWN_REALTY": ['Y', 'N'],
            "EMERGENCYSTATE_MODE": ['Yes', 'No'],
        },
        "binary": [
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
            "REG_REGION_NOT_LIVE_REGION",
            "REG_REGION_NOT_WORK_REGION",
            "LIVE_REGION_NOT_WORK_REGION",
            "REG_CITY_NOT_WORK_CITY",
            "LIVE_CITY_NOT_WORK_CITY",
            "FLAG_DOCUMENT_3",
            "FLAG_DOCUMENT_5",
            "FLAG_DOCUMENT_6",
            "FLAG_DOCUMENT_8",
            "FLAG_DOCUMENT_9",
            "REGION_RATING_CLIENT",
            "REGION_RATING_CLIENT_W_CITY",
        ],
        "onehot": {
            "NAME_TYPE_SUITE": ["Unaccompanied", "Family", "Spouse, partner", "Children", "Other_A", "Other_B",
                                "Group of people"],
            "NAME_INCOME_TYPE": ["Working", "State servant", "Commercial associate", "Pensioner", "Unemployed",
                                 "Student", "Businessman", "Maternity leave"],
            "NAME_EDUCATION_TYPE": ["Secondary / secondary special", "Higher education", "Incomplete higher",
                                    "Lower secondary", "Academic degree"],
            "NAME_FAMILY_STATUS": ["Single / not married", "Married", "Civil marriage", "Widow", "Separated",
                                   "Unknown"],
            "NAME_HOUSING_TYPE": ["House / apartment", "Rented apartment", "With parents", "Municipal apartment",
                                  "Office apartment", "Co-op apartment"],
            "OCCUPATION_TYPE": ["Laborers", "Core staff", "Accountants", "Managers", "Drivers", "Sales staff",
                                "Cleaning staff", "Cooking staff", "Private service staff", "Medicine staff",
                                "Security staff", "High skill tech staff", "Waiters/barmen staff", "Low-skill Laborers",
                                "Realty agents", "Secretaries", "IT staff", "HR staff"],
            "ORGANIZATION_TYPE": ["Business Entity Type 3", "School", "Government", "Religion", "Other", "XNA",
                                  "Electricity", "Medicine", "Business Entity Type 2", "Self-employed",
                                  "Transport: type 2", "Construction", "Housing", "Kindergarten", "Trade: type 7",
                                  "Industry: type 11", "Military", "Services", "Security Ministries",
                                  "Transport: type 4", "Industry: type 1", "Emergency", "Security", "Trade: type 2",
                                  "University", "Transport: type 3", "Police", "Business Entity Type 1", "Postal",
                                  "Industry: type 4", "Agriculture", "Restaurant", "Culture", "Hotel",
                                  "Industry: type 7", "Trade: type 3", "Industry: type 3", "Bank", "Industry: type 9",
                                  "Insurance", "Trade: type 6", "Industry: type 2", "Transport: type 1",
                                  "Industry: type 12", "Mobile", "Trade: type 1", "Industry: type 5",
                                  "Industry: type 10", "Legal Services", "Advertising", "Trade: type 5", "Cleaning",
                                  "Industry: type 13", "Trade: type 4", "Telecom", "Industry: type 8", "Realtor",
                                  "Industry: type 6"],
            "FONDKAPREMONT_MODE": ["reg oper account", "org spec account", "reg oper spec account", "not specified"],
            "HOUSETYPE_MODE": ["block of flats", "terraced house", "specific housing"],
            "WALLSMATERIAL_MODE": ["Stone, brick", "Block", "Panel", "Mixed", "Wooden", "Others", "Monolithic"],
        }
    }

    for k in dict_feat:
        if k == "binary_default":
            for cname in dict_feat[k]:
                # get default value
                default_val = dict_feat[k][cname][0]

                # convert category to binary
                feat_name = "is_" + cname
                select_features.append(feat_name)
                pdf_data[feat_name] = pdf_data[cname].apply(lambda x: int(x == default_val))
        elif k == "binary":
            # rename only
            for cname in dict_feat[k]:
                feat_name = "is_" + cname
                select_features.append(feat_name)
                pdf_data[feat_name] = pdf_data[cname]
        elif k == "onehot":
            for cname in dict_feat[k]:
                ls_vals = dict_feat[k][cname]
                for val in ls_vals:
                    try:
                        new_name = "{}_{}".format(cname, val.replace(" ", "_") \
                                                  .replace(":", "_") \
                                                  .replace("/", "_") \
                                                  .replace("-", "_"))

                        select_features.append(new_name)
                        pdf_data[new_name] = pdf_data[cname].apply(lambda x: int(x == val))
                    except Exception as err:
                        print("One hot for {}-{}. Error: {}".format(cname, val, err))

    return pdf_data[["SK_ID_CURR"] + select_features]


def gen_one_hot_feat(pdf_input, dict_feat, main_key="SK_ID_CURR"):
    pdf_data = pdf_input.copy()
    select_features = []

    for cname in dict_feat:
        ls_vals = dict_feat[cname]
        for val in ls_vals:
            try:
                new_name = "{}_{}".format(cname, val.replace(" ", "_") \
                                          .replace(":", "_") \
                                          .replace("/", "_") \
                                          .replace("-", "_"))

                select_features.append(new_name)
                pdf_data[new_name] = pdf_data[cname].apply(lambda x: int(x == val))
            except Exception as err:
                print("One hot for {}-{}. Error: {}".format(cname, val, err))

    return pdf_data[[main_key] + select_features]


def agg_common_data(pdf_input, ls_func, main_key="SK_ID_CURR"):
    ls_agg_name = [cname for cname in pdf_input.columns if cname != main_key]

    # define agg
    dict_agg = {}
    for name in ls_agg_name:
        dict_agg[name] = ls_func
    display(dict_agg)

    # do agg
    pdf_agg = pdf_input.groupby(main_key).agg(dict_agg)
    print("After agg: {}".format(pdf_agg.shape))

    # rename columns
    name01 = pdf_agg.columns.get_level_values(0)
    name02 = pdf_agg.columns.get_level_values(1)
    rename_cols = ["{}_{}".format(tpl[0], tpl[1]) for tpl in zip(name01, name02)]
    pdf_agg.columns = rename_cols

    return pdf_agg


# 
def mean_encoding(pdf_input, col_to_encode):
    s_mean_encoding = pdf_input.groupby(col_to_encode)["TARGET"].mean()
    pdf_mean_encoding = pdf_input[["SK_ID_CURR", col_to_encode]].copy()
    pdf_mean_encoding["{}_mean_encoding".format(col_to_encode)] = pdf_mean_encoding[col_to_encode].apply(
        lambda t: s_mean_encoding.loc[t] if t is not np.nan else t)
    pdf_mean_encoding = pdf_mean_encoding.drop(columns=[col_to_encode])

    return pdf_mean_encoding, s_mean_encoding


# 
def agg_mean_encoding(pdf_input, ls_cat):
    ls_pdf_encoding = []
    dict_encoding_map = {}
    for col_to_encode in ls_cat:
        print("Encoding {}...".format(col_to_encode))
        pdf_mean_encoding, s_mean_encoding = mean_encoding(pdf_input, col_to_encode)

        #
        dict_encoding_map[col_to_encode] = s_mean_encoding
        ls_pdf_encoding.append(pdf_mean_encoding)

    # join all encoded columns
    pdf_agg = ls_pdf_encoding[0]
    for pdf in ls_pdf_encoding[1:]:
        pdf_agg = pdf_agg.merge(pdf, on="SK_ID_CURR")

    return pdf_agg, dict_encoding_map


# 
def mean_encode_mapping(pdf_input, dict_encoding_map):
    # mapping for other set
    ls_encode_col = dict_encoding_map.keys()
    pdf_encoded = pdf_input[["SK_ID_CURR"] + ls_encode_col].copy()

    for col_to_encode in ls_encode_col:
        print("Encoding {}...".format(col_to_encode))
        s_mean_encoding = dict_encoding_map[col_to_encode]
        pdf_encoded["{}_mean_encoding".format(col_to_encode)] = pdf_encoded[col_to_encode].apply(
            lambda t: s_mean_encoding.loc[t] if t is not np.nan else t)

    pdf_encoded = pdf_encoded.drop(columns=ls_encode_col)
    return pdf_encoded
