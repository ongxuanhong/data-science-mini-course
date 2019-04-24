import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 
def grid_bar_charts(pdf, ls_cname, ncols = 3):
    """
    Vẽ nhiều bar chart cho các thuộc tính được xếp vào grid
    Cho số lượng grid column, ta sẽ fill out bar chart cho từng cell của grid
    """
    
    # tính số dòng cần cho grid
    n_cat = len(ls_cname)    
    nrows = int(math.ceil(n_cat * 1.0 / ncols))

    # khởi tạo figure gồm nrows * ncols cho grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3 * nrows))
    
    # dùng tuỳ chọn này để các chart được rời nhau
    fig.set_tight_layout(False)
    
    # fill out grid
    for i in range(nrows):
        for j in range(ncols):
            # xác định vị trí tên column trong danh sách dựa vào (i, j, ncols)
            idx = i * ncols + j
            
            # khi plot hết thì dừng
            if idx == n_cat:
                break
                
            # lấy tên column cần plot
            cname = ls_cname[idx]
            s00 = pdf[~pdf[cname].isna()]
            s00 = s00.groupby(cname).size()
            
            # sắp giá trị giảm dần trước khi plot
            if nrows == 1:
                s00.sort_values(ascending=False).plot.bar(ax=axes[idx], rot=45)
            else:
                s00.sort_values(ascending=False).plot.bar(ax=axes[i][j], rot=45)

    # plot grid
    plt.tight_layout()
    plt.show()
    
#     
def plot_wordcloud(pdf, ls_cname):
    """
    Vẽ wordcloud cho biến có nhiều giá trị categories
    """
    
    for cname in ls_cname:
        # get sequence of types
        s00 = pdf[~pdf[cname].isna()][cname]
        text = " ".join(s00.astype(str).tolist())

        # generate wordcloud
        wordcloud = WordCloud(background_color="white", width=1600, height=800).generate(text)

        # 
        fig, ax = plt.subplots(figsize=(15, 15))
        fig.set_tight_layout(False)
        
        # plot wordcloud
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Word cloud of {}".format(cname), fontsize=20)
        
        #
        plt.tight_layout()
        plt.show()    
        
# 
def grid_histogram(pdf, ls_cname, ncols = 3):
    """
    Vẽ nhiều histogram cho các thuộc tính được xếp vào grid
    Cho số lượng grid column, ta sẽ fill out histogram cho từng cell của grid
    """
    
    # tính số dòng cần cho grid
    n_cat = len(ls_cname)    
    nrows = int(math.ceil(n_cat * 1.0 / ncols))

    # khởi tạo figure gồm nrows * ncols cho grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4 * nrows))
    
    # dùng tuỳ chọn này để các chart được rời nhau
    fig.set_tight_layout(False)
    
    # fill out grid
    for i in range(nrows):
        for j in range(ncols):
            
            # xác định vị trí tên column trong danh sách dựa vào (i, j, ncols)
            idx = i * ncols + j
            
            # khi plot hết thì dừng
            if idx == n_cat:
                break
                
            cname = ls_cname[idx]
            s00 = pdf[~pdf[cname].isna()][cname]            
            if nrows == 1:
                s00.plot(kind="hist", ax=axes[idx], rot=45, title=cname)
            else:
                s00.plot(kind="hist", ax=axes[i][j], rot=45, title=cname)
            
    plt.tight_layout()
    plt.show()        
    
# 
def plot_continuous_data(s00, title):
    """
    Quan sát continuous data bằng histogram và boxplot
    """
    
    # khởi tạo figure
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
    # plot
    s00.hist(bins=30, ax=ax1)
    s00.plot.box(ax=ax2)
    
    #
    plt.suptitle(title)
    plt.show()
    
#
def visualize_meta_based(pdf_input, pdf_meta, params={}):
    
    default_params = {
        "bar_cate_cols": 3,
        "bar_num_cols": 3,
        "hist_num_cols": 3,
    }
    
    default_params.update(params)
    params = default_params
    
    print("Visualize for categorical data...")
    # check categorical data attributes
    ls_cat_name = pdf_meta[pdf_meta["sub_type"] == "object"]["name"].tolist()
    
    # if number of category is small we could use bar chart, otherwise use cloud chart
    pdf_meta00 = pdf_meta[pdf_meta["name"].isin(ls_cat_name)][["name", "n_distinct"]]
    pdf_meta00["chart"] = pdf_meta00["n_distinct"].apply(lambda x: "wordcloud" if int(x.split()[0]) > 10 else "bar")
    
    #
    ls_cat_bar = pdf_meta00.query("chart == 'bar'")["name"].tolist()
    grid_bar_charts(pdf_input, ls_cat_bar, ncols=params["bar_cate_cols"])
    
    #
    ls_cat_wordcloud = pdf_meta00.query("chart == 'wordcloud'")["name"].tolist()
    plot_wordcloud(pdf_input, ls_cat_wordcloud)
    
    print("Visualization for numerical data...")
    # check numerical data attributes
    ls_num_name = pdf_meta[pdf_meta["sub_type"] == "int64"]["name"].tolist()
    
    # if number of distinct values is small we could use bar chart, otherwise use histogram
    pdf_meta00 = pdf_meta[pdf_meta["name"].isin(ls_num_name)][["name", "n_distinct"]]
    pdf_meta00["chart"] = pdf_meta00["n_distinct"].apply(lambda x: "histogram" if int(x.split()[0]) > 10 else "bar")
    
    #
    ls_num_bar = pdf_meta00.query("chart == 'bar'")["name"].tolist()
    grid_bar_charts(pdf_input, ls_num_bar, ncols=params["bar_num_cols"])
    
    #
    ls_num_hist = pdf_meta00.query("chart == 'histogram'")["name"].tolist()
    grid_histogram(pdf_input, ls_num_hist, ncols=params["hist_num_cols"])
    
    print("Visualization for continuous data...")
    # check continuous data attributes
    ls_continuous_name = pdf_meta[pdf_meta["sub_type"] == "float64"]["name"].tolist()
    
    for cname in ls_continuous_name:
        s00 = pdf_input[~pdf_input[cname].isna()][cname]    
        plot_continuous_data(s00, cname)