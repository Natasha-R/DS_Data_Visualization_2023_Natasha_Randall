import pandas as pd
import numpy as np
import scipy as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import bokeh
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, CustomJS, Div, Select, Label
from bokeh.models.tools import TapTool
from bokeh.transform import factor_cmap, factor_mark
from bokeh.layouts import row, column

def format_data(path="processed_code_solutions/features_data.csv"):
    """
    Processes the dataset to a form suitable to create the plot:
    Non-continuous features are dropped and labels are given readable names.
    
    Parameters
    ----------
    path : str
        The path to the csv file containing the raw dataset.
    
    Returns
    -------
    data : pandas.DataFrame
        The formatted dataset.
    """
    data = pd.read_csv(path)
    data = data.drop(["used_boolean", "used_List", "used_Integer", "used_Point", "used_ArrayList", "used_StringBuilder"], axis=1)
    data["source"] = data["source"].replace({"bard":"Bard", "gpt3.5":"ChatGPT-3.5", "bing":"Bing", "gpt4":"ChatGPT-4", "student":"Human"})
    data.columns = ["name", "source", "style", "version", "code", "Number of characters", "Number of lines", 
                    "Average line length", "Maximum line length", "Number of comments (scaled to length of code)", 
                    "Number of if statements (scaled to length of code)", "Number of for loops (scaled to length of code)", 
                    "Number of switch statements (scaled to length of code)", "Number of digits (scaled to length of code)", 
                    "Number of exceptions thrown (scaled to length of code)", "Number of empty lines (scaled to length of code)", 
                    "Number of print statements (scaled to length of code)", "Number of files", 
                    "Number of method declarations (scaled to length of code)", "Number of field variables declared (scaled to length of code)", 
                    "Number of local variables declared (scaled to length of code)", "Number of classes (scaled to length of code)", 
                    "Number of variables referenced (scaled to length of code)", "Number of method invocations (scaled to length of code)", 
                    "Number of imports (scaled to length of code)", "Average variable name length", "Maximum variable name length", 
                    "Average comment length", "Maximum comment length"]
    data["binary_source"] = data["source"].apply(lambda row: 1 if row=="Human" else 0)
    data["source"] = pd.Categorical(data["source"], ["Bard", "ChatGPT-3.5", "Bing", "ChatGPT-4", "Human"])
    data = data.sort_values("source").reset_index(drop=True)
    data["name"] = data["name"].str.replace("student", "")
    data["name"] = data["source"].astype(str) + "_" + data["name"]
    data.loc[data["source"]=="Bing", "name"] = data[data["source"]=="Bing"]["name"] + "_" + data[data["source"]=="Bing"]["version"]
    data["code"] = data["code"].str.replace("\n", "<br>")
    
    return data

def sort_features(data):
    """
    Fits a random forest classifier to the full set of features in the data,
    classifying between human and AI generated code solutions, 
    in order to determine which features are most important for classification.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The dataset that has been formatted by format_data.
    
    Returns
    -------
    sorted_features : list
        A list of feature names, ordered by how important they were to classify the data.
        The metric used for importance is sklearn's feature_importances_.
    """ 
    all_features = list(data.columns[5:-1])
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(data[all_features].values, 
                                                                            data["binary_source"], 
                                                                            test_size=0.3, 
                                                                            random_state=0)
    full_forest = RandomForestClassifier(random_state=0, 
                                         max_features=1, 
                                         n_estimators=160, 
                                         max_depth=10)
    full_forest.fit(X_train_full, y_train_full)
    sorted_features = list(pd.Series(full_forest.feature_importances_, index=all_features).sort_values(ascending=False).index)
    
    return sorted_features

def fit_model(data, feature_1, feature_2):
    """
    Fits a random forest classifier to the given two features.
    This method is used by create_decision_boundaries and does not need to be called directly.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset that has been formatted by format_data.
    feature_1 : str
        The name of the first feature.
    feature_2 : str
        The name of the second feature.

    Returns
    -------
    forest_model : sklearn.ensemble._forest.RandomForestClassifier
        The fitted model.
    accuracy : str
        The accuracy of the fitted model as calculated on the test set,
        rounded to 3 d.p., given as a percentage.
    """
    X = data[[feature_1, feature_2]].values
    y = data["binary_source"].values
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.4, 
                                                        random_state=0)
    forest_model = RandomForestClassifier(random_state=0, 
                                          max_features="sqrt", 
                                          n_estimators=180, 
                                          max_depth=5)
    forest_model.fit(X_train, y_train)
    accuracy = "{:.1f}".format(accuracy_score(y_test, forest_model.predict(X_test))*100)

    return forest_model, accuracy

def create_decision_boundaries(data, sorted_features):
    """
    Calculates meshes for the decision boundaries of a classifier fitted to every permutation of two features.
    Also calculates the accuracy of the classifiers on a test dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset that has been formatted by format_data.
    sorted_features : list
        A list of feature names, sorted by how important they are for classification.
        This list is generated by sort_features.

    Returns
    -------
    all_meshes : pandas.DataFrame
        Each column name begins with the two features that it is related to.
        Columns with suffix : xx represent the x-coordinate of mesh relating to the respective features.
        Columns with suffix : yy represent the y-coordinate of the mesh relating to the respective features.
        Columns with suffix : pred represent the classifiers' class prediction at the given coordinate, relating to the respective features.
        Columns with suffix : colours represent the colour representing the class prediction.
    all_accuracy : dict
        The keys are strings of each two feature pairs, in the format: "feature1_feature2".
        The values are the accuracy of a classifier applied to those two features, on a test dataset.

    """
    all_meshes = []
    all_accuracy = dict()
    db_colours = {0:"#bae083", 1:"#a2457d"}

    for feature_1 in sorted_features:
        for feature_2 in sorted_features:
            
            forest_model, accuracy = fit_model(data, feature_1, feature_2)
            all_accuracy[f"{feature_1}_{feature_2}"] = accuracy
            X = data[[feature_1, feature_2]].values
            
            min_feature_1, max_feature_1 = X[:, 0].min(), X[:, 0].max()
            min_feature_2, max_feature_2 = X[:, 1].min(), X[:, 1].max()
            mesh = np.meshgrid(np.linspace(min_feature_1, max_feature_1, 80), np.linspace(min_feature_2, max_feature_2, 85))
            predictions = forest_model.predict(np.c_[mesh[0].ravel(), mesh[1].ravel()])
            predictions = predictions.reshape(mesh[0].shape)    
            mesh = pd.DataFrame(np.c_[mesh[0].ravel(), 
                                      mesh[1].ravel(), 
                                      predictions.ravel()], 
                                columns=[f"{feature_1}_{feature_2}_xx", f"{feature_1}_{feature_2}_yy", f"{feature_1}_{feature_2}_pred"])
            mesh[f"{feature_1}_{feature_2}_colours"] = mesh[f"{feature_1}_{feature_2}_pred"].map(db_colours)
            
            all_meshes.append(mesh)
    all_meshes = pd.concat(all_meshes, axis=1)
    
    return all_meshes, all_accuracy

def create_plot(data, all_meshes, all_accuracy, sorted_features, starting_x="Number of lines", starting_y="Number of field variables declared (scaled to length of code)"):
    """
    Uses Bokeh to create a scatterplot of two (selectable) features from the dataset, coloured by source.
    A decision boundary (between human and AI) from a random forest classifier applied to the two plotted features, and its accuracy, is drawn on the plot.
    When a point is clicked on, the related code solution is printed on the right side.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset that has been formatted by format_data.
    all_meshes : pandas.DataFrame
        A dataframe containing a mesh for the decision boundary for each permutation of two features from sorted_features.
        Contains columns for each permutation of the x and y coordinates of the mesh, and the colour representing the class prediction.
        Generated by create_decision boundaries.
    all_accuracy : dict
        The keys are strings of each two feature pairs, in the format: "feature1_feature2".
        The values are the accuracy of a classifier applied to those two features, on a test dataset.
        Generated by create_decision_boundaries.
    sorted_features : list
        A list of feature names, sorted by how important they are for classification.
        This list is generated by sort_features.
    starting_x : str
        The feature shown initially on the x axis of the plot.
    starting_y : str
        The feature shown initially on the y axis of the plot.

    """
    cds_data = ColumnDataSource(data)
    cds_mesh = ColumnDataSource(all_meshes)

    markers = ["plus", "hex", "triangle", "square", "circle"]
    category = ["Bard", "ChatGPT-3.5", "Bing", "ChatGPT-4", "Human"]
    colours = ["#e5c949", "#e99675", "#95a3c3", "#a2c865", "#db96c0"]
    db_colours = {0:"#bae083", 1:"#a2457d"}

    plot = figure(plot_width=600, 
                  plot_height=600, 
                  tools=["tap", "save", "reset"],
                  x_axis_label=starting_x, 
                  y_axis_label=starting_y)
    click_desc = Div(text="""<div style='width: 600px; margin: 0 auto;'><p style='font-size: 15px; font-weight: bold; text-align: center;'>
                             Click on a point to see its associated code solution!</p></div>""")
    db_desc = Div(text="<div style='width: 610px; font-size: 14px;'>The features are ordered by how well they classify the data into AI versus human written code.\
                                                      The accuracy of a classifier trained on the currently displayed features is shown under the legend.</div>")

    mesh = plot.square(source=cds_mesh, 
                       x=f"{starting_x}_{starting_y}_xx", 
                       y=f"{starting_x}_{starting_y}_yy", 
                       fill_color=f"{starting_x}_{starting_y}_colours", 
                       size=6, 
                       line_alpha=0, 
                       fill_alpha=0.2, 
                       name="Mesh")
    mesh.nonselection_glyph = None

    scatterpoints = plot.scatter(source=cds_data, 
                                 x=starting_x, 
                                 y=starting_y, 
                                 size=12, 
                                 line_width=0.5, 
                                 line_color="#686768", 
                                 legend_field="source", 
                                 name="Points",
                                 marker=factor_mark(field_name="source", 
                                                    markers=markers, 
                                                    factors=category),
                                 color=factor_cmap(field_name="source", 
                                                   palette=colours, 
                                                   factors=category))

    accuracy_label = Label(x=390, 
                           y=340, 
                           x_units="screen", 
                           y_units="screen", 
                           text_font_size="14px", 
                           text_font_style="bold", 
                           text_baseline="bottom",
                           text=all_accuracy[f"{starting_x}_{starting_y}"] + "% accuracy", 
                           render_mode="canvas")

    db_label_1 = Label(x=385, 
                       y=385, 
                       x_units="screen", 
                       y_units="screen",
                       text="AI Predictions", 
                       render_mode="canvas",
                       border_line_alpha=0.2,
                       background_fill_alpha=0.4,
                       text_font_size="14px",
                       background_fill_color=db_colours[0])

    db_label_2 = Label(x=385, 
                       y=365, 
                       x_units="screen", 
                       y_units="screen",
                       text="Human Predictions", 
                       render_mode="canvas",
                       border_line_alpha=0.2,
                       background_fill_alpha=0.2,
                       text_font_size="14px",
                       background_fill_color=db_colours[1])

    plot.axis.axis_label_text_font_style = 'normal'

    code_text = Div()
    callback_click = CustomJS(args=dict(source=cds_data, 
                                        div=code_text), 
        code="""
        var index = source.selected.indices
        if (index.length == 0) {
            div.text = "";
            div.style = {};
        }
        else {
            div.text = source.data['name'][index[0]] + '<pre>' + source.data['code'][index[0]] + '</pre>';
            div.style = {'border': '1px solid black', 'padding': '10px'};
        }
        """)
    cds_data.selected.js_on_change('indices', callback_click)

    select_x = Select(title="Choose the feature for the x-axis:", 
                      value=starting_x, 
                      options=sorted_features)
    select_y = Select(title="Choose the feature for the y-axis:", 
                      value=starting_y, 
                      options=sorted_features)
    callback_select = CustomJS(args=dict(source=cds_data, 
                                         mesh_data=cds_mesh, 
                                         plot=plot, 
                                         scatterpoints_renderer=scatterpoints, 
                                         mesh_renderer=mesh,
                                         x_select=select_x, 
                                         y_select=select_y, 
                                         accuracy_label=accuracy_label, 
                                         accuracy_dict=all_accuracy,
                                         xaxis=plot.xaxis[0], 
                                         yaxis=plot.yaxis[0]),             
        code="""
        scatterpoints_renderer.glyph.x = {field: x_select.value};
        scatterpoints_renderer.glyph.y = {field: y_select.value};

        xaxis.axis_label = x_select.value;
        yaxis.axis_label = y_select.value;

        var current_vars = x_select.value + "_" + y_select.value;
        var current_xx = current_vars + "_xx";
        var current_yy = current_vars + "_yy";
        var current_colour = current_vars + "_colours";
        mesh_renderer.glyph.x = {field: current_xx};
        mesh_renderer.glyph.y = {field: current_yy};
        mesh_renderer.glyph.fill_color = {field: current_colour};

        accuracy_label.text = accuracy_dict[current_vars] + "% accuracy";

    """)

    select_x.js_on_change('value', callback_select)
    select_y.js_on_change('value', callback_select)

    layout = row(column(select_x, select_y, db_desc, plot, click_desc), code_text)
    for label in [accuracy_label, db_label_1, db_label_2]:
        plot.add_layout(label)
    show(layout)