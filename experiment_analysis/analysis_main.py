import json

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from analyse_explanation_ranking import get_interactive_ranking, get_static_ranking
from experiment_analysis.analysis_data_holder import AnalysisDataHolder
from handle_feedback import get_button_feedback
from load_data import load_data
from plot_overviews import plot_questions_tornado
from process_mining.process_mining import ProcessMining
from statistical_tests import is_t_test_applicable, plot_box_with_significance_bars, print_correlation_ranking

analysis_steps = ["plot_statistical_boxplots"]

sns.set_theme(context='paper', style='darkgrid', palette='colorblind')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

id_to_question_mapping = {
    23: "top3Features",
    11: "anchor",
    24: "shapAllFeatures",
    27: "least3Features",
    25: "ceterisParibus",
    13: "featureStatistics",
    7: "counterfactualAnyChange",
    0: "followUp",
    1: "whyExplanation",
    100: "notXaiMethod",
    99: "greeting",
    -1: "None"
}

dispaly_mappping = {
    "followUp": "Follow Up",
    "whyExplanation": "Why Explanation",
    "counterfactualAnyChange": "Counterfactual",
    "anchor": "Anchors",
    "featureStatistics": "Feature Statistics",
    "top3Features": "Top 3 Features",
    "shapAllFeatures": "Feature Influences",
    "ceterisParibus": "Ceteris Paribus",
    "least3Features": "Least 3 Features",
    "greeting": "Greeting",
    "notXaiMethod": "Not XAI Method",
}


def get_wort_and_best_users(df_with_score, score_name):
    # Calculate thresholds for the top 5% and bottom 5% based on the score
    top_5_percent_threshold = df_with_score[score_name].quantile(0.85)
    bottom_5_percent_threshold = df_with_score[score_name].quantile(0.15)

    # Filter best users = model_score >= top 5% threshold
    best_users = df_with_score[df_with_score[score_name] >= top_5_percent_threshold]

    # Filter worst users = model_score <= bottom 5% threshold
    worst_users = df_with_score[df_with_score[score_name] <= bottom_5_percent_threshold]

    if len(worst_users) > len(best_users):
        # Order by model_score and take the worst users
        worst_users = worst_users.sort_values(score_name, ascending=True)
        worst_users = worst_users.head(len(best_users))
    else:
        # Order by model_score and take the worst users
        best_users = best_users.sort_values(score_name, ascending=False)
        best_users = best_users.head(len(worst_users))
    # return user ids
    return best_users["id"].to_list(), worst_users["id"].to_list()


def make_question_count_df(questions_over_time_df, user_df):
    # For each user calculate the frequency of each question type
    user_question_freq = questions_over_time_df.groupby("user_id")["question_id"].value_counts().unstack()
    # Combine question id 25 and question id 13 to question id 99
    user_question_freq["feature_specific"] = user_question_freq[25] + user_question_freq[13]
    # Add model_score to the user_question_freq
    # Combine the rest of the questions to question id 100
    user_question_freq["general"] = user_question_freq[7] + user_question_freq[23] + user_question_freq[11] + \
                                    user_question_freq[27] + user_question_freq[24]
    user_question_freq = user_question_freq.merge(user_df[["id", "final_irt_score_mean", "study_condition"]],
                                                  left_on="user_id", right_on="id")
    return user_question_freq


def main():
    # Load data
    with open('db_config.json', 'r') as config_file:
        config = json.load(config_file)
    data = load_data(config["result_folder_path"])
    data["user_df"] = data['user_df_with_irt_scores']
    groups = ["static", "interactive"]

    # Get best and worst users from the interactive group for some analysis
    interactive_group = data["user_df"][data["user_df"]["study_condition"] == "interactive"]
    best_ids, wors_ids = get_wort_and_best_users(interactive_group, "final_irt_score_mean")

    if "plot_questions_tornado" in analysis_steps:
        # Replace question_id by string mapping
        data["questions_over_time_df"]["question_id"] = data["questions_over_time_df"]["question_id"].map(
            id_to_question_mapping)
        # And apply display mapping
        data["questions_over_time_df"]["question_id"] = data["questions_over_time_df"]["question_id"].map(
            dispaly_mappping)
        plot_questions_tornado(data["questions_over_time_df"], best_ids, wors_ids, save=True,
                               group1_name="highest U_model", group2_name="lowest U_model")

    if "plot_question_raking" in analysis_steps:
        print("Ranking of interactive questions for best")
        get_interactive_ranking(data["user_df"][data["user_df"]["study_condition"] == "interactive"],
                                filter_ids=best_ids)
        print("Ranking of interactive questions for worst")
        get_interactive_ranking(data["user_df"][data["user_df"]["study_condition"] == "interactive"],
                                filter_ids=wors_ids)

    if "print_buttons_feedback" in analysis_steps:
        # Get All Feedback button feedback per question
        print(get_button_feedback(data['event_df']))

    if "print_correlations" in analysis_steps:
        keep_cols = ["total_learning_time", "exp_instruction_time", "total_exp_time", "ml_knowledge",
                     "final_avg_confidence", "intro_avg_confidence", "final_irt_score_mean", "intro_irt_score_mean",
                     "accuracy_over_time"]
        print_correlation_ranking(data["user_df"], "final_irt_score_mean", "static", keep_cols=keep_cols)
        print_correlation_ranking(data["user_df"], "final_irt_score_mean", "interactive", keep_cols=keep_cols)

        user_question_freq = make_question_count_df(data["questions_over_time_df"], data["user_df"])

        print_correlation_ranking(user_question_freq, "final_irt_score_mean", "interactive")
        print_correlation_ranking(user_question_freq, "final_irt_score_mean", "static")

    # Confidence Plot
    if "plot_statistical_boxplots" in analysis_steps:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), sharey="all")
        for position, (variable_name, title_name, y_label_name) in enumerate(zip(
                ["intro_avg_confidence", "final_avg_confidence"],
                ["Initial Confidence", "Model Confidence"],
                ["Confidence", "Confidence"])):
            t_test = is_t_test_applicable(data["user_df"], "study_condition", variable_name,
                                          groups_to_compare=groups)
            plot_box_with_significance_bars(data["user_df"], "study_condition", variable_name,
                                            f'{title_name}', ttest=t_test, ax=axes[position],
                                            y_label_name=y_label_name)
        plt.tight_layout()
        path = "analysis_plots/" + f"initial_model_confidence.pdf"
        plt.savefig(path)

    # Model Understanding Plot
    if "plot_statistical_boxplots" in analysis_steps:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), sharey="all")
        for position, (variable_name, title_name, y_label_name) in enumerate(zip(
                ["intro_irt_score_mean", "final_irt_score_mean"],
                ["Initial Understanding", "Model Understanding"],
                ["IRT Scores", "IRT Scores"])):
            # t_test = is_t_test_applicable(data["user_df"], "study_condition", variable_name, groups)
            t_test = False
            plot_box_with_significance_bars(data["user_df"], "study_condition", variable_name,
                                            f'{title_name}', ttest=t_test,
                                            ax=axes[position],
                                            y_label_name=y_label_name,
                                            groups_to_compare=groups)
        plt.tight_layout()
        # plt.show()
        path = "analysis_plots/" + f"initial_model_understanding.pdf"
        plt.savefig(path)

    # Subjective Understanding Plot
    if "plot_statistical_boxplots" in analysis_steps:
        # Add subjective understanding col
        adh = AnalysisDataHolder(data["user_df"], data["event_df"], data["user_completed_df"])
        adh.add_self_assessment_value_column()

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3), sharey="all")
        for position, (variable_name, title_name, y_label_name) in enumerate(zip(
                ["subjective_understanding", "subjective_understanding"],
                ["Subjective Understanding", "Subjective Understanding"],
                ["Understanding Score", "Understanding Score"])):
            t_test = is_t_test_applicable(data["user_df"], "study_condition", variable_name, groups)
            plot_box_with_significance_bars(data["user_df"], "study_condition", variable_name,
                                            f'{title_name}', ttest=t_test,
                                            ax=axes[position],
                                            y_label_name=y_label_name,
                                            groups_to_compare=groups)
        plt.tight_layout()
        # plt.show()
        path = "analysis_plots/" + f"subjective_understanding.pdf"
        plt.savefig(path)

    # Process Mining
    if "process_mining" in analysis_steps:
        pm = ProcessMining()
        pm.create_pm_csv(data["questions_over_time_df"],
                         datapoint_count=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         target_user_ids=wors_ids,
                         target_group_name="worst")


if __name__ == "__main__":
    main()
