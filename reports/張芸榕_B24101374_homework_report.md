The dataset is about 847 reddit posts categorized by positive, neutral or negative sentiment label (load_dataset_1)and there's no missing values within(check_missing_4), but 23 duplicates exist(check_duplicates_6). Those duplicates might be spam, so I set drop=true and rerun the check_duplicates_tool. The tool reported 23 duplicates, but 34 rows were removed (check_duplicates_7), because keep=False also removes the original copy. So 11 different texts were posted more than once, which looks like spam. 32 of the 34 removed posts were neutral (0.0), so the duplicates were probably not promotional spam. maybe they were automated or very short common replies (describe_data_8). Since both the document length and the total vocabulary size decreased relative to 20 Newsgroups, the two effects balanced each other out (99.5249%) (dtm_10).

![Document length by sentiment](../plots/張芸榕_homework_plot_describe_data_8.png)


After running the term_frequency_tool, I found one interesting fact is that "he" is counted 132 times and ranks 15th in this reddit dataset (term_freq_11) while "she" isn't in the top 25 list.

Variance tool kept only 93 of 4,313 terms (2.2%) (variance_filter_13), and useful words like “stocks”, “analysis”, and “down” were removed, just like “targa” in 20 Newsgroups. From pearson test(pearson_filter_14), we can see positive adjectives do appear in the positive sentiment label texts, while from pearson_filter_15 for negative texts, the kept terms include "he","his", "him", "trump", and swear word. This results can further explain why "he" is counted 132 times and who "he" probably refers to. However, in pearson_filter_15, it can be observed that 'shit', 'delusional', and 'clown' are removed probably because neutral posts also use them, so they are not distinctive enough. In the bar chart of pearson_filter_14, all the terms are positive, but the values are low (below 0.16), and several terms have exactly the same value, such as “main”, “tldr”, “huge”, and “nice”. This suggests they come from the same one or two posts.
![Pearson correlation for positive posts](../plots/張芸榕_homework_plot_turn108.png)

Spearman_filter_16 results turned out to be very similar to pearson_filter_15. 8 of the top 10 terms are the same (['he', 'his', 'will', 'trump', 'fuck','him', 'economy',  'until']). In 20 Newsgroups, the two methods were very different because of very long outlier posts.However, Even the longest Reddit post (2,753 characters) (describe_data_8) is much shorter than the longest 20 Newsgroups post, so term counts stay small and the two methods give similar results.

we can see three sentiment classes are completely mixed (reduce_tsne_17). wording alone seems difficult to separate sentiments.
![t-SNE plot](../plots/張芸榕_homework_plot_reduce_tsne_17.png)

Since Reddit posts are short, length has less effect compared with 20 Newsgroups. 
![Feature correlation matrix](../plots/張芸榕_homework_plot_feat_corr_18.png)


From mine_patterns_19, “trump” (42 posts) and “china” (33 posts) are the top patterns among negative posts. All top patterns are single words, maybe because posts are short, so two words rarely appear together. 


