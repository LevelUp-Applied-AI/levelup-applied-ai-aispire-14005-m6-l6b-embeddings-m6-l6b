# Stretch Analysis: Embedding Space Explorer

## Method

This stretch assignment visualizes two different embedding spaces:

1. A word-level embedding space using 200 selected GloVe word vectors.
2. A document-level embedding space using 20 BBC News article embeddings extracted with DistilBERT.

For the word-level visualization, I selected 200 words from the GloVe vocabulary and organized them into five semantic categories: sports, technology, business/finance, politics/government, and entertainment/media. These categories were chosen because they align closely with the BBC News corpus categories and make the word-level visualization easier to compare with the document-level visualization.

For dimensionality reduction, I used t-SNE for both plots. I chose t-SNE because it is useful for preserving local neighborhood structure, which makes it easier to see whether semantically related words or documents form visible clusters in 2D space. The GloVe word vectors were reduced from 50 dimensions to 2 dimensions, and the DistilBERT document embeddings were reduced from 768 dimensions to 2 dimensions.

---

## Word Embedding Plot Analysis

The GloVe word embedding visualization shows clear semantic structure across the five selected word categories.

The sports words form one of the cleanest clusters in the lower-right area of the plot. Words such as `football`, `team`, and `goal` appear close together, which makes sense because these words commonly occur in similar sports contexts such as matches, teams, players, competitions, and scoring. This suggests that GloVe captures sports vocabulary very strongly.

Technology words form another clear cluster in the upper-right area of the plot. Words such as `computer`, `mobile`, and `internet` appear close to each other, reflecting their shared connection to digital systems, devices, networks, and online services. This category is also relatively close to the entertainment/media region, which makes sense because modern media often overlaps with technology through television, digital platforms, mobile devices, and online distribution.

Business and finance words are mostly clustered in the upper-left area. Words such as `profit`, `market`, and `bank` appear in the same general region, showing that GloVe places financial and economic vocabulary near each other. This cluster is fairly distinct, although some business-related words naturally spread toward politics because financial news often overlaps with government policy, regulation, trade, and public economic decisions.

Politics and government words appear mostly in the left and central parts of the plot. Words such as `minister`, `government`, and `election` are related but not as tightly clustered as the sports words. This makes sense because politics is a broad semantic category. It includes elections, law, government institutions, leadership, public policy, conflict, and international relations, so the words are connected but cover several subtopics.

Entertainment and media words appear mostly on the right side of the plot. Words such as `television`, `film`, and `music` are in the same broad region, but the category is more spread out than sports. This is expected because entertainment/media includes several different subdomains such as film, music, television, journalism, performance, and broadcasting.

Overall, the word embedding plot shows that GloVe preserves meaningful semantic relationships. Sports and business/finance show strong cluster separation, while technology and entertainment/media show some natural overlap. This overlap is useful rather than problematic because it reflects real-world language relationships between digital technology, broadcasting, and media content.

---

## Document Embedding Plot Analysis

The DistilBERT document embedding visualization also shows meaningful structure across the 20 selected BBC News articles.

The sport articles form the clearest document cluster in the upper-right region of the plot. Documents `spor_2`, `spor_3`, and `spor_4` are very close to each other, while `spor_1` is slightly lower but still in the same general region. This suggests that DistilBERT strongly captures the shared sports context across these articles.

The tech articles form a clear cluster in the lower-right region. The four tech documents are separated from business and politics, which suggests that DistilBERT recognizes their shared digital and technology-related context. This cluster likely reflects vocabulary and meaning related to software, internet services, mobile technology, devices, online systems, or digital platforms.

The business documents appear in the lower-middle and central area of the plot. Some business documents are closer to the politics cluster than to the tech cluster. This pattern makes sense because business news often overlaps with government decisions, economic policy, markets, regulation, taxes, and public financial issues. For example, `busi_3` appears closer to the politics region than the lower business points, suggesting that this article may contain policy or public-economic context.

The politics documents form a fairly clear cluster in the middle-upper part of the plot. Documents `poli_1`, `poli_2`, `poli_3`, and `poli_4` are relatively close to each other, showing that DistilBERT captures common political context. This area is also not too far from some business documents, which reflects the natural relationship between politics and economics in news articles.

Entertainment is the most dispersed category in the document plot. Documents `ente_1` and `ente_4` are grouped on the far-left side, but `ente_2` appears closer to the sport region and `ente_3` appears closer to the tech region. This suggests that entertainment articles are more diverse than the other categories. Some entertainment articles may focus on television or drama, while others may involve digital media, websites, games, or online platforms. This shows that DistilBERT organizes articles by contextual meaning rather than only by the dataset category label.

Overall, the document embedding plot shows that DistilBERT preserves meaningful topic relationships. Sport and tech form the clearest clusters, business and politics show natural overlap, and entertainment is more spread out because its articles cover a wider range of subtopics.

---

## Comparison Between Word and Document Spaces

The word-level and document-level visualizations show similar high-level patterns.

In the GloVe word plot, sports words form a clean cluster, and in the DistilBERT document plot, sport articles also form one of the clearest clusters. This consistency suggests that sports vocabulary is semantically distinctive at both the word level and the document level.

Technology also appears clearly structured in both plots. Technology words cluster together in the GloVe plot, and tech articles cluster together in the DistilBERT plot. This suggests that technology-related language is strongly represented in both static word embeddings and contextual document embeddings.

Business and politics show more overlap in both spaces. In the word plot, business/finance and politics/government are near each other compared with categories like sports. In the document plot, some business articles appear closer to politics articles. This reflects the real-world overlap between economic topics and government policy.

Entertainment/media is more spread out in both plots. In the word plot, entertainment/media overlaps somewhat with technology, and in the document plot, entertainment articles are the most dispersed. This suggests that entertainment is a broad category with multiple subtopics, including film, TV, music, media platforms, and sometimes digital technology.

---

## Conclusion

The embedding visualizations reveal that both GloVe and DistilBERT capture meaningful semantic structure.

GloVe organizes individual words by broad semantic relationships. It produces strong clusters for categories like sports, technology, and business/finance, while also showing natural overlap between related areas such as technology and media.

DistilBERT organizes full news articles by contextual meaning. The document plot shows clear clusters for sport and tech, natural overlap between business and politics, and more dispersion in entertainment because entertainment articles cover a wider variety of subtopics.

Together, the two plots show why embedding visualizations are useful. They make high-dimensional semantic relationships visible, helping us understand which categories are clearly separated, which categories overlap, and which items behave like outliers.