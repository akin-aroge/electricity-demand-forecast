""" Comments on the analysis and modelling process"""

ON_24HR_TREND_INTRO = """

## Daily Trends

Based on intuition, one could explore various trajectories which might be 
relevant to electricity demand. This may include factors based on time of the day, 
day of the week and month of the year. However, another way might be to leverage the 
power of unsupervised techniques in discerning patterns from the data with limited assumptions.

"""

ON_24HR_TREND_PLOT = """

The plot shows randomly sampled 24-hour profiles color-coded by months. This provides a lot of insights into demand pattern by time and month. 

It appears there are two major general patterns, one with 2 peaks, and the other with one peaks. The legend reveals those to correspond to seasons: summer and winter months.

We would leverage the decompositional power of pricipal component analysis to further investigate this while taking all the data into account, rather than a random sample.
"""


ON_PCA_RESULT = """

The plot above indicates that >95\% of the variance in the 24-hour profiles may be captures by projection onto two principal axes. This suggests that there are two major categories of daily electricity usage patterns as suggested earlier. 

But first let's project the 24-hour profiles on the two principal axes and plot the resulting coefficients:

"""

ON_PCA_COEFFS = """

The projection reveals the two distinct groups with high magnitude of principal coefficients on the the 1st or 2nd axes. Furthermore the load magnitude overlay shows that a wide range of load magnitude may be seen across the clusters.

We would try to explicitly exrtact the two clusters in order to identify the underlying days, by mapping the cluster number to the dates. To achive that, the unsupervised gaussian mixture model is employed.


"""

ON_PCA_COEFFS_CLUSTER = """

The plot shows the obtained clusters. With the clusters obtained, the are 
be mapped back to the dates and grouped accordingly.


"""

ON_PCA_CLUSTER_PROFILES = """

The above plot represents the mean 24-hour profile for each of the clusters base on the two principal axes.

What we have here are really two major daily profiles representative of what happens across the year. 

Cluster 0, features a daily pattern which includes a declining demand from midnight till very early in the morning following by a gradual increase which peaks in the evening around 17:00 and then declines again towards midnight. 

In clsuter 1, there are two peaks, one around 8am and the other around 8 pm, with intermediate dips around 3:00 and 16:00.

To shed more light on what separates these two days, let's bring in the different months and overlay on the plot of the coefficients along the PCs.


"""

ON_PCA_COEFFS_MONTHS = """

mid-year cluster 0:  summer months : 
    - demand: colling during the mid-day period
    - typically no need for morning cooling
January/December are cluster 1: winter months
    - demand: warming in the morning | warming in the evenings

The picture becomes clear now. It appears that cluster 0 are mostly days in the middle of the year, while cluster 1 is dominated by days in the beginning and ending months of the year. This essentially represents the two major seasons of summer and winter months.

So without any initial assumption, we have noticed two major daily patterns and then discovered how those represent summer and winter electricity usage patterns when we overlay the months on the principal co-efficients. While there are some similarities in both seasons, there are notable differences. In the summer months there is a simgle peak which occurs in the evening around 6 pm while here are two peaks in the winter months; one in the morning aroun 8am and the other around 8pm.

To understand this summer winter patterns,, it may be worth considering the nature of electricity use in those tmes. In the summer months, there is often need for cooling in the relatively warmer afternoons, which explains the rise from morning till the early evening where the need for cooling is reduced. In the winter months however, heating would be needed in the mornings and evenings when it is cool and least required in the afternoon where the temperature is typically highest.

This analysis also underscore the strong influence of temperature/weather on energy usage. North Carolina is located well North of the eqautor with significiant temperature swings.  It is possible to imagine that areas closer to the equator where temperatures are more stable year round do not hvae as strong correlation with temperature.

"""
