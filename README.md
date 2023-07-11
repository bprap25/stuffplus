# stuffplus
An MIT Baseball Student-Athlete's attempt at building a stuff+ model
## Overview
Stuff+ is a metric used to quantify the quality of a pitch based solely on the pitch's attributes. Typically, most models are kept protected by their owners, whether they belong to an MLB org, or an independent R&D org. This is an individual's attempt at training a model that can be used by individuals in their own work, accessed via the website (can search by player name or input your own metrics), and read/understood openly on this repo.
### Training Methodology and Process
The input used to train this model is included, but not limited to: velocity, horizontal break, vertical break, release metrics, spin metrics, etc. The neural network trains these inputs against run value. Run value was decided upon as the optimal statistic to train the model on, as opposed to wOBA, xwOBA, etc. While run value carries some inherent biases (as it does not do the best in accounting for quality of contact/luck), it was decided on as the best statistic. Given the large number of pitches thrown over the course of a season, this normalizes for outliers, in expectation.
#### Understanding Stuff+ as a Statistic
The + scale is relatively new, which may seem intimidating, but is actually really easy to understand. A stuff+ of 100 is league average. Every 1 point above/below 100 means that the player has a pitch 1% better/worse than league average. For example, a score of 120 indicates that a pitch is 20% better than league average.
##### Credits
Data was obtained using [pybaseball](https://github.com/jldbc/pybaseball) and [Baseball Savant Run Value](https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats?type=pitcher&pitchType=&year=2023&team=&min=1).