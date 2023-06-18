# weather-markov-model
    Isaac Perks
    06-17-2023

# Description
A basic weather prediction hidden markov model using TensorFlow.

Same info on the data provided:
- Cold days are 0 and hot days are 1
- First day has 80% chance of being cold
- A cold day has a 70% chance of staying cold
- A hot day has a 80% chance of staying hot
- Each day the tempature is distributed with mean, and standard deviation of:
    - 0 and 5 for cold days
    - 15 and 10 on hot days

- Created distrobutions for use in markov model
    - Init distrobution, a categorical of .8 and .2 for our initial states
    - Transitional distrobution, a categorical of .7,.3 and .2,.8 for respective states
    - Observational distrobution, a normal with mean and standard deviation of [0, 15] and [5, 10]
- Set up markov model
    - Added in each distrobution to the core provided function
    - Set steps to 7
- Ran markov model to gather the means, and printed the values provided
