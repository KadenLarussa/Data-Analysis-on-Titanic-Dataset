#!/usr/bin/env python
# coding: utf-8

# # <center>CHALLENGE 4.1<center>

# <p>Team Name:Group 1
# <p>Student Names:Kaden Larussa, Brooks Schafer, Nick Solari
# <p>Student W#s:w0762283, w0711688, w0755643

# ## Instructions
# Use <b>generic</b> coding style unless hard-coded values are really necessary.<br>
# Your code must be efficient and use self-explanatory naming.<br>
# Use appropriate Python library methods for each task instead of using loops.<br>
# Run your entire code and save <b>BEFORE</b> submitting. Then, submit this <b>saved</b> copy.

# ## Definitions

# AGE_12   : Passengers with age in \[1, 13)<br>
# AGE_TEEN : Passengers with age in \[13, 20)<br>
# AGE_YOUNG: Passengers with age in \[20, 31)<br>
# AGE_OTHER: Passengers with age >= 31

# ## Imports

import pandas as pd
import matplotlib as plt
import numpy as np
import os


# ## Loading Data

# Load "titanic.csv" into the data frame object "data"</br>
# [train.csv](https://github.com/wesm/pydata-book/blob/2nd-edition/datasets/titanic/train.csv)

data = pd.read_csv('titanic.csv')


# ## Calculate Counts

# Calculate joint counts for each {Survived_i, Age_j}, i = {Yes, No}, j = {_12, _Teen, _Young, _Other}:

# Define age categories as per the provided specifications
def age_category(age):
    if age <= 12:
        return '_12'
    elif age > 12 and age < 20:
        return '_Teen'
    elif age >= 20 and age < 40:
        return '_Young'
    else:
        return '_Other'

data['Age_Category'] = data['Age'].apply(age_category)

data['Survived'] = data['Survived'].map({1: 'Yes', 0: 'No'})

joint_counts = pd.crosstab(data['Survived'], data['Age_Category'])

joint_counts


# Calculate marginal counts for survive categories:

marginal_counts_survived_corrected = data['Survived'].value_counts()

marginal_counts_survived_corrected


# Calculate marginal counts for age categories:

marginal_counts_age = data['Age_Category'].value_counts()

marginal_counts_age


# Calculate total count of samples:

total_count_samples = data.shape[0]  

total_count_samples


# Store all of the counts calculated above into the data frame "Count"<br>
# .    Column headers: Age categories and TOTAL<br>
# .    Row headers: survive categories, TOTAL<br>
# Print the object Count's content:

# Calculate the joint counts again to reset the previous manipulation
joint_counts = pd.crosstab(data['Survived'], data['Age_Category'])

# Add marginal totals for rows at the end of the joint_counts DataFrame
joint_counts.loc['TOTAL', :] = joint_counts.sum(axis=0)

# Add marginal totals for columns
joint_counts['TOTAL'] = joint_counts.sum(axis=1)

# The corrected "Count" DataFrame
Count = joint_counts


Count


# Save Counts into the text file "Count.csv" under the folder "OUTPUT".<br>
# .    Column headers: Age categories and TOTAL<br>
# .    Row headers: survive categories, TOTAL:

# Since there is no existing "OUTPUT" folder mentioned, it will be created in the specified directory if not present.
output_directory = 'OUTPUT/'
os.makedirs(output_directory, exist_ok=True)  # Create the OUTPUT directory if it doesn't exist

# Define the file path
output_file_path = os.path.join(output_directory, 'Count.csv')

# Save the Count dataframe to a CSV file
Count.to_csv(output_file_path, header=True, index=True)  # Include headers and index in the output file

output_file_path  # Return the path of the created file for access


# ## Calculate Probabilities

# Generate the Probability Table T2- P() as described. Store into the data frame object "P". Print the object's content:

P = Count.copy()  # Create a copy of the Count DataFrame

# Renaming the index to match the provided image for the probability table
P = P.rename(index={'Not Survived = 0': 'Not Survived', 'Survived = 1': 'Survived'})

# Renaming the columns to match the provided image for the probability table
P.columns = ['Age_12', 'Age_Other', 'Age_Teen', 'Age_Young', 'TOTAL']

# reorder because im retarded
P = P[['Age_12', 'Age_Teen', 'Age_Young', 'Age_Other', 'TOTAL']]

# Display the reordered 'P' DataFrame

P


# Generate the Probability Table T3_1- P(Survive | Age) as described. Store into the data frame object "P_SgA". Print the object's content:

# Calculating P(Survive | Age) which is the probability of survival within each age category
# This requires dividing the count of survivors by the total count within each age category

# Isolating the counts of survivors and non-survivors for each age category from the 'Count' DataFrame
survivors_count = Count.loc['Yes', ['_12', '_Teen', '_Young', '_Other']]
total_age_count = Count.loc['TOTAL', ['_12', '_Teen', '_Young', '_Other']]

# Calculating the conditional probability of survival within each age category
P_SgA = survivors_count / total_age_count

# Creating a DataFrame from the series and transposing it to match the requested format
P_SgA = pd.DataFrame(P_SgA).T

# Renaming index and columns to match the structure provided in the image
P_SgA.index = ['Survived | Age']
P_SgA.columns = ['Age_12', 'Age_Teen', 'Age_Young', 'Age_Other']

# Adding a row for 'Not Survived | Age' by subtracting 'Survived | Age' from 1
P_SgA.loc['Not Survived | Age'] = 1 - P_SgA.loc['Survived | Age']

# Adding the 'TOTAL' row for marginal probabilities of survival, which we already have in DataFrame 'P'
P_SgA.loc['TOTAL'] = P.loc['TOTAL', ['Age_12', 'Age_Teen', 'Age_Young', 'Age_Other']]

# Displaying the "P_SgA" DataFrame
P_SgA



# Generate the Probability Table T3_2- P(Survive | Age) P(Age) as described. Store into the data frame object "P_SgA_A". Print the object's content:

# Recalculating P(Survive | Age) P(Age) with the correct interpretation
# First, we need to re-calculate the probabilities of survival given an age category, and multiply by the probabilities of the age categories

# Re-extracting P(Age) for each age category from the 'P' DataFrame
P_Age = P.loc['TOTAL', ['Age_12', 'Age_Teen', 'Age_Young', 'Age_Other']]

# Re-calculating the conditional probabilities P(Survive | Age)
P_Survive_Given_Age = P_SgA.drop('TOTAL')

# Multiplying P(Survive | Age) by P(Age) to get the joint probabilities P(Survive and Age)
P_SgA_A = P_Survive_Given_Age.mul(P_Age, axis=1)

# Adding a 'TOTAL' row that sums the joint probabilities for each age category
P_SgA_A.loc['TOTAL'] = P_SgA_A.sum(axis=0)

# Displaying the DataFrame 'P_SgA_A' which now should match the structure of T3_2
P_SgA_A


# Generate the Probability Table T4_1- P(Age | Survive) as described. Store into the data frame object "P_AgS". Print the object's content:

# First, we'll separate the counts of survivors and non-survivors for each age category
survivors = Count.loc['Yes', :].drop('TOTAL')  # Drops the 'TOTAL' column to keep only age categories
non_survivors = Count.loc['No', :].drop('TOTAL')

# Now, we calculate the total number of survivors and non-survivors
total_survivors = survivors.sum()
total_non_survivors = non_survivors.sum()

# Calculate the conditional probabilities P(Age | Survived) and P(Age | Not Survived)
P_Age_given_Survived = survivors / total_survivors
P_Age_given_Not_Survived = non_survivors / total_non_survivors

# Combine the conditional probabilities into a DataFrame
P_AgS = pd.DataFrame({
    'Age_12': [P_Age_given_Survived['_12'], P_Age_given_Not_Survived['_12']],
    'Age_Teen': [P_Age_given_Survived['_Teen'], P_Age_given_Not_Survived['_Teen']],
    'Age_Young': [P_Age_given_Survived['_Young'], P_Age_given_Not_Survived['_Young']],
    'Age_Other': [P_Age_given_Survived['_Other'], P_Age_given_Not_Survived['_Other']]
}, index=['Survived', 'Not Survived'])

# Adding a 'TOTAL' column that sums the conditional probabilities for 'Survived' and 'Not Survived'
P_AgS['TOTAL'] = P_AgS.sum(axis=1)

P_AgS


# Generate the Probability Table T4_2- P(Age | Survive) P(Survive) as described. Store into the data frame object "P_AgS_S". Print the object's content:

# Assuming we have already calculated P_AgS which contains P(Age | Survive)

# Calculate marginal probabilities of survival and non-survival
P_Survive = Count.loc['Yes', 'TOTAL'] / total_count_samples
P_Not_Survive = Count.loc['No', 'TOTAL'] / total_count_samples

# Initialize P_AgS_S DataFrame to store P(Age | Survive) P(Survive)
P_AgS_S = P_AgS.copy()

# Multiply conditional probabilities by the marginal probabilities
P_AgS_S.loc['Survived'] *= P_Survive
P_AgS_S.loc['Not Survived'] *= P_Not_Survive

# Compute total joint probabilities
P_AgS_S.loc['TOTAL'] = P_AgS_S.sum(axis=0)

P_AgS_S
P_Survive = Count.loc['Yes', 'TOTAL'] / total_count_samples
P_Not_Survive = Count.loc['No', 'TOTAL'] / total_count_samples

# Initialize P_AgS_S DataFrame to store P(Age | Survive) P(Survive)
P_AgS_S = P_AgS.copy()

# Multiply conditional probabilities by the marginal probabilities
P_AgS_S.loc['Survived'] *= P_Survive
P_AgS_S.loc['Not Survived'] *= P_Not_Survive

# Compute total joint probabilities
P_AgS_S.loc['TOTAL'] = P_AgS_S.sum(axis=0)

P_AgS_S


# ## Compare

# Compare the row TOTAL of T3_1- P(Survive | Age) with the row TOTAL of T2- P(). Explain:

print(P_SgA, '\n\n', P)

# T3_1 'TOTAL' row shows conditional probabilities of survival per age group (sum should be 1 per group).
# T2 'TOTAL' row shows actual counts of people per age group, regardless of survival status


# Compare the row TOTAL of T3_2- P(Survive | Age) P(Age) with the row TOTAL of T2- P(). Explain:

print(P_SgA_A, '\n\n', P_AgS)

# T3_2 'TOTAL' row represents the combined probability of survival and age group distribution
# T2 'TOTAL' row shows actual counts of survival status.
# T3_2's probabilities reflect theoretical likelihoods, while T2's counts are empirical data


# Compare the column TOTAL of T4_1- P(Age | Survive) with the column TOTAL of T2- P(). Explain:

print(P_AgS, '\n\n', P)
# T4_1 'TOTAL' column shows probability distribution of age groups within each survival status (sums to 1)
# T2 'TOTAL' column shows actual counts of survivors and non-survivors, regardless of age group
# T4_1 provides theoretical likelihoods within survival categories, while T2 provides empirical data


# Compare the column TOTAL of T4_2- P(Age | Survive) P(Survive) with the column TOTAL of T2- P(). Explain:

print(P_AgS_S, '\n\n', P)

# T4_2 'TOTAL' column represents the combined probability of each age group and survival status
# T2 'TOTAL' column shows actual counts of survivors and non-survivors across age groups
# T4_2's probabilities reflect theoretical likelihoods, while T2's counts are empirical data


# ## <center> REFERENCES </center>
# List resources (book, internet page, etc.) that you used to complete this challenge.

# 
