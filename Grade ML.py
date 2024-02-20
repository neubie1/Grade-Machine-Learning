import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssl
from warnings import simplefilter
from prettytable import PrettyTable
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

simplefilter(action = 'ignore', category = FutureWarning)

# Read in data and and corresponding column names.
ssl._create_default_https_context = ssl._create_stdlib_context
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00623/DATA.csv',
                   header=0, sep=';')

# ==== Visualize the data using Bar Plots ====

# Visualize the amount of female and male students
female = 0  # counter for female students
male = 0  # counter for male students

gender = data['2']

# Count the total of male and female students.
for g in gender:
    # If 1 then female, if 2 then male.
    if g == 1:
        female = female + 1
    else:
        male = male + 1

f_m_data = {'Female': female, 'Male': male}

plt.bar(f_m_data.keys(), f_m_data.values())

plt.ylabel('Number of Students')
plt.xlabel('Gender')
plt.title(label='The Number of Students that are Female and Male')

plt.show()

# Visualize how a mother’s education correlate with grade
mom_data = data[['11', 'GRADE']]

FAIL = [0, 0, 0, 0, 0, 0, 0, 0]
DD = [0, 0, 0, 0, 0, 0, 0, 0]
DC = [0, 0, 0, 0, 0, 0, 0, 0]
CC = [0, 0, 0, 0, 0, 0, 0, 0]
CB = [0, 0, 0, 0, 0, 0, 0, 0]
BB = [0, 0, 0, 0, 0, 0, 0, 0]
BA = [0, 0, 0, 0, 0, 0, 0, 0]
AA = [0, 0, 0, 0, 0, 0, 0, 0]

for index, row in mom_data.iterrows():
    if row['GRADE'] == 0:
        if row['11'] == 1:
            FAIL[0] += 1
        elif row['11'] == 2:
            FAIL[1] += 1
        elif row['11'] == 3:
            FAIL[2] += 1
        elif row['11'] == 4:
            FAIL[3] += 1
        elif row['11'] == 5:
            FAIL[4] += 1
        else:
            FAIL[5] += 1
    if row['GRADE'] == 1:
        if row['11'] == 1:
            DD[0] += 1
        elif row['11'] == 2:
            DD[1] += 1
        elif row['11'] == 3:
            DD[2] += 1
        elif row['11'] == 4:
            DD[3] += 1
        elif row['11'] == 5:
            DD[4] += 1
        else:
            DD[5] += 1
    if row['GRADE'] == 2:
        if row['11'] == 1:
            DC[0] += 1
        elif row['11'] == 2:
            DC[1] += 1
        elif row['11'] == 3:
            DC[2] += 1
        elif row['11'] == 4:
            DC[3] += 1
        elif row['11'] == 5:
            DC[4] += 1
        else:
            DC[5] += 1
    if row['GRADE'] == 3:
        if row['11'] == 1:
            CC[0] += 1
        elif row['11'] == 2:
            CC[1] += 1
        elif row['11'] == 3:
            CC[2] += 1
        elif row['11'] == 4:
            CC[3] += 1
        elif row['11'] == 5:
            CC[4] += 1
        else:
            CC[5] += 1
    if row['GRADE'] == 4:
        if row['11'] == 1:
            CB[0] += 1
        elif row['11'] == 2:
            CB[1] += 1
        elif row['11'] == 3:
            CB[2] += 1
        elif row['11'] == 4:
            CB[3] += 1
        elif row['11'] == 5:
            CB[4] += 1
        else:
            CB[5] += 1
    if row['GRADE'] == 5:
        if row['11'] == 1:
            BB[0] += 1
        elif row['11'] == 2:
            BB[1] += 1
        elif row['11'] == 3:
            BB[2] += 1
        elif row['11'] == 4:
            BB[3] += 1
        elif row['11'] == 5:
            BB[4] += 1
        else:
            BB[5] += 1
    if row['GRADE'] == 6:
        if row['11'] == 1:
            BA[0] += 1
        elif row['11'] == 2:
            BA[1] += 1
        elif row['11'] == 3:
            BA[2] += 1
        elif row['11'] == 4:
            BA[3] += 1
        elif row['11'] == 5:
            BA[4] += 1
        else:
            BA[5] += 1
    if row['GRADE'] == 7:
        if row['11'] == 1:
            AA[0] += 1
        elif row['11'] == 2:
            AA[1] += 1
        elif row['11'] == 3:
            AA[2] += 1
        elif row['11'] == 4:
            AA[3] += 1
        elif row['11'] == 5:
            AA[4] += 1
        else:
            AA[5] += 1

grades = ('Fail', 'DD', 'DC', 'CC', 'CB', 'BB', 'BA', 'AA')
mom_ed = {'Primary School': (FAIL[0], DD[0], DC[0], CC[0], CB[0], BB[0], BA[0], AA[0]),
          'Secondary School': (FAIL[1], DD[1], DC[1], CC[1], CB[1], BB[1], BA[1], AA[1]),
          'High School': (FAIL[2], DD[2], DC[2], CC[2], CB[2], BB[2], BA[2], AA[2]),
          'University': (FAIL[3], DD[3], DC[3], CC[3], CB[3], BB[3], BA[3], AA[3]),
          'MSc.': (FAIL[4], DD[4], DC[4], CC[4], CB[4], BB[4], BA[4], AA[4]),
          'Ph.D.': (FAIL[5], DD[5], DC[5], CC[5], CB[5], BB[5], BA[5], AA[5])}

# Arrange bar plot.
x = np.arange(len(grades))
width = 0.14
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for education, total in mom_ed.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, total, width, label=education)
    ax.bar_label(rects, padding=1)
    multiplier += 1

ax.set_ylabel('Number of Students')
ax.set_xlabel('Grade')
ax.set_title('Mother\'s education per Grade')
ax.set_xticks(x + width, grades)
ax.legend(loc='upper left', ncols=1)
ax.set_ylim(0, 25)

plt.show()

# Visualize how a fathers’s education correlate with grade
dad_data = data[['12', 'GRADE']]

FAIL = [0, 0, 0, 0, 0, 0, 0, 0]
DD = [0, 0, 0, 0, 0, 0, 0, 0]
DC = [0, 0, 0, 0, 0, 0, 0, 0]
CC = [0, 0, 0, 0, 0, 0, 0, 0]
CB = [0, 0, 0, 0, 0, 0, 0, 0]
BB = [0, 0, 0, 0, 0, 0, 0, 0]
BA = [0, 0, 0, 0, 0, 0, 0, 0]
AA = [0, 0, 0, 0, 0, 0, 0, 0]

for index, row in dad_data.iterrows():
    if row['GRADE'] == 0:
        if row['12'] == 1:
            FAIL[0] += 1
        elif row['12'] == 2:
            FAIL[1] += 1
        elif row['12'] == 3:
            FAIL[2] += 1
        elif row['12'] == 4:
            FAIL[3] += 1
        elif row['12'] == 5:
            FAIL[4] += 1
        else:
            FAIL[5] += 1
    if row['GRADE'] == 1:
        if row['12'] == 1:
            DD[0] += 1
        elif row['12'] == 2:
            DD[1] += 1
        elif row['12'] == 3:
            DD[2] += 1
        elif row['12'] == 4:
            DD[3] += 1
        elif row['12'] == 5:
            DD[4] += 1
        else:
            DD[5] += 1
    if row['GRADE'] == 2:
        if row['12'] == 1:
            DC[0] += 1
        elif row['12'] == 2:
            DC[1] += 1
        elif row['12'] == 3:
            DC[2] += 1
        elif row['12'] == 4:
            DC[3] += 1
        elif row['12'] == 5:
            DC[4] += 1
        else:
            DC[5] += 1
    if row['GRADE'] == 3:
        if row['12'] == 1:
            CC[0] += 1
        elif row['12'] == 2:
            CC[1] += 1
        elif row['12'] == 3:
            CC[2] += 1
        elif row['12'] == 4:
            CC[3] += 1
        elif row['12'] == 5:
            CC[4] += 1
        else:
            CC[5] += 1
    if row['GRADE'] == 4:
        if row['12'] == 1:
            CB[0] += 1
        elif row['12'] == 2:
            CB[1] += 1
        elif row['12'] == 3:
            CB[2] += 1
        elif row['12'] == 4:
            CB[3] += 1
        elif row['12'] == 5:
            CB[4] += 1
        else:
            CB[5] += 1
    if row['GRADE'] == 5:
        if row['12'] == 1:
            BB[0] += 1
        elif row['12'] == 2:
            BB[1] += 1
        elif row['12'] == 3:
            BB[2] += 1
        elif row['12'] == 4:
            BB[3] += 1
        elif row['12'] == 5:
            BB[4] += 1
        else:
            BB[5] += 1
    if row['GRADE'] == 6:
        if row['12'] == 1:
            BA[0] += 1
        elif row['12'] == 2:
            BA[1] += 1
        elif row['12'] == 3:
            BA[2] += 1
        elif row['12'] == 4:
            BA[3] += 1
        elif row['12'] == 5:
            BA[4] += 1
        else:
            BA[5] += 1
    if row['GRADE'] == 7:
        if row['12'] == 1:
            AA[0] += 1
        elif row['12'] == 2:
            AA[1] += 1
        elif row['12'] == 3:
            AA[2] += 1
        elif row['12'] == 4:
            AA[3] += 1
        elif row['12'] == 5:
            AA[4] += 1
        else:
            AA[5] += 1

grades = ('Fail', 'DD', 'DC', 'CC', 'CB', 'BB', 'BA', 'AA')
dad_ed = {'Primary School': (FAIL[0], DD[0], DC[0], CC[0], CB[0], BB[0], BA[0], AA[0]),
          'Secondary School': (FAIL[1], DD[1], DC[1], CC[1], CB[1], BB[1], BA[1], AA[1]),
          'High School': (FAIL[2], DD[2], DC[2], CC[2], CB[2], BB[2], BA[2], AA[2]),
          'University': (FAIL[3], DD[3], DC[3], CC[3], CB[3], BB[3], BA[3], AA[3]),
          'MSc.': (FAIL[4], DD[4], DC[4], CC[4], CB[4], BB[4], BA[4], AA[4]),
          'Ph.D.': (FAIL[5], DD[5], DC[5], CC[5], CB[5], BB[5], BA[5], AA[5])}

# Arrange bar plot.
x = np.arange(len(grades))
width = 0.14
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for education, total in dad_ed.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, total, width, label=education)
    ax.bar_label(rects, padding=1)
    multiplier += 1

ax.set_ylabel('Number of Students')
ax.set_xlabel('Grade')
ax.set_title('Father\'s Education per Grade')
ax.set_xticks(x + width, grades)
ax.legend(loc='upper left', ncols=1)
ax.set_ylim(0, 25)

plt.show()

# Visualize how a parental relationship affects grade.
p_relation_data = data[['14', 'GRADE']]

FAIL = [0, 0, 0, 0, 0, 0, 0, 0]
DD = [0, 0, 0, 0, 0, 0, 0, 0]
DC = [0, 0, 0, 0, 0, 0, 0, 0]
CC = [0, 0, 0, 0, 0, 0, 0, 0]
CB = [0, 0, 0, 0, 0, 0, 0, 0]
BB = [0, 0, 0, 0, 0, 0, 0, 0]
BA = [0, 0, 0, 0, 0, 0, 0, 0]
AA = [0, 0, 0, 0, 0, 0, 0, 0]

for index, row in p_relation_data.iterrows():
    if row['GRADE'] == 0:
        if row['14'] == 1:
            FAIL[0] += 1
        elif row['14'] == 2:
            FAIL[1] += 1
        else:
            FAIL[2] += 1
    if row['GRADE'] == 1:
        if row['14'] == 1:
            DD[0] += 1
        elif row['14'] == 2:
            DD[1] += 1
        else:
            DD[2] += 1
    if row['GRADE'] == 2:
        if row['14'] == 1:
            DC[0] += 1
        elif row['14'] == 2:
            DC[1] += 1
        else:
            DC[2] += 1
    if row['GRADE'] == 3:
        if row['14'] == 1:
            CC[0] += 1
        elif row['14'] == 2:
            CC[1] += 1
        else:
            CC[2] += 1
    if row['GRADE'] == 4:
        if row['14'] == 1:
            CB[0] += 1
        elif row['14'] == 2:
            CB[1] += 1
        else:
            CB[2] += 1
    if row['GRADE'] == 5:
        if row['14'] == 1:
            BB[0] += 1
        elif row['14'] == 2:
            BB[1] += 1
        else:
            BB[2] += 1
    if row['GRADE'] == 6:
        if row['14'] == 1:
            BA[0] += 1
        elif row['14'] == 2:
            BA[1] += 1
        else:
            BA[2] += 1
    if row['GRADE'] == 7:
        if row['14'] == 1:
            AA[0] += 1
        elif row['14'] == 2:
            AA[1] += 1
        else:
            AA[2] += 1

grades = ('Fail', 'DD', 'DC', 'CC', 'CB', 'BB', 'BA', 'AA')
p_status = {'Married': (FAIL[0], DD[0], DC[0], CC[0], CB[0], BB[0], BA[0], AA[0]),
            'Divorced': (FAIL[1], DD[1], DC[1], CC[1], CB[1], BB[1], BA[1], AA[1]),
            'Deceased': (FAIL[2], DD[2], DC[2], CC[2], CB[2], BB[2], BA[2], AA[2])}

# Arrange bar plot.
x = np.arange(len(grades))
width = 0.25
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for status, total in p_status.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, total, width, label=status)
    ax.bar_label(rects, padding=1)
    multiplier += 1

ax.set_ylabel('Number of Students')
ax.set_xlabel('Grade')
ax.set_title('Parental Status per Grade')
ax.set_xticks(x + width, grades)
ax.legend(loc='upper left', ncols=1)
ax.set_ylim(0, 50)

plt.show()

# Visualize how attendance affects grade.
attendace_data = data[['22', 'GRADE']]

FAIL = [0, 0, 0, 0, 0, 0, 0, 0]
DD = [0, 0, 0, 0, 0, 0, 0, 0]
DC = [0, 0, 0, 0, 0, 0, 0, 0]
CC = [0, 0, 0, 0, 0, 0, 0, 0]
CB = [0, 0, 0, 0, 0, 0, 0, 0]
BB = [0, 0, 0, 0, 0, 0, 0, 0]
BA = [0, 0, 0, 0, 0, 0, 0, 0]
AA = [0, 0, 0, 0, 0, 0, 0, 0]

for index, row in attendace_data.iterrows():
    if row['GRADE'] == 0:
        if row['22'] == 1:
            FAIL[0] += 1
        elif row['22'] == 2:
            FAIL[1] += 1
        else:
            FAIL[2] += 1
    if row['GRADE'] == 1:
        if row['22'] == 1:
            DD[0] += 1
        elif row['22'] == 2:
            DD[1] += 1
        else:
            DD[2] += 1
    if row['GRADE'] == 2:
        if row['22'] == 1:
            DC[0] += 1
        elif row['22'] == 2:
            DC[1] += 1
        else:
            DC[2] += 1
    if row['GRADE'] == 3:
        if row['22'] == 1:
            CC[0] += 1
        elif row['22'] == 2:
            CC[1] += 1
        else:
            CC[2] += 1
    if row['GRADE'] == 4:
        if row['22'] == 1:
            CB[0] += 1
        elif row['22'] == 2:
            CB[1] += 1
        else:
            CB[2] += 1
    if row['GRADE'] == 5:
        if row['22'] == 1:
            BB[0] += 1
        elif row['22'] == 2:
            BB[1] += 1
        else:
            BB[2] += 1
    if row['GRADE'] == 6:
        if row['22'] == 1:
            BA[0] += 1
        elif row['22'] == 2:
            BA[1] += 1
        else:
            BA[2] += 1
    if row['GRADE'] == 7:
        if row['22'] == 1:
            AA[0] += 1
        elif row['22'] == 2:
            AA[1] += 1
        else:
            AA[2] += 1

grades = ('Fail', 'DD', 'DC', 'CC', 'CB', 'BB', 'BA', 'AA')
attendance = {'Always': (FAIL[0], DD[0], DC[0], CC[0], CB[0], BB[0], BA[0], AA[0]),
              'Sometimes': (FAIL[1], DD[1], DC[1], CC[1], CB[1], BB[1], BA[1], AA[1]),
              'Never': (FAIL[2], DD[2], DC[2], CC[2], CB[2], BB[2], BA[2], AA[2])}

# Arrange bar plot.
x = np.arange(len(grades))
width = 0.25
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for status, total in attendance.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, total, width, label=status)
    ax.bar_label(rects, padding=1)
    multiplier += 1

ax.set_ylabel('Number of Students')
ax.set_xlabel('Grade')
ax.set_title('Attendance per Grade')
ax.set_xticks(x + width, grades)
ax.legend(loc='upper left', ncols=1)
ax.set_ylim(0, 40)

plt.show()

# Visualize how study time correlates with grade
study_data = data[['17', 'GRADE']]

FAIL = [0, 0, 0, 0, 0, 0, 0, 0]
DD = [0, 0, 0, 0, 0, 0, 0, 0]
DC = [0, 0, 0, 0, 0, 0, 0, 0]
CC = [0, 0, 0, 0, 0, 0, 0, 0]
CB = [0, 0, 0, 0, 0, 0, 0, 0]
BB = [0, 0, 0, 0, 0, 0, 0, 0]
BA = [0, 0, 0, 0, 0, 0, 0, 0]
AA = [0, 0, 0, 0, 0, 0, 0, 0]

for index, row in study_data.iterrows():
    if row['GRADE'] == 0:
        if row['17'] == 1:
            FAIL[0] += 1
        elif row['17'] == 2:
            FAIL[1] += 1
        elif row['17'] == 3:
            FAIL[2] += 1
        elif row['17'] == 4:
            FAIL[3] += 1
        else:
            FAIL[4] += 1
    if row['GRADE'] == 1:
        if row['17'] == 1:
            DD[0] += 1
        elif row['17'] == 2:
            DD[1] += 1
        elif row['17'] == 3:
            DD[2] += 1
        elif row['17'] == 4:
            DD[3] += 1
        else:
            DD[4] += 1
    if row['GRADE'] == 2:
        if row['17'] == 1:
            DC[0] += 1
        elif row['17'] == 2:
            DC[1] += 1
        elif row['17'] == 3:
            DC[2] += 1
        elif row['17'] == 4:
            DC[3] += 1
        else:
            DC[4] += 1
    if row['GRADE'] == 3:
        if row['17'] == 1:
            CC[0] += 1
        elif row['17'] == 2:
            CC[1] += 1
        elif row['17'] == 3:
            CC[2] += 1
        elif row['17'] == 4:
            CC[3] += 1
        else:
            CC[4] += 1
    if row['GRADE'] == 4:
        if row['17'] == 1:
            CB[0] += 1
        elif row['17'] == 2:
            CB[1] += 1
        elif row['17'] == 3:
            CB[2] += 1
        elif row['17'] == 4:
            CB[3] += 1
        else:
            CB[4] += 1
    if row['GRADE'] == 5:
        if row['17'] == 1:
            BB[0] += 1
        elif row['17'] == 2:
            BB[1] += 1
        elif row['17'] == 3:
            BB[2] += 1
        elif row['17'] == 4:
            BB[3] += 1
        else:
            BB[4] += 1
    if row['GRADE'] == 6:
        if row['17'] == 1:
            BA[0] += 1
        elif row['17'] == 2:
            BA[1] += 1
        elif row['17'] == 3:
            BA[2] += 1
        elif row['17'] == 4:
            BA[3] += 1
        else:
            BA[4] += 1
    if row['GRADE'] == 7:
        if row['17'] == 1:
            AA[0] += 1
        elif row['17'] == 2:
            AA[1] += 1
        elif row['17'] == 3:
            AA[2] += 1
        elif row['17'] == 4:
            AA[3] += 1
        else:
            AA[4] += 1

grades = ('Fail', 'DD', 'DC', 'CC', 'CB', 'BB', 'BA', 'AA')
study_time = {'None': (FAIL[0], DD[0], DC[0], CC[0], CB[0], BB[0], BA[0], AA[0]),
              '<5 Hours': (FAIL[1], DD[1], DC[1], CC[1], CB[1], BB[1], BA[1], AA[1]),
              '6-10 Hours': (FAIL[2], DD[2], DC[2], CC[2], CB[2], BB[2], BA[2], AA[2]),
              '11-20 Hours': (FAIL[3], DD[3], DC[3], CC[3], CB[3], BB[3], BA[3], AA[3]),
              'More Than 20 Hours': (FAIL[4], DD[4], DC[4], CC[4], CB[4], BB[4], BA[4], AA[4])}

# Arrange bar plot.
x = np.arange(len(grades))
width = 0.15
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for time, total in study_time.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, total, width, label=time)
    ax.bar_label(rects, padding=1)
    multiplier += 1

ax.set_ylabel('Number of Students')
ax.set_xlabel('Grade')
ax.set_title('Weekly Study Time per Grade')
ax.set_xticks(x + width, grades)
ax.legend(loc='upper left', ncols=1)
ax.set_ylim(0, 30)

plt.show()

# Visualize how note taking affects grade.
note_data = data[['25', 'GRADE']]

FAIL = [0, 0, 0, 0, 0, 0, 0, 0]
DD = [0, 0, 0, 0, 0, 0, 0, 0]
DC = [0, 0, 0, 0, 0, 0, 0, 0]
CC = [0, 0, 0, 0, 0, 0, 0, 0]
CB = [0, 0, 0, 0, 0, 0, 0, 0]
BB = [0, 0, 0, 0, 0, 0, 0, 0]
BA = [0, 0, 0, 0, 0, 0, 0, 0]
AA = [0, 0, 0, 0, 0, 0, 0, 0]

for index, row in note_data.iterrows():
    if row['GRADE'] == 0:
        if row['25'] == 1:
            FAIL[0] += 1
        elif row['25'] == 2:
            FAIL[1] += 1
        else:
            FAIL[2] += 1
    if row['GRADE'] == 1:
        if row['25'] == 1:
            DD[0] += 1
        elif row['25'] == 2:
            DD[1] += 1
        else:
            DD[2] += 1
    if row['GRADE'] == 2:
        if row['25'] == 1:
            DC[0] += 1
        elif row['25'] == 2:
            DC[1] += 1
        else:
            DC[2] += 1
    if row['GRADE'] == 3:
        if row['25'] == 1:
            CC[0] += 1
        elif row['25'] == 2:
            CC[1] += 1
        else:
            CC[2] += 1
    if row['GRADE'] == 4:
        if row['25'] == 1:
            CB[0] += 1
        elif row['25'] == 2:
            CB[1] += 1
        else:
            CB[2] += 1
    if row['GRADE'] == 5:
        if row['25'] == 1:
            BB[0] += 1
        elif row['25'] == 2:
            BB[1] += 1
        else:
            BB[2] += 1
    if row['GRADE'] == 6:
        if row['25'] == 1:
            BA[0] += 1
        elif row['25'] == 2:
            BA[1] += 1
        else:
            BA[2] += 1
    if row['GRADE'] == 7:
        if row['25'] == 1:
            AA[0] += 1
        elif row['25'] == 2:
            AA[1] += 1
        else:
            AA[2] += 1

grades = ('Fail', 'DD', 'DC', 'CC', 'CB', 'BB', 'BA', 'AA')
note_taking = {'Never': (FAIL[0], DD[0], DC[0], CC[0], CB[0], BB[0], BA[0], AA[0]),
               'Sometimes': (FAIL[1], DD[1], DC[1], CC[1], CB[1], BB[1], BA[1], AA[1]),
               'Always': (FAIL[2], DD[2], DC[2], CC[2], CB[2], BB[2], BA[2], AA[2])}

# Arrange bar plot.
x = np.arange(len(grades))
width = 0.25
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for status, total in note_taking.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, total, width, label=status)
    ax.bar_label(rects, padding=1)
    multiplier += 1

ax.set_ylabel('Number of Students')
ax.set_xlabel('Grade')
ax.set_title('Note Taking per Grade')
ax.set_xticks(x + width, grades)
ax.legend(loc='upper left', ncols=1)
ax.set_ylim(0, 40)

plt.show()

# ==== Preprocessing ====
# Remove the Student ID and Course ID columns.
data = data.drop(columns='STUDENT ID')
data = data.drop(columns='COURSE ID')

# Remove the columns based on opinions.
data = data.drop(columns='21')
data = data.drop(columns='27')
data = data.drop(columns='28')

# Let y be the dependant variable, label.
y = data['GRADE']

# Let X be the independent variable, aka every other column in data.
X = data.drop(columns='GRADE')
X_columns = X.columns
X = X.values

# ==== KNN ====
dataTable = PrettyTable(['', 'KNN Accuracy'])

# Set k
knn = KNeighborsClassifier(n_neighbors=77)

# 5-Fold Cross Validation
k_data = KFold(n_splits=5, shuffle=True)

accuracy_total = 0

for i, (train, test) in enumerate(k_data.split(X)):
    # Use Kfold indexes for train and test data.
    X_train = X[train]
    y_train = y[train]
    X_train = pd.DataFrame(X_train)
    X_train.columns = X_columns

    X_test = X[test]
    y_test = y[test]
    X_test = pd.DataFrame(X_test)
    X_test.columns = X_columns

    # Train data.
    knn.fit(X_train, y_train)

    # Predict with test data for accuracy.
    pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    accuracy_total = accuracy_total + accuracy

    dataTable.add_row(['Fold ' + str(i + 1), accuracy])

# Calculate average scores.
average_accuracy = accuracy_total / 5

# Add average scores to table
dataTable.add_row(['Average', average_accuracy])

print(dataTable)

# ==== Logistic Regression ====
dataTable = PrettyTable(['', 'Logistic Regression Accuracy'])

# 5-Fold Cross Validation
k_data = KFold(n_splits=5, shuffle=True)

accuracy_total = 0

for i, (train, test) in enumerate(k_data.split(X)):
    # Use Kfold indexes for train and test data.
    X_train = X[train]
    y_train = y[train]
    X_train = pd.DataFrame(X_train)
    X_train.columns = X_columns

    X_test = X[test]
    y_test = y[test]
    X_test = pd.DataFrame(X_test)
    X_test.columns = X_columns

    # Train data.
    log_reg = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=10000)
    log_reg.fit(X_train, y_train)

    # Predict with test data for accuracy.
    pred = log_reg.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    accuracy_total = accuracy_total + accuracy

    dataTable.add_row(['Fold ' + str(i + 1), accuracy])

# Calculate average scores.
average_accuracy = accuracy_total / 5

# Add average scores to table
dataTable.add_row(['Average', average_accuracy])

print(dataTable)

# ==== SVM ====
dataTable = PrettyTable(['', 'Linear SVM', 'Poly SVM', 'RBF SVM'])


def linear(X_train, y_train, X_test, y_test):
    kernel = SVC(kernel='linear')

    kernel.fit(X_train, y_train)

    return kernel.score(X_test, y_test)


def poly(X_train, y_train, X_test, y_test):
    kernel = SVC(kernel='poly')

    kernel.fit(X_train, y_train)

    return kernel.score(X_test, y_test)


def rbf(X_train, y_train, X_test, y_test):
    kernel = SVC(kernel='rbf')

    kernel.fit(X_train, y_train)

    return kernel.score(X_test, y_test)


# 5-Fold Cross Validation
k_data = KFold(n_splits=5, shuffle=True)

l_total = 0
p_total = 0
r_total = 0

for i, (train, test) in enumerate(k_data.split(X)):
    # Use Kfold indexes for train and test data.
    X_train = X[train]
    y_train = y[train]
    X_train = pd.DataFrame(X_train)
    X_train.columns = X_columns

    X_test = X[test]
    y_test = y[test]
    X_test = pd.DataFrame(X_test)
    X_test.columns = X_columns

    l_score = linear(X_train, y_train, X_test, y_test)
    p_score = poly(X_train, y_train, X_test, y_test)
    r_score = rbf(X_train, y_train, X_test, y_test)

    l_total = l_total + l_score
    p_total = p_total + p_score
    r_total = r_total + r_score

    dataTable.add_row(['Fold ' + str(i + 1), l_score, p_score, r_score])

# Calculate average scores.
l_average = l_total / 5
p_average = p_total / 5
r_average = r_total / 5

# Add average scores to table
dataTable.add_row(['Average', l_average, p_average, r_average])

print(dataTable)

# ==== Decision Tree ====
dataTable = PrettyTable(['', 'Decision Tree Accuracy'])

# 5-Fold Cross Validation
k_data = KFold(n_splits=5, shuffle=True)

accuracy_total = 0

for i, (train, test) in enumerate(k_data.split(X)):
    # Use Kfold indexes for train and test data.
    X_train = X[train]
    y_train = y[train]
    X_train = pd.DataFrame(X_train)
    X_train.columns = X_columns

    X_test = X[test]
    y_test = y[test]
    X_test = pd.DataFrame(X_test)
    X_test.columns = X_columns

    dtree = DecisionTreeClassifier()

    # Train data.
    dtree = dtree.fit(X_train, y_train)

    # Predict with test data for accuracy.
    pred = dtree.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    accuracy_total = accuracy_total + accuracy

    dataTable.add_row(['Fold ' + str(i + 1), accuracy])

# Calculate average scores.
average_accuracy = accuracy_total / 5

# Add average scores to table
dataTable.add_row(['Average', average_accuracy])

print(dataTable)

# ==== Neural Network ====
dataTable = PrettyTable(['', 'Neural Network Accuracy'])

# 5-Fold Cross Validation
k_data = KFold(n_splits=5, shuffle=True)

accuracy_total = 0
accuracy = [0, 0, 0, 0, 0]

for i, (train, test) in enumerate(k_data.split(X)):
    # Use Kfold indexes for train and test data.
    X_train = X[train]
    y_train = y[train]
    X_train = pd.DataFrame(X_train)
    X_train.columns = X_columns

    X_test = X[test]
    y_test = y[test]
    X_test = pd.DataFrame(X_test)
    X_test.columns = X_columns

    # Define neural network architecture.
    nn = MLPClassifier(solver='lbfgs', alpha=0.0001,
                       hidden_layer_sizes=30, random_state=1)

    # Train data.
    nn.fit(X_train, y_train)

    # Predict with test data for accuracy.
    pred = nn.predict(X_test)

    accuracy[i] = accuracy_score(y_test, pred)

# Calculate average scores.
average_accuracy = accuracy_total / 5

# Add average scores to table
dataTable.add_row(['Fold 1', accuracy[0]])
dataTable.add_row(['Fold 2', accuracy[1]])
dataTable.add_row(['Fold 3', accuracy[2]])
dataTable.add_row(['Fold 4', accuracy[3]])
dataTable.add_row(['Fold 5', accuracy[4]])
dataTable.add_row(['Average', (accuracy[0] + accuracy[1] + accuracy[2] +
                               accuracy[3] + accuracy[4]) / 5])

print(dataTable)
