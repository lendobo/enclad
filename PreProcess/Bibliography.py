import re
from collections import Counter
import matplotlib.pyplot as plt

# Bibliography data as text
bibliography = """
1. 2023
2. 2023
3. 2023
4. 2023
5. 2023
6. 2017
7. 2023
8. 2022
9. 2015
10. 2019
11. 2021
12. 2018
13. 2020
14. 2023
15. 2019
16. 2021
17. 2020
18. 2022
19. 2015
20. 2020
21. 2021
22. 2015
23. 2022
25. 2021
26. 2022
27. 2023
29. 2022
30. 2021
31. ?
32. 2021
33. 2022
34. 2022
35. 2021
36. 2020
37. 2020
38. 2016
39. 2019
40. 2018
41. 2020
42. 2021
43. 2017
44. 2020
45. 2021
46. ?
47. 2018
48. 2020
49. ?
50. 2022
51. 2022
52. 2022
53. ?
54. 2020
55. 2015
56. 2021
57. 2021
58. 2022
59. ?
60. 2022
61. 2023
62. 2023
63. 2018
64. 2019
65. 1996
66. 2019
67. 2020
68. 2020
69. 2020
70. 2020
71. 2017
72. 2022
73. 2023
74. 1982
76. 2018
77. 2021
78. 2022
79. 2023
80. 1999
81. 2009
82. 2021
83. 2021
84. 2022
85. 1990
86. 1998
87. 2022
88. 2021
89. 2021
90. 2020
"""

# Extract publication years
years = re.findall(r'\d{4}', bibliography)

# Convert to integers and filter out years before 2010
years = [int(year) for year in years if int(year) > 2010]

# Count the number of publications per year
year_counts = Counter(years)

# Plotting the histogram
plt.figure(figsize=(12, 6))
plt.bar(year_counts.keys(), year_counts.values(), color='#ffb66c')
plt.xlabel('Publication Year', fontsize=16)
plt.ylabel('Number of Publications', fontsize=16)
plt.title('Number of Publications Per Year', fontsize=20)
plt.xticks(sorted(year_counts.keys()))
plt.yticks(range(0, max(year_counts.values()) + 1))

# increase font size of ticks
plt.tick_params(axis='both', which='major', labelsize=14)

# increase font size of title

# # Annotate the bars
# for x, y in year_counts.items():
#     plt.text(x, y, str(y), ha='center', va='bottom')

plt.show()
