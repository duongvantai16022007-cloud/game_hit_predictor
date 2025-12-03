import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('dataset.csv')
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df[df['Hit_Target'] == 1], x='Name_Length', fill=True, color='green', label='HIT Games', alpha=0.3)
sns.kdeplot(data=df[df['Hit_Target'] == 0], x='Name_Length', fill=True, color='red', label='FLOP Games', alpha=0.3)
plt.title('NAME LENGTH DISTRIBUTION', fontsize=15)
plt.xlabel('Length', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlim(0, 80)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()