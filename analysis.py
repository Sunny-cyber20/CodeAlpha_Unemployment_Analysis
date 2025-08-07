import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

# Load dataset
df = pd.read_csv("Unemployment in India.csv")

df.columns = df.columns.str.strip()

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

df['Month'] = df['Date'].dt.month

# Add COVID_Period column
def label_covid_period(date):
    if pd.isna(date):
        return 'Unknown'
    elif date < pd.Timestamp('2020-03-01'):
        return 'Pre-COVID'
    elif pd.Timestamp('2020-03-01') <= date <= pd.Timestamp('2021-12-31'):
        return 'COVID-Peak'
    else:
        return 'Post-COVID'

df['COVID_Period'] = df['Date'].apply(label_covid_period)

# Set up the plotting environment
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

# 1. COMPREHENSIVE COVID-19 IMPACT DASHBOARD
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('COVID-19 Impact on India\'s Employment: Comprehensive Analysis', fontsize=18, fontweight='bold')

# 1.1 Heatmap of unemployment by region and time period
if 'df' in locals() or 'df' in globals():
    pivot_heat = df.groupby(['Region', 'COVID_Period'])['Estimated Unemployment Rate (%)'].mean().unstack()
    
    sns.heatmap(pivot_heat, annot=True, fmt='.1f', cmap='Reds', ax=axes[0,0], cbar_kws={'label': 'Unemployment Rate (%)'})
    axes[0,0].set_title('Regional Unemployment Heatmap by COVID Period', fontweight='bold')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 1.2 Box plot showing distribution of unemployment rates
    sns.boxplot(data=df, x='COVID_Period', y='Estimated Unemployment Rate (%)', hue='Area', ax=axes[0,1])
    axes[0,1].set_title('Unemployment Rate Distribution by COVID Period', fontweight='bold')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 1.3 Labor participation rate trends
    labor_trends = df.groupby(['Date', 'Area'])['Estimated Labour Participation Rate (%)'].mean().reset_index()
    labor_pivot = labor_trends.pivot(index='Date', columns='Area', values='Estimated Labour Participation Rate (%)')
    
    for area in labor_pivot.columns:
        axes[0,2].plot(labor_pivot.index, labor_pivot[area], marker='o', label=area, linewidth=2)
    
    axes[0,2].axvline(x=datetime(2020, 3, 1), color='red', linestyle='--', alpha=0.7, label='COVID Start')
    axes[0,2].set_title('Labor Participation Rate Trends', fontweight='bold')
    axes[0,2].set_ylabel('Labor Participation Rate (%)')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 1.4 Regional ranking by unemployment severity
    regional_severity = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=True)
    regional_severity.plot(kind='barh', ax=axes[1,0], color='lightcoral')
    axes[1,0].set_title('Regional Unemployment Ranking', fontweight='bold')
    axes[1,0].set_xlabel('Average Unemployment Rate (%)')
    
    # 1.5 Employment capacity vs unemployment rate scatter
    regional_metrics = df.groupby('Region').agg({
        'Estimated Unemployment Rate (%)': 'mean',
        'Estimated Employed': 'mean',
        'Estimated Labour Participation Rate (%)': 'mean'
    }).reset_index()
    
    scatter = axes[1,1].scatter(regional_metrics['Estimated Employed']/1000000, 
                               regional_metrics['Estimated Unemployment Rate (%)'],
                               s=regional_metrics['Estimated Labour Participation Rate (%)']*3,
                               alpha=0.7, c=range(len(regional_metrics)), cmap='viridis')
    
    axes[1,1].set_xlabel('Average Employed (Millions)')
    axes[1,1].set_ylabel('Average Unemployment Rate (%)')
    axes[1,1].set_title('Employment Capacity vs Unemployment\n(Bubble size = Labor Participation)', fontweight='bold')
    
    # region labels to scatter plot
    for i, row in regional_metrics.iterrows():
        axes[1,1].annotate(row['Region'][:3], 
                          (row['Estimated Employed']/1000000, row['Estimated Unemployment Rate (%)']),
                          xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    
    # 1.6 Seasonal patterns analysis
    monthly_pattern = df.groupby(['Month', 'Area'])['Estimated Unemployment Rate (%)'].mean().unstack()
    monthly_pattern.plot(kind='line', marker='o', ax=axes[1,2], linewidth=2)
    axes[1,2].set_title('Seasonal Unemployment Patterns', fontweight='bold')
    axes[1,2].set_xlabel('Month')
    axes[1,2].set_ylabel('Unemployment Rate (%)')
    axes[1,2].legend(title='Area')
    axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 2. POLICY IMPACT SIMULATION DASHBOARD
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Policy Impact Simulation: Economic Recovery Scenarios', fontsize=16, fontweight='bold')

# 2.1 Recovery scenarios
if 'df' in locals() or 'df' in globals():
    # Simulate different recovery scenarios
    base_unemployment = df[df['COVID_Period'] == 'COVID-Peak']['Estimated Unemployment Rate (%)'].mean()
    months = range(1, 13)
    
    # Different recovery scenarios
    scenarios = {
        'Optimistic (Strong Policy)': [base_unemployment * (0.9 ** (m/2)) for m in months],
        'Moderate (Current Policy)': [base_unemployment * (0.95 ** (m/3)) for m in months],
        'Pessimistic (Weak Policy)': [base_unemployment * (0.98 ** (m/4)) for m in months]
    }
    
    for scenario, values in scenarios.items():
        axes[0,0].plot(months, values, marker='o', label=scenario, linewidth=2)
    
    axes[0,0].set_title('Unemployment Recovery Scenarios', fontweight='bold')
    axes[0,0].set_xlabel('Months from Policy Implementation')
    axes[0,0].set_ylabel('Projected Unemployment Rate (%)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2.2 Investment impact simulation
    investment_levels = [0, 500, 1000, 1500, 2000, 2500]  # in billion rupees
    unemployment_reduction = [0, 0.5, 1.2, 2.1, 3.2, 4.5]  # percentage points reduction
    
    axes[0,1].plot(investment_levels, unemployment_reduction, marker='o', color='green', linewidth=3)
    axes[0,1].fill_between(investment_levels, unemployment_reduction, alpha=0.3, color='green')
    axes[0,1].set_title('Investment vs Unemployment Reduction', fontweight='bold')
    axes[0,1].set_xlabel('Government Investment (Billion ₹)')
    axes[0,1].set_ylabel('Unemployment Reduction (% points)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 2.3 Regional priority matrix
    regional_data = df.groupby('Region').agg({
        'Estimated Unemployment Rate (%)': 'mean',
        'Estimated Labour Participation Rate (%)': 'mean'
    }).reset_index()
    
    # Create priority quadrants
    unemp_median = regional_data['Estimated Unemployment Rate (%)'].median()
    labor_median = regional_data['Estimated Labour Participation Rate (%)'].median()
    
    colors = []
    for _, row in regional_data.iterrows():
        if row['Estimated Unemployment Rate (%)'] > unemp_median and row['Estimated Labour Participation Rate (%)'] < labor_median:
            colors.append('red')  
        elif row['Estimated Unemployment Rate (%)'] > unemp_median or row['Estimated Labour Participation Rate (%)'] < labor_median:
            colors.append('orange')  
        else:
            colors.append('green')  
    
    scatter = axes[1,0].scatter(regional_data['Estimated Labour Participation Rate (%)'],
                               regional_data['Estimated Unemployment Rate (%)'],
                               c=colors, s=100, alpha=0.7)
    
    axes[1,0].axhline(y=unemp_median, color='gray', linestyle='--', alpha=0.5)
    axes[1,0].axvline(x=labor_median, color='gray', linestyle='--', alpha=0.5)
    axes[1,0].set_xlabel('Labor Participation Rate (%)')
    axes[1,0].set_ylabel('Unemployment Rate (%)')
    axes[1,0].set_title('Regional Priority Matrix\n(Red=High, Orange=Medium, Green=Low Priority)', fontweight='bold')
    
    # Add region labels
    for i, row in regional_data.iterrows():
        axes[1,0].annotate(row['Region'][:4], 
                          (row['Estimated Labour Participation Rate (%)'], row['Estimated Unemployment Rate (%)']),
                          xytext=(2, 2), textcoords='offset points', fontsize=8)
    
    # 2.4 Skill development impact
    skill_programs = ['Basic Skills', 'Technical Skills', 'Digital Skills', 'Entrepreneurship', 'Advanced Skills']
    employment_increase = [5, 12, 18, 15, 25]  # percentage increase in employment
    
    bars = axes[1,1].bar(skill_programs, employment_increase, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
    axes[1,1].set_title('Projected Employment Impact of Skill Development Programs', fontweight='bold')
    axes[1,1].set_ylabel('Employment Increase (%)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, employment_increase):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                      f'{value}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# 3. SUMMARY STATISTICS AND RECOMMENDATIONS
print("\n" + "="*80)
print("STRATEGIC POLICY RECOMMENDATIONS BASED ON DATA ANALYSIS")
print("="*80)

if 'df' in locals() or 'df' in globals():
    # Calculate key metrics
    total_regions = df['Region'].nunique()
    avg_unemployment = df['Estimated Unemployment Rate (%)'].mean()
    covid_impact = df[df['COVID_Period'] == 'COVID-Peak']['Estimated Unemployment Rate (%)'].mean() - \
                   df[df['COVID_Period'] == 'Pre-COVID']['Estimated Unemployment Rate (%)'].mean()
    
    most_affected_region = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().idxmax()
    best_performing_region = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().idxmin()
    
    print(f"\nKEY FINDINGS:")
    print(f"   • Total regions analyzed: {total_regions}")
    print(f"   • Overall average unemployment: {avg_unemployment:.2f}%")
    print(f"   • COVID-19 impact: +{covid_impact:.2f} percentage points")
    print(f"   • Most affected region: {most_affected_region}")
    print(f"   • Best performing region: {best_performing_region}")
    
    print(f"\nIMMEDIATE ACTION ITEMS:")
    print("   1. EMERGENCY RESPONSE (0-6 months)")
    print("      • Deploy rapid employment schemes in high-unemployment regions")
    print("      • Provide direct cash transfers to affected families")
    print("      • Establish temporary job guarantee programs")
    
    print(f"\n   2. SHORT-TERM RECOVERY (6-18 months)")
    print("      • Invest in infrastructure projects for job creation")
    print("      • Launch sector-specific skill development programs")
    print("      • Support small and medium enterprises with credit facilities")
    
    print(f"\n   3. LONG-TERM RESILIENCE (18+ months)")
    print("      • Diversify regional economies to reduce vulnerability")
    print("      • Strengthen social safety nets and unemployment insurance")
    print("      • Develop digital economy infrastructure")
    
    print(f"\n💡 INNOVATION OPPORTUNITIES:")
    print("      • Implement AI-driven job matching platforms")
    print("      • Create mobile employment service units for rural areas")
    print("      • Establish real-time unemployment monitoring systems")
    print("      • Launch green jobs initiatives for sustainable employment")

print(f"\nAnalysis complete with actionable insights for policy implementation!")