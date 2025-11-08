import os
import re
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np


DATA_DIRS = {
    '1e8 steps': 'output10_8',
    '1e9 steps': 'increased_output10_9',
    '2e9 steps': 'increased_output2_10_9'
}
OUTPUT_CSV_FILE = 'pi_summary_combined.csv'
PI_TRUE = 3.1415926535


TIME_PATTERN = r'(\d+\.\d+(?:[Ee][+-]?\d+)?|\d+)'
PI_CALC_PATTERN = r'(\d+\.\d+)' 
STEPS_PATTERN = r'with (\d+) steps/guesses'

SERIAL_TIME_LINE = re.compile(r'Time to calculate Pi serially.*is: ' + TIME_PATTERN)
PARALLEL_TIME_LINE = re.compile(r'Time to calculate Pi in // with \d+ steps is: ' + TIME_PATTERN)
MC_TIME_LINE = re.compile(r'Time to calculate Pi in // with \d+ guesses is: ' + TIME_PATTERN)
PI_VALUE_LINE = re.compile(r'Pi is ' + PI_CALC_PATTERN)
STEPS_COUNT_LINE = re.compile(r'Starting serial experiment for \d+ runs ' + STEPS_PATTERN)

aggregated_full_data = {}
all_accuracy_data = [] 


def safe_avg(times):
    """Calculates average time, handling empty lists gracefully."""
    return sum(times) / len(times) if times else float('nan')

def calculate_absolute_error(pi_values):
    """Calculates the absolute error against the true value of Pi."""
    return [abs(float(pi) - PI_TRUE) for pi in pi_values]


for dataset_name, dir_name in DATA_DIRS.items():
    
    if not os.path.isdir(dir_name):
        print(f"Directory not found: {dir_name}. Skipping.")
        continue

    aggregated_data = {}
    problem_steps = None 

    print(f"\n--- Processing Directory: {dir_name} ({dataset_name}) ---")
    
    for filename in os.listdir(dir_name):
        if not filename.endswith('.out'):
            continue
        
         
        parts = filename.split('_')
        method_type = parts[2]
        try:
            threads = int(parts[3].split('.')[0])
        except (IndexError, ValueError):
             print(f"Skipping malformed filename: {filename}")
             continue
        
        file_path = os.path.join(dir_name, filename)
        
        current_times = {'Serial': [], 'Parallel': [], 'MC': []}
        current_pi_values = {'Serial': [], 'Parallel': [], 'MC': []}

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            continue
        
         
        last_detected_method = None
        for line in lines:
            
             
            if problem_steps is None:
                steps_match = STEPS_COUNT_LINE.search(line)
                if steps_match:
                    problem_steps = int(steps_match.group(1))

            if SERIAL_TIME_LINE.search(line):
                time_match = SERIAL_TIME_LINE.search(line)
                current_times['Serial'].append(float(time_match.group(1)))
                last_detected_method = 'Serial'
            elif PARALLEL_TIME_LINE.search(line):
                time_match = PARALLEL_TIME_LINE.search(line)
                current_times['Parallel'].append(float(time_match.group(1)))
                last_detected_method = 'Parallel'
            elif MC_TIME_LINE.search(line):
                time_match = MC_TIME_LINE.search(line)
                current_times['MC'].append(float(time_match.group(1)))
                last_detected_method = 'MC'
            
            elif last_detected_method:
                pi_match = PI_VALUE_LINE.search(line)
                if pi_match:
                    current_pi_values[last_detected_method].append(pi_match.group(1))
                last_detected_method = None 

        
         
        if threads not in aggregated_data:
            aggregated_data[threads] = {
                'Serial': [], 'Critical': [], 'Atomic': [], 'MonteCarlo': [],
                'Error_Serial': [], 'Error_Critical': [], 'Error_Atomic': [], 'Error_MonteCarlo': []
            }
        
        error_serial = calculate_absolute_error(current_pi_values['Serial'])
        error_parallel = calculate_absolute_error(current_pi_values['Parallel'])
        error_mc = calculate_absolute_error(current_pi_values['MC'])

         
        if not aggregated_data[threads]['Serial']:
             aggregated_data[threads]['Serial'].extend(current_times['Serial'])
             aggregated_data[threads]['MonteCarlo'].extend(current_times['MC'])
             aggregated_data[threads]['Error_Serial'].extend(error_serial)
             aggregated_data[threads]['Error_MonteCarlo'].extend(error_mc)

         
        if method_type == 'critical':
             aggregated_data[threads]['Parallel_Critical'] = current_times['Parallel']
             aggregated_data[threads]['Error_Critical'] = error_parallel
        elif method_type == 'atomic':
             aggregated_data[threads]['Parallel_Atomic'] = current_times['Parallel']
             aggregated_data[threads]['Error_Atomic'] = error_parallel


     
    final_data = []
    for threads in sorted(aggregated_data.keys()):
        data = aggregated_data[threads]
        
         
        avg_serial = safe_avg(data['Serial'])
        avg_critical = safe_avg(data.get('Parallel_Critical', []))
        avg_atomic = safe_avg(data.get('Parallel_Atomic', []))
        avg_mc = safe_avg(data.get('MonteCarlo', []))
        
         
        avg_error_serial = safe_avg(data.get('Error_Serial', []))
        avg_error_critical = safe_avg(data.get('Error_Critical', []))
        avg_error_atomic = safe_avg(data.get('Error_Atomic', []))
        avg_error_mc = safe_avg(data.get('Error_MonteCarlo', []))
        
        final_data.append({
            'Threads': threads,
            'N': problem_steps,
            'Serial_Time': avg_serial,
            'Parallel_Critical_Time': avg_critical,
            'Parallel_Atomic_Time': avg_atomic,
            'MonteCarlo_Time': avg_mc,
            'Serial_Error': avg_error_serial,
            'Parallel_Critical_Error': avg_error_critical,
            'Parallel_Atomic_Error': avg_error_atomic,
            'MonteCarlo_Error': avg_error_mc
        })

    df = pd.DataFrame(final_data)
    
     
    aggregated_full_data[dataset_name] = df
    
     
    if problem_steps is not None:
        rep_row = df[df['Threads'] == 128].iloc[0] if 128 in df['Threads'].values else df.iloc[-1]
        all_accuracy_data.append({
            'N': problem_steps,
            'Serial_Error': rep_row['Serial_Error'],
            'Parallel_Critical_Error': rep_row['Parallel_Critical_Error'],
            'MonteCarlo_Error': rep_row['MonteCarlo_Error']
        })

    
    threads = df['Threads']
    
    plt.figure(figsize=(10, 6))

    y_data = {
        'Serial': df['Serial_Time'],
        'Parallel Integration (Critical)': df['Parallel_Critical_Time'],
        'Parallel Integration (Atomic)': df['Parallel_Atomic_Time'],
        'Monte Carlo': df['MonteCarlo_Time']
    }

     
    for label, data in y_data.items():
        plt.plot(threads, data, label=label, marker='o', linestyle='--' if label == 'Serial' else '-')
        
        if not data.empty and threads.iloc[-1] == 128 and not pd.isna(data.iloc[-1]):
            last_time = data.iloc[-1]
            plt.annotate(
                f'{last_time:.4f}s', 
                (threads.iloc[-1], last_time), 
                textcoords="offset points", 
                xytext=(0, 10 if last_time > 0.01 else -15),
                ha='center',
                fontsize=8
            )

    plt.title(f'Execution Time vs. Thread Count ({dataset_name})')
    plt.xlabel('Number of Threads (Log Scale X)')
    plt.ylabel('Average Execution Time (s) [Log Scale Y]')
    plt.xscale('log', base=2)
     
    plt.yscale('log') 
    plt.xticks(threads, labels=[str(t) for t in threads]) 
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plot_filename = f'time_vs_threads_{dir_name}.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Generated plot: {plot_filename}")


if not aggregated_full_data:
    print("\n--- FATAL ERROR: No data was successfully aggregated. Check your DATA_DIRS paths. ---")
    exit() 

 
combined_df = pd.concat(list(aggregated_full_data.values()), ignore_index=True)

column_order = [
    'Threads', 'N', 
    'Serial_Time', 'Parallel_Critical_Time', 'Parallel_Atomic_Time', 'MonteCarlo_Time',
    'Serial_Error', 'Parallel_Critical_Error', 'Parallel_Atomic_Error', 'MonteCarlo_Error'
]
combined_df = combined_df[column_order]

combined_df.to_csv(OUTPUT_CSV_FILE, index=False, float_format='%.10e') 
print(f"\n--- Data Processing Complete ---")
print(f"Combined summary data saved to: {OUTPUT_CSV_FILE}")

 
accuracy_df = pd.DataFrame(all_accuracy_data).sort_values(by='N')

 
ERROR_FLOOR = 1e-18
accuracy_df['Serial_Error_Plot'] = np.where(accuracy_df['Serial_Error'] > 0, accuracy_df['Serial_Error'], ERROR_FLOOR)
accuracy_df['Parallel_Critical_Error_Plot'] = np.where(accuracy_df['Parallel_Critical_Error'] > 0, accuracy_df['Parallel_Critical_Error'], ERROR_FLOOR)
accuracy_df['MonteCarlo_Error_Plot'] = np.where(accuracy_df['MonteCarlo_Error'] > 0, accuracy_df['MonteCarlo_Error'], ERROR_FLOOR)


plt.figure(figsize=(10, 6))

 
plt.plot(accuracy_df['N'], accuracy_df['Serial_Error_Plot'], label='Integration Error (Serial Baseline)', marker='o', linestyle='--')
plt.plot(accuracy_df['N'], accuracy_df['Parallel_Critical_Error_Plot'], label='Integration Error (Parallel)', marker='o')
plt.plot(accuracy_df['N'], accuracy_df['MonteCarlo_Error_Plot'], label='Monte Carlo Error', marker='o')

plt.title('Accuracy Convergence: Error vs. Problem Size (Steps/Guesses)')
plt.xlabel('Number of Steps/Guesses (N) - Log Scale')
plt.ylabel('Average Absolute Error (Log Scale)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", ls="--")
plot_filename = 'accuracy_vs_steps.png'
plt.savefig(plot_filename)
plt.close()
print(f"Generated plot: {plot_filename}")

print("\nAll plots generated successfully.")