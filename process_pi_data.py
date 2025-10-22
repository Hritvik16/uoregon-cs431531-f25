import os
import re
import pandas as pd


OUTPUT_DIR = 'final_output' 
OUTPUT_CSV_FILE = 'pi_summary_data.csv'


SERIAL_PATTERN = re.compile(r'Time to calculate Pi serially.*is: (\d+\.\d+)')

PARALLEL_PATTERN = re.compile(r'Time to calculate Pi in // with \d+ steps is: (\d+\.\d+)')

MC_PATTERN = re.compile(r'Time to calculate Pi in // with \d+ guesses is: (\d+\.\d+)')


summary_data = []




for filename in os.listdir(OUTPUT_DIR):
    if filename.endswith('.out'):

        parts = filename.split('_')
        

        method = parts[2] 
        

        threads = int(parts[3].split('.')[0])
        

        file_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue


        serial_times = [float(t) for t in SERIAL_PATTERN.findall(content)]
        parallel_times = [float(t) for t in PARALLEL_PATTERN.findall(content)]
        mc_times = [float(t) for t in MC_PATTERN.findall(content)]


        if len(serial_times) != 20 or len(parallel_times) != 20 or len(mc_times) != 20:
            print(f"Warning: {filename} did not contain 20 runs for all methods. Found S:{len(serial_times)}, P:{len(parallel_times)}, MC:{len(mc_times)}")


        if serial_times:
            avg_serial = sum(serial_times) / len(serial_times)
        else:
            avg_serial = float('nan')
            
        if parallel_times:
            avg_parallel = sum(parallel_times) / len(parallel_times)
        else:
            avg_parallel = float('nan')

        if mc_times:
            avg_mc = sum(mc_times) / len(mc_times)
        else:
            avg_mc = float('nan')

        
        summary_data.append({
            'Method': method,
            'Threads': threads,
            'Avg_Time_Serial': avg_serial,
            'Avg_Time_Parallel': avg_parallel,
            'Avg_Time_MonteCarlo': avg_mc
        })


df = pd.DataFrame(summary_data)


df_sorted = df.sort_values(by=['Method', 'Threads'])


df_sorted.to_csv(OUTPUT_CSV_FILE, index=False)

print(f"\n--- Data Processing Complete ---")
print(f"Averaged data saved to: {OUTPUT_CSV_FILE}")
print("\nSummary Table:")
print(df_sorted.to_markdown(index=False))